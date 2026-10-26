"""Unit tests for jitrl.estimator: Algorithm 1 steps 2-5 (Eq. 4-7, 25)."""

from __future__ import annotations

import math
import random

import pytest

from jitrl.estimator import (
    EPSILON,
    aggregate_by_action,
    estimate_values,
    summarize_action,
)
from jitrl.models import Neighbor


def make_neighbor(
    entry_id: int,
    action_key: str,
    discounted_return: float,
    *,
    similarity: float = 0.9,
    weight: float = 1.0,
    reward: float = 0.0,
) -> Neighbor:
    """Build a Neighbor carrying exactly the fields the estimator reads."""
    return Neighbor(
        entry_id=entry_id,
        similarity=similarity,
        action_key=action_key,
        action_text=action_key.upper(),
        discounted_return=discounted_return,
        similarity_weight=weight,
        reward=reward,
    )


def weighted_sample() -> list[Neighbor]:
    """Three neighbours with distinct weights, covering actions a and b."""
    return [
        make_neighbor(1, "a", 1.0, similarity=0.9, weight=9.0, reward=1.0),
        make_neighbor(2, "a", 0.0, similarity=0.8, weight=8.0, reward=0.0),
        make_neighbor(3, "b", -1.0, similarity=0.7, weight=7.0, reward=-1.0),
    ]


def test_state_value_matches_eq_4() -> None:
    estimate = estimate_values(weighted_sample(), ["a", "b"], exploration_rate=0.0)
    # V = (9*1.0 + 8*0.0 + 7*(-1.0)) / 24 = 2/24 = 1/12.
    assert estimate.v_hat == pytest.approx(1.0 / 12.0)
    assert estimate.neighbor_count == 3
    assert estimate.mean_similarity == pytest.approx(0.8)


def test_known_action_values_match_eq_5() -> None:
    estimate = estimate_values(weighted_sample(), ["a", "b"], exploration_rate=0.0)
    # Q(s, a) = 9/17 because only the two 'a' rows count, Q(s, b) = -1 exactly.
    assert estimate.q_hat["a"] == pytest.approx(9.0 / 17.0)
    assert estimate.q_hat["b"] == pytest.approx(-1.0)
    assert estimate.action_counts["a"] == 2
    assert estimate.action_counts["b"] == 1
    assert estimate.explored == set()


def test_advantage_matches_eq_7() -> None:
    estimate = estimate_values(weighted_sample(), ["a", "b"], exploration_rate=0.0)
    assert estimate.advantage["a"] == pytest.approx(9.0 / 17.0 - 1.0 / 12.0)
    assert estimate.advantage["b"] == pytest.approx(-1.0 - 1.0 / 12.0)


def test_normalized_advantage_matches_eq_25() -> None:
    estimate = estimate_values(weighted_sample(), ["a", "b"], exploration_rate=0.0)
    max_abs = 13.0 / 12.0
    assert estimate.advantage_norm["a"] == pytest.approx(
        (9.0 / 17.0 - 1.0 / 12.0) / (max_abs + EPSILON)
    )
    assert estimate.advantage_norm["b"] == pytest.approx(-max_abs / (max_abs + EPSILON))
    # The largest magnitude saturates at -1 and never exceeds it.
    assert -1.0 < estimate.advantage_norm["b"] <= -1.0 + 1e-6


def test_raw_advantages_are_copied_when_normalization_is_off() -> None:
    estimate = estimate_values(
        weighted_sample(), ["a", "b"], exploration_rate=0.0, normalize=False
    )
    assert estimate.advantage_norm == estimate.advantage
    assert estimate.advantage_norm is not estimate.advantage
    assert estimate.advantage_norm["a"] == pytest.approx(9.0 / 17.0 - 1.0 / 12.0)


def test_non_positive_weights_fall_back_to_equal_average() -> None:
    neighbors = [
        make_neighbor(1, "a", 1.0, weight=0.0),
        make_neighbor(2, "a", 3.0, weight=-2.0),
        make_neighbor(3, "b", 5.0, weight=0.0),
    ]
    estimate = estimate_values(neighbors, ["a", "b"], exploration_rate=0.0)
    assert estimate.v_hat == pytest.approx(3.0)
    assert estimate.q_hat["a"] == pytest.approx(2.0)
    assert estimate.q_hat["b"] == pytest.approx(5.0)


def test_uniform_weights_are_equivalent_to_equal_average() -> None:
    weighted = [
        make_neighbor(1, "a", 1.0, weight=1.0),
        make_neighbor(2, "a", 3.0, weight=1.0),
        make_neighbor(3, "b", 5.0, weight=1.0),
    ]
    plain = [
        make_neighbor(n.entry_id, n.action_key, n.discounted_return, weight=0.0)
        for n in weighted
    ]
    weighted_estimate = estimate_values(weighted, ["a", "b"], exploration_rate=0.0)
    plain_estimate = estimate_values(plain, ["a", "b"], exploration_rate=0.0)
    assert weighted_estimate.v_hat == pytest.approx(plain_estimate.v_hat)
    assert weighted_estimate.q_hat == pytest.approx(plain_estimate.q_hat)
    assert weighted_estimate.advantage == pytest.approx(plain_estimate.advantage)


def test_unseen_action_gets_optimism_bonus_at_injected_rate() -> None:
    neighbors = [
        make_neighbor(1, "known", 1.0, similarity=0.8),
        make_neighbor(2, "known", 3.0, similarity=0.8),
    ]
    rng = random.Random(0)
    hits = 0
    draws = 200
    for _ in range(draws):
        estimate = estimate_values(
            neighbors,
            ["known", "novel"],
            rng=rng,
            exploration_rate=0.3,
            ucb_alpha=5.0,
        )
        assert estimate.q_hat["known"] == pytest.approx(2.0)
        assert estimate.action_counts["novel"] == 0
        if "novel" in estimate.explored:
            hits += 1
            # Eq. 6: V(s) + alpha / |N(s)|.
            assert estimate.q_hat["novel"] == pytest.approx(estimate.v_hat + 5.0 / 2)
            assert estimate.advantage_norm["novel"] == pytest.approx(1.0, abs=1e-6)
        else:
            assert estimate.q_hat["novel"] == 0.0
    assert hits > 0
    assert 0.2 <= hits / draws <= 0.4


def test_empty_neighbourhood_is_all_zero_without_division_error() -> None:
    estimate = estimate_values(
        [], ["x", "y"], exploration_rate=1.0, rng=random.Random(0)
    )
    assert estimate.neighbor_count == 0
    assert estimate.v_hat == 0.0
    assert estimate.mean_similarity == 0.0
    assert estimate.q_hat == {"x": 0.0, "y": 0.0}
    assert estimate.advantage == {"x": 0.0, "y": 0.0}
    assert estimate.advantage_norm == {"x": 0.0, "y": 0.0}
    assert estimate.action_counts == {"x": 0, "y": 0}
    assert estimate.explored == set()
    assert all(math.isfinite(value) for value in estimate.advantage_norm.values())


def test_all_tied_advantages_normalize_to_zero_not_nan() -> None:
    neighbors = [
        make_neighbor(1, "a", 1.0, similarity=0.8, weight=1.0),
        make_neighbor(2, "b", 1.0, similarity=0.8, weight=1.0),
    ]
    estimate = estimate_values(neighbors, ["a", "b"], exploration_rate=0.0)
    assert estimate.advantage["a"] == 0.0
    assert estimate.advantage["b"] == 0.0
    assert estimate.advantage_norm == {"a": 0.0, "b": 0.0}
    assert not any(math.isnan(value) for value in estimate.advantage_norm.values())


def test_min_count_turns_a_thin_action_into_an_unknown_action() -> None:
    neighbors = [make_neighbor(1, "seen", 2.0)]
    unknown = estimate_values(neighbors, ["seen"], min_count=2, exploration_rate=0.0)
    assert unknown.action_counts["seen"] == 1
    assert unknown.q_hat["seen"] == 0.0
    assert unknown.explored == set()

    known = estimate_values(neighbors, ["seen"], min_count=1, exploration_rate=0.0)
    assert known.q_hat["seen"] == pytest.approx(2.0)

    # min_count is clamped to one, so an action without support stays unknown.
    clamped = estimate_values([], ["seen"], min_count=0, exploration_rate=1.0)
    assert clamped.q_hat["seen"] == 0.0


def test_empty_candidate_set_keeps_state_statistics() -> None:
    estimate = estimate_values(weighted_sample(), [], exploration_rate=0.0)
    assert estimate.q_hat == {}
    assert estimate.advantage == {}
    assert estimate.advantage_norm == {}
    assert estimate.action_counts == {}
    assert estimate.explored == set()
    assert estimate.neighbor_count == 3
    assert estimate.v_hat == pytest.approx(1.0 / 12.0)
    assert estimate.mean_similarity == pytest.approx(0.8)


def test_candidate_deduplication_preserves_first_seen_order() -> None:
    estimate = estimate_values(
        weighted_sample(), ["b", "a", "b", "c"], exploration_rate=0.0
    )
    assert list(estimate.q_hat) == ["b", "a", "c"]
    assert list(estimate.advantage) == ["b", "a", "c"]
    assert list(estimate.advantage_norm) == ["b", "a", "c"]
    assert list(estimate.action_counts) == ["b", "a", "c"]
    assert estimate.q_hat["c"] == 0.0


def test_fixed_seed_reproduces_identical_estimates() -> None:
    neighbors = [
        make_neighbor(1, "x", 1.0, similarity=0.9),
        make_neighbor(2, "y", -1.0, similarity=0.5),
    ]
    first = estimate_values(neighbors, ["x", "z"], rng=random.Random(11))
    second = estimate_values(neighbors, ["x", "z"], rng=random.Random(11))
    assert first.q_hat == second.q_hat
    assert first.advantage_norm == second.advantage_norm
    assert first.explored == second.explored


def test_aggregate_by_action_preserves_retrieval_order() -> None:
    neighbors = [
        make_neighbor(1, "a", 1.0),
        make_neighbor(2, "b", 2.0),
        make_neighbor(3, "a", 3.0),
    ]
    grouped = aggregate_by_action(neighbors)
    assert list(grouped) == ["a", "b"]
    assert [neighbor.entry_id for neighbor in grouped["a"]] == [1, 3]
    assert [neighbor.entry_id for neighbor in grouped["b"]] == [2]


def test_summarize_action_reports_evidence() -> None:
    neighbors = [
        make_neighbor(1, "a", 1.0, similarity=0.9, reward=1.0),
        make_neighbor(2, "a", -1.0, similarity=0.8, reward=0.0),
        make_neighbor(3, "a", 2.0, similarity=0.7, reward=1.0),
    ]
    summary = summarize_action(neighbors)
    assert summary["count"] == 3
    assert summary["mean_return"] == pytest.approx(2.0 / 3.0)
    assert summary["mean_reward"] == pytest.approx(2.0 / 3.0)
    assert summary["positive_rate"] == pytest.approx(2.0 / 3.0)
    assert summary["mean_similarity"] == pytest.approx(0.8)


def test_summarize_action_handles_empty_input() -> None:
    summary = summarize_action([])
    assert summary["count"] == 0
    assert summary["mean_return"] == 0.0
    assert summary["mean_reward"] == 0.0
    assert summary["positive_rate"] == 0.0
    assert summary["mean_similarity"] == 0.0
