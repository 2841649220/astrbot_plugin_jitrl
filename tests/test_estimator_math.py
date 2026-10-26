"""Property tests for the paper math in jitrl.estimator (Eq. 4-7 and Eq. 25).

These tests are independent of tests/test_estimator.py.  They pin the
definitions the module documents:

  Eq. 4  V(s) is the similarity-weight (kernel-weight) mean of the neighbour
         returns, degrading to the unweighted mean when no positive weight
         exists;
  Eq. 5  Q(s, a) is the same weighted mean restricted to N(s, a), with a
         per-group unweighted fallback;
  Eq. 6  for an unseen action: with probability lambda Q = V(s) + alpha/|N(s)|,
         otherwise Q = 0; never when the neighbourhood is empty;
  Eq. 7  A(s, a) = Q(s, a) - V(s);
  Eq. 25 A~ = A / (max|A| + eps) in (-1, 1], with all-ties -> zeros (no NaN).

Random datasets are generated from a fixed rng seed, so any failure is
reproducible.  The expected values are recomputed here from first principles
instead of calling the implementation's helpers.
"""

from __future__ import annotations

import math
import random

import pytest

from jitrl.estimator import EPSILON, estimate_values
from jitrl.models import Neighbor


def _neighbor(
    entry_id: int,
    action_key: str,
    discounted_return: float,
    *,
    similarity: float = 0.8,
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
        state_text="",
        scope="s1",
        created_at=0.0,
    )


def _random_neighbors(
    rng: random.Random, count: int, actions: tuple[str, ...] = ("a", "b", "c", "d")
) -> list[Neighbor]:
    """Generate neighbours whose weights and returns are deliberately mixed."""
    neighbours: list[Neighbor] = []
    for index in range(count):
        positive = round(rng.uniform(0.01, 5.0), 6)
        weight = rng.choice((positive, 0.0, -round(rng.uniform(0.01, 5.0), 6)))
        neighbours.append(
            _neighbor(
                index + 1,
                rng.choice(actions),
                round(rng.uniform(-4.0, 4.0), 6),
                similarity=round(rng.uniform(0.05, 1.0), 6),
                weight=weight,
                reward=round(rng.uniform(-1.0, 1.0), 6),
            )
        )
    return neighbours


def _grouped(neighbours: list[Neighbor]) -> dict[str, list[Neighbor]]:
    """Independent re-implementation of the action grouping."""
    grouped: dict[str, list[Neighbor]] = {}
    for neighbour in neighbours:
        grouped.setdefault(neighbour.action_key, []).append(neighbour)
    return grouped


def _weighted_mean(neighbours: list[Neighbor]) -> float:
    """Eq. 4/5 definition: positive-weight mean, unweighted fallback."""
    positives = [
        neighbour for neighbour in neighbours if neighbour.similarity_weight > 0.0
    ]
    if positives:
        total = sum(neighbour.similarity_weight for neighbour in positives)
        return (
            sum(
                neighbour.similarity_weight * neighbour.discounted_return
                for neighbour in positives
            )
            / total
        )
    if not neighbours:
        return 0.0
    return sum(neighbour.discounted_return for neighbour in neighbours) / len(
        neighbours
    )


# ---------------------------------------------------------------------------
# Eq. 4: state value
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_eq4_state_value_is_the_weighted_mean(seed: int) -> None:
    """V(s) must equal the positive-weight return mean of the neighbourhood."""
    rng = random.Random(seed)
    neighbours = _random_neighbors(rng, rng.randint(1, 30))
    estimate = estimate_values(neighbours, [], exploration_rate=0.0)
    assert estimate.neighbor_count == len(neighbours)
    assert estimate.v_hat == pytest.approx(_weighted_mean(neighbours), abs=1e-12)
    expected_mean = sum(item.similarity for item in neighbours) / len(neighbours)
    assert estimate.mean_similarity == pytest.approx(expected_mean, abs=1e-12)


def test_eq4_unweighted_fallback_when_no_positive_weight_exists() -> None:
    """Without any positive kernel weight V(s) is the plain mean of returns."""
    neighbours = [
        _neighbor(1, "a", 1.0, weight=0.0),
        _neighbor(2, "a", 3.0, weight=-2.0),
        _neighbor(3, "b", 5.0, weight=-0.5),
    ]
    estimate = estimate_values(neighbours, ["a", "b"], exploration_rate=0.0)
    assert estimate.v_hat == pytest.approx(3.0)
    assert estimate.q_hat["a"] == pytest.approx(2.0)
    assert estimate.q_hat["b"] == pytest.approx(5.0)


def test_eq4_ignores_non_positive_weight_rows_when_a_positive_one_exists() -> None:
    """A zero/negative-weight row contributes nothing to the weighted mean."""
    neighbours = [
        _neighbor(1, "a", 4.0, weight=2.0),
        _neighbor(2, "a", 100.0, weight=0.0),
        _neighbor(3, "b", -100.0, weight=-1.0),
    ]
    estimate = estimate_values(neighbours, ["a", "b"], exploration_rate=0.0)
    assert estimate.v_hat == pytest.approx(4.0)
    assert estimate.q_hat["a"] == pytest.approx(4.0)
    # This group has no positive weight at all, so it uses the plain mean.
    assert estimate.q_hat["b"] == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# Eq. 5: known-action values
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_eq5_known_action_values_are_group_weighted_means(seed: int) -> None:
    """Q(s, a) must be the weighted mean of exactly the rows that took a."""
    rng = random.Random(seed)
    neighbours = _random_neighbors(rng, rng.randint(2, 30))
    grouped = _grouped(neighbours)
    estimate = estimate_values(neighbours, list(grouped), exploration_rate=0.0)
    for key, group in grouped.items():
        assert estimate.q_hat[key] == pytest.approx(_weighted_mean(group), abs=1e-12)
        assert estimate.action_counts[key] == len(group)
    assert estimate.explored == set()


def test_eq5_group_without_positive_weight_falls_back_to_plain_mean() -> None:
    """The unweighted fallback is decided per action group, not globally."""
    neighbours = [
        _neighbor(1, "a", 2.0, weight=0.0),
        _neighbor(2, "a", 6.0, weight=-1.0),
        _neighbor(3, "b", 1.0, weight=2.0),
    ]
    estimate = estimate_values(neighbours, ["a", "b"], exploration_rate=0.0)
    assert estimate.q_hat["a"] == pytest.approx(4.0)
    assert estimate.q_hat["b"] == pytest.approx(1.0)
    assert estimate.v_hat == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Eq. 6: optimism for unseen actions
# ---------------------------------------------------------------------------
def test_eq6_lambda_zero_gives_unseen_actions_a_zero_value() -> None:
    """With lambda = 0 the exploration branch is never taken."""
    neighbours = [
        _neighbor(1, "seen", 2.0, weight=1.0, similarity=0.8),
        _neighbor(2, "seen", 4.0, weight=1.0, similarity=0.6),
    ]
    estimate = estimate_values(
        neighbours,
        ["seen", "novel"],
        exploration_rate=0.0,
        rng=random.Random(0),
        exploration_estimate="sample",
    )
    assert estimate.q_hat["novel"] == 0.0
    assert estimate.explored == set()
    assert estimate.advantage["novel"] == pytest.approx(-estimate.v_hat)
    assert estimate.advantage_norm["novel"] == pytest.approx(-1.0, abs=1e-6)


def test_eq6_lambda_one_always_grants_the_optimism_bonus() -> None:
    """With lambda = 1 every unseen action gets V(s) + alpha/|N(s)|."""
    neighbours = [
        _neighbor(1, "seen", 2.0, weight=1.0, similarity=0.8),
        _neighbor(2, "seen", 2.0, weight=1.0, similarity=0.6),
    ]
    estimate = estimate_values(
        neighbours,
        ["seen", "novel"],
        exploration_rate=1.0,
        ucb_alpha=2.5,
        rng=random.Random(1),
    )
    assert estimate.v_hat == pytest.approx(2.0)
    assert estimate.q_hat["novel"] == pytest.approx(estimate.v_hat + 2.5 / 2)
    assert estimate.advantage["novel"] == pytest.approx(2.5 / 2)
    assert estimate.advantage["novel"] > 0.0
    assert estimate.explored == {"novel"}
    assert estimate.advantage_norm["novel"] == pytest.approx(1.0, abs=1e-6)


def test_eq6_bonus_uses_the_neighbourhood_size_for_every_unseen_action() -> None:
    """alpha/|N(s)| is shared by all unseen candidates of one query state."""
    rng = random.Random(4)
    neighbours = _random_neighbors(rng, 4)
    estimate = estimate_values(
        neighbours,
        ["u1", "u2"],
        exploration_rate=1.0,
        ucb_alpha=2.0,
        rng=random.Random(0),
    )
    assert estimate.q_hat["u1"] == pytest.approx(estimate.v_hat + 2.0 / 4)
    assert estimate.q_hat["u2"] == pytest.approx(estimate.v_hat + 2.0 / 4)
    assert estimate.explored == {"u1", "u2"}


def test_eq6_known_actions_never_receive_the_bonus_even_at_lambda_one() -> None:
    """A supported action keeps its Eq. 5 value even when lambda = 1."""
    neighbours = [
        _neighbor(1, "known", 1.0, weight=1.0),
        _neighbor(2, "known", 1.0, weight=1.0),
    ]
    estimate = estimate_values(
        neighbours,
        ["known"],
        exploration_rate=1.0,
        ucb_alpha=100.0,
        rng=random.Random(0),
    )
    assert estimate.q_hat["known"] == pytest.approx(1.0)
    assert estimate.explored == set()


def test_eq6_sample_mode_reproduces_the_paper_draw_frequency() -> None:
    """exploration_estimate="sample" is the paper's Bernoulli branch."""
    neighbours = [_neighbor(1, "known", 1.0, weight=1.0)]
    rate = 0.3
    draws = 400
    hits = 0
    rng = random.Random(17)
    for _ in range(draws):
        estimate = estimate_values(
            neighbours,
            ["known", "novel"],
            exploration_rate=rate,
            rng=rng,
            exploration_estimate="sample",
        )
        if "novel" in estimate.explored:
            hits += 1
            assert estimate.q_hat["novel"] == pytest.approx(
                estimate.v_hat + 5.0 / len(neighbours)
            )
        else:
            assert estimate.q_hat["novel"] == 0.0
    assert 0.15 * draws <= hits <= 0.45 * draws


def test_eq6_expected_mode_is_the_closed_form_expectation() -> None:
    """Expected mode: A(s, a) = lambda * alpha / |N(s)| for unseen actions."""
    neighbours = [
        _neighbor(1, "known", 4.0, weight=1.0),
        _neighbor(2, "known", 2.0, weight=1.0),
    ]
    estimate = estimate_values(
        neighbours,
        ["known", "novel"],
        exploration_rate=0.3,
        ucb_alpha=5.0,
    )
    bonus = 0.3 * 5.0 / len(neighbours)
    assert estimate.q_hat["novel"] == pytest.approx(estimate.v_hat + bonus)
    assert estimate.advantage["novel"] == pytest.approx(bonus)
    assert estimate.explored == {"novel"}
    # The neutral baseline is Q = V, not the paper's Q = 0: an untried action is
    # never punished with -V just for being untried.
    assert estimate.advantage["novel"] > 0.0
    # Deterministic: no RNG is consumed, so repeated calls agree exactly.
    again = estimate_values(
        neighbours, ["known", "novel"], exploration_rate=0.3, ucb_alpha=5.0
    )
    assert again.q_hat == estimate.q_hat
    assert again.explored == estimate.explored


def test_eq6_expected_mode_scales_with_lambda_and_neighbourhood_size() -> None:
    """A_unseen = lambda * alpha / |N| is monotone in lambda and shrinks with |N|."""
    for size in (1, 4, 10):
        neighbours = [
            _neighbor(index + 1, f"known-{index}", 1.0, weight=1.0)
            for index in range(size)
        ]
        for rate in (0.0, 0.5, 1.0):
            estimate = estimate_values(
                neighbours,
                ["novel"],
                exploration_rate=rate,
                ucb_alpha=5.0,
            )
            assert estimate.advantage["novel"] == pytest.approx(rate * 5.0 / size)
            assert ("novel" in estimate.explored) is (rate > 0.0)


# ---------------------------------------------------------------------------
# Eq. 7: advantage
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_eq7_advantage_is_q_minus_v(seed: int) -> None:
    """A(s, a) = Q(s, a) - V(s) for known, unseen and optimistic actions."""
    rng = random.Random(seed)
    neighbours = _random_neighbors(rng, rng.randint(1, 20))
    candidates = list(_grouped(neighbours)) + ["unseen-1", "unseen-2"]
    estimate = estimate_values(
        neighbours,
        candidates,
        exploration_rate=0.5,
        rng=random.Random(seed + 100),
    )
    for key in candidates:
        assert estimate.advantage[key] == pytest.approx(
            estimate.q_hat[key] - estimate.v_hat, abs=1e-12
        )


# ---------------------------------------------------------------------------
# Eq. 25: advantage normalisation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(10))
def test_eq25_normalised_advantage_matches_the_definition(seed: int) -> None:
    """A~ = A / (max|A| + eps), with the largest magnitude close to one."""
    rng = random.Random(seed)
    neighbours = _random_neighbors(rng, rng.randint(1, 20))
    candidates = list(_grouped(neighbours)) + ["unseen"]
    estimate = estimate_values(
        neighbours,
        candidates,
        exploration_rate=0.5,
        rng=random.Random(seed + 7),
    )
    max_abs = max(abs(value) for value in estimate.advantage.values())
    assert max_abs > 0.0
    for key, value in estimate.advantage.items():
        expected = value / (max_abs + EPSILON)
        assert estimate.advantage_norm[key] == pytest.approx(expected, abs=1e-12)
        assert math.isfinite(estimate.advantage_norm[key])
        assert -1.0 < estimate.advantage_norm[key] <= 1.0
    biggest = max(estimate.advantage_norm.values(), key=abs)
    assert abs(biggest) == pytest.approx(max_abs / (max_abs + EPSILON), abs=1e-12)


def test_eq25_epsilon_argument_guards_the_denominator() -> None:
    """A caller-supplied epsilon is added to max|A| exactly like the paper's."""
    neighbours = [
        _neighbor(1, "a", 10.0, weight=1.0),
        _neighbor(2, "b", 0.0, weight=1.0),
    ]
    estimate = estimate_values(
        neighbours, ["a", "b"], exploration_rate=0.0, epsilon=0.5
    )
    max_abs = max(abs(value) for value in estimate.advantage.values())
    for key, value in estimate.advantage.items():
        assert estimate.advantage_norm[key] == pytest.approx(value / (max_abs + 0.5))


def test_eq25_normalize_off_returns_the_raw_advantages() -> None:
    """normalize=False exposes raw advantages, including values above one."""
    neighbours = [
        _neighbor(1, "a", 10.0, weight=1.0),
        _neighbor(2, "b", 0.0, weight=1.0),
    ]
    estimate = estimate_values(
        neighbours, ["a", "b"], exploration_rate=0.0, normalize=False
    )
    assert estimate.advantage_norm == estimate.advantage
    assert estimate.advantage_norm is not estimate.advantage
    assert estimate.advantage["a"] == pytest.approx(5.0)
    assert estimate.advantage_norm["a"] > 1.0


@pytest.mark.parametrize("seed", range(5))
def test_eq25_all_ties_normalise_to_zero_without_nan(seed: int) -> None:
    """Equal action values give zero advantages and therefore zero norms."""
    rng = random.Random(seed)
    tied_return = rng.uniform(-2.0, 2.0)
    neighbours = [
        _neighbor(index + 1, key, tied_return, weight=1.0)
        for index, key in enumerate(("a", "b", "c", "d"))
    ]
    estimate = estimate_values(neighbours, ["a", "b", "c", "d"], exploration_rate=0.0)
    assert estimate.v_hat == pytest.approx(tied_return, abs=1e-12)
    for key in ("a", "b", "c", "d"):
        assert estimate.advantage[key] == pytest.approx(0.0, abs=1e-12)
        assert estimate.advantage_norm[key] == 0.0
        assert not math.isnan(estimate.advantage_norm[key])


# ---------------------------------------------------------------------------
# degenerate inputs, determinism, candidate handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("exploration_rate", [0.0, 0.3, 1.0])
def test_empty_neighbourhood_is_all_zero_and_finite(exploration_rate: float) -> None:
    """|N(s)| = 0 must yield zeros: alpha/0 would be undefined."""
    estimate = estimate_values(
        [], ["x", "y", "z"], exploration_rate=exploration_rate, rng=random.Random(0)
    )
    assert estimate.neighbor_count == 0
    assert estimate.v_hat == 0.0
    assert estimate.mean_similarity == 0.0
    assert estimate.q_hat == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert estimate.advantage == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert estimate.advantage_norm == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert estimate.action_counts == {"x": 0, "y": 0, "z": 0}
    assert estimate.explored == set()
    for value in estimate.advantage_norm.values():
        assert math.isfinite(value)


def test_empty_candidate_set_keeps_state_statistics() -> None:
    """No candidates means empty dicts but a valid V(s) and |N(s)|."""
    rng = random.Random(2)
    neighbours = _random_neighbors(rng, 6)
    estimate = estimate_values(neighbours, [], exploration_rate=0.0)
    assert estimate.q_hat == {}
    assert estimate.advantage == {}
    assert estimate.advantage_norm == {}
    assert estimate.action_counts == {}
    assert estimate.explored == set()
    assert estimate.neighbor_count == 6
    assert estimate.v_hat == pytest.approx(_weighted_mean(neighbours), abs=1e-12)


def test_fixed_seed_reproduces_every_estimate_exactly() -> None:
    """The same seed and candidate order reproduce identical estimates."""
    rng = random.Random(3)
    neighbours = _random_neighbors(rng, 8)
    candidates = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    kwargs = {
        "exploration_rate": 0.5,
        "ucb_alpha": 3.0,
        "min_count": 2,
    }
    first = estimate_values(neighbours, candidates, rng=random.Random(1234), **kwargs)
    second = estimate_values(neighbours, candidates, rng=random.Random(1234), **kwargs)
    assert first.v_hat == second.v_hat
    assert first.q_hat == second.q_hat
    assert first.advantage == second.advantage
    assert first.advantage_norm == second.advantage_norm
    assert first.action_counts == second.action_counts
    assert first.explored == second.explored
    assert first.mean_similarity == second.mean_similarity


def test_min_count_shrinks_thin_actions_toward_the_state_value() -> None:
    """Below min_count Q is shrunk towards V(s) instead of flipped to optimism.

    With kappa = min_count the estimate is (n * Q_empirical + kappa * V) / (n + kappa),
    so a single observation cannot swing the advantage between +1 and -1 from one
    request to the next; an action with no support at all still explores.
    """
    neighbours = [
        _neighbor(1, "single", 5.0, weight=1.0, similarity=0.9),
        _neighbor(2, "double", 1.0, weight=1.0, similarity=0.7),
        _neighbor(3, "double", 3.0, weight=1.0, similarity=0.5),
    ]
    shrunk = estimate_values(
        neighbours, ["single", "double"], min_count=2, exploration_rate=0.0
    )
    v_hat = (5.0 + 1.0 + 3.0) / 3.0
    assert shrunk.action_counts == {"single": 1, "double": 2}
    assert shrunk.v_hat == pytest.approx(v_hat)
    assert shrunk.q_hat["single"] == pytest.approx((5.0 + 2 * v_hat) / 3.0)
    assert shrunk.q_hat["double"] == pytest.approx(2.0)
    assert shrunk.explored == set()

    trusted = estimate_values(
        neighbours, ["single", "double"], min_count=1, exploration_rate=0.0
    )
    assert trusted.q_hat["single"] == pytest.approx(5.0)

    # No support at all keeps the paper's exploration branch.
    explored = estimate_values(
        neighbours,
        ["double", "novel"],
        min_count=2,
        exploration_rate=1.0,
        ucb_alpha=3.0,
    )
    assert explored.q_hat["novel"] == pytest.approx(explored.v_hat + 3.0 / 3)
    assert explored.explored == {"novel"}


def test_min_count_below_one_is_clamped_to_one() -> None:
    """min_count <= 0 behaves exactly like min_count == 1."""
    rng = random.Random(8)
    neighbours = _random_neighbors(rng, 6)
    candidates = list(_grouped(neighbours)) + ["novel"]
    kwargs = {"exploration_rate": 0.5, "ucb_alpha": 2.0, "rng": random.Random(99)}
    clamped = estimate_values(neighbours, candidates, min_count=0, **kwargs)
    one = estimate_values(neighbours, candidates, min_count=1, **kwargs)
    assert clamped.q_hat == one.q_hat
    assert clamped.advantage == one.advantage
    assert clamped.advantage_norm == one.advantage_norm
    assert clamped.explored == one.explored
    assert clamped.action_counts == one.action_counts


@pytest.mark.parametrize("seed", range(5))
def test_candidate_deduplication_preserves_first_seen_order(seed: int) -> None:
    """Duplicates are dropped but the caller's ordering is preserved."""
    rng = random.Random(seed)
    neighbours = _random_neighbors(rng, 10)
    keys = [rng.choice(("a", "b", "c", "d", "e")) for _ in range(25)]
    unique: list[str] = []
    for key in keys:
        if key not in unique:
            unique.append(key)
    estimate = estimate_values(
        neighbours, keys, exploration_rate=0.5, rng=random.Random(seed + 50)
    )
    assert list(estimate.q_hat) == unique
    assert list(estimate.advantage) == unique
    assert list(estimate.advantage_norm) == unique
    assert list(estimate.action_counts) == unique
    twin = estimate_values(
        neighbours, unique, exploration_rate=0.5, rng=random.Random(seed + 50)
    )
    assert estimate.q_hat == twin.q_hat
    assert estimate.advantage_norm == twin.advantage_norm
    assert estimate.explored == twin.explored


def test_k_is_accepted_without_truncating_the_neighbourhood() -> None:
    """The memory layer already truncated N(s); k must not drop rows here."""
    neighbours = _random_neighbors(random.Random(5), 12)
    candidates = ["a", "b", "c", "d"]
    baseline = estimate_values(neighbours, candidates, exploration_rate=0.0)
    tiny_k = estimate_values(neighbours, candidates, k=1, exploration_rate=0.0)
    assert tiny_k.neighbor_count == 12
    assert tiny_k.v_hat == baseline.v_hat
    assert tiny_k.q_hat == baseline.q_hat
    assert tiny_k.action_counts == baseline.action_counts


@pytest.mark.parametrize("seed", range(5))
def test_counts_and_explored_cover_the_candidate_set(seed: int) -> None:
    """action_counts mirrors the group size and explored is the unseen subset."""
    rng = random.Random(seed)
    neighbours = _random_neighbors(rng, rng.randint(1, 15))
    grouped = _grouped(neighbours)
    candidates = list(grouped) + ["novel-1", "novel-2"]
    estimate = estimate_values(
        neighbours,
        candidates,
        min_count=2,
        exploration_rate=1.0,
        rng=random.Random(seed + 1),
    )
    # Only actions without any support take the exploration branch; thin but
    # observed actions are shrunk instead (see the min_count test).
    unseen = {key for key in candidates if len(grouped.get(key, ())) == 0}
    assert estimate.explored == unseen
    for key in candidates:
        assert estimate.action_counts[key] == len(grouped.get(key, ()))
        assert 0.0 <= estimate.action_counts[key] <= len(neighbours)
