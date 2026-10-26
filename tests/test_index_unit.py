"""Unit tests for the inverted-index retrieval accelerator (jitrl.index).

The contract tests in tests/test_index.py pin the exact behaviour of
MemoryBank.topk end to end. These tests target the index itself: the pruning
rules must be lossless, tombstones and compaction must not change any result,
and the query's own unknown tokens must still count towards the Jaccard
denominator (dropping them would inflate every similarity).
"""

from __future__ import annotations

import random

import pytest

from jitrl.index import ExperienceIndex
from jitrl.state import jaccard

NOW = 1_800_000_000.0
VOCAB = tuple(f"tok{index:03d}" for index in range(200))


def _fill(
    rng: random.Random, size: int
) -> tuple[ExperienceIndex, dict[int, frozenset[str]]]:
    """Build an index and the ground-truth token sets it was fed.

    Args:
        rng: Seeded random source.
        size: Number of rows.

    Returns:
        (index, rows) where rows maps entry_id to its token set.
    """
    index = ExperienceIndex()
    rows: dict[int, frozenset[str]] = {}
    for position in range(size):
        tokens = frozenset(rng.sample(VOCAB, rng.randint(1, 40)))
        entry_id = 10_000 + position
        rows[entry_id] = tokens
        index.append(
            entry_id=entry_id,
            tokens=tokens,
            scope="s1",
            surface="web",
            action_key=f"tool:a{position % 3}(q=<str>)",
            discounted_return=rng.uniform(-3.0, 3.0),
            reward=rng.uniform(-3.0, 3.0),
            created_at=1000.0 + position,
        )
    return index, rows


def _reference(
    rows: dict[int, frozenset[str]],
    query: frozenset[str],
    *,
    threshold: float,
    k: int,
) -> list[int]:
    """Brute-force Eq. 24 ranking, used as the ground truth.

    Args:
        rows: Stored token sets.
        query: Query token set.
        threshold: Minimum Jaccard similarity (inclusive).
        k: Maximum number of results.

    Returns:
        The entry ids in retrieval order.
    """
    scored = []
    for entry_id, tokens in rows.items():
        similarity = jaccard(query, tokens)
        if similarity <= 0.0 or similarity < threshold:
            continue
        scored.append((-similarity, -1000.0, -entry_id))
    return [-item[2] for item in sorted(scored)[:k]]


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("threshold", [0.05, 0.3, 0.6, 0.9])
def test_pruning_is_lossless(seed: int, threshold: float) -> None:
    """Candidate pruning must never drop a row the exhaustive scan would keep."""
    rng = random.Random(seed)
    index, rows = _fill(rng, rng.randint(5, 120))
    for _ in range(6):
        query = frozenset(rng.sample(VOCAB, rng.randint(1, 50)))
        hits = index.search(
            query,
            k=10,
            threshold=threshold,
            scope="s1",
            prefilter_limit=200000,
            recency_half_life_days=0.0,
            now=NOW,
        )
        assert [hit.entry_id for hit in hits] == _reference(
            rows, query, threshold=threshold, k=10
        )


def test_unknown_query_tokens_stay_in_the_denominator() -> None:
    """Tokens absent from the bank must lower similarity, not be ignored."""
    index = ExperienceIndex()
    index.append(
        entry_id=1,
        tokens=frozenset({"alpha", "beta"}),
        scope="s1",
        surface="",
        action_key="tool:x",
        discounted_return=1.0,
        reward=1.0,
        created_at=NOW,
    )
    # Two known tokens plus two that never occurred: J = 2 / (2 + 4 - 2) = 0.5.
    query = frozenset({"alpha", "beta", "never-seen", "also-never"})
    hits = index.search(query, k=5, threshold=0.5, recency_half_life_days=0.0)
    assert len(hits) == 1
    assert hits[0].similarity == pytest.approx(0.5)
    assert index.vocab_size == 2


def test_boundary_threshold_keeps_exactly_equal_similarity() -> None:
    """The threshold is inclusive, including at a length-boundary quantisation."""
    index = ExperienceIndex()
    index.append(
        entry_id=7,
        tokens=frozenset({"a1", "a2", "a3"}),
        scope="",
        surface="",
        action_key="tool:x",
        discounted_return=0.0,
        reward=0.0,
        created_at=NOW,
    )
    query = frozenset({"a1", "a2", "a3", "a4"})
    # Exactly 3 / 4 = 0.75.
    assert index.search(query, k=1, threshold=0.75, recency_half_life_days=0.0)
    assert not index.search(query, k=1, threshold=0.7501, recency_half_life_days=0.0)


def test_remove_update_and_compaction_are_exact() -> None:
    """Tombstones and the compaction pass must not change retrieval results."""
    rng = random.Random(99)
    index, rows = _fill(rng, 180)
    victims = rng.sample(sorted(rows), 120)
    for entry_id in victims:
        rows.pop(entry_id)
        assert index.remove(entry_id)
    assert len(index) == len(rows)
    # Compaction ran during the churn (the table shrank below the original size)
    # and the remaining tombstones stay below the compaction trigger.
    assert len(index._entry_ids) < 180
    assert index._stale < 64

    survivor = next(iter(rows))
    index.update_return(survivor, discounted_return=77.0, reward=7.0)
    for _ in range(4):
        query = frozenset(rng.sample(VOCAB, rng.randint(2, 30)))
        hits = index.search(
            query,
            k=8,
            threshold=0.2,
            scope="s1",
            prefilter_limit=200000,
            recency_half_life_days=0.0,
            now=NOW,
        )
        assert [hit.entry_id for hit in hits] == _reference(
            rows, query, threshold=0.2, k=8
        )
        for hit in hits:
            if hit.entry_id == survivor:
                assert hit.discounted_return == 77.0


def test_reappending_the_same_entry_replaces_it() -> None:
    """A rewritten row (same primary key) is not indexed twice."""
    index = ExperienceIndex()
    for tokens in ({"alpha"}, {"alpha", "beta"}):
        index.append(
            entry_id=5,
            tokens=frozenset(tokens),
            scope="",
            surface="",
            action_key="tool:x",
            discounted_return=1.0,
            reward=1.0,
            created_at=NOW,
        )
    assert len(index) == 1
    hits = index.search(
        frozenset({"alpha", "beta"}), k=1, threshold=0.5, recency_half_life_days=0.0
    )
    assert len(hits) == 1
    assert hits[0].similarity == pytest.approx(1.0)


def test_full_index_rejects_appends() -> None:
    """The row cap is a hard stop so the owner can fall back to the SQL scan."""
    index = ExperienceIndex(max_entries=2)
    assert index.append(
        entry_id=1,
        tokens=frozenset({"alpha"}),
        scope="",
        surface="",
        action_key="tool:x",
        discounted_return=0.0,
        reward=0.0,
        created_at=NOW,
    )
    assert index.append(
        entry_id=2,
        tokens=frozenset({"beta"}),
        scope="",
        surface="",
        action_key="tool:x",
        discounted_return=0.0,
        reward=0.0,
        created_at=NOW,
    )
    assert index.is_full
    assert not index.append(
        entry_id=3,
        tokens=frozenset({"gamma"}),
        scope="",
        surface="",
        action_key="tool:x",
        discounted_return=0.0,
        reward=0.0,
        created_at=NOW,
    )
    assert len(index) == 2


def test_tokenless_rows_are_skipped() -> None:
    """A state without tokens can never be a neighbour."""
    index = ExperienceIndex()
    assert not index.append(
        entry_id=1,
        tokens=(),
        scope="",
        surface="",
        action_key="tool:x",
        discounted_return=0.0,
        reward=0.0,
        created_at=NOW,
    )
    assert len(index) == 0
    assert index.search(frozenset({"alpha"}), threshold=0.5) == []


def test_cross_scope_penalty_is_applied_after_the_threshold() -> None:
    """Hybrid mode thresholds on the raw score, then dampens the ranking."""
    index = ExperienceIndex()
    for entry_id, scope in ((1, "s1"), (2, "s2")):
        index.append(
            entry_id=entry_id,
            tokens=frozenset({"alpha", "beta", "gamma", "delta"}),
            scope=scope,
            surface="",
            action_key="tool:x",
            discounted_return=0.0,
            reward=0.0,
            created_at=NOW,
        )
    hits = index.search(
        frozenset({"alpha", "beta", "gamma", "delta"}),
        k=2,
        threshold=0.9,
        scope="s1",
        scope_mode="hybrid",
        recency_half_life_days=0.0,
        now=NOW,
    )
    by_scope = {hit.scope: hit.similarity for hit in hits}
    assert by_scope["s1"] == pytest.approx(1.0)
    assert by_scope["s2"] == pytest.approx(0.85)
    assert hits[0].scope == "s1"
