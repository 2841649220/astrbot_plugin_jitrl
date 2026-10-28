"""Exact retrieval must survive small budgets, unseen scopes and SQL fallback."""

from __future__ import annotations

import random

import pytest

import jitrl.memory as memory_module
from jitrl.index import ExperienceIndex
from jitrl.memory import MemoryBank
from jitrl.models import StateView

NOW = 1_800_000_000.0


def test_high_overlap_does_not_displace_a_better_jaccard_match():
    index = ExperienceIndex()
    query = frozenset(f"q{i}" for i in range(10))
    # The exact row is old and has less raw overlap than every padded decoy.
    for entry_id, tokens in enumerate(
        [query - {"q9"}] + [query | {f"extra{i}" for i in range(8)}] * 40, 1
    ):
        index.append(
            entry_id=entry_id,
            tokens=tokens,
            scope="s1",
            surface="web",
            action_key="tool:search()",
            discounted_return=float(entry_id),
            reward=0.0,
            created_at=NOW + entry_id,
        )
    hits = index.search(query, k=3, threshold=0.5, prefilter_limit=3, now=NOW)
    assert [hit.entry_id for hit in hits] == [1, 41, 40]


@pytest.mark.parametrize("mode", ["session", "hybrid", "global"])
@pytest.mark.parametrize("threshold", [0.0, 0.1, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("scope", ["s1", "never-recorded", ""])
def test_budgeted_search_matches_exhaustive_ranking(mode, threshold, scope):
    rng = random.Random(4014)
    query = frozenset(f"q{i}" for i in range(10))
    index = ExperienceIndex()
    expected = []
    for entry_id in range(1, 121):
        tokens = frozenset(rng.sample(sorted(query), rng.randint(1, 10)))
        tokens |= frozenset(f"extra{i}" for i in range(rng.randrange(12)))
        row_scope = f"s{entry_id % 3}"
        surface = "web" if entry_id % 4 else "chat"
        created_at = NOW + entry_id % 7
        index.append(
            entry_id=entry_id,
            tokens=tokens,
            scope=row_scope,
            surface=surface,
            action_key="tool:search()",
            discounted_return=1.0,
            reward=0.0,
            created_at=created_at,
        )
        if surface != "web" or (mode == "session" and scope and scope != row_scope):
            continue
        similarity = len(query & tokens) / len(query | tokens)
        if similarity < threshold:
            continue
        if mode == "hybrid" and scope and scope != row_scope:
            similarity *= 0.85
        expected.append((similarity, created_at, entry_id))
    hits = index.search(
        query,
        k=7,
        threshold=threshold,
        scope=scope,
        scope_mode=mode,
        surface="web",
        surface_gate=True,
        prefilter_limit=7,
        now=NOW,
    )
    assert [(hit.similarity, hit.created_at, hit.entry_id) for hit in hits] == sorted(
        expected, reverse=True
    )[:7]


@pytest.mark.asyncio
@pytest.mark.parametrize("force_sql", [False, True])
async def test_scope_surface_threshold_and_weights_match_oracle(
    tmp_path, monkeypatch, force_sql
):
    rng = random.Random(2214)
    query_tokens = frozenset(f"q{i}" for i in range(8))
    specs = [
        {
            "episode_id": f"row-{i}",
            "scope": f"s{i % 3}",
            "surface": "web" if i % 4 else "chat",
            "state_tokens": rng.sample(sorted(query_tokens), rng.randint(1, 8))
            + [f"extra{i % 4}"],
            "action_key": "tool:search()",
            "created_at": NOW - (i % 13) * 86400.0,
            "discounted_return": float(i),
        }
        for i in range(60)
    ]
    bank = MemoryBank(tmp_path / "oracle.sqlite3", checkpoint_interval=0)
    await bank.initialize()
    try:
        await bank.import_json({"experiences": specs})
        if force_sql:
            monkeypatch.setattr(memory_module, "_INDEX_MAX_ENTRIES", 1)
        rows = (await bank.export_json(limit=1000))["experiences"]
        query = StateView(signature="q", tokens=query_tokens, text="q", surface="web")
        for mode in ("global", "session", "hybrid"):
            for scope in ("s1", "unseen", ""):
                for threshold in (0.0, 0.4, 0.7):
                    expected = []
                    for row in rows:
                        if row["surface"] != "web" or (
                            mode == "session" and scope and row["scope"] != scope
                        ):
                            continue
                        tokens = frozenset(row["state_tokens"])
                        similarity = len(query_tokens & tokens) / len(
                            query_tokens | tokens
                        )
                        if similarity < threshold:
                            continue
                        if mode == "hybrid" and scope and row["scope"] != scope:
                            similarity *= 0.85
                        expected.append(
                            (similarity, row["created_at"], row["entry_id"])
                        )
                    expected = sorted(expected, reverse=True)[:5]
                    hits = await bank.topk(
                        query,
                        k=5,
                        threshold=threshold,
                        scope=scope,
                        scope_mode=mode,
                        surface_gate=True,
                        prefilter_limit=5,
                        now=NOW,
                    )
                    assert [hit.entry_id for hit in hits] == [
                        item[2] for item in expected
                    ]
                    for hit, (similarity, timestamp, _) in zip(hits, expected):
                        weight = similarity * (
                            0.5 + 0.5 * 0.5 ** ((NOW - timestamp) / (86400 * 30))
                        )
                        assert hit.similarity == pytest.approx(similarity)
                        assert hit.similarity_weight == pytest.approx(weight)
    finally:
        await bank.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("force_sql", [False, True])
async def test_old_best_match_survives_small_sql_batches(
    tmp_path, monkeypatch, threshold, force_sql
):
    bank = MemoryBank(tmp_path / "memory.sqlite3", checkpoint_interval=0)
    await bank.initialize()
    try:
        await bank.import_json(
            {
                "experiences": [
                    {
                        "episode_id": f"e{i}",
                        "scope": "s1",
                        "state_tokens": ["target"] if i == 0 else [f"filler{i}"],
                        "action_key": "tool:search()",
                        "created_at": NOW + i,
                        "discounted_return": float(i),
                    }
                    for i in range(35)
                ]
            }
        )
        if force_sql:
            # Exercise the production fallback when the bank exceeds its cap.
            monkeypatch.setattr(memory_module, "_INDEX_MAX_ENTRIES", 1)
        query = StateView(
            signature="target", tokens=frozenset({"target"}), text="target"
        )
        hits = await bank.topk(
            query, k=2, threshold=threshold, prefilter_limit=2, now=NOW
        )
        assert hits
        assert hits[0].discounted_return == 0.0
        assert hits[0].similarity == 1.0
        assert len(hits) == (2 if threshold == 0.0 else 1)
    finally:
        await bank.close()
