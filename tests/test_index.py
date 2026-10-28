"""Adversarial contract tests for MemoryBank.topk (Eq. 24 retrieval).

These tests are independent of tests/test_memory.py.  A brute-force reference
implementation below reads the stored rows back through the public export API
and re-ranks them with jitrl.state.jaccard; MemoryBank.topk must agree with it
exactly (entry ids, order, similarity, recency weight, action fields) whenever
prefilter_limit is large enough to score every row.

Contract pinned here:
  * candidate filter: optional surface equality and, in "session" mode, scope
    equality;
  * similarity = Jaccard(query.tokens, row.tokens); a row is dropped when
    similarity < threshold (the threshold is inclusive) and rows without
    tokens are never neighbours;
  * in "hybrid" mode a cross-scope row's similarity is multiplied by 0.85
    *after* the threshold test;
  * similarity_weight = similarity * (0.5 + 0.5 * 0.5 ** (age_days / half_life)),
    and weight == similarity exactly when half_life <= 0;
  * ranking is (-similarity, -created_at, -id) and at most k rows are returned;
  * prefilter_limit bounds SQL batch size and the returned count, not the
    candidate set. All admissible rows participate in the exact ranking.
"""

from __future__ import annotations

import asyncio
import math
import random

import pytest

from jitrl.memory import MemoryBank
from jitrl.models import Episode, StateView, StepRecord
from jitrl.state import jaccard, normalize_action

CROSS_SCOPE_PENALTY = 0.85
RECENCY_FLOOR = 0.5
SECONDS_PER_DAY = 86400.0

NOW = 1_800_000_000.0
HALF_LIFE = 30.0
BIG_PREFILTER = 200_000

_VOCAB = tuple(f"tok{index:03d}" for index in range(240))
_ACTIONS = (("tool", "search"), ("tool", "click"), ("reply", "reply"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _state(tokens, surface: str = "") -> StateView:
    """Build a query StateView directly from a token set."""
    ordered = sorted(str(token) for token in tokens)
    return StateView(
        signature=f"query|surface={surface}",
        tokens=frozenset(ordered),
        text=" ".join(ordered)[:120],
        surface=surface,
    )


def _spec(
    tokens,
    *,
    scope: str = "s1",
    surface: str = "web",
    created_at: float = NOW,
    return_: float = 0.0,
    reward: float = 0.0,
    action: tuple[str, str] = ("tool", "search"),
) -> dict:
    """Describe one experience row to store."""
    return {
        "tokens": frozenset(str(token) for token in tokens),
        "scope": scope,
        "surface": surface,
        "created_at": created_at,
        "return": return_,
        "reward": reward,
        "action": action,
    }


def _episode_from_spec(index: int, spec: dict) -> Episode:
    """Build a one-step episode from a row spec."""
    kind, name = spec["action"]
    action = normalize_action(
        kind=kind, name=name, args={"page": "home"}, text=f"{name} home"
    )
    state = StateView(
        signature=f"surface={spec['surface']}|episode={index}",
        tokens=spec["tokens"],
        text=f"stored episode {index}",
        surface=spec["surface"],
    )
    return Episode(
        episode_id=f"episode-{index:04d}",
        scope=spec["scope"],
        created_at=spec["created_at"],
        steps=[
            StepRecord(
                index=0,
                state=state,
                action=action,
                reward=spec["reward"],
                discounted_return=spec["return"],
                outcome="ok",
            )
        ],
    )


async def _seed(path, specs: list[dict]) -> list[dict]:
    """Create a database, store one row per spec and return the exported rows."""
    bank = MemoryBank(path, max_entries=0, checkpoint_interval=0)
    await bank.initialize()
    try:
        for index, spec in enumerate(specs):
            await bank.add_episode(_episode_from_spec(index, spec))
        payload = await bank.export_json(limit=1_000_000)
    finally:
        await bank.close()
    return payload["experiences"]


async def _open_bank(path) -> MemoryBank:
    """Open an existing database for a single test."""
    bank = MemoryBank(path, max_entries=0, checkpoint_interval=0)
    await bank.initialize()
    return bank


# ---------------------------------------------------------------------------
# brute-force reference implementation
# ---------------------------------------------------------------------------
def _reference(
    rows: list[dict],
    query_tokens,
    *,
    query_surface: str,
    scope,
    k: int,
    threshold: float,
    surface_gate: bool,
    scope_mode: str,
    half_life: float,
    now: float,
) -> list[dict]:
    """Rank every stored row exactly as the documented contract says."""
    query_scope = str(scope or "")
    minimum = max(0.0, min(1.0, float(threshold)))
    mode = scope_mode if scope_mode in {"session", "global", "hybrid"} else "hybrid"
    tokens = frozenset(query_tokens)
    scored: list[dict] = []
    for row in rows:
        row_scope = str(row.get("scope") or "")
        if surface_gate and str(row.get("surface") or "") != (query_surface or ""):
            continue
        if mode == "session" and query_scope and row_scope != query_scope:
            continue
        row_tokens = frozenset(str(token) for token in row.get("state_tokens") or ())
        if not row_tokens:
            continue
        similarity = jaccard(tokens, row_tokens)
        if similarity < minimum:
            continue
        if mode == "hybrid" and query_scope and row_scope != query_scope:
            similarity *= CROSS_SCOPE_PENALTY
        created_at = float(row.get("created_at") or 0.0)
        if half_life <= 0:
            recency = 1.0
        else:
            age_days = max(0.0, (now - created_at) / SECONDS_PER_DAY)
            recency = 0.5 ** (age_days / half_life)
        weight = similarity * (RECENCY_FLOOR + (1.0 - RECENCY_FLOOR) * recency)
        scored.append(
            {
                "entry_id": int(row["entry_id"]),
                "similarity": similarity,
                "weight": weight,
                "created_at": created_at,
                "action_key": str(row.get("action_key") or ""),
                "discounted_return": float(row.get("discounted_return") or 0.0),
            }
        )
    scored.sort(
        key=lambda item: (-item["similarity"], -item["created_at"], -item["entry_id"])
    )
    return scored[: max(1, int(k))]


def _assert_matches(got, expected: list[dict], context: str) -> None:
    """Assert that a topk result equals the reference ranking."""
    assert len(got) == len(expected), (
        f"{context}: expected entry ids {[item['entry_id'] for item in expected]}, "
        f"got {[neighbour.entry_id for neighbour in got]}"
    )
    for neighbour, want in zip(got, expected):
        label = f"{context} | entry_id={want['entry_id']}"
        assert neighbour.entry_id == want["entry_id"], label
        assert neighbour.similarity == pytest.approx(want["similarity"], abs=1e-12), (
            label
        )
        assert neighbour.similarity_weight == pytest.approx(
            want["weight"], abs=1e-12
        ), label
        assert neighbour.created_at == pytest.approx(want["created_at"], abs=1e-9), (
            label
        )
        assert neighbour.action_key == want["action_key"], label
        assert neighbour.discounted_return == pytest.approx(
            want["discounted_return"], abs=1e-12
        ), label
        assert math.isfinite(neighbour.similarity_weight), label
        assert -1e-12 <= neighbour.similarity <= 1.0 + 1e-12, label


# ---------------------------------------------------------------------------
# randomised dataset
# ---------------------------------------------------------------------------
def _random_specs(rng: random.Random, count: int) -> list[dict]:
    """Generate overlapping, identical, disjoint and variable-size states."""
    sizes = (1, 1, 2, 3, 5, 10, 25, 60, 120)
    specs: list[dict] = []
    for index in range(count):
        size = min(rng.choice(sizes), len(_VOCAB))
        tokens = set(rng.sample(_VOCAB, size))
        roll = rng.random()
        if specs and roll < 0.12:
            tokens = set(specs[rng.randrange(len(specs))]["tokens"])  # duplicate state
        elif specs and roll < 0.28:
            base = specs[rng.randrange(len(specs))]["tokens"]
            take = max(1, min(len(base), rng.randint(1, max(1, size))))
            tokens = set(rng.sample(sorted(base), take))  # subset, high overlap
        elif specs and roll < 0.38:
            base = specs[rng.randrange(len(specs))]["tokens"]
            pool = [token for token in _VOCAB if token not in base]
            tokens = set(rng.sample(pool, min(size, len(pool))))  # disjoint
        created_at = NOW - rng.uniform(0.0, 365.0) * SECONDS_PER_DAY
        if specs and rng.random() < 0.12:
            created_at = specs[-1]["created_at"]  # timestamp tie
        specs.append(
            _spec(
                tokens,
                scope=rng.choice(("s1", "s2", "s3")),
                surface=rng.choice(("web", "cli", "api")),
                created_at=created_at,
                return_=round(rng.uniform(-3.0, 3.0), 6),
                reward=round(rng.uniform(-1.0, 1.0), 6),
                action=rng.choice(_ACTIONS),
            )
        )
    return specs


def _make_queries(specs: list[dict]) -> list[dict]:
    """Build query states that exercise identical, partial and mixed overlaps."""
    base = specs[0]["tokens"]
    ordered = sorted(base)
    partial = frozenset(ordered[: max(1, len(ordered) // 2)] + list(_VOCAB[200:203]))
    mixed = frozenset(set(specs[3]["tokens"]) | set(specs[7]["tokens"]))
    return [
        {
            "label": "exact-row",
            "tokens": frozenset(base),
            "surface": specs[0]["surface"],
        },
        {"label": "partial", "tokens": partial, "surface": "web"},
        {"label": "union", "tokens": mixed, "surface": "cli"},
    ]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    """One randomised 90-row bank shared by the parameter matrix."""
    path = tmp_path_factory.mktemp("retrieval") / "memory.db"
    rng = random.Random(20260501)
    specs = _random_specs(rng, 90)
    rows = asyncio.run(_seed(path, specs))
    return {"path": path, "specs": specs, "rows": rows, "queries": _make_queries(specs)}


# ---------------------------------------------------------------------------
# parameter matrix: topk == brute force
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("k", [1, 5, 10, 64])
@pytest.mark.parametrize("threshold", [0.05, 0.3, 0.5, 0.8, 0.95])
@pytest.mark.parametrize("scope_mode", ["session", "global", "hybrid"])
@pytest.mark.parametrize("surface_gate", [True, False], ids=["gate", "no-gate"])
@pytest.mark.parametrize("query_scope", ["s1", None], ids=["qscope-s1", "qscope-none"])
async def test_topk_matches_bruteforce_reference(
    dataset, k, threshold, scope_mode, surface_gate, query_scope
) -> None:
    """topk must equal the brute-force ranking of every stored row."""
    bank = await _open_bank(dataset["path"])
    try:
        for query in dataset["queries"]:
            expected = _reference(
                dataset["rows"],
                query["tokens"],
                query_surface=query["surface"],
                scope=query_scope,
                k=k,
                threshold=threshold,
                surface_gate=surface_gate,
                scope_mode=scope_mode,
                half_life=HALF_LIFE,
                now=NOW,
            )
            got = await bank.topk(
                _state(query["tokens"], query["surface"]),
                scope=query_scope,
                k=k,
                threshold=threshold,
                surface_gate=surface_gate,
                prefilter_limit=BIG_PREFILTER,
                recency_half_life_days=HALF_LIFE,
                scope_mode=scope_mode,
                now=NOW,
            )
            _assert_matches(
                got,
                expected,
                f"k={k} threshold={threshold} mode={scope_mode} "
                f"gate={surface_gate} scope={query_scope!r} query={query['label']}",
            )
    finally:
        await bank.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("half_life", [0.0, -5.0, 1.0, 7.0, 30.0, 365.0])
@pytest.mark.parametrize("threshold", [0.05, 0.5, 0.9])
@pytest.mark.parametrize("scope_mode", ["session", "global", "hybrid"])
async def test_recency_disabled_and_enabled_match_reference(
    dataset, half_life, threshold, scope_mode
) -> None:
    """The recency factor must follow 0.5 + 0.5 * 0.5 ** (age / half_life)."""
    bank = await _open_bank(dataset["path"])
    try:
        for query in dataset["queries"]:
            expected = _reference(
                dataset["rows"],
                query["tokens"],
                query_surface=query["surface"],
                scope="s1",
                k=64,
                threshold=threshold,
                surface_gate=False,
                scope_mode=scope_mode,
                half_life=half_life,
                now=NOW,
            )
            got = await bank.topk(
                _state(query["tokens"], query["surface"]),
                scope="s1",
                k=64,
                threshold=threshold,
                prefilter_limit=BIG_PREFILTER,
                recency_half_life_days=half_life,
                scope_mode=scope_mode,
                now=NOW,
            )
            _assert_matches(
                got,
                expected,
                f"half_life={half_life} threshold={threshold} "
                f"mode={scope_mode} query={query['label']}",
            )
            if half_life <= 0:
                for neighbour in got:
                    assert neighbour.similarity_weight == neighbour.similarity
    finally:
        await bank.close()


# ---------------------------------------------------------------------------
# prefilter_limit semantics
# ---------------------------------------------------------------------------
EXACT_TOKENS = frozenset({"alpha", "beta", "gamma", "delta"})


def _filler_specs(count: int, now: float = NOW) -> list[dict]:
    """Create newer rows that share no token with EXACT_TOKENS."""
    return [
        _spec({f"noise{index:03d}"}, created_at=now - index, return_=float(index))
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_large_cap_scores_every_stored_row(tmp_path) -> None:
    """With a cap above the row count, an old exact match must be found."""
    specs = _filler_specs(300)
    specs.append(
        _spec(EXACT_TOKENS, created_at=NOW - 400 * SECONDS_PER_DAY, return_=99.0)
    )
    bank, _ = await _seed_and_open(tmp_path / "large-cap.db", specs)
    try:
        hits = await bank.topk(
            _state(EXACT_TOKENS),
            scope="s1",
            k=1,
            threshold=0.9,
            prefilter_limit=BIG_PREFILTER,
            recency_half_life_days=0.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert len(hits) == 1
    assert hits[0].similarity == pytest.approx(1.0)
    assert hits[0].discounted_return == pytest.approx(99.0)


@pytest.mark.asyncio
async def test_small_cap_is_a_work_cap_not_a_newest_window(tmp_path) -> None:
    """EXPECTED-BEFORE-CHANGE: prefilter_limit caps rows *scored*, not rows kept.

    An exactly matching row that is older than the cap must still win: the
    pre-change "newest N rows" window drops it and returns [].
    """
    specs = _filler_specs(300)
    specs.append(
        _spec(EXACT_TOKENS, created_at=NOW - 400 * SECONDS_PER_DAY, return_=99.0)
    )
    bank, _ = await _seed_and_open(tmp_path / "small-cap.db", specs)
    try:
        hits = await bank.topk(
            _state(EXACT_TOKENS),
            scope="s1",
            k=1,
            threshold=0.9,
            prefilter_limit=64,
            recency_half_life_days=0.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert len(hits) == 1
    assert hits[0].similarity == pytest.approx(1.0)
    assert hits[0].discounted_return == pytest.approx(99.0)


@pytest.mark.asyncio
async def test_small_cap_prefers_overlap_over_recency(tmp_path) -> None:
    """EXPECTED-BEFORE-CHANGE: overlap must outrank raw recency under a small cap.

    Three older rows partially overlap the query; 120 newer rows do not.  A
    cap of 10 has to score the overlapping rows, not merely the newest ten.
    """
    query_tokens = frozenset({"target", "alpha", "beta", "gamma"})
    overlapping = [
        frozenset({"target", "alpha"}),
        frozenset({"target", "beta"}),
        frozenset({"target", "gamma"}),
    ]
    specs = [
        _spec({f"filler{index:03d}"}, created_at=NOW - index) for index in range(120)
    ]
    specs.extend(
        _spec(tokens, created_at=NOW - 500 * SECONDS_PER_DAY, return_=float(position))
        for position, tokens in enumerate(overlapping)
    )
    bank, _ = await _seed_and_open(tmp_path / "overlap-cap.db", specs)
    try:
        hits = await bank.topk(
            _state(query_tokens),
            scope="s1",
            k=3,
            threshold=0.3,
            prefilter_limit=10,
            recency_half_life_days=0.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert len(hits) == 3
    # 2 shared tokens / |{target, alpha, beta, gamma} + 2 - 2| = 2 / 4 = 0.5.
    assert all(hit.similarity == pytest.approx(0.5) for hit in hits)
    assert sorted(hit.discounted_return for hit in hits) == [0.0, 1.0, 2.0]


@pytest.mark.asyncio
async def test_cap_bounds_the_number_of_returned_rows(tmp_path) -> None:
    """A cap of N can never yield more than N neighbours, even when k > N."""
    shared = frozenset({"shared", "state"})
    specs = [_spec(shared, created_at=NOW - index) for index in range(40)]
    bank, _ = await _seed_and_open(tmp_path / "bound-cap.db", specs)
    try:
        hits = await bank.topk(
            _state(shared),
            scope="s1",
            k=10,
            threshold=0.9,
            prefilter_limit=7,
            recency_half_life_days=0.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert len(hits) == 7


# ---------------------------------------------------------------------------
# ordering, threshold and recency invariants
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ordering_is_similarity_then_newest_then_id(tmp_path) -> None:
    """Equal similarities break ties on created_at and then on descending id."""
    shared = frozenset({"shared", "state"})
    specs = [
        _spec(shared, created_at=NOW - 300.0, return_=1.0),
        _spec(shared, created_at=NOW - 200.0, return_=2.0),
        _spec(shared, created_at=NOW - 200.0, return_=3.0),
        _spec(shared, created_at=NOW - 200.0, return_=4.0),
    ]
    bank, rows = await _seed_and_open(tmp_path / "ties.db", specs)
    try:
        hits = await bank.topk(
            _state(shared),
            scope="s1",
            k=10,
            threshold=0.9,
            recency_half_life_days=0.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert [hit.discounted_return for hit in hits] == [4.0, 3.0, 2.0, 1.0]
    assert [hit.entry_id for hit in hits] == [
        rows[3]["entry_id"],
        rows[2]["entry_id"],
        rows[1]["entry_id"],
        rows[0]["entry_id"],
    ]


@pytest.mark.asyncio
async def test_ordering_uses_similarity_not_recency_weight(tmp_path) -> None:
    """An old exact match outranks a fresh fuzzy one despite its lower weight."""
    query_tokens = frozenset({"a", "b", "c", "d", "e"})
    specs = [
        _spec(
            {"a", "b", "c", "d", "e"},
            created_at=NOW - 3650 * SECONDS_PER_DAY,
            return_=1.0,
        ),
        _spec({"a", "b", "c", "d"}, created_at=NOW, return_=2.0),
    ]
    bank, _ = await _seed_and_open(tmp_path / "order.db", specs)
    try:
        hits = await bank.topk(
            _state(query_tokens),
            scope="s1",
            k=2,
            threshold=0.5,
            recency_half_life_days=30.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert [hit.discounted_return for hit in hits] == [1.0, 2.0]
    assert hits[0].similarity == pytest.approx(1.0)
    assert hits[1].similarity == pytest.approx(0.8)
    # The exact match is almost a year and a half old, so it is damped to the
    # floor while the fresh fuzzy row keeps its full weight.
    assert hits[0].similarity_weight < hits[1].similarity_weight


@pytest.mark.asyncio
async def test_threshold_is_inclusive(tmp_path) -> None:
    """A row is kept when similarity >= threshold, not only when it is above."""
    query_tokens = frozenset({"a", "b", "c", "d", "e"})
    specs = [
        _spec({"a", "b"}, return_=0.4),
        _spec({"a", "b", "c", "d", "e"}, return_=1.0),
        _spec({"a", "b", "c", "d", "e", "f", "g", "h", "i", "j"}, return_=0.5),
    ]
    bank, _ = await _seed_and_open(tmp_path / "threshold.db", specs)
    try:

        async def returns(threshold: float) -> list[float]:
            hits = await bank.topk(
                _state(query_tokens),
                scope="s1",
                k=10,
                threshold=threshold,
                recency_half_life_days=0.0,
                scope_mode="global",
                now=NOW,
            )
            return [hit.discounted_return for hit in hits]

        assert await returns(0.4) == [1.0, 0.5, 0.4]
        assert await returns(0.4000001) == [1.0, 0.5]
        assert await returns(0.5) == [1.0, 0.5]
        assert await returns(0.5000001) == [1.0]
    finally:
        await bank.close()


@pytest.mark.asyncio
async def test_cross_scope_penalty_is_applied_after_the_threshold(tmp_path) -> None:
    """Hybrid mode penalises cross-scope rows only after the threshold gate."""
    query_tokens = frozenset({"a", "b"})
    specs = [
        _spec(query_tokens, scope="s1", return_=1.0),
        _spec(query_tokens, scope="s2", return_=2.0),
    ]
    bank, _ = await _seed_and_open(tmp_path / "scope.db", specs)
    try:

        async def fetch(scope, threshold: float, mode: str):
            return await bank.topk(
                _state(query_tokens),
                scope=scope,
                k=10,
                threshold=threshold,
                recency_half_life_days=0.0,
                scope_mode=mode,
                now=NOW,
            )

        # The cross-scope row passes threshold 0.9 on its raw similarity of 1.0
        # and is then damped to 0.85, i.e. below the threshold.
        hybrid = await fetch("s1", 0.9, "hybrid")
        assert [hit.scope for hit in hybrid] == ["s1", "s2"]
        assert hybrid[0].similarity == pytest.approx(1.0)
        assert hybrid[1].similarity == pytest.approx(CROSS_SCOPE_PENALTY)

        # Both rows are exact matches and share a timestamp, so global mode
        # falls through to the descending-id tie-break (s2 was inserted last).
        glob = await fetch("s1", 0.9, "global")
        assert [hit.scope for hit in glob] == ["s2", "s1"]
        assert all(hit.similarity == pytest.approx(1.0) for hit in glob)

        session = await fetch("s1", 0.9, "session")
        assert [hit.scope for hit in session] == ["s1"]

        unscoped = await fetch(None, 0.9, "hybrid")
        assert all(hit.similarity == pytest.approx(1.0) for hit in unscoped)
    finally:
        await bank.close()


@pytest.mark.asyncio
async def test_recency_weight_formula(tmp_path) -> None:
    """weight = similarity * (0.5 + 0.5 * 0.5 ** (age_days / half_life))."""
    now = NOW
    specs = [
        _spec({"recency", "a"}, created_at=now - 30 * SECONDS_PER_DAY),
        _spec({"recency", "b"}, created_at=now + 100 * SECONDS_PER_DAY),
        _spec({"recency", "c"}, created_at=now - 30 * SECONDS_PER_DAY),
        _spec({"recency", "d"}, created_at=now - SECONDS_PER_DAY),
    ]
    bank, _ = await _seed_and_open(tmp_path / "recency.db", specs)
    try:

        async def fetch_weight(query: str, half_life: float) -> tuple[float, float]:
            hits = await bank.topk(
                _state({"recency", query}),
                scope="s1",
                k=1,
                threshold=0.9,
                recency_half_life_days=half_life,
                scope_mode="global",
                now=now,
            )
            assert len(hits) == 1
            return hits[0].similarity, hits[0].similarity_weight

        similarity, weight = await fetch_weight("a", 30.0)
        assert similarity == pytest.approx(1.0)
        assert weight == pytest.approx(0.75)

        # A creation time in the future is clamped to age zero.
        assert (await fetch_weight("b", 30.0))[1] == pytest.approx(1.0)

        disabled = await fetch_weight("c", 0.0)
        assert disabled[1] == disabled[0]

        negative = await fetch_weight("c", -5.0)
        assert negative[1] == negative[0]

        similarity, weight = await fetch_weight("d", 60.0)
        expected = similarity * (0.5 + 0.5 * 0.5 ** (1.0 / 60.0))
        assert weight == pytest.approx(expected, rel=1e-12)
    finally:
        await bank.close()


# ---------------------------------------------------------------------------
# neighbour payload and empty-input safety
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_neighbour_fields_echo_the_stored_row(tmp_path) -> None:
    """Every Neighbor field must be copied verbatim from the stored row."""
    tokens = frozenset({"alpha", "beta"})
    specs = [
        _spec(
            tokens,
            scope="s1",
            surface="web",
            created_at=NOW - 12345.0,
            return_=1.5,
            reward=0.5,
            action=("tool", "search"),
        )
    ]
    bank, rows = await _seed_and_open(tmp_path / "fields.db", specs)
    try:
        hits = await bank.topk(
            _state(tokens),
            scope="s1",
            k=1,
            threshold=0.9,
            recency_half_life_days=0.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert len(hits) == 1
    hit = hits[0]
    row = rows[0]
    assert hit.entry_id == row["entry_id"]
    assert hit.similarity == pytest.approx(1.0)
    assert hit.action_key == row["action_key"]
    assert hit.action_text == row["action_text"]
    assert hit.discounted_return == pytest.approx(1.5)
    assert hit.reward == pytest.approx(0.5)
    assert hit.state_text == row["state_text"]
    assert hit.scope == row["scope"]
    assert hit.created_at == pytest.approx(NOW - 12345.0)


@pytest.mark.asyncio
async def test_empty_query_and_tokenless_rows_are_never_returned(tmp_path) -> None:
    """An empty query matches nothing; a stored row without tokens is invisible."""
    specs = [_spec(set()), _spec({"alpha", "beta"})]
    bank, _ = await _seed_and_open(tmp_path / "empty.db", specs)
    try:
        assert await bank.topk(_state(set()), scope="s1", now=NOW) == []
        hits = await bank.topk(
            _state({"alpha", "beta"}),
            scope="s1",
            k=10,
            threshold=0.5,
            recency_half_life_days=0.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert len(hits) == 1
    assert hits[0].similarity == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_empty_bank_returns_no_neighbours(tmp_path) -> None:
    """Retrieval on an empty database is an empty list, not an error."""
    bank, _ = await _seed_and_open(tmp_path / "none.db", [])
    try:
        hits = await bank.topk(
            _state({"alpha"}), scope="s1", k=5, threshold=0.05, now=NOW
        )
    finally:
        await bank.close()
    assert hits == []


@pytest.mark.asyncio
async def test_disjoint_query_matches_nothing(tmp_path) -> None:
    """Zero token overlap must stay below even a very low threshold."""
    bank, _ = await _seed_and_open(tmp_path / "disjoint.db", [_spec({"alpha", "beta"})])
    try:
        hits = await bank.topk(
            _state({"zzz", "yyy"}),
            scope="s1",
            k=10,
            threshold=0.05,
            recency_half_life_days=0.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert hits == []


@pytest.mark.asyncio
async def test_duplicate_states_are_all_returned(tmp_path) -> None:
    """Identical stored states are distinct neighbours, ordered newest first."""
    shared = frozenset({"same", "state"})
    specs = [
        _spec(shared, created_at=NOW - index, return_=float(index))
        for index in range(5)
    ]
    bank, _ = await _seed_and_open(tmp_path / "dupes.db", specs)
    try:
        hits = await bank.topk(
            _state(shared),
            scope="s1",
            k=10,
            threshold=0.95,
            recency_half_life_days=0.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert [hit.discounted_return for hit in hits] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(hit.similarity == pytest.approx(1.0) for hit in hits)


@pytest.mark.asyncio
async def test_at_most_k_rows_are_returned(tmp_path) -> None:
    """k truncates the ranking even when more rows clear the threshold."""
    shared = frozenset({"same", "state"})
    specs = [_spec(shared, created_at=NOW - index) for index in range(20)]
    bank, _ = await _seed_and_open(tmp_path / "k.db", specs)
    try:
        hits = await bank.topk(
            _state(shared),
            scope="s1",
            k=5,
            threshold=0.9,
            recency_half_life_days=0.0,
            scope_mode="global",
            now=NOW,
        )
    finally:
        await bank.close()
    assert len(hits) == 5
    assert all(hit.similarity == pytest.approx(1.0) for hit in hits)


async def _seed_and_open(path, specs: list[dict]) -> tuple[MemoryBank, list[dict]]:
    """Seed a database and reopen it for a single test."""
    rows = await _seed(path, specs)
    bank = await _open_bank(path)
    return bank, rows
