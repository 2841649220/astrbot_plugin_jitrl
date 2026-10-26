"""Unit tests for jitrl.memory: SQLite persistence and Eq. 24 retrieval."""

from __future__ import annotations

import pytest

from jitrl.memory import MemoryBank
from jitrl.models import Episode, StateView, StepRecord
from jitrl.state import abstract_state, normalize_action


def build_state(observation: str, surface: str = "web") -> StateView:
    """Build a query state whose tokens are deterministic.

    Args:
        observation: Observation text to abstract.
        surface: Coarse partition key stored on the state.

    Returns:
        The abstracted state.
    """
    return abstract_state(task="task", observation=observation, surface=surface)


def build_episode(
    episode_id: str,
    rewards: list[float],
    *,
    scope: str = "s1",
    created_at: float = 1000.0,
    gamma: float = 0.5,
    observation: str = "alpha beta gamma",
    surface: str = "web",
    user_text: str = "",
    success: bool | None = None,
) -> Episode:
    """Build an episode whose steps share one observation.

    Args:
        episode_id: Unique episode identifier.
        rewards: One reward per step; also used as the discounted return.
        scope: Isolation key of the episode.
        created_at: Unix timestamp of the episode.
        gamma: Discount factor stored in the episode header.
        observation: Observation used for every step.
        surface: Surface attached to every step state.
        user_text: Raw user request.
        success: Optional success flag.

    Returns:
        The constructed episode.
    """
    steps = [
        StepRecord(
            index=index,
            state=abstract_state(task="task", observation=observation, surface=surface),
            action=normalize_action(
                kind="tool", name="search", args={"query": "lookup words"}
            ),
            reward=reward,
            discounted_return=reward,
            outcome="ok",
        )
        for index, reward in enumerate(rewards)
    ]
    return Episode(
        episode_id=episode_id,
        scope=scope,
        created_at=created_at,
        user_text=user_text,
        success=success,
        steps=steps,
        meta={"gamma": gamma},
    )


@pytest.mark.asyncio
async def test_initialize_is_idempotent(tmp_path) -> None:
    """Initialising and closing twice must be safe."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.initialize()
    assert await bank.count() == 0
    await bank.close()
    await bank.close()


@pytest.mark.asyncio
async def test_add_episode_writes_every_step(tmp_path) -> None:
    """Every step becomes one experience row, scoped by the episode scope."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    episode = build_episode("e1", [1.0, 2.0, 3.0])
    assert await bank.add_episode(episode) == 3
    assert await bank.count() == 3
    assert await bank.count(scope="s1") == 3
    assert await bank.count(scope="other") == 0
    assert await bank.add_episode(episode) == 3
    assert await bank.count() == 3
    await bank.close()


@pytest.mark.asyncio
async def test_topk_exact_match_and_threshold(tmp_path) -> None:
    """Retrieval ranks by Jaccard similarity and honours the threshold."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.add_episode(
        build_episode("e1", [1.0], observation="weather sunny beijing")
    )
    await bank.add_episode(build_episode("e2", [2.0], observation="stock price apple"))
    exact = build_state("weather sunny beijing")
    hits = await bank.topk(exact, scope="s1", now=2000.0)
    assert len(hits) == 1
    assert hits[0].similarity == pytest.approx(1.0)
    assert hits[0].action_key == "tool:search(query=<str>)"
    assert hits[0].scope == "s1"
    assert hits[0].discounted_return == pytest.approx(1.0)

    fuzzy = build_state("weather sunny shanghai")
    assert await bank.topk(fuzzy, scope="s1", threshold=0.8, now=2000.0) == []
    relaxed = await bank.topk(fuzzy, scope="s1", threshold=0.5, now=2000.0)
    assert len(relaxed) == 1
    assert 0.5 <= relaxed[0].similarity < 0.8
    await bank.close()


@pytest.mark.asyncio
async def test_topk_respects_k_and_recency(tmp_path) -> None:
    """k truncates the result and recency weights newer experiences higher."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    now = 1_000_000.0
    await bank.add_episode(
        build_episode(
            "old",
            [1.0],
            created_at=now - 90 * 86400,
            observation="weather sunny beijing",
        )
    )
    await bank.add_episode(
        build_episode("new", [2.0], created_at=now, observation="weather sunny beijing")
    )
    query = build_state("weather sunny beijing")
    hits = await bank.topk(query, scope="s1", k=2, now=now)
    assert len(hits) == 2
    assert [hit.created_at for hit in hits] == [now, now - 90 * 86400]
    assert hits[0].similarity_weight == pytest.approx(1.0, abs=1e-6)
    assert hits[1].similarity_weight == pytest.approx(0.5625, abs=1e-6)
    assert hits[1].similarity == pytest.approx(1.0)

    truncated = await bank.topk(query, scope="s1", k=1, now=now)
    assert len(truncated) == 1

    disabled = await bank.topk(query, scope="s1", recency_half_life_days=0, now=now)
    assert disabled[1].similarity_weight == pytest.approx(disabled[1].similarity)
    await bank.close()


@pytest.mark.asyncio
async def test_topk_prefilter_limit(tmp_path) -> None:
    """The SQL prefilter only scores the newest prefilter_limit rows."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    for index in range(5):
        await bank.add_episode(
            build_episode(
                f"e{index}",
                [float(index)],
                created_at=1000.0 + index,
                observation="weather sunny beijing",
            )
        )
    query = build_state("weather sunny beijing")
    assert len(await bank.topk(query, scope="s1", now=2000.0)) == 5
    limited = await bank.topk(query, scope="s1", prefilter_limit=2, now=2000.0)
    assert len(limited) == 2
    assert [hit.created_at for hit in limited] == [1004.0, 1003.0]
    await bank.close()


@pytest.mark.asyncio
async def test_topk_surface_gate(tmp_path) -> None:
    """The surface gate restricts retrieval to the query surface."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.add_episode(
        build_episode(
            "web",
            [1.0],
            created_at=1000.0,
            observation="weather sunny beijing",
            surface="web",
        )
    )
    await bank.add_episode(
        build_episode(
            "cli",
            [2.0],
            created_at=2000.0,
            observation="weather sunny beijing",
            surface="cli",
        )
    )
    query = build_state("weather sunny beijing", surface="web")
    assert len(await bank.topk(query, scope="s1", now=3000.0)) == 2
    gated = await bank.topk(query, scope="s1", surface_gate=True, now=3000.0)
    assert len(gated) == 1
    assert gated[0].created_at == pytest.approx(1000.0)
    await bank.close()


@pytest.mark.asyncio
async def test_topk_scope_modes(tmp_path) -> None:
    """Hybrid mode penalises cross-scope rows; session mode hides them."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.add_episode(
        build_episode(
            "same",
            [1.0],
            scope="s1",
            created_at=2000.0,
            observation="weather sunny beijing",
        )
    )
    await bank.add_episode(
        build_episode(
            "other",
            [2.0],
            scope="s2",
            created_at=2000.0,
            observation="weather sunny beijing",
        )
    )
    query = build_state("weather sunny beijing")

    hybrid = await bank.topk(query, scope="s1", scope_mode="hybrid", now=3000.0)
    assert len(hybrid) == 2
    by_scope = {hit.scope: hit for hit in hybrid}
    assert by_scope["s1"].similarity == pytest.approx(1.0)
    assert by_scope["s2"].similarity == pytest.approx(0.85)
    assert hybrid[0].scope == "s1"

    global_hits = await bank.topk(query, scope="s1", scope_mode="global", now=3000.0)
    assert {hit.scope for hit in global_hits} == {"s1", "s2"}
    assert all(hit.similarity == pytest.approx(1.0) for hit in global_hits)

    session_hits = await bank.topk(query, scope="s1", scope_mode="session", now=3000.0)
    assert [hit.scope for hit in session_hits] == ["s1"]

    unscoped = await bank.topk(query, scope=None, scope_mode="session", now=3000.0)
    assert len(unscoped) == 2
    await bank.close()


@pytest.mark.asyncio
async def test_topk_empty_query(tmp_path) -> None:
    """A query without tokens never matches anything."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.add_episode(
        build_episode("e1", [1.0], observation="weather sunny beijing")
    )
    empty = StateView(signature="", tokens=frozenset(), text="")
    assert await bank.topk(empty, scope="s1") == []
    await bank.close()


@pytest.mark.asyncio
async def test_action_stats(tmp_path) -> None:
    """Action statistics aggregate counts and returns per action key."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    episode = Episode(
        episode_id="stats",
        scope="s1",
        created_at=1000.0,
        steps=[
            StepRecord(
                index=0,
                state=build_state("alpha beta"),
                action=normalize_action(
                    kind="tool", name="search", args={"query": "first query"}
                ),
                reward=1.0,
                discounted_return=2.0,
            ),
            StepRecord(
                index=1,
                state=build_state("gamma delta"),
                action=normalize_action(
                    kind="tool", name="search", args={"query": "second query"}
                ),
                reward=1.0,
                discounted_return=-1.0,
            ),
            StepRecord(
                index=2,
                state=build_state("epsilon zeta"),
                action=normalize_action(kind="reply", name="reply", text="done"),
                reward=0.5,
                discounted_return=0.5,
            ),
        ],
    )
    await bank.add_episode(episode)
    stats = await bank.action_stats()
    assert len(stats) == 2
    top = stats[0]
    assert top["action_key"] == "tool:search(query=<str>)"
    assert top["action_name"] == "search"
    assert top["count"] == 2
    assert top["mean_return"] == pytest.approx(0.5)
    assert top["mean_reward"] == pytest.approx(1.0)
    assert top["positive_rate"] == pytest.approx(0.5)
    assert set(top) == {
        "action_key",
        "action_name",
        "action_text",
        "count",
        "mean_return",
        "mean_reward",
        "positive_rate",
    }
    assert await bank.action_stats(min_count=3) == []
    assert await bank.action_stats(scope="other") == []
    await bank.close()


@pytest.mark.asyncio
async def test_learning_curve(tmp_path) -> None:
    """Experiences are bucketed chronologically for the learning curve."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    steps = [
        StepRecord(
            index=index,
            state=build_state(f"obs {index}"),
            action=normalize_action(
                kind="tool", name="search", args={"query": "lookup words"}
            ),
            reward=float(index),
            discounted_return=float(index),
        )
        for index in range(10)
    ]
    await bank.add_episode(
        Episode(episode_id="curve", scope="s1", created_at=1000.0, steps=steps)
    )
    curve = await bank.learning_curve(buckets=2)
    assert len(curve) == 2
    assert curve[0]["bucket"] == 0
    assert curve[0]["count"] == 5
    assert curve[0]["mean_return"] == pytest.approx(2.0)
    assert curve[1]["mean_return"] == pytest.approx(7.0)
    assert curve[1]["mean_reward"] == pytest.approx(7.0)

    fine = await bank.learning_curve(buckets=100)
    assert len(fine) == 10
    assert all(bucket["count"] == 1 for bucket in fine)
    assert await bank.learning_curve(buckets=10) != []
    await bank.close()


@pytest.mark.asyncio
async def test_learning_curve_empty(tmp_path) -> None:
    """An empty bank produces an empty curve."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    assert await bank.learning_curve() == []
    await bank.close()


@pytest.mark.asyncio
async def test_recent_episodes_metadata(tmp_path) -> None:
    """Episode headers expose decoded metadata and a step count."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.add_episode(
        build_episode(
            "e1",
            [1.0, 2.0],
            created_at=1000.0,
            user_text="help me at bob@example.com",
            success=True,
        )
    )
    episodes = await bank.recent_episodes()
    assert len(episodes) == 1
    header = episodes[0]
    assert header["episode_id"] == "e1"
    assert header["step_count"] == 2
    assert header["success"] is True
    assert header["gamma"] == pytest.approx(0.5)
    assert header["user_text"] == "help me at <email>"
    assert header["meta"] == {"gamma": 0.5}
    assert await bank.recent_episodes(scope="other") == []
    await bank.close()


@pytest.mark.asyncio
async def test_storage_redacts_free_text(tmp_path) -> None:
    """Secrets are scrubbed before they hit the database."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    state = abstract_state(
        observation="login with password=hunter2 and key=sk-abcdefgh12345678"
    )
    action = normalize_action(
        kind="tool",
        name="login",
        args={"secret": "sk-abcdefgh12345678"},
        text="login with api_key=abcdef",
    )
    await bank.add_episode(
        Episode(
            episode_id="r1",
            scope="s1",
            created_at=1000.0,
            user_text="mail bob@example.com",
            steps=[StepRecord(index=0, state=state, action=action)],
        )
    )
    payload = await bank.export_json()
    row = payload["experiences"][0]
    assert "hunter2" not in row["state_text"]
    assert "sk-abcdefgh12345678" not in row["action_text"]
    assert "sk-abcdefgh12345678" not in row["action_args"]
    assert row["action_key"] == action.key
    assert payload["episodes"][0]["user_text"] == "mail <email>"
    await bank.close()


@pytest.mark.asyncio
async def test_storage_can_disable_redaction(tmp_path) -> None:
    """Disabling redaction stores the raw user text."""
    bank = MemoryBank(tmp_path / "memory.db", redact=False)
    await bank.initialize()
    await bank.add_episode(build_episode("e1", [1.0], user_text="mail bob@example.com"))
    payload = await bank.export_json()
    assert payload["episodes"][0]["user_text"] == "mail bob@example.com"
    await bank.close()


@pytest.mark.asyncio
async def test_forget_filters(tmp_path) -> None:
    """forget only deletes rows matching an explicit filter."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.add_episode(build_episode("a", [1.0], scope="s1", created_at=1000.0))
    await bank.add_episode(build_episode("b", [2.0], scope="s2", created_at=2000.0))
    assert await bank.forget() == 0
    assert await bank.count() == 2
    assert await bank.forget(scope="s2") == 1
    assert await bank.count() == 1
    assert await bank.count(scope="s2") == 0
    assert [item["episode_id"] for item in await bank.recent_episodes()] == ["a"]
    await bank.close()


@pytest.mark.asyncio
async def test_forget_by_action_and_age(tmp_path) -> None:
    """Action-key and age filters delete exactly the matching rows."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.add_episode(build_episode("old", [1.0], created_at=1000.0))
    await bank.add_episode(
        Episode(
            episode_id="reply",
            scope="s1",
            created_at=2000.0,
            steps=[
                StepRecord(
                    index=0,
                    state=build_state("alpha beta"),
                    action=normalize_action(kind="reply", name="reply", text="done"),
                )
            ],
        )
    )
    assert await bank.forget(older_than=1500.0) == 1
    assert await bank.count() == 1
    assert await bank.forget(action_key="reply:reply()") == 1
    assert await bank.count() == 0
    await bank.close()


@pytest.mark.asyncio
async def test_prune_low_value_removes_whole_episodes(tmp_path) -> None:
    """The low-value policy evicts entire episodes, lowest return first."""
    bank = MemoryBank(tmp_path / "memory.db", max_entries=4, prune_strategy="low_value")
    await bank.initialize()
    await bank.add_episode(build_episode("low", [0.0, 0.0], created_at=1000.0))
    await bank.add_episode(build_episode("mid", [3.0, 3.0], created_at=2000.0))
    assert await bank.count() == 4
    await bank.add_episode(build_episode("high", [9.0, 9.0], created_at=3000.0))
    assert await bank.count() == 4
    survivors = await bank.recent_episodes()
    assert [item["episode_id"] for item in survivors] == ["high", "mid"]
    assert all(item["step_count"] == 2 for item in survivors)
    await bank.close()


@pytest.mark.asyncio
async def test_prune_oldest_and_hard_cap(tmp_path) -> None:
    """The oldest policy and the none policy both trim by creation time."""
    oldest = MemoryBank(tmp_path / "oldest.db", max_entries=2, prune_strategy="oldest")
    await oldest.initialize()
    for index, created in enumerate([1000.0, 2000.0, 3000.0]):
        await oldest.add_episode(build_episode(f"e{index}", [1.0], created_at=created))
    assert await oldest.count() == 2
    assert [item["episode_id"] for item in await oldest.recent_episodes()] == [
        "e2",
        "e1",
    ]
    await oldest.close()

    none = MemoryBank(tmp_path / "none.db", max_entries=1, prune_strategy="none")
    await none.initialize()
    await none.add_episode(build_episode("first", [1.0], created_at=1000.0))
    await none.add_episode(build_episode("second", [1.0], created_at=2000.0))
    assert await none.count() == 1
    assert [item["episode_id"] for item in await none.recent_episodes()] == ["second"]
    await none.close()


@pytest.mark.asyncio
async def test_prune_explicit_call(tmp_path) -> None:
    """prune() evicts when the cap is lowered at runtime."""
    bank = MemoryBank(tmp_path / "memory.db", max_entries=100)
    await bank.initialize()
    await bank.add_episode(build_episode("x", [1.0, 1.0], created_at=1000.0))
    await bank.add_episode(build_episode("y", [2.0, 2.0], created_at=2000.0))
    assert await bank.count() == 4
    bank.max_entries = 2
    assert await bank.prune() == 2
    assert await bank.count() == 2
    await bank.close()


@pytest.mark.asyncio
async def test_prune_disabled(tmp_path) -> None:
    """A non-positive cap disables pruning entirely."""
    bank = MemoryBank(tmp_path / "memory.db", max_entries=0)
    await bank.initialize()
    for index in range(3):
        await bank.add_episode(
            build_episode(f"e{index}", [1.0], created_at=1000.0 + index)
        )
    assert await bank.count() == 3
    assert await bank.prune() == 0
    assert await bank.count() == 3
    await bank.close()


@pytest.mark.asyncio
async def test_export_import_roundtrip(tmp_path) -> None:
    """Exporting and re-importing preserves retrieval behaviour."""
    source = MemoryBank(tmp_path / "source.db")
    await source.initialize()
    await source.add_episode(
        build_episode("e1", [1.0, 2.0], observation="weather sunny beijing")
    )
    await source.add_episode(
        build_episode("e2", [3.0], observation="stock price apple")
    )
    payload = await source.export_json()
    assert payload["version"] == 1
    assert len(payload["episodes"]) == 2
    assert len(payload["experiences"]) == 3
    assert isinstance(payload["experiences"][0]["state_tokens"], list)
    assert isinstance(payload["experiences"][0]["action_args"], dict)

    target = MemoryBank(tmp_path / "target.db")
    await target.initialize()
    assert await target.import_json(payload) == 3
    assert await target.count() == 3
    query = build_state("weather sunny beijing")
    before = await source.topk(query, scope="s1", now=2000.0)
    after = await target.topk(query, scope="s1", now=2000.0)
    assert [(hit.action_key, round(hit.similarity, 6)) for hit in before] == [
        (hit.action_key, round(hit.similarity, 6)) for hit in after
    ]
    assert await source.action_stats() == await target.action_stats()
    assert await target.import_json({}) == 0
    await source.close()
    await target.close()


@pytest.mark.asyncio
async def test_import_skips_corrupt_rows(tmp_path) -> None:
    """Corrupt token payloads are stored but never break retrieval."""
    source = MemoryBank(tmp_path / "source.db")
    await source.initialize()
    await source.add_episode(
        build_episode("e1", [1.0], observation="weather sunny beijing")
    )
    payload = await source.export_json()
    payload["experiences"][0]["state_tokens"] = "not-json"
    payload["experiences"].append({"episode_id": "e1", "state_tokens": None})
    target = MemoryBank(tmp_path / "target.db")
    await target.initialize()
    assert await target.import_json(payload) == 2
    assert await target.topk(build_state("weather sunny beijing"), scope="s1") == []
    assert await target.count() == 2
    await source.close()
    await target.close()


@pytest.mark.asyncio
async def test_apply_user_feedback_recomputes_returns(tmp_path) -> None:
    """Feedback shifts every reward and rebuilds G_t with the stored gamma."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.add_episode(build_episode("fb", [1.0, 2.0, 4.0], gamma=0.5))
    assert await bank.apply_user_feedback("fb", 1.0) == 3
    payload = await bank.export_json()
    rows = {row["step_index"]: row for row in payload["experiences"]}
    assert rows[2]["reward"] == pytest.approx(5.0)
    assert rows[2]["discounted_return"] == pytest.approx(5.0)
    assert rows[1]["reward"] == pytest.approx(3.0)
    assert rows[1]["discounted_return"] == pytest.approx(5.5)
    assert rows[0]["reward"] == pytest.approx(2.0)
    assert rows[0]["discounted_return"] == pytest.approx(4.75)
    assert await bank.apply_user_feedback("missing", 1.0) == 0
    await bank.close()


@pytest.mark.asyncio
async def test_clear_empty_bank(tmp_path) -> None:
    """Clearing an empty bank is a harmless no-op."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    assert await bank.clear() == 0
    assert await bank.count() == 0
    await bank.close()


@pytest.mark.asyncio
async def test_clear_removes_experiences_and_headers(tmp_path) -> None:
    """clear() wipes every experience and its episode header."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.add_episode(build_episode("e1", [1.0, 2.0, 3.0], created_at=1000.0))
    await bank.add_episode(
        build_episode("e2", [4.0, 5.0], scope="s2", created_at=2000.0)
    )
    assert await bank.count() == 5
    assert await bank.clear() == 5
    assert await bank.count() == 0
    assert await bank.count(scope="s2") == 0
    assert await bank.recent_episodes() == []
    payload = await bank.export_json()
    assert payload["experiences"] == []
    assert payload["episodes"] == []
    await bank.close()


@pytest.mark.asyncio
async def test_clear_keeps_the_bank_usable(tmp_path) -> None:
    """The connection survives a clear, so learning can continue."""
    bank = MemoryBank(tmp_path / "memory.db")
    await bank.initialize()
    await bank.add_episode(build_episode("e1", [1.0, 2.0], created_at=1000.0))
    assert await bank.clear() == 2
    assert await bank.add_episode(build_episode("e2", [3.0], created_at=2000.0)) == 1
    assert await bank.count() == 1
    assert [item["episode_id"] for item in await bank.recent_episodes()] == ["e2"]
    hits = await bank.topk(build_state("alpha beta gamma"), scope="s1", now=3000.0)
    assert len(hits) == 1
    await bank.close()


@pytest.mark.asyncio
async def test_clear_requires_initialization(tmp_path) -> None:
    """Calling clear() before initialize() raises RuntimeError."""
    bank = MemoryBank(tmp_path / "memory.db")
    with pytest.raises(RuntimeError):
        await bank.clear()
