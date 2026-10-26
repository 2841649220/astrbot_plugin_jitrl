"""Neural hooks exercised through a minimal host and real persisted audit data."""

from __future__ import annotations

import ast
import asyncio
import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio
from test_checkpoint_plugin import Event, _seed, command
from test_checkpoint_plugin import host as host

from jitrl.evaluator import RewardResult
from jitrl.neural.contracts import (
    Intervention,
    LensDecision,
    MemoryRecord,
    RetrievalSnapshot,
)
from jitrl.neural.runtime import render_evidence

pytestmark = pytest.mark.asyncio


class Tools:
    """Tool set whose identity and contents must survive advisory inference."""

    def __init__(self, *names):
        self.items = tuple(names or ("search", "inspect", "delete"))

    def names(self):
        return self.items


def request(messages=(), *, tools=None):
    return SimpleNamespace(
        prompt="Find the report",
        system_prompt="Cached system prefix.\nKeep every byte unchanged.",
        contexts=list(messages),
        func_tool=tools or Tools(),
        extra_user_content_parts=[],
    )


def call(identifier, name="search"):
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": identifier,
                "type": "function",
                "function": {"name": name, "arguments": '{"q":"report"}'},
            }
        ],
    }


def result(identifier, text="Found the requested report"):
    return {"role": "tool", "tool_call_id": identifier, "content": text}


def audit_rows(plugin, query):
    with sqlite3.connect(plugin._neural.store.path) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute(query)]


@pytest_asyncio.fixture
async def plugin(host):
    context = SimpleNamespace(register_web_api=lambda *args: None)
    instance = host.module.JitRLLearner(
        context,
        {
            "learning": {
                "evaluator_mode": "heuristic",
                "async_learn": False,
                "success_detection": "off",
            },
            "memory": {"checkpoint_interval": 0},
            "neural": {"mode": "shadow"},
            "policy": {"tool_gate": True},
            "commands": {"admin_only": False},
        },
    )
    instance.cfg.neural = replace(
        instance.cfg.neural,
        mode="shadow",
        min_samples=2,
        hidden_dim=24,
        embedding_dim=32,
        summary_dim=16,
        blocks=1,
        engram_buckets=31,
    )
    try:
        yield instance
    finally:
        await instance.terminate()


def fixed_evaluator(monkeypatch, host, rewards):
    captured = []

    class Evaluator:
        def __init__(self, *args, **kwargs):
            pass

        async def score(self, **kwargs):
            captured.append(kwargs)
            assert len(kwargs["actions"]) == len(rewards)
            return RewardResult(rewards=list(rewards), source="heuristic", ok=True)

    monkeypatch.setattr(host.module, "StepRewardEvaluator", Evaluator)
    return captured


async def active_proposal(plugin, monkeypatch):
    """Provide a valid nonempty PRE snapshot without requiring model training."""
    plugin.cfg.neural = replace(plugin.cfg.neural, mode="active")
    runtime = await plugin._ensure_neural()
    proposed = []

    async def prepare(state):
        past = replace(
            state,
            decision_id="historical-decision",
            episode_id="historical-episode",
            observed_at=state.observed_at - 3,
        )
        memory = MemoryRecord(
            experience_id="memory-1",
            state=past,
            action_key="tool:search",
            action_name="search",
            intervention="NO_OP",
            return_g=2.0,
            reward_source="environment",
            reliability=1.0,
            settled_at=state.observed_at - 2,
            similarity=1.0,
        )
        snapshot = RetrievalSnapshot(
            "selected-snapshot", state.scope, state.observed_at, (memory,)
        )
        pre = Intervention("SELECT_EVIDENCE", (memory.experience_id,), "TEST_SUPPORT")
        decision = LensDecision(
            state,
            snapshot,
            pre,
            render_evidence(pre, snapshot, 256),
            "test-checkpoint",
        )
        proposed.append(decision)
        return decision, runtime.store.epoch(state.scope)

    monkeypatch.setattr(runtime, "prepare", prepare)
    return proposed


async def test_shadow_preserves_request_and_tool_identity(plugin, host, monkeypatch):
    async def legacy_must_not_run(*args, **kwargs):
        raise AssertionError("neural mode must not run the legacy hint path")

    monkeypatch.setattr(plugin, "_decide", legacy_must_not_run)
    event = Event(text="Find the report")
    tools = Tools("search", "delete")
    req = request(tools=tools)
    original_system = req.system_prompt
    original_contexts = list(req.contexts)
    await plugin.on_llm_request(event, req)

    assert req.system_prompt == original_system
    assert req.contexts == original_contexts
    assert req.func_tool is tools
    assert req.func_tool.names() == ("search", "delete")
    assert req.extra_user_content_parts == []
    assert plugin._bank is None
    run = event.get_extra(host.module._NEURAL_KEY)
    assert len(run.frames) == 1
    assert run.frames[0].decision.intervention.kind == "NO_OP"
    assert run.frames[0].decision.intervention.reason == "SHADOW_PURE_NO_OP"
    rows = audit_rows(plugin, "SELECT rendered,intervention_json FROM decisions")
    assert rows[0]["rendered"] == ""
    assert json.loads(rows[0]["intervention_json"])["kind"] == "NO_OP"


async def test_active_pre_is_temporary_and_audit_matches_attached_text(
    plugin, host, monkeypatch
):
    proposed = await active_proposal(plugin, monkeypatch)
    event = Event(text="Find the report")
    req = request()
    original_system, tools = req.system_prompt, req.func_tool
    await plugin.on_llm_request(event, req)

    assert len(req.extra_user_content_parts) == 1
    part = req.extra_user_content_parts[0]
    assert part.temporary is True
    assert part.text == proposed[0].rendered
    assert req.system_prompt == original_system
    assert req.func_tool is tools
    frame = event.get_extra(host.module._NEURAL_KEY).frames[0]
    assert frame.decision.snapshot is proposed[0].snapshot
    assert frame.decision.state.decision_id == proposed[0].state.decision_id
    row = audit_rows(
        plugin, "SELECT rendered,intervention_json,snapshot_json FROM decisions"
    )[0]
    assert row["rendered"] == part.text
    assert json.loads(row["intervention_json"])["kind"] == "SELECT_EVIDENCE"
    assert json.loads(row["snapshot_json"])["snapshot_id"] == "selected-snapshot"
    assert plugin._stats["biased"] == 1


@pytest.mark.parametrize("host_part_available", [True, False])
async def test_apply_off_or_missing_host_part_records_actual_no_op(
    plugin, host, monkeypatch, host_part_available
):
    await active_proposal(plugin, monkeypatch)
    if host_part_available:
        plugin.cfg.apply_mode = "off"
        expected = "MASTER_APPLY_OFF"
    else:
        monkeypatch.setattr(host.module, "_make_temporary_part", lambda text: None)
        expected = "HOST_TEXT_PART_UNAVAILABLE"
    event = Event(text="Find the report")
    req = request()
    await plugin.on_llm_request(event, req)

    assert req.extra_user_content_parts == []
    assert req.system_prompt.startswith("Cached system prefix.")
    frame = event.get_extra(host.module._NEURAL_KEY).frames[0]
    assert frame.decision.intervention.kind == "NO_OP"
    assert frame.decision.intervention.reason == expected
    row = audit_rows(plugin, "SELECT rendered,intervention_json FROM decisions")[0]
    assert row["rendered"] == ""
    assert json.loads(row["intervention_json"])["reason"] == expected


async def test_multistep_requests_observe_latest_results_with_fresh_decisions(
    plugin, host
):
    event = Event(text="Find the report")
    prior = [call("old"), result("old", "Prior conversation result")]
    await plugin.on_llm_request(event, request(prior))
    await plugin.on_llm_request(
        event,
        request([*prior, call("first"), result("first", "Report page one")]),
    )
    await plugin.on_llm_request(
        event,
        request(
            [
                *prior,
                call("first"),
                result("first", "Report page one"),
                call("second", "inspect"),
                result("second", "Latest report page two"),
            ]
        ),
    )

    frames = event.get_extra(host.module._NEURAL_KEY).frames
    assert [frame.decision.state.observation for frame in frames] == [
        "Prior conversation result",
        "Report page one",
        "Latest report page two",
    ]
    assert [frame.decision.state.step_index for frame in frames] == [0, 1, 2]
    assert len({frame.decision.state.decision_id for frame in frames}) == 3
    assert len({frame.decision.state.episode_id for frame in frames}) == 1
    assert [frame.observed_calls for frame in frames] == [
        frozenset(),
        frozenset({"first"}),
        frozenset({"first", "second"}),
    ]
    rows = audit_rows(plugin, "SELECT state_json,snapshot_json FROM decisions")
    for row in rows:
        state, snapshot = (
            json.loads(row["state_json"]),
            json.loads(row["snapshot_json"]),
        )
        assert snapshot["cutoff"] <= state["observed_at"]
        assert snapshot["scope"] == state["scope"]


async def test_done_labels_only_observed_current_actions_and_associates_final_reply(
    plugin, host, monkeypatch
):
    scored = fixed_evaluator(monkeypatch, host, [1.0, 2.0, 3.0])
    event = Event(text="Find the report")
    prior = [
        {"role": "user", "content": "Old task"},
        call("old", "delete"),
        result("old", "Old result"),
        {"role": "assistant", "content": "Old reply"},
    ]
    current = {"role": "user", "content": "Find the report"}
    await plugin.on_llm_request(event, request([*prior, current]))
    first_frame = event.get_extra(host.module._NEURAL_KEY).frames[0]
    first = [call("first"), result("first", "Located report directory")]
    await plugin.on_llm_request(event, request([*prior, current, *first]))
    second_frame = event.get_extra(host.module._NEURAL_KEY).frames[1]
    messages = [
        *prior,
        current,
        *first,
        call("missing", "delete"),  # A proposal, with no result.
        call("second", "inspect"),
        result("second", "Report contents validated"),
        {"role": "assistant", "content": "Intermediate wording"},
    ]
    await plugin.on_agent_done(
        event,
        SimpleNamespace(messages=messages),
        SimpleNamespace(completion_text="Here is the requested report"),
    )

    assert len(scored) == 1
    assert len(scored[0]["actions"]) == 3
    assert "delete" not in " ".join(scored[0]["actions"])
    assert "Old result" not in scored[0]["observations"]
    assert scored[0]["actions"][-1] == "Here is the requested report"
    records = audit_rows(
        plugin,
        "SELECT x.action_name,x.decision_id,s.reward,s.return_g FROM executions x "
        "JOIN settlements s USING(experience_id) ORDER BY x.execution_index",
    )
    assert [row["action_name"] for row in records] == ["search", "inspect", "reply"]
    assert [row["decision_id"] for row in records] == [
        first_frame.decision.state.decision_id,
        second_frame.decision.state.decision_id,
        second_frame.decision.state.decision_id,
    ]
    assert [row["reward"] for row in records] == [1.0, 2.0, 3.0]
    assert [row["return_g"] for row in records] == [2.75, 3.5, 3.0]
    assert plugin._neural.store.stats(event.unified_msg_origin) == {
        "decisions": 2,
        "executions": 3,
        "settlements": 3,
    }


async def test_evaluator_off_retains_pending_execution_without_zero_labels(plugin):
    plugin.cfg.learning.evaluator_mode = "off"
    event = Event(text="Find the report")
    await plugin.on_llm_request(event, request())
    await plugin.on_agent_done(
        event,
        SimpleNamespace(messages=[call("first"), result("first")]),
        SimpleNamespace(completion_text=""),
    )
    assert plugin._neural.store.stats(event.unified_msg_origin) == {
        "decisions": 1,
        "executions": 1,
        "settlements": 0,
    }
    assert plugin._neural.store.replay(event.unified_msg_origin) == []
    text = await command(plugin.jitrl_feedback(event, "good"))
    assert "0 个步骤" in text
    assert plugin._neural.store.stats(event.unified_msg_origin)["settlements"] == 0


async def test_duplicate_agent_done_cannot_relabel_or_recount(
    plugin, host, monkeypatch
):
    scored = fixed_evaluator(monkeypatch, host, [2.0])
    event = Event(text="Find the report")
    await plugin.on_llm_request(event, request())
    run_context = SimpleNamespace(messages=[call("first"), result("first")])
    response = SimpleNamespace(completion_text="")
    await plugin.on_agent_done(event, run_context, response)
    await plugin.on_agent_done(event, run_context, response)
    assert len(scored) == 1
    assert plugin._neural.store.stats(event.unified_msg_origin)["settlements"] == 1
    assert plugin._stats["episodes"] == 1
    assert plugin._stats["experiences"] == 1


async def test_reset_during_reward_evaluation_discards_stale_labels(
    plugin, host, monkeypatch
):
    started, release = asyncio.Event(), asyncio.Event()

    class Evaluator:
        def __init__(self, *args, **kwargs):
            pass

        async def score(self, **kwargs):
            started.set()
            await release.wait()
            return RewardResult(rewards=[2.0], source="heuristic", ok=True)

    monkeypatch.setattr(host.module, "StepRewardEvaluator", Evaluator)
    event = Event(text="Find the report")
    await plugin.on_llm_request(event, request())
    old_epoch = plugin._neural.store.epoch(event.unified_msg_origin)
    learning = asyncio.create_task(
        plugin.on_agent_done(
            event,
            SimpleNamespace(messages=[call("first"), result("first")]),
            SimpleNamespace(completion_text=""),
        )
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        assert plugin._neural.store.stats(event.unified_msg_origin)["executions"] == 1
        await plugin._neural.reset(event.unified_msg_origin)
    finally:
        release.set()
        await learning
    assert plugin._neural.store.epoch(event.unified_msg_origin) == old_epoch + 1
    assert plugin._neural.store.stats(event.unified_msg_origin) == {
        "decisions": 0,
        "executions": 0,
        "settlements": 0,
    }
    assert plugin._stats["experiences"] == 0


async def test_reset_between_prepare_and_commit_attaches_no_stale_context(
    plugin, host, monkeypatch
):
    proposed = await active_proposal(plugin, monkeypatch)
    runtime = plugin._neural
    original_prepare = runtime.prepare

    async def prepare_then_reset(state):
        decision, epoch = await original_prepare(state)
        await runtime.reset(state.scope)
        return decision, epoch

    monkeypatch.setattr(runtime, "prepare", prepare_then_reset)
    event = Event(text="Find the report")
    req = request()
    await plugin.on_llm_request(event, req)
    assert proposed
    assert req.extra_user_content_parts == []
    assert event.get_extra(host.module._NEURAL_KEY).frames == []
    assert runtime.store.stats(event.unified_msg_origin)["decisions"] == 0
    assert plugin._stats["biased"] == 0


@pytest_asyncio.fixture
async def legacy_plugin(host):
    """An unchanged V0.1.X-style configuration without any neural section."""
    routes = []
    instance = host.module.JitRLLearner(
        SimpleNamespace(register_web_api=lambda *args: routes.append(args)),
        {
            "apply_mode": "hint",
            "learning": {"evaluator_mode": "heuristic", "async_learn": False},
            "retrieval": {"scope_isolation": "session"},
            "memory": {"checkpoint_interval": 0},
            "commands": {"admin_only": False},
        },
    )
    instance.routes = routes
    try:
        yield instance
    finally:
        await instance.terminate()


async def seed_legacy_memory(instance):
    await instance._ensure_ready()
    for tool, reward in (("search", 2.0), ("inspect", -2.0)):
        await _seed(instance._bank, "trusted task", tool, reward)


async def test_existing_config_uses_legacy_memory_and_temporary_injection(
    legacy_plugin, host
):
    assert legacy_plugin.cfg.neural.mode == "off"
    assert legacy_plugin._neural is None
    await seed_legacy_memory(legacy_plugin)
    event = Event(text="trusted task")
    req = request(tools=Tools("search", "inspect"))
    system_prefix, tools = req.system_prompt, req.func_tool
    await legacy_plugin.on_llm_request(event, req)

    assert len(req.extra_user_content_parts) == 1
    assert req.extra_user_content_parts[0].temporary is True
    assert "<jitrl_experience>" in req.extra_user_content_parts[0].text
    assert req.system_prompt == system_prefix
    assert req.func_tool is tools
    assert legacy_plugin._neural is None
    assert event.get_extra(host.module._NEURAL_KEY) is None
    assert event.get_extra(host.module._EVENT_EXTRA_KEY)["decision"].applied
    assert legacy_plugin._decision_cache
    assert legacy_plugin._stats["biased"] == 1
    assert await legacy_plugin._bank.count("s1") == 2
    assert not legacy_plugin._neural_root.exists()


async def test_existing_dashboard_route_names_and_verbs_are_preserved(legacy_plugin):
    expected = {
        "overview": ["GET"],
        "curve": ["GET"],
        "actions": ["GET"],
        "episodes": ["GET"],
        "forget": ["POST"],
        "export": ["GET"],
        "checkpoints": ["GET"],
        "checkpoint": ["POST"],
        "rollback": ["POST"],
        "checkpoint_delete": ["POST"],
    }
    registered = {
        route: (handler, methods) for route, handler, methods, _ in legacy_plugin.routes
    }
    for name, verbs in expected.items():
        handler, methods = registered[f"/astrbot_plugin_jitrl/{name}"]
        assert methods == verbs
        assert handler.__self__ is legacy_plugin
        assert handler.__name__ == f"_web_{name}"
    assert legacy_plugin._neural is None


async def test_existing_command_registration_names_are_preserved(host):
    # The minimal host strips decorator metadata, so inspect the public command
    # declarations and additionally exercise existing commands below.
    source = await asyncio.to_thread(
        Path(host.module.__file__).read_text, encoding="utf-8"
    )
    tree = ast.parse(source)
    registrations = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "jitrl_group"
                and decorator.func.attr == "command"
            ):
                registrations[decorator.args[0].value] = node.name
    expected = {
        "status": "jitrl_status",
        "stats": "jitrl_stats",
        "curve": "jitrl_curve",
        "recall": "jitrl_recall_cmd",
        "forget": "jitrl_forget",
        "feedback": "jitrl_feedback",
        "export": "jitrl_export",
        "episodes": "jitrl_episodes",
        "checkpoint": "jitrl_checkpoint",
        "checkpoints": "jitrl_checkpoints",
        "rollback": "jitrl_rollback",
        "checkpoint_delete": "jitrl_checkpoint_delete",
    }
    assert {name: registrations[name] for name in expected} == expected
    assert registrations["lens"] == "jitrl_lens"


async def test_overview_adds_neural_without_changing_existing_payload(legacy_plugin):
    await seed_legacy_memory(legacy_plugin)
    await legacy_plugin.on_llm_request(Event(), request())
    response = await legacy_plugin._web_overview()
    assert response.status_code == 200
    payload = response.payload
    assert {
        "enabled",
        "apply_mode",
        "total",
        "stats",
        "hyper",
        "last_decision",
        "db_path",
    } <= payload.keys()
    assert payload["enabled"] is True
    assert payload["apply_mode"] == "hint"
    assert payload["total"] == 2
    assert payload["db_path"] == str(legacy_plugin._db_path)
    assert payload["stats"] == legacy_plugin._stats
    assert {
        "k",
        "threshold",
        "gamma",
        "beta",
        "exploration_rate",
        "ucb_alpha",
        "scope",
        "evaluator",
        "weighting",
        "exploration",
    } <= payload["hyper"].keys()
    assert payload["last_decision"] is not None
    assert {
        "reason",
        "chosen",
        "v_hat",
        "neighbors",
        "probabilities",
        "advantages",
    } <= payload["last_decision"].keys()
    assert payload["neural"] == {"mode": "off"}
    assert legacy_plugin._neural is None


async def test_existing_read_commands_continue_to_use_legacy_bank(legacy_plugin):
    await seed_legacy_memory(legacy_plugin)
    event = Event(text="trusted task")
    assert "本会话 2 条" in await command(legacy_plugin.jitrl_status(event))
    stats = await command(legacy_plugin.jitrl_stats(event))
    assert "search" in stats and "inspect" in stats
    assert "search" in await command(
        legacy_plugin.jitrl_recall_cmd(event, "trusted task")
    )
    assert legacy_plugin._neural is None


@pytest.mark.parametrize("action", ["status", "train", "reset"])
async def test_lens_command_obeys_global_command_disable(plugin, action):
    plugin.cfg.commands.enabled = False
    text = await command(plugin.jitrl_lens(Event(), action, "confirm"))
    assert "禁用" in text
    assert plugin._neural is None


@pytest.mark.parametrize("action", ["status", "train", "reset"])
async def test_lens_command_obeys_configured_admin_only(plugin, action):
    plugin.cfg.commands.admin_only = True
    text = await command(plugin.jitrl_lens(Event(admin=False), action, "confirm"))
    assert "管理员" in text
    assert plugin._neural is None


@pytest.mark.parametrize("action", ["train", "reset"])
async def test_lens_mutation_stays_admin_only_when_read_commands_are_public(
    plugin, action
):
    plugin.cfg.commands.admin_only = False
    text = await command(plugin.jitrl_lens(Event(admin=False), action, "confirm"))
    assert "管理员" in text
    assert plugin._neural is None


async def test_lens_status_reads_only_the_callers_scope(plugin):
    await plugin.on_llm_request(Event(scope="private-a"), request())
    await plugin.on_llm_request(Event(scope="private-b"), request())
    await plugin.on_llm_request(Event(scope="private-b"), request())
    text = await command(plugin.jitrl_lens(Event(admin=False, scope="private-a")))
    assert "Neural Lens：shadow" in text
    assert "决策 1 / 执行事实 0 / 回报事件 0" in text
    assert "检查点：untrained" in text
    assert plugin._neural.store.stats()["decisions"] == 3


async def test_lens_train_passes_exact_scope_and_reports_local_training_only(
    plugin, monkeypatch
):
    runtime = await plugin._ensure_neural()
    scopes = []

    async def train(scope):
        scopes.append(scope)
        return {
            "status": "trained",
            "checkpoint_id": "local-checkpoint",
            "samples": 4,
            "before_loss": 1.0,
            "after_loss": 0.4,
        }

    monkeypatch.setattr(runtime, "train", train)
    text = await command(plugin.jitrl_lens(Event(scope="private-a"), "train"))
    assert scopes == ["private-a"]
    assert "local-checkpoint" in text
    assert "样本 4" in text
    assert "不代表真实任务提升" in text


async def test_lens_reset_requires_confirm_and_invalidates_only_its_scope(
    plugin, host, monkeypatch
):
    await seed_legacy_memory(plugin)
    first = Event(scope="private-a")
    other = Event(scope="private-b")
    await plugin.on_llm_request(first, request())
    await plugin.on_llm_request(other, request())
    runtime = plugin._neural
    frame = first.get_extra(host.module._NEURAL_KEY).frames[0]
    old_epoch = runtime.store.epoch("private-a")
    text = await command(plugin.jitrl_lens(Event(scope="private-a"), "reset"))
    assert "confirm" in text
    assert runtime.store.epoch("private-a") == old_epoch
    assert runtime.store.stats("private-a")["decisions"] == 1

    text = await command(
        plugin.jitrl_lens(Event(scope="private-a"), "reset", "confirm")
    )
    assert "已重置当前会话" in text
    assert "V0.1 经验库独立保留" in text
    assert runtime.store.epoch("private-a") == old_epoch + 1
    assert runtime.store.stats("private-a")["decisions"] == 0
    assert runtime.store.stats("private-b")["decisions"] == 1
    assert await plugin._bank.count("s1") == 2
    assert not await runtime.commit(frame.decision, frame.epoch)

    # A callback created before the confirmed command cannot recreate deleted
    # facts or settlements after the command changes the scope generation.
    scored = fixed_evaluator(monkeypatch, host, [2.0])
    await plugin.on_agent_done(
        first,
        SimpleNamespace(messages=[call("stale"), result("stale")]),
        SimpleNamespace(completion_text=""),
    )
    assert scored == []
    assert runtime.store.stats("private-a") == {
        "decisions": 0,
        "executions": 0,
        "settlements": 0,
    }


async def completed_neural_episode(plugin, host, monkeypatch, *, scope="private-a"):
    """Two observed actions give known discounted returns and feedback credit."""
    fixed_evaluator(monkeypatch, host, [1.0, -1.0])
    event = Event(scope=scope, text="Find the report")
    await plugin.on_llm_request(event, request())
    await plugin.on_agent_done(
        event,
        SimpleNamespace(
            messages=[
                call("search-report"),
                result("search-report", "Located report"),
                call("inspect-report", "inspect"),
                result("inspect-report", "Report unreadable"),
            ]
        ),
        SimpleNamespace(completion_text=""),
    )
    assert plugin._neural.store.stats(scope)["settlements"] == 2
    return event


def settlement_history(plugin, scope):
    with sqlite3.connect(plugin._neural.store.path) as conn:
        conn.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in conn.execute(
                "SELECT x.execution_index,s.reward,s.return_g,s.source,s.sequence "
                "FROM settlements s JOIN executions x USING(experience_id) "
                "JOIN decisions d USING(decision_id) WHERE d.scope=? "
                "ORDER BY x.execution_index,s.sequence",
                (scope,),
            )
        ]


@pytest.mark.parametrize(
    "verdict,rewards,returns",
    [
        ("good", [3.0, 1.0], [3.5, 1.0]),
        ("bad", [-1.0, -3.0], [-2.5, -3.0]),
    ],
)
async def test_manual_neural_feedback_appends_history_and_uses_episode_gamma(
    plugin, host, monkeypatch, verdict, rewards, returns
):
    await completed_neural_episode(plugin, host, monkeypatch)
    original = settlement_history(plugin, "private-a")
    assert [row["return_g"] for row in original] == [0.5, -1.0]
    # This later decision has already captured historical rewards. Feedback
    # changes future retrieval but must never rewrite its causal snapshot.
    await plugin.on_llm_request(
        Event(scope="private-a", text="Find the report"), request()
    )
    snapshots = audit_rows(plugin, "SELECT decision_id,snapshot_json FROM decisions")
    assert any(json.loads(row["snapshot_json"])["memories"] for row in snapshots)
    plugin.cfg.learning.gamma = 0.9
    text = await command(plugin.jitrl_feedback(Event(scope="private-a"), verdict))
    assert "2 个步骤" in text
    assert "追加结算" in text
    assert "private-a" not in plugin._neural_last_episode

    history = settlement_history(plugin, "private-a")
    assert [row for row in history if row["source"] == "heuristic"] == original
    feedback = [row for row in history if row["source"] == "user"]
    assert [row["reward"] for row in feedback] == rewards
    assert [row["return_g"] for row in feedback] == returns
    assert (
        audit_rows(plugin, "SELECT decision_id,snapshot_json FROM decisions")
        == snapshots
    )
    replay = plugin._neural.store.replay("private-a")
    assert sorted(sample.return_g for sample in replay) == sorted(returns)
    repeated = await command(plugin.jitrl_feedback(Event(scope="private-a"), verdict))
    assert "0 个步骤" in repeated
    assert settlement_history(plugin, "private-a") == history


async def test_manual_neural_feedback_never_uses_another_scopes_last_episode(
    plugin, host, monkeypatch
):
    await completed_neural_episode(plugin, host, monkeypatch, scope="private-a")
    history = settlement_history(plugin, "private-a")
    text = await command(plugin.jitrl_feedback(Event(scope="private-b"), "bad"))
    assert "0 个步骤" in text
    assert settlement_history(plugin, "private-a") == history
    assert "private-a" in plugin._neural_last_episode
    assert plugin._neural.store.stats("private-b")["settlements"] == 0
    text = await command(plugin.jitrl_feedback(Event(scope="private-a"), "good"))
    assert "2 个步骤" in text


@pytest.mark.parametrize(
    "keyword,expected_rewards", [("谢谢！", [3.0, 1.0]), ("不对", [-1.0, -3.0])]
)
async def test_keyword_feedback_runs_once_at_start_of_next_neural_run(
    plugin, host, monkeypatch, keyword, expected_rewards
):
    await completed_neural_episode(plugin, host, monkeypatch)
    originals = audit_rows(plugin, "SELECT decision_id,snapshot_json FROM decisions")
    feedback_event = Event(scope="private-a", text=keyword)
    await plugin.on_llm_request(feedback_event, request())
    history = settlement_history(plugin, "private-a")
    user_events = [row for row in history if row["source"] == "user"]
    assert [row["reward"] for row in user_events] == expected_rewards
    assert "private-a" not in plugin._neural_last_episode
    assert (
        audit_rows(
            plugin,
            "SELECT decision_id,snapshot_json FROM decisions ORDER BY created_at LIMIT 1",
        )
        == originals
    )

    # Another request inside that same agent run cannot apply the keyword again.
    await plugin.on_llm_request(feedback_event, request())
    assert len(feedback_event.get_extra(host.module._NEURAL_KEY).frames) == 2
    assert settlement_history(plugin, "private-a") == history
    await plugin.on_llm_request(Event(scope="private-a", text=keyword), request())
    assert settlement_history(plugin, "private-a") == history


async def test_keyword_feedback_is_scope_isolated_and_honors_disable(
    plugin, host, monkeypatch
):
    await completed_neural_episode(plugin, host, monkeypatch)
    history = settlement_history(plugin, "private-a")
    await plugin.on_llm_request(Event(scope="private-b", text="不对"), request())
    assert settlement_history(plugin, "private-a") == history
    plugin.cfg.feedback.enabled = False
    await plugin.on_llm_request(Event(scope="private-a", text="谢谢"), request())
    assert settlement_history(plugin, "private-a") == history
    assert "private-a" in plugin._neural_last_episode
    # This setting disables keyword detection; the existing explicit command
    # retains its legacy meaning and can still apply an authorized verdict.
    text = await command(plugin.jitrl_feedback(Event(scope="private-a"), "good"))
    assert "2 个步骤" in text


@pytest.mark.parametrize("manual", [True, False])
async def test_expired_neural_feedback_does_not_create_user_labels(
    plugin, host, monkeypatch, manual
):
    await completed_neural_episode(plugin, host, monkeypatch)
    episode, timestamp, gamma, epoch = plugin._neural_last_episode["private-a"]
    plugin._neural_last_episode["private-a"] = (
        episode,
        timestamp - plugin.cfg.feedback.window_seconds - 1,
        gamma,
        epoch,
    )
    history = settlement_history(plugin, "private-a")
    if manual:
        text = await command(plugin.jitrl_feedback(Event(scope="private-a"), "good"))
        assert "0 个步骤" in text
    else:
        await plugin.on_llm_request(Event(scope="private-a", text="谢谢"), request())
    assert settlement_history(plugin, "private-a") == history
    assert "private-a" not in plugin._neural_last_episode


@pytest.mark.parametrize("manual", [True, False])
async def test_neural_reset_rejects_cached_feedback_from_old_generation(
    plugin, host, monkeypatch, manual
):
    await completed_neural_episode(plugin, host, monkeypatch)
    previous_epoch = plugin._neural.store.epoch("private-a")
    await command(plugin.jitrl_lens(Event(scope="private-a"), "reset", "confirm"))
    assert plugin._neural.store.epoch("private-a") == previous_epoch + 1
    if manual:
        text = await command(plugin.jitrl_feedback(Event(scope="private-a"), "bad"))
        assert "0 个步骤" in text
    else:
        await plugin.on_llm_request(Event(scope="private-a", text="不对"), request())
    assert "private-a" not in plugin._neural_last_episode
    assert plugin._neural.store.stats("private-a")["settlements"] == 0
    assert plugin._neural.store.replay("private-a") == []


async def test_neural_reset_during_manual_feedback_rejects_inflight_settlement(
    plugin, host, monkeypatch
):
    await completed_neural_episode(plugin, host, monkeypatch)
    runtime = plugin._neural
    original_apply = runtime.apply_user_feedback
    started, release = asyncio.Event(), asyncio.Event()

    async def delayed_apply(**kwargs):
        started.set()
        await release.wait()
        return await original_apply(**kwargs)

    monkeypatch.setattr(runtime, "apply_user_feedback", delayed_apply)
    feedback = asyncio.create_task(
        command(plugin.jitrl_feedback(Event(scope="private-a"), "good"))
    )
    try:
        await asyncio.wait_for(started.wait(), timeout=3)
        await command(plugin.jitrl_lens(Event(scope="private-a"), "reset", "confirm"))
    finally:
        release.set()
    text = await asyncio.wait_for(feedback, timeout=3)
    assert "0 个步骤" in text
    assert runtime.store.stats("private-a") == {
        "decisions": 0,
        "executions": 0,
        "settlements": 0,
    }


@dataclass
class HostMessage:
    role: str
    content: object = None
    tool_calls: list | None = None
    tool_call_id: str | None = None


@dataclass
class HostToolPacket:
    tool_calls_info: object
    tool_calls_result: list


def host_packet(identifier, text, *, object_messages=False, name="search"):
    """ToolCallsResult shape, including message/content objects used by hosts."""
    if object_messages:
        assistant = HostMessage(
            role="assistant",
            tool_calls=[
                SimpleNamespace(
                    id=identifier,
                    function=SimpleNamespace(name=name, arguments='{"q":"report"}'),
                )
            ],
        )
        tool_result = HostMessage(
            role="tool",
            tool_call_id=identifier,
            content=[SimpleNamespace(type="text", text=text)],
        )
        return HostToolPacket(assistant, [tool_result])
    return {
        "tool_calls_info": call(identifier, name),
        "tool_calls_result": [result(identifier, text)],
    }


@pytest.mark.parametrize("packet_list", [False, True])
@pytest.mark.parametrize("object_messages", [False, True])
async def test_singular_or_list_provider_tool_packets_update_observed_state(
    plugin, host, packet_list, object_messages
):
    event = Event(text="Find the report")
    await plugin.on_llm_request(event, request())
    packet = host_packet(
        "packet-search",
        "Report located in the shared directory",
        object_messages=object_messages,
    )
    req = request(
        [
            SimpleNamespace(role="system", content="Host context prefix"),
            HostMessage(role="user", content="Find the report"),
        ]
    )
    req.tool_calls_result = [packet] if packet_list else packet
    contexts, packets, tools = req.contexts, req.tool_calls_result, req.func_tool
    prefix = req.system_prompt
    await plugin.on_llm_request(event, req)

    frames = event.get_extra(host.module._NEURAL_KEY).frames
    assert len(frames) == 2
    latest = frames[-1]
    assert latest.decision.state.observation == "Report located in the shared directory"
    assert latest.decision.state.step_index == 1
    assert latest.observed_calls == frozenset({"packet-search"})
    assert latest.decision.state.events.count("tool_call:search") == 1
    assert latest.decision.state.events.count("tool_result:observed") == 1
    assert req.contexts is contexts and len(contexts) == 2
    assert req.tool_calls_result is packets
    assert req.func_tool is tools
    assert req.system_prompt == prefix
    assert req.extra_user_content_parts == []


async def test_duplicate_tool_packets_and_context_results_are_not_counted_twice(
    plugin, host
):
    event = Event(text="Find the report")
    await plugin.on_llm_request(event, request())
    already = host_packet("already", "Initial report listing", object_messages=True)
    fresh = host_packet(
        "fresh", "Latest report details", object_messages=True, name="inspect"
    )
    # The same first result is present in contexts and twice in the packet list.
    # A second packet contributes one new result, also repeated in that list.
    req = request([already.tool_calls_info, *already.tool_calls_result])
    req.tool_calls_result = [already, already, fresh, fresh]
    contexts, packets = req.contexts, req.tool_calls_result
    first_calls = already.tool_calls_info.tool_calls
    second_results = fresh.tool_calls_result
    await plugin.on_llm_request(event, req)
    frame = event.get_extra(host.module._NEURAL_KEY).frames[-1]
    assert frame.observed_calls == frozenset({"already", "fresh"})
    assert frame.decision.state.step_index == 2
    assert frame.decision.state.observation == "Latest report details"
    assert frame.decision.state.events.count("tool_call:search") == 1
    assert frame.decision.state.events.count("tool_call:inspect") == 1
    assert frame.decision.state.events.count("tool_result:observed") == 2
    assert req.contexts is contexts and len(contexts) == 2
    assert req.tool_calls_result is packets and len(packets) == 4
    assert already.tool_calls_info.tool_calls is first_calls and len(first_calls) == 1
    assert fresh.tool_calls_result is second_results and len(second_results) == 1

    await plugin.on_llm_request(event, req)
    latest = event.get_extra(host.module._NEURAL_KEY).frames[-1]
    assert latest.observed_calls == frame.observed_calls
    assert latest.decision.state.step_index == 2
    assert latest.decision.state.events == frame.decision.state.events


async def test_request_model_partitions_neural_history_without_mutating_request(
    plugin, host, monkeypatch
):
    fixed_evaluator(monkeypatch, host, [2.0])
    event = Event(scope="private-a", text="Find the report")
    initial = request()
    initial.model = "provider/model-a"
    await plugin.on_llm_request(event, initial)
    await plugin.on_agent_done(
        event,
        SimpleNamespace(messages=[call("report"), result("report")]),
        SimpleNamespace(completion_text=""),
    )
    frames = []
    for model in ("provider/model-b", "provider/model-a", None):
        current = Event(scope="private-a", text="Find the report")
        req = request()
        req.model = model
        prefix, tools = req.system_prompt, req.func_tool
        await plugin.on_llm_request(current, req)
        frame = current.get_extra(host.module._NEURAL_KEY).frames[0]
        frames.append(frame)
        assert req.model == model
        assert req.system_prompt == prefix
        assert req.func_tool is tools
        assert req.extra_user_content_parts == []
    other_model, same_model, unknown_model = frames
    assert other_model.decision.state.environment.endswith(":model=provider/model-b")
    assert same_model.decision.state.environment.endswith(":model=provider/model-a")
    assert unknown_model.decision.state.environment.endswith(":model=unknown")
    assert other_model.decision.snapshot.memories == ()
    assert len(same_model.decision.snapshot.memories) == 1
    assert unknown_model.decision.snapshot.memories == ()
    assert len({frame.decision.state.environment for frame in frames}) == 3
