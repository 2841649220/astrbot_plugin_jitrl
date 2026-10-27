"""End-to-end tests that drive the plugin class through AstrBot's contracts.

The core algorithm is covered by pure unit tests that never import AstrBot.
This module instead exercises the wiring: the real ``JitRLLearner`` class, the
real ``MemoryBank``, the real estimator/policy chain, and a fake event plus a
real ``ProviderRequest`` so that a regression in the hook contract fails here
instead of in production.

Run with::

    python -m pytest tests/test_integration.py -q
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

astrbot = pytest.importorskip("astrbot")
if getattr(astrbot, "__jitrl_stub__", False):
    pytest.skip(
        "AstrBot is not installed; the integration suite needs the real host",
        allow_module_level=True,
    )

from astrbot.core.agent.tool import FunctionTool, ToolSet  # noqa: E402
from astrbot.core.provider.entities import ProviderRequest  # noqa: E402

from jitrl.state import abstract_state, normalize_action  # noqa: E402
from jitrl.trajectory import Transition, build_episode  # noqa: E402

pytestmark = pytest.mark.asyncio
"""Every test here drives an async hook or an async command."""

UMO = "aiocqhttp:FriendMessage:10001"
"""Unified message origin used as the memory scope in every test."""

TASK = "帮我查一下明天北京的天气怎么样"
"""User request reused across tests so that Jaccard similarity is 1.0."""

CONFIG: dict[str, Any] = {
    "enabled": True,
    "apply_mode": "hint",
    "learning": {
        "enabled": True,
        "evaluator_mode": "heuristic",
        "async_learn": False,
        "gamma": 0.5,
        "min_steps": 1,
        "success_detection": "off",
    },
    "retrieval": {"k": 10, "similarity_threshold": 0.5, "scope_isolation": "session"},
    "policy": {"beta": 1.5, "candidate_limit": 3},
    "memory": {"max_entries": 5000, "redact_secrets": True},
    "feedback": {"enabled": True, "window_seconds": 900},
    "commands": {"enabled": True, "admin_only": False, "enable_agent_tools": True},
}
"""Plugin configuration used by the integration tests."""


class FakeEvent:
    """Minimal stand-in for ``AstrMessageEvent``.

    Only the attributes the plugin actually reads are implemented, so any new
    dependency on the event surface shows up as an ``AttributeError`` here.
    """

    def __init__(
        self,
        text: str = TASK,
        umo: str = UMO,
        platform_id: str = "platform-1",
        self_id: str = "bot-1",
    ) -> None:
        """Create a fake event.

        Args:
            text: Plain-text message content.
            umo: Unified message origin.
        """
        self.unified_msg_origin = umo
        self.platform_id = platform_id
        self.message_obj = SimpleNamespace(self_id=self_id)
        self._text = text
        self._extras: dict[str, Any] = {}
        self.results: list[str] = []

    def get_message_str(self) -> str:
        """Return the message text.

        Returns:
            The text handed to the constructor.
        """
        return self._text

    def set_extra(self, key: str, value: Any) -> None:
        """Store an event-scoped extra, mirroring AstrBot's API.

        Args:
            key: Extra key.
            value: Extra value.
        """
        self._extras[key] = value

    def get_extra(self, key: str | None = None, default: Any = None) -> Any:
        """Read an event-scoped extra.

        Args:
            key: Extra key.
            default: Value returned when the key is absent.

        Returns:
            The stored value or ``default``.
        """
        if key is None:
            return dict(self._extras)
        return self._extras.get(key, default)

    def get_platform_name(self) -> str:
        """Return the platform adapter name.

        Returns:
            Always ``"aiocqhttp"``.
        """
        return "aiocqhttp"

    def get_platform_id(self) -> str:
        """Return the configured platform-instance ID."""
        return self.platform_id

    def get_self_id(self) -> str:
        """Return the connected robot account ID."""
        return self.message_obj.self_id

    def is_admin(self) -> bool:
        """Report whether the sender is an AstrBot admin.

        Returns:
            Always ``True`` so command tests are not blocked.
        """
        return True

    def plain_result(self, text: str) -> str:
        """Record and return a plain-text result.

        Args:
            text: Message body.

        Returns:
            The same text, so the caller can ``yield`` it.
        """
        self.results.append(text)
        return text


class FakeProviderManager:
    """Provider manager stub: no provider is ever resolvable."""

    async def get_provider_by_id(self, provider_id: str) -> None:
        """Resolve a provider id.

        Args:
            provider_id: Requested provider id.

        Returns:
            Always ``None``.
        """
        return None


class FakeContext:
    """Plugin context stub carrying only what the plugin touches."""

    def __init__(self) -> None:
        """Create the stub context."""
        self.provider_manager = FakeProviderManager()
        self.registered_web_apis: list[tuple[str, Any, list[str], str]] = []
        self.provider: Any = None

    def register_web_api(
        self,
        route: str,
        view_handler: Any,
        methods: list[str],
        desc: str,
    ) -> None:
        """Record a dashboard endpoint registration.

        Args:
            route: API route.
            view_handler: Async handler.
            methods: HTTP methods.
            desc: Human-readable description.
        """
        self.registered_web_apis.append((route, view_handler, methods, desc))

    async def get_using_provider_async(self, umo: str | None = None) -> None:
        """Resolve the session provider.

        Args:
            umo: Unified message origin.

        Returns:
            The stub provider assigned by the test, or ``None`` by default.
        """
        return self.provider


class FakeProvider:
    """Provider stub that returns a canned completion and counts calls."""

    def __init__(self, completion: str) -> None:
        """Create the stub provider.

        Args:
            completion: Text returned by every completion call.
        """
        self.completion = completion
        self.calls: list[dict[str, Any]] = []

    async def text_chat(self, **kwargs: Any) -> Any:
        """Record the call and return the canned completion.

        Args:
            **kwargs: Forwarded arguments recorded for assertions.

        Returns:
            An object exposing ``completion_text``.
        """
        self.calls.append(kwargs)
        return SimpleNamespace(completion_text=self.completion)


def _make_tool(name: str) -> FunctionTool:
    """Build a schema-only function tool.

    Args:
        name: Tool name.

    Returns:
        A ``FunctionTool`` with empty parameters and no handler.
    """
    return FunctionTool(
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {}},
        handler=None,
    )


def _tool_set(*names: str) -> ToolSet:
    """Build a tool set from names.

    Args:
        *names: Tool names to include.

    Returns:
        A ``ToolSet`` holding one schema-only tool per name.
    """
    tool_set = ToolSet()
    for name in names:
        tool_set.add_tool(_make_tool(name))
    return tool_set


def _request(system_prompt: str = "你是一个乐于助人的助手。") -> ProviderRequest:
    """Build an outgoing provider request with two tools.

    Args:
        system_prompt: Base system prompt to extend.

    Returns:
        A ``ProviderRequest`` exposing ``web_search`` and ``bad_tool``.
    """
    return ProviderRequest(
        prompt=TASK,
        system_prompt=system_prompt,
        func_tool=_tool_set("web_search", "bad_tool"),
        contexts=[],
    )


def _injected_text(request: ProviderRequest) -> str:
    """Return the text of the temporary parts the plugin appended to the user message.

    The plugin is not allowed to rewrite ``system_prompt`` (that would break
    prompt caching), so the experience block arrives as an extra user content
    part. AstrBot accepts both ``ContentPart`` objects and the equivalent dict.

    Args:
        request: The outgoing provider request.

    Returns:
        The concatenated text of every injected content part.
    """
    return "\n".join(
        part["text"] if isinstance(part, dict) else part.text
        for part in request.extra_user_content_parts
    )


@pytest.fixture
def plugin(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """Create a plugin instance backed by a throwaway database.

    Args:
        monkeypatch: Pytest monkeypatch helper.
        tmp_path: Temporary directory for the SQLite file.

    Returns:
        A configured ``JitRLLearner`` whose data path points at ``tmp_path``.
    """
    from astrbot_plugin_jitrl.main import JitRLLearner

    monkeypatch.setattr(
        "astrbot_plugin_jitrl.main.get_astrbot_plugin_data_path",
        lambda: str(tmp_path),
    )
    instance = JitRLLearner(FakeContext(), dict(CONFIG))
    assert instance._legacy_db_path == (
        tmp_path / "astrbot_plugin_jitrl" / "memory.sqlite3"
    )
    return instance


async def _seed(
    bank: Any, task: str, tool: str, reward: float, scope: str = UMO
) -> None:
    """Persist a single-step episode with a known reward.

    Args:
        bank: An initialised ``MemoryBank``.
        task: User request that determines the stored state.
        tool: Tool name used as the action.
        reward: Step reward to store.
        scope: Memory scope.
    """
    transition = Transition(
        state=abstract_state(task=task),
        action=normalize_action(kind="tool", name=tool, args={"q": "x"}),
        observation="ok",
        outcome="ok",
    )
    episode = build_episode(
        episode_id=uuid.uuid4().hex,
        scope=scope,
        user_text=task,
        transitions=[transition],
        rewards=[reward],
        gamma=0.5,
    )
    await bank.add_episode(episode)


def _agent_messages() -> list[dict[str, Any]]:
    """Build a realistic assistant/tool message history.

    Returns:
        A list of OpenAI-shaped messages with one tool call and a final reply.
    """
    return [
        {"role": "system", "content": "你是一个乐于助人的助手。"},
        {"role": "user", "content": TASK},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"q": "北京 明天 天气"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "晴，21 度"},
        {"role": "assistant", "content": "明天北京晴，最高 21 度。"},
    ]


async def test_request_without_memory_does_not_touch_prompt(plugin: Any) -> None:
    """An empty memory bank must leave the request untouched."""
    event = FakeEvent()
    request = _request()
    original = request.system_prompt
    await plugin.on_llm_request(event, request)
    assert request.system_prompt == original
    assert request.extra_user_content_parts == []
    assert request.func_tool is not None
    assert set(request.func_tool.names()) == {"web_search", "bad_tool"}
    assert plugin._stats["requests"] == 0


async def test_request_injects_advantage_ranked_block(plugin: Any) -> None:
    """Retrieval must yield a signed, number-carrying preference block."""
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 2.0)
    await _seed(plugin._bank, TASK, "bad_tool", -2.0)

    event = FakeEvent()
    request = _request()
    original = request.system_prompt
    await plugin.on_llm_request(event, request)

    block = _injected_text(request)
    assert "<jitrl_experience>" in block
    assert "</jitrl_experience>" in block
    assert "web_search" in block
    assert "bad_tool" in block
    assert "优势" in block
    assert "不要生搬硬套" in block
    assert request.system_prompt == original

    context = event.get_extra("_jitrl_context")
    assert context is not None
    decision = context["decision"]
    assert decision.chosen is not None
    assert set(decision.probabilities) == set(decision.estimate.advantage_norm)
    assert plugin._stats["requests"] == 1
    assert plugin._stats["biased"] == 1


async def test_injected_block_is_temporary_and_keeps_the_cache_prefix(
    plugin: Any,
) -> None:
    """The block must ride on the user message, not on the cached system prompt.

    AstrBot's plugin review rejects per-request ``system_prompt`` rewrites: the
    system prompt is the cached context prefix, so changing it lowers the prompt
    cache hit rate. The injected part must also be marked provider-only so it is
    never written into the persisted conversation history.
    """
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 2.0)
    await _seed(plugin._bank, TASK, "bad_tool", -2.0)

    request = _request()
    original = request.system_prompt
    await plugin.on_llm_request(FakeEvent(), request)

    assert request.system_prompt == original
    assert len(request.extra_user_content_parts) == 1
    part = request.extra_user_content_parts[0]
    assert part.type == "text"
    assert part._no_save is True
    assert "<jitrl_experience>" in part.text


async def test_request_is_injected_at_most_once(plugin: Any) -> None:
    """A second hook call on the same event must not duplicate the block."""
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 2.0)
    event = FakeEvent()
    request = _request()
    await plugin.on_llm_request(event, request)
    first = _injected_text(request)
    await plugin.on_llm_request(event, request)
    assert _injected_text(request) == first
    assert _injected_text(request).count("<jitrl_experience>") == 1


async def test_tool_gate_masks_confidently_negative_tool(plugin: Any) -> None:
    """The gate must drop the losing tool but keep the winning one."""
    plugin.cfg.policy.tool_gate = True
    plugin.cfg.policy.tool_gate_min_count = 1
    plugin.cfg.policy.tool_gate_threshold = -0.3
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 2.0)
    await _seed(plugin._bank, TASK, "bad_tool", -2.0)

    request = _request()
    await plugin.on_llm_request(FakeEvent(), request)

    assert request.func_tool is not None
    assert request.func_tool.names() == ["web_search"]


async def test_agent_done_persists_episode(plugin: Any) -> None:
    """A finished agent run must become retrievable experience."""
    event = FakeEvent()
    await plugin.on_llm_request(event, _request())
    run_context = SimpleNamespace(messages=_agent_messages())
    response = SimpleNamespace(completion_text="明天北京晴，最高 21 度。")

    await plugin.on_agent_done(event, run_context, response)
    await asyncio.sleep(0)

    assert plugin._bank is not None
    assert await plugin._bank.count() > 0
    episodes = await plugin._bank.recent_episodes(limit=5)
    assert episodes
    assert episodes[0]["step_count"] >= 1
    assert plugin._stats["episodes"] == 1
    assert any(key.endswith(f":{UMO}") for key in plugin._last_episode)


async def test_captured_experience_biases_the_next_request(plugin: Any) -> None:
    """The full loop: learn a run, then retrieve it on the next request."""
    event = FakeEvent()
    await plugin.on_llm_request(event, _request())
    await plugin.on_agent_done(
        event,
        SimpleNamespace(messages=_agent_messages()),
        SimpleNamespace(completion_text="明天北京晴，最高 21 度。"),
    )

    next_event = FakeEvent()
    next_request = _request()
    await plugin.on_llm_request(next_event, next_request)
    assert "<jitrl_experience>" in _injected_text(next_request)


async def test_user_feedback_updates_returns(plugin: Any) -> None:
    """A short verdict must be folded into the previous episode."""
    event = FakeEvent()
    await plugin.on_llm_request(event, _request())
    await plugin.on_agent_done(
        event,
        SimpleNamespace(messages=_agent_messages()),
        SimpleNamespace(completion_text="明天北京晴，最高 21 度。"),
    )
    before = await plugin._bank.export_json()
    returns_before = [row["discounted_return"] for row in before["experiences"]]

    feedback_event = FakeEvent(text="不对")
    await plugin.on_llm_request(feedback_event, _request())

    after = await plugin._bank.export_json()
    returns_after = [row["discounted_return"] for row in after["experiences"]]
    assert len(returns_after) == len(returns_before)
    assert returns_after != returns_before
    assert all(
        after_value < before_value
        for before_value, after_value in zip(returns_before, returns_after, strict=True)
    )
    assert not any(key.endswith(f":{UMO}") for key in plugin._last_episode)


async def test_status_command_reports_state(plugin: Any) -> None:
    """The status command must return a readable summary."""
    event = FakeEvent()
    results = [chunk async for chunk in plugin.jitrl_status(event)]
    assert results
    assert "JitRL" in results[0]
    assert "记忆" in results[0]


async def test_stats_and_curve_commands_handle_empty_memory(plugin: Any) -> None:
    """Empty-memory commands must answer instead of raising."""
    stats = [chunk async for chunk in plugin.jitrl_stats(FakeEvent())]
    curve = [chunk async for chunk in plugin.jitrl_curve(FakeEvent())]
    assert stats and "经验" in stats[0]
    assert curve and "学习曲线" in curve[0]


async def test_recall_tool_returns_report(plugin: Any) -> None:
    """The agent tool must answer even when nothing matches."""
    empty = await plugin.jitrl_recall(FakeEvent(), "完全陌生的情景 xyz")
    assert "No relevant experience" in empty

    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 2.0)
    await _seed(plugin._bank, TASK, "bad_tool", -2.0)
    report = await plugin.jitrl_recall(FakeEvent(), TASK)
    assert "Retrieved" in report
    assert "web_search" in report


async def test_forget_command_clears_scope(plugin: Any) -> None:
    """The forget command must remove the current scope's memory."""
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 2.0)
    assert await plugin._bank.count() == 1
    results = [chunk async for chunk in plugin.jitrl_forget(FakeEvent(), "session")]
    assert results
    assert await plugin._bank.count() == 0


async def test_terminate_closes_the_bank(plugin: Any) -> None:
    """Unloading the plugin must release the database handle."""
    await plugin._ensure_ready(FakeEvent())
    assert plugin._bank is not None
    await plugin.terminate()
    assert plugin._bank is None


async def test_apply_mode_off_never_injects(plugin: Any) -> None:
    """Mode ``off`` still learns but must not bias requests."""
    plugin.cfg.apply_mode = "off"
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 2.0)
    request = _request()
    original = request.system_prompt
    await plugin.on_llm_request(FakeEvent(), request)
    assert request.system_prompt == original


async def test_learning_can_be_disabled(plugin: Any) -> None:
    """Disabling learning must skip persistence entirely."""
    plugin.cfg.learning.enabled = False
    event = FakeEvent()
    await plugin.on_agent_done(
        event,
        SimpleNamespace(messages=_agent_messages()),
        SimpleNamespace(completion_text="明天北京晴。"),
    )
    assert plugin._bank is None


async def test_disabled_plugin_is_inert(plugin: Any) -> None:
    """The master switch must short-circuit both hooks."""
    plugin.cfg.enabled = False
    request = _request()
    original = request.system_prompt
    await plugin.on_llm_request(FakeEvent(), request)
    await plugin.on_agent_done(
        FakeEvent(),
        SimpleNamespace(messages=_agent_messages()),
        SimpleNamespace(completion_text="x"),
    )
    assert request.system_prompt == original
    assert plugin._bank is None


class _FakeHttpRequest:
    """Minimal stand-in for the Starlette request behind PluginRequest."""

    def __init__(
        self,
        method: str = "GET",
        query: dict[str, str] | None = None,
        body: Any = None,
    ) -> None:
        """Create a fake HTTP request.

        Args:
            method: HTTP method.
            query: Query parameter mapping.
            body: Parsed JSON body.
        """
        self.method = method
        self.url = SimpleNamespace(path="/api/v1/plugins/extensions")
        self.headers = {"content-type": "application/json"}
        self.cookies: dict[str, str] = {}
        self.client = SimpleNamespace(host="127.0.0.1")
        self.query_params = SimpleNamespace(
            multi_items=lambda: list((query or {}).items())
        )
        self._body = body if body is not None else {}

    async def json(self) -> Any:
        """Return the parsed JSON body.

        Returns:
            The body handed to the constructor.
        """
        return self._body

    async def body(self) -> bytes:
        """Return an empty raw body.

        Returns:
            An empty byte string.
        """
        return b""


async def _call_web(plugin: Any, handler_name: str, http: Any) -> Any:
    """Invoke a dashboard handler inside a bound request context.

    Args:
        plugin: The plugin instance.
        handler_name: Name of the handler method.
        http: Fake HTTP request.

    Returns:
        The handler response object.
    """
    from astrbot.api.web import PluginRequest, bind_request_context

    handler = getattr(plugin, handler_name)
    with bind_request_context(PluginRequest(http)):
        return await handler()


def _payload(response: Any) -> Any:
    """Decode a JSONResponse body.

    Args:
        response: Response returned by a dashboard handler.

    Returns:
        The decoded JSON payload.
    """
    import json

    return json.loads(bytes(response.body).decode("utf-8"))


async def test_web_apis_are_registered(plugin: Any) -> None:
    """All dashboard routes must be registered on the context."""
    routes = {entry[0] for entry in plugin.context.registered_web_apis}
    assert routes == {
        "/astrbot_plugin_jitrl/robots",
        "/astrbot_plugin_jitrl/overview",
        "/astrbot_plugin_jitrl/curve",
        "/astrbot_plugin_jitrl/actions",
        "/astrbot_plugin_jitrl/episodes",
        "/astrbot_plugin_jitrl/forget",
        "/astrbot_plugin_jitrl/export",
        "/astrbot_plugin_jitrl/checkpoints",
        "/astrbot_plugin_jitrl/checkpoint",
        "/astrbot_plugin_jitrl/rollback",
        "/astrbot_plugin_jitrl/checkpoint_delete",
    }
    for _route, handler, methods, desc in plugin.context.registered_web_apis:
        assert callable(handler)
        assert methods in (["GET"], ["POST"])
        assert desc


async def test_web_overview_reports_state(plugin: Any) -> None:
    """The overview endpoint must expose counters and hyperparameters."""
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 2.0)
    data = _payload(await _call_web(plugin, "_web_overview", _FakeHttpRequest()))
    assert data["total"] == 1
    assert data["apply_mode"] == "hint"
    assert data["hyper"]["k"] == 10
    assert data["db_path"].endswith("memory.sqlite3")


async def test_web_curve_and_actions_return_lists(plugin: Any) -> None:
    """Curve, actions and episodes endpoints must return JSON lists."""
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 2.0)
    curve = _payload(
        await _call_web(plugin, "_web_curve", _FakeHttpRequest(query={"buckets": "4"}))
    )
    assert isinstance(curve["buckets"], list)
    assert len(curve["buckets"]) >= 1
    actions = _payload(await _call_web(plugin, "_web_actions", _FakeHttpRequest()))
    assert actions["actions"]
    assert actions["actions"][0]["action_name"] == "web_search"
    episodes = _payload(await _call_web(plugin, "_web_episodes", _FakeHttpRequest()))
    assert len(episodes["episodes"]) == 1


async def test_web_forget_all_clears_memory(plugin: Any) -> None:
    """The forget endpoint must clear the bank when scope is all."""
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 2.0)
    data = _payload(
        await _call_web(
            plugin,
            "_web_forget",
            _FakeHttpRequest(method="POST", body={"scope": "all", "confirm": True}),
        )
    )
    assert data["removed"] == 1
    assert await plugin._bank.count() == 0


async def test_web_forget_rejects_bad_scope(plugin: Any) -> None:
    """An invalid scope must return the error envelope, not raise."""
    await plugin._ensure_ready(FakeEvent())
    data = _payload(
        await _call_web(
            plugin,
            "_web_forget",
            _FakeHttpRequest(method="POST", body={"scope": "bogus"}),
        )
    )
    assert data["status"] == "error"


async def test_learning_prefers_the_rewarding_action(plugin: Any) -> None:
    """Algorithm 1 must rank the useful action above the harmful one.

    Episodes are replayed into the memory bank with distinct returns, then the
    request-time estimate must assign a positive advantage to the action that
    historically paid off and a negative one to the action that did not.
    """
    await plugin._ensure_ready(FakeEvent())
    for _ in range(4):
        await _seed(plugin._bank, TASK, "web_search", 3.0)
        await _seed(plugin._bank, TASK, "bad_tool", -3.0)

    request = _request()
    event = FakeEvent()
    await plugin.on_llm_request(event, request)
    decision = event.get_extra("_jitrl_context")["decision"]

    advantages = decision.estimate.advantage_norm
    good = next(key for key in advantages if "web_search" in key)
    bad = next(key for key in advantages if "bad_tool" in key)
    assert advantages[good] > 0.9
    assert advantages[bad] < -0.9
    assert decision.probabilities[good] > decision.probabilities[bad]
    assert "<jitrl_experience>" in _injected_text(request)


async def test_advantage_normalisation_bounds_the_update(plugin: Any) -> None:
    """Eq. 25 must keep every normalised advantage inside [-1, 1]."""
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 9.0)
    await _seed(plugin._bank, TASK, "bad_tool", -1.0)
    event = FakeEvent()
    await plugin.on_llm_request(event, _request())
    decision = event.get_extra("_jitrl_context")["decision"]
    assert decision.estimate.advantage_norm
    for value in decision.estimate.advantage_norm.values():
        assert -1.0 <= value <= 1.0


async def test_learning_curve_is_time_ordered(plugin: Any) -> None:
    """Buckets must be ordered from the oldest to the newest episode."""
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", -3.0)
    await _seed(plugin._bank, TASK, "web_search", 3.0)
    rows = await plugin._bank.learning_curve(buckets=2)
    assert len(rows) == 2
    assert rows[0]["mean_return"] < rows[-1]["mean_return"]


async def test_episode_persists_the_configured_gamma(plugin: Any) -> None:
    """Feedback recomputation must use the gamma the episode was scored with."""
    plugin.cfg.learning.gamma = 0.1
    event = FakeEvent()
    await plugin.on_agent_done(
        event,
        SimpleNamespace(messages=_agent_messages()),
        SimpleNamespace(completion_text="明天北京晴，最高 21 度。"),
    )
    exported = await plugin._bank.export_json()
    assert exported["episodes"]
    assert exported["episodes"][0]["gamma"] == 0.1


async def test_unknown_session_feedback_is_ignored(plugin: Any) -> None:
    """A verdict without a recent episode must be a no-op, not an error."""
    await plugin.on_llm_request(FakeEvent(text="不对"), _request())
    assert plugin._bank is not None
    assert await plugin._bank.count() == 0


async def test_logit_mode_uses_verbalized_confidences(plugin: Any) -> None:
    """The paper's logit path must turn confidences into base logits."""
    import math

    plugin.cfg.apply_mode = "logit"
    plugin.context.provider = FakeProvider(
        '{"scores": [{"id": 1, "confidence": 100}, {"id": 2, "confidence": 50}], "best": 1}'
    )
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 3.0)
    await _seed(plugin._bank, TASK, "bad_tool", -3.0)

    event = FakeEvent()
    request = _request()
    await plugin.on_llm_request(event, request)

    provider = plugin.context.provider
    assert len(provider.calls) == 1
    assert "confidence" in provider.calls[0]["prompt"]

    decision = event.get_extra("_jitrl_context")["decision"]
    assert decision.mode == "logit"
    logits = sorted(decision.logits_before.values())
    assert logits[0] == pytest.approx(math.log(0.5), abs=1e-6)
    assert logits[1] == pytest.approx(0.0, abs=1e-6)
    assert decision.chosen is not None
    assert "按闭式更新后的策略采样" in _injected_text(request)


async def test_logit_mode_survives_a_broken_reranker(plugin: Any) -> None:
    """A garbage model answer must degrade to a neutral prior, not crash."""
    plugin.cfg.apply_mode = "logit"
    plugin.context.provider = FakeProvider("sorry, I cannot answer that")
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 3.0)

    event = FakeEvent()
    request = _request()
    await plugin.on_llm_request(event, request)

    def _neutral(confidence: float) -> float:
        import math

        return math.log(max(confidence, 1e-4))

    decision = event.get_extra("_jitrl_context")["decision"]
    assert decision.chosen is not None
    for value in decision.logits_before.values():
        assert value == pytest.approx(_neutral(0.5), abs=1e-6)
    assert "<jitrl_experience>" in _injected_text(request)


async def test_logit_mode_without_provider_keeps_zero_prior(plugin: Any) -> None:
    """With no provider the reranker must be skipped and z(s, a) stays 0."""
    plugin.cfg.apply_mode = "logit"
    plugin.context.provider = None
    await plugin._ensure_ready(FakeEvent())
    await _seed(plugin._bank, TASK, "web_search", 3.0)

    event = FakeEvent()
    await plugin.on_llm_request(event, _request())
    decision = event.get_extra("_jitrl_context")["decision"]
    assert decision.chosen is not None
    assert all(value == 0.0 for value in decision.logits_before.values())
