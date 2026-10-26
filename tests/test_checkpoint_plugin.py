"""Checkpoint wiring tests with a small host adapter, no AstrBot install needed.

The optional integration suite still checks the real AstrBot contracts. These
tests exercise plugin behaviour and races on every CI run, using the real bank.
"""

from __future__ import annotations

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import pytest_asyncio

from jitrl.evaluator import RewardResult
from jitrl.trajectory import extract_transitions

pytestmark = pytest.mark.asyncio
ROOT = Path(__file__).resolve().parents[1]
MESSAGES = [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "search", "arguments": '{"q":"task"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c1", "content": "ok"},
]


class Event:
    def __init__(self, *, admin=True, scope="s1", text="trusted task"):
        self.unified_msg_origin = scope
        self.admin = admin
        self.text = text
        self.extra = {}

    def is_admin(self):
        return self.admin

    def get_message_str(self):
        return self.text

    def get_platform_name(self):
        return "test"

    def get_extra(self, key, default=None):
        return self.extra.get(key, default)

    def set_extra(self, key, value):
        self.extra[key] = value

    def plain_result(self, text):
        return text


@pytest.fixture
def host(monkeypatch, tmp_path):
    names = (
        "astrbot",
        "astrbot.api",
        "astrbot.api.event",
        "astrbot.api.provider",
        "astrbot.api.star",
        "astrbot.api.web",
        "astrbot.core",
        "astrbot.core.utils",
        "astrbot.core.utils.astrbot_path",
    )
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def decorator(*args, **kwargs):
        return lambda fn: fn

    def command_group(*args, **kwargs):
        def decorate(fn):
            fn.command = decorator
            return fn

        return decorate

    class Star:
        def __init__(self, context):
            self.context = context
            self.logger = logging.getLogger("checkpoint-test")

    modules["astrbot.api"].logger = logging.getLogger("checkpoint-test")
    modules["astrbot.api.event"].AstrMessageEvent = Event
    modules["astrbot.api.event"].filter = SimpleNamespace(
        on_llm_request=decorator,
        on_agent_done=decorator,
        llm_tool=decorator,
        command_group=command_group,
    )
    modules["astrbot.api.provider"].ProviderRequest = SimpleNamespace
    modules["astrbot.api.star"].Star = Star
    modules["astrbot.api.star"].Context = SimpleNamespace
    modules["astrbot.core.utils.astrbot_path"].get_astrbot_plugin_data_path = lambda: (
        str(tmp_path)
    )
    web = modules["astrbot.api.web"]
    web.json_response = lambda payload: SimpleNamespace(
        status_code=200, payload=payload
    )
    web.error_response = lambda message, status_code=400: SimpleNamespace(
        status_code=status_code, payload={"status": "error", "message": message}
    )
    package = ModuleType("_jitrl_checkpoint_tests")
    package.__path__ = [str(ROOT)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    spec = importlib.util.spec_from_file_location(
        f"{package.__name__}.main", ROOT / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return SimpleNamespace(module=module, web=web)


@pytest_asyncio.fixture
async def plugin(host):
    routes = []
    context = SimpleNamespace(register_web_api=lambda *args: routes.append(args))
    plugin = host.module.JitRLLearner(
        context,
        {
            "learning": {"evaluator_mode": "heuristic", "async_learn": False},
            "memory": {"checkpoint_interval": 0},
            "commands": {"admin_only": False},
        },
    )
    plugin.routes = routes
    try:
        await plugin._ensure_ready()
        yield plugin
    finally:
        await plugin.terminate()


async def command(generator):
    return "\n".join([message async for message in generator])


async def web_call(host, plugin, method, payload):
    async def json_body(default=None):
        return payload

    host.web.request = SimpleNamespace(json=json_body)
    return await getattr(plugin, method)()


async def learn(plugin, scope="s1"):
    await plugin._learn(
        umo=scope,
        user_text="trusted task",
        surface="test",
        transitions=extract_transitions(
            messages=MESSAGES, user_text="trusted task", surface="test"
        ),
        reply_text="ok",
    )


async def test_checkpoint_routes_and_manual_command(plugin):
    routes = {row[0]: row[2] for row in plugin.routes}
    assert routes["/astrbot_plugin_jitrl/checkpoints"] == ["GET"]
    for name in ("checkpoint", "rollback", "checkpoint_delete"):
        assert routes[f"/astrbot_plugin_jitrl/{name}"] == ["POST"]
    await learn(plugin)
    text = await command(plugin.jitrl_checkpoint(Event(), "stable"))
    saved = (await plugin._bank.list_checkpoints())[0]
    assert saved["id"] in text and saved["name"] == "stable"
    assert saved["id"] in await command(plugin.jitrl_checkpoints(Event()))


@pytest.mark.parametrize(
    "name,args",
    [
        ("jitrl_checkpoint", ()),
        ("jitrl_checkpoints", ()),
        ("jitrl_rollback", ("0" * 32, "all", "confirm")),
        ("jitrl_checkpoint_delete", ("0" * 32, "confirm")),
    ],
)
async def test_global_checkpoint_management_stays_admin_only(plugin, name, args):
    text = await command(getattr(plugin, name)(Event(admin=False), *args))
    assert "管理员" in text
    assert await plugin._bank.list_checkpoints() == []


async def test_rollback_confirmation_and_session_scope(plugin):
    await learn(plugin)
    saved = await plugin._bank.create_checkpoint()
    await learn(plugin)
    await learn(plugin, "s2")
    generation = plugin._bank.generation
    text = await command(plugin.jitrl_rollback(Event(), saved["id"]))
    assert "confirm" in text
    assert plugin._bank.generation == generation
    assert await plugin._bank.count() == 3
    text = await command(
        plugin.jitrl_rollback(Event(), saved["id"], "session", "confirm")
    )
    assert "已回退当前会话" in text
    assert await plugin._bank.count("s1") == await plugin._bank.count("s2") == 1
    assert not plugin._last_episode and not plugin._last_decision


async def test_forget_rejects_typo_and_clears_all_session_caches(plugin):
    await learn(plugin)
    await learn(plugin, "s2")
    text = await command(plugin.jitrl_forget(Event(), "al"))
    assert "未删除" in text and await plugin._bank.count() == 2
    text = await command(plugin.jitrl_forget(Event(), "all"))
    assert "自动备份" in text and await plugin._bank.count() == 0
    assert not plugin._last_episode and not plugin._last_decision


async def test_web_checkpoint_restore_and_delete_flow(host, plugin):
    await learn(plugin)
    response = await web_call(host, plugin, "_web_checkpoint", {"name": "clean"})
    saved = response.payload["checkpoint"]
    assert response.status_code == 200
    inventory = await plugin._web_checkpoints()
    assert inventory.payload["checkpoints"][0]["id"] == saved["id"]
    await learn(plugin)
    response = await web_call(
        host, plugin, "_web_rollback", {"id": saved["id"], "confirm": True}
    )
    assert response.status_code == 200
    assert response.payload["safety_checkpoint"]["experiences"] == 2
    assert await plugin._bank.count() == 1
    assert not plugin._last_episode and not plugin._last_decision
    response = await web_call(
        host, plugin, "_web_checkpoint_delete", {"id": saved["id"], "confirm": True}
    )
    assert response.payload["deleted"] == saved["id"]


@pytest.mark.parametrize("confirm", [None, False, "true", 1])
async def test_web_requires_explicit_boolean_confirmation(host, plugin, confirm):
    await learn(plugin)
    saved = await plugin._bank.create_checkpoint()
    for name in ("_web_rollback", "_web_checkpoint_delete"):
        response = await web_call(
            host, plugin, name, {"id": saved["id"], "confirm": confirm}
        )
        assert response.status_code == 400
    assert await plugin._bank.count() == 1
    assert len(await plugin._bank.list_checkpoints()) == 1


@pytest.mark.parametrize(
    "method,payload,status",
    [
        ("_web_checkpoint", [], 400),
        ("_web_checkpoint", {"name": {}}, 400),
        ("_web_checkpoint", {"name": "a" * 81}, 400),
        ("_web_forget", [], 400),
        ("_web_rollback", {"id": "../memory", "confirm": True}, 400),
        ("_web_rollback", {"id": "0" * 32, "confirm": True}, 404),
        ("_web_rollback", {"scope": "bad", "confirm": True}, 400),
        ("_web_rollback", {"scope": "session", "confirm": True, "umo": ""}, 400),
    ],
)
async def test_web_rejects_bad_requests_without_mutation(
    host, plugin, method, payload, status
):
    await learn(plugin)
    response = await web_call(host, plugin, method, payload)
    assert response.status_code == status
    assert await plugin._bank.count() == 1


async def test_agent_started_before_restore_cannot_learn_after_it(plugin):
    await learn(plugin)
    saved = await plugin._bank.create_checkpoint()
    plugin.cfg.apply_mode = "off"
    event = Event()
    await plugin.on_llm_request(event, SimpleNamespace(prompt="trusted task"))
    await plugin._bank.restore_checkpoint(saved["id"])
    stats = dict(plugin._stats)
    await plugin.on_agent_done(
        event, SimpleNamespace(messages=MESSAGES), SimpleNamespace(completion_text="ok")
    )
    assert await plugin._bank.count() == 1
    assert plugin._stats == stats
    # A request started after the restore can learn normally.
    event = Event()
    await plugin.on_llm_request(event, SimpleNamespace(prompt="trusted task"))
    await plugin.on_agent_done(
        event, SimpleNamespace(messages=MESSAGES), SimpleNamespace(completion_text="ok")
    )
    assert await plugin._bank.count() == 2


async def test_evaluation_in_flight_is_discarded_after_restore(
    host, plugin, monkeypatch
):
    await learn(plugin)
    saved = await plugin._bank.create_checkpoint()
    started, resume = asyncio.Event(), asyncio.Event()

    async def delayed_score(self, **kwargs):
        started.set()
        await resume.wait()
        return RewardResult(rewards=[1.0], source="test", ok=True)

    monkeypatch.setattr(host.module.StepRewardEvaluator, "score", delayed_score)
    pending = asyncio.create_task(learn(plugin))
    await asyncio.wait_for(started.wait(), 3)
    stats = dict(plugin._stats)
    try:
        await web_call(
            host, plugin, "_web_rollback", {"id": saved["id"], "confirm": True}
        )
    finally:
        resume.set()
    await asyncio.wait_for(pending, 3)
    assert await plugin._bank.count() == 1
    assert plugin._stats == stats
    assert not plugin._last_episode and not plugin._last_decision


async def test_rerank_in_flight_cannot_inject_old_policy_after_restore(
    plugin, monkeypatch
):
    await learn(plugin)
    saved = await plugin._bank.create_checkpoint()
    plugin.cfg.apply_mode = "logit"
    started, resume = asyncio.Event(), asyncio.Event()

    async def delayed_rerank(**kwargs):
        started.set()
        await resume.wait()
        return {}

    monkeypatch.setattr(plugin, "_rerank", delayed_rerank)
    event = Event()
    request = SimpleNamespace(
        prompt="trusted task",
        system_prompt="base",
        func_tool=SimpleNamespace(names=lambda: ["search"]),
    )
    pending = asyncio.create_task(plugin.on_llm_request(event, request))
    await asyncio.wait_for(started.wait(), 3)
    try:
        await plugin._bank.restore_checkpoint(saved["id"])
        plugin._invalidate_memory_cache()
    finally:
        resume.set()
    await asyncio.wait_for(pending, 3)
    assert request.system_prompt == "base"
    assert not plugin._last_decision
