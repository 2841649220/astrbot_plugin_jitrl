"""Checkpoint wiring tests with a small host adapter, no AstrBot install needed.

The optional integration suite still checks the real AstrBot contracts. These
tests exercise plugin behaviour and races on every CI run, using the real bank.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import pytest_asyncio

from jitrl.evaluator import RewardResult
from jitrl.memory import MemoryBank
from jitrl.models import ValueEstimate
from jitrl.state import abstract_state, normalize_action
from jitrl.trajectory import (
    Transition,
    build_episode,
    extract_transitions,
)

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
    def __init__(
        self,
        *,
        admin=True,
        scope="s1",
        text="trusted task",
        platform_id="platform-1",
        self_id="bot-1",
    ):
        self.unified_msg_origin = scope
        self.admin = admin
        self.text = text
        self.extra = {}
        self.platform_id = platform_id
        self.self_id = self_id

    def is_admin(self):
        return self.admin

    def get_message_str(self):
        return self.text

    def get_platform_name(self):
        return "test"

    def get_platform_id(self):
        return self.platform_id

    def get_self_id(self):
        return self.self_id

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
        "astrbot.core.agent",
        "astrbot.core.agent.message",
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

    class TextPart:
        """Minimal stand-in for ``astrbot.core.agent.message.TextPart``."""

        def __init__(self, *, text):
            self.text = text
            self.temporary = False

        def mark_as_temp(self):
            self.temporary = True
            return self

    modules["astrbot.api"].logger = logging.getLogger("checkpoint-test")
    modules["astrbot.core.agent.message"].TextPart = TextPart
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
        await plugin._ensure_ready(Event())
        yield plugin
    finally:
        await plugin.terminate()


async def command(generator):
    return "\n".join([message async for message in generator])


async def web_call(host, plugin, method, payload):
    async def json_body(default=None):
        return payload

    def query_get(key, default=None, type=None):
        value = payload.get(key, default) if isinstance(payload, dict) else default
        return type(value) if type and value is not None else value

    host.web.request = SimpleNamespace(
        json=json_body, query=SimpleNamespace(get=query_get)
    )
    return await getattr(plugin, method)()


async def learn(plugin, scope="s1"):
    event = Event(scope=scope)
    robot_id = plugin._robot_key_for_event(event)
    await plugin._ensure_ready(event)
    await plugin._learn(
        umo=scope,
        user_text="trusted task",
        surface="test",
        transitions=extract_transitions(
            messages=MESSAGES, user_text="trusted task", surface="test"
        ),
        reply_text="ok",
        cache_scope=plugin._cache_scope(event),
        robot_id=robot_id,
    )


async def _seed(bank, task, tool, reward, scope="s1"):
    """Store one single-step episode with a known reward."""
    transition = Transition(
        state=abstract_state(task=task),
        action=normalize_action(kind="tool", name=tool, args={"q": "x"}),
        observation="ok",
        outcome="ok",
    )
    await bank.add_episode(
        build_episode(
            episode_id=uuid.uuid4().hex,
            scope=scope,
            user_text=task,
            transitions=[transition],
            rewards=[reward],
            gamma=0.5,
        )
    )


@pytest.mark.parametrize(
    ("advantages", "expected"),
    [
        ({"tool:bad(a)": -0.004, "tool:good()": 0.004}, {"bad", "good"}),
        (
            {"tool:bad(a)": -2.0, "tool:bad(b)": 1.0, "tool:good()": 2.0},
            {"bad", "good"},
        ),
        ({"tool:bad(a)": -2.0, "tool:bad(b)": -2.0, "tool:good()": 2.0}, {"good"}),
        ({"tool:bad(a)": -2.0, "tool:good()": -1.0}, {"bad", "good"}),
    ],
)
async def test_tool_gate_requires_raw_evidence_for_every_variant(
    plugin, advantages, expected
):
    names = {"bad", "good"}
    request = SimpleNamespace(
        func_tool=SimpleNamespace(names=lambda: sorted(names), remove_tool=names.remove)
    )
    scale = max(abs(value) for value in advantages.values())
    estimate = ValueEstimate(
        advantage=advantages,
        advantage_norm={key: value / scale for key, value in advantages.items()},
        action_counts=dict.fromkeys(advantages, 3),
        neighbor_count=3 * len(advantages),
        max_abs_advantage=scale,
    )
    plugin._apply_tool_gate(request, estimate)
    assert names == expected


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


async def test_robot_instances_use_independent_banks_stats_and_feedback(plugin):
    first = Event(platform_id="platform-a", self_id="bot-a")
    second = Event(platform_id="platform-b", self_id="bot-a")
    first_bank = await plugin._ensure_ready(first)
    await _seed(first_bank, "trusted task", "search", 2.0)
    first_key = plugin._robot_key_for_event(first)
    first_scope = plugin._cache_scope(first)
    plugin._stats["requests"] = 4
    plugin._last_episode[first_scope] = ("first-episode", 1.0)

    second_bank = await plugin._ensure_ready(second)
    second_key = plugin._robot_key_for_event(second)
    second_scope = plugin._cache_scope(second)
    assert second_bank is not first_bank
    assert plugin._bank_paths[second_key] != plugin._bank_paths[first_key]
    assert await second_bank.count() == 0
    assert await first_bank.count() == 1
    assert plugin._stats["requests"] == 0
    assert second_scope not in plugin._last_episode
    plugin._stats["requests"] = 2
    plugin._last_episode[second_scope] = ("second-episode", 1.0)

    await plugin._ensure_ready(first)
    assert plugin._stats["requests"] == 4
    assert plugin._last_episode[first_scope][0] == "first-episode"
    assert plugin._state_for(
        first_key, "s1", "same task", "test"
    ) is not plugin._state_for(second_key, "s1", "same task", "test")
    assert first_scope != second_scope
    assert await first_bank.count() == 1
    assert await second_bank.count() == 0


async def test_concurrent_robot_requests_keep_their_context_across_awaits(
    plugin, monkeypatch
):
    first = Event(platform_id="platform-a", self_id="bot-a")
    second = Event(platform_id="platform-b", self_id="bot-a")
    await plugin._ensure_ready(first)
    await plugin._ensure_ready(second)
    paths = {
        plugin._robot_key_for_event(event): plugin._bank_paths[
            plugin._robot_key_for_event(event)
        ]
        for event in (first, second)
    }
    observed = []

    async def interleaved_decide(**kwargs):
        before = plugin._db_path
        await asyncio.sleep(0)
        after = plugin._db_path
        observed.append((before, after))
        return None

    monkeypatch.setattr(plugin, "_decide", interleaved_decide)
    first_request = SimpleNamespace(
        prompt="first task", func_tool=None, extra_user_content_parts=[]
    )
    second_request = SimpleNamespace(
        prompt="second task", func_tool=None, extra_user_content_parts=[]
    )
    await asyncio.gather(
        plugin.on_llm_request(first, first_request),
        plugin.on_llm_request(second, second_request),
    )

    expected = set(paths.values())
    assert len(observed) == 2
    assert all(before == after and before in expected for before, after in observed)
    assert {before for before, _after in observed} == expected


async def test_legacy_bank_and_checkpoints_move_to_first_robot(host, tmp_path):
    data_root = (
        Path(host.module.get_astrbot_plugin_data_path()) / "astrbot_plugin_jitrl"
    )
    legacy_path = data_root / "memory.sqlite3"
    legacy = MemoryBank(legacy_path, checkpoint_dir=data_root / "checkpoints")
    await legacy.initialize()
    await _seed(legacy, "old trusted task", "search", 2.0)
    saved = await legacy.create_checkpoint("legacy")
    await legacy.close()

    instance = host.module.JitRLLearner(
        SimpleNamespace(register_web_api=lambda *args: None),
        {"memory": {"checkpoint_interval": 0}},
    )
    try:
        event = Event(self_id="first-after-upgrade")
        bank = await instance._ensure_ready(event)
        assert await bank.count() == 1
        assert saved["id"] in {row["id"] for row in await bank.list_checkpoints()}
        assert not legacy_path.exists()
        assert instance._bank_paths[instance._robot_key_for_event(event)] == (
            instance._robot_dir(instance._robot_key_for_event(event)) / "memory.sqlite3"
        )
    finally:
        await instance.terminate()


async def test_legacy_conflicts_are_preserved_and_never_reassigned(plugin):
    first_robot_dir = plugin._robot_dir("b" * 64)
    first_robot_dir.mkdir(parents=True)
    robot_db = first_robot_dir / "memory.sqlite3"
    robot_db.write_bytes(b"existing robot database")
    plugin._legacy_db_path.write_bytes(b"legacy shared database")
    legacy_checkpoints = plugin._data_root / "checkpoints"
    legacy_checkpoints.mkdir()
    (legacy_checkpoints / "snapshot.json").write_text("snapshot", encoding="utf-8")
    legacy_neural = plugin._data_root / "neural"
    legacy_neural.mkdir()
    (legacy_neural / "neural.sqlite3").write_bytes(b"neural")
    legacy_export = plugin._data_root / "export-legacy.json"
    legacy_export.write_text("{}", encoding="utf-8")

    plugin._migrate_legacy_data(first_robot_dir, robot_db)

    assert robot_db.read_bytes() == b"existing robot database"
    assert plugin._legacy_db_path.read_bytes() == b"legacy shared database"
    assert (legacy_checkpoints / "snapshot.json").read_text(
        encoding="utf-8"
    ) == "snapshot"
    assert (first_robot_dir / "neural" / "neural.sqlite3").read_bytes() == b"neural"
    assert legacy_export.read_text(encoding="utf-8") == "{}"

    second_robot_dir = plugin._robot_dir("c" * 64)
    second_robot_dir.mkdir(parents=True)
    plugin._migrate_legacy_data(second_robot_dir, second_robot_dir / "memory.sqlite3")
    assert not (second_robot_dir / "memory.sqlite3").exists()
    assert json.loads(
        (plugin._data_root / ".legacy-migration.json").read_text(encoding="utf-8")
    ) == {"robot_id": first_robot_dir.name}


async def test_legacy_database_migration_resumes_after_sidecar_move(plugin):
    first_robot_dir = plugin._robot_dir("d" * 64)
    first_robot_dir.mkdir(parents=True)
    robot_db = first_robot_dir / "memory.sqlite3"
    plugin._legacy_db_path.write_bytes(b"legacy database")
    legacy_wal = plugin._legacy_db_path.parent / "memory.sqlite3-wal"
    target_wal = robot_db.parent / "memory.sqlite3-wal"
    legacy_wal.write_bytes(b"legacy wal")
    legacy_wal.replace(target_wal)

    plugin._migrate_legacy_data(first_robot_dir, robot_db)

    assert robot_db.read_bytes() == b"legacy database"
    assert target_wal.read_bytes() == b"legacy wal"
    assert not legacy_wal.exists()
    assert not plugin._legacy_db_path.exists()
    assert (plugin._data_root / ".legacy-migration.json").exists()


async def test_web_robot_selection_prevents_cross_instance_management(host, plugin):
    first = Event(platform_id="platform-a", self_id="bot-a")
    second = Event(platform_id="platform-b", self_id="bot-a")
    first_bank = await plugin._ensure_ready(first)
    second_bank = await plugin._ensure_ready(second)
    await _seed(first_bank, "trusted task", "search", 1.0)
    await _seed(second_bank, "trusted task", "inspect", 2.0)

    robots = (await plugin._web_robots()).payload["robots"]
    assert {row["id"] for row in robots} >= {
        plugin._robot_key_for_event(first),
        plugin._robot_key_for_event(second),
    }
    missing = await web_call(
        host, plugin, "_web_forget", {"scope": "all", "confirm": True}
    )
    assert missing.status_code == 400
    assert await first_bank.count() == await second_bank.count() == 1

    selected_id = plugin._robot_key_for_event(first)
    overview = await web_call(host, plugin, "_web_overview", {"robot": selected_id})
    assert overview.payload["robot_id"] == selected_id
    assert overview.payload["total"] == 1
    forgotten = await web_call(
        host,
        plugin,
        "_web_forget",
        {"robot": selected_id, "scope": "all", "confirm": True},
    )
    assert forgotten.payload["removed"] == 1
    assert await first_bank.count() == 0
    assert await second_bank.count() == 1


async def test_one_platform_instance_is_one_robot_although_self_id_varies(plugin):
    """qq_official changes self_id per message shape; the robot must not."""
    self_ids = [
        "qq_official",
        "unknown_selfid",
        "C3DC36F1480C2129659672A89D9D04FD",
        "",
    ]
    events = [Event(platform_id="爱弥斯", self_id=self_id) for self_id in self_ids]
    keys = {plugin._robot_key_for_event(event) for event in events}
    assert len(keys) == 1
    for event in events:
        await plugin._ensure_ready(event)

    robot_key = keys.pop()
    robots = {row["id"]: row for row in plugin._known_robots()}
    assert robot_key in robots
    assert robots[robot_key]["label"] == "test · 爱弥斯"
    assert robots[robot_key]["platform_id"] == "爱弥斯"
    assert robots[robot_key]["self_id"] == self_ids[-2]

    metadata = json.loads(
        (plugin._robot_dir(robot_key) / "robot.json").read_text(encoding="utf-8")
    )
    assert metadata["identity"] == "platform:爱弥斯"
    assert metadata["self_id"] == self_ids[-2], "an empty self_id must not erase it"
    assert await plugin._bank.count() == 0


async def test_missing_platform_id_falls_back_without_self_id(plugin):
    """A degenerate adapter id falls back to the adapter type, not self_id."""
    anonymous = Event(platform_id="", self_id="bot-x")
    other = Event(platform_id="", self_id="bot-y")
    assert plugin._robot_key_for_event(anonymous) == plugin._robot_key("adapter:test")
    assert plugin._robot_key_for_event(other) == plugin._robot_key("adapter:test")


async def test_robot_richness_reads_db_paths_containing_hash(host, tmp_path):
    """URI quoting must not let a '#' in the data path skew promotion."""
    directory = tmp_path / "hash#dir"
    directory.mkdir()
    bank = MemoryBank(
        directory / "memory.sqlite3",
        checkpoint_dir=directory / "checkpoints",
        checkpoint_interval=0,
    )
    await bank.initialize()
    await _seed(bank, "hash path", "search", 1.0)
    await bank.close()
    rows, size, has_neural, _modified = host.module.JitRLLearner._robot_richness(
        directory
    )
    assert rows == 1
    assert size > 0
    assert has_neural == 0


async def test_known_robots_ignores_hidden_corrupt_and_metadata_free_dirs(plugin):
    """Only real robot directories may appear in the dashboard selector."""
    robots_root = plugin._data_root / "robots"
    hidden = robots_root / ".legacy" / "archived"
    hidden.mkdir(parents=True)
    (hidden / "robot.json").write_text("{}", encoding="utf-8")
    corrupt = robots_root / "corrupt"
    corrupt.mkdir()
    (corrupt / "robot.json").write_text("{not json", encoding="utf-8")
    metadata_free = robots_root / "no-metadata"
    metadata_free.mkdir()
    identity_free = robots_root / "solo-copy"
    identity_free.mkdir()
    (identity_free / "robot.json").write_text('{"self_id": "solo"}', encoding="utf-8")

    robots = {row["id"]: row for row in plugin._known_robots()}
    assert plugin._robot_key_for_event(Event()) in robots
    assert "corrupt" not in robots
    assert "no-metadata" not in robots
    assert ".legacy" not in robots
    assert robots["solo-copy"]["label"] == "unrecognized · solo-copy"


async def test_legacy_per_self_id_directories_are_consolidated_and_merged(host):
    """Legacy platform_id:self_id dirs for one instance become one merged bank."""
    data_root = (
        Path(host.module.get_astrbot_plugin_data_path()) / "astrbot_plugin_jitrl"
    )
    robots_root = data_root / "robots"
    platform_id = "爱弥斯"
    self_ids = ("qq_official", "unknown_selfid", "C3DC36F1480C2129659672A89D9D04FD")
    for index, self_id in enumerate(self_ids):
        legacy_key = hashlib.sha256(f"{platform_id}:{self_id}".encode()).hexdigest()
        directory = robots_root / legacy_key
        directory.mkdir(parents=True)
        (directory / "robot.json").write_text(
            json.dumps(
                {
                    "platform_id": platform_id,
                    "platform_name": "qq_official",
                    "self_id": self_id,
                    "label": f"qq_official · {self_id}",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        bank = MemoryBank(
            directory / "memory.sqlite3",
            checkpoint_dir=directory / "checkpoints",
            checkpoint_interval=0,
        )
        await bank.initialize()
        await _seed(
            bank, f"task {index}", f"tool-{index}", float(index + 1), scope=f"s{index}"
        )
        await bank.close()

    instance = host.module.JitRLLearner(
        SimpleNamespace(register_web_api=lambda *args: None),
        {"memory": {"checkpoint_interval": 0}},
    )
    try:
        canonical_key = instance._robot_key_for_event(
            Event(platform_id=platform_id, self_id=self_ids[0])
        )
        canonical_dir = robots_root / canonical_key
        assert canonical_dir.is_dir()
        assert (canonical_dir / "robot.json").is_file()
        assert [row["id"] for row in instance._known_robots()] == [canonical_key]

        bank = await instance._ensure_ready(
            Event(platform_id=platform_id, self_id=self_ids[1])
        )
        assert await bank.count() == len(self_ids)
        payload = await bank.export_json()
        assert {row["episode_id"] for row in payload["experiences"]} == {
            row["episode_id"] for row in payload["episodes"]
        }
        assert len(payload["episodes"]) == len(self_ids)

        archive_root = robots_root / ".legacy" / canonical_key
        archived = sorted(path for path in archive_root.iterdir() if path.is_dir())
        assert len(archived) == len(self_ids) - 1
        assert all((path / "memory.sqlite3.merged").is_file() for path in archived)

        retry = host.module.JitRLLearner(
            SimpleNamespace(register_web_api=lambda *args: None),
            {"memory": {"checkpoint_interval": 0}},
        )
        try:
            again = await retry._ensure_ready(
                Event(platform_id=platform_id, self_id=self_ids[2])
            )
            assert await again.count() == len(self_ids)
            assert len(await again.recent_episodes(limit=10)) == len(self_ids)
        finally:
            await retry.terminate()
    finally:
        await instance.terminate()


async def test_archived_neural_store_is_adopted_when_canonical_has_none(host):
    """A lens trained in a non-richest legacy dir must not disappear."""
    data_root = (
        Path(host.module.get_astrbot_plugin_data_path()) / "astrbot_plugin_jitrl"
    )
    robots_root = data_root / "robots"
    platform_id = "爱弥斯"
    legacy_keys = {}
    for index, (self_id, episodes, has_neural) in enumerate(
        (("rich", 2, False), ("with-lens", 1, True))
    ):
        legacy_key = hashlib.sha256(f"{platform_id}:{self_id}".encode()).hexdigest()
        legacy_keys[self_id] = legacy_key
        directory = robots_root / legacy_key
        directory.mkdir(parents=True)
        (directory / "robot.json").write_text(
            json.dumps(
                {
                    "platform_id": platform_id,
                    "platform_name": "qq_official",
                    "self_id": self_id,
                    "label": f"qq_official · {self_id}",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        bank = MemoryBank(
            directory / "memory.sqlite3",
            checkpoint_dir=directory / "checkpoints",
            checkpoint_interval=0,
        )
        await bank.initialize()
        for step in range(episodes):
            await _seed(bank, f"task {index}-{step}", f"tool-{step}", 1.0)
        await bank.close()
        if has_neural:
            neural = directory / "neural"
            neural.mkdir()
            (neural / "neural.sqlite3").write_bytes(b"trained lens")

    instance = host.module.JitRLLearner(
        SimpleNamespace(register_web_api=lambda *args: None),
        {"memory": {"checkpoint_interval": 0}},
    )
    try:
        canonical_key = instance._robot_key_for_event(
            Event(platform_id=platform_id, self_id="rich")
        )
        canonical_dir = robots_root / canonical_key
        assert (canonical_dir / "neural").exists() is False
        bank = await instance._ensure_ready(Event(platform_id=platform_id))
        assert await bank.count() == 3
        adopted = canonical_dir / "neural" / "neural.sqlite3"
        assert adopted.read_bytes() == b"trained lens"
        archived_lens = (
            robots_root / ".legacy" / canonical_key / legacy_keys["with-lens"]
        )
        assert not (archived_lens / "neural").exists()
    finally:
        await instance.terminate()


async def test_newest_archived_neural_store_is_adopted(host):
    """Two archived lenses: the largest/newest store wins, not name order."""
    data_root = (
        Path(host.module.get_astrbot_plugin_data_path()) / "astrbot_plugin_jitrl"
    )
    robots_root = data_root / "robots"
    platform_id = "爱弥斯"
    specs = (
        ("rich", 2, None),
        ("old-lens", 1, (b"old lens", 1_600_000_000)),
        ("new-lens", 1, (b"new lens", 1_700_000_000)),
    )
    legacy_keys = {}
    for index, (self_id, episodes, lens) in enumerate(specs):
        legacy_key = hashlib.sha256(f"{platform_id}:{self_id}".encode()).hexdigest()
        legacy_keys[self_id] = legacy_key
        directory = robots_root / legacy_key
        directory.mkdir(parents=True)
        (directory / "robot.json").write_text(
            json.dumps(
                {
                    "platform_id": platform_id,
                    "platform_name": "qq_official",
                    "self_id": self_id,
                    "label": f"qq_official · {self_id}",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        bank = MemoryBank(
            directory / "memory.sqlite3",
            checkpoint_dir=directory / "checkpoints",
            checkpoint_interval=0,
        )
        await bank.initialize()
        for step in range(episodes):
            await _seed(bank, f"task {index}-{step}", f"tool-{step}", 1.0)
        await bank.close()
        if lens is not None:
            payload, modified = lens
            neural = directory / "neural"
            neural.mkdir()
            store = neural / "neural.sqlite3"
            store.write_bytes(payload)
            os.utime(store, (modified, modified))

    instance = host.module.JitRLLearner(
        SimpleNamespace(register_web_api=lambda *args: None),
        {"memory": {"checkpoint_interval": 0}},
    )
    try:
        canonical_key = instance._robot_key_for_event(
            Event(platform_id=platform_id, self_id="rich")
        )
        bank = await instance._ensure_ready(Event(platform_id=platform_id))
        assert await bank.count() == 4
        canonical_lens = robots_root / canonical_key / "neural" / "neural.sqlite3"
        assert canonical_lens.read_bytes() == b"new lens"

        archive_root = robots_root / ".legacy" / canonical_key
        old_archive = archive_root / legacy_keys["old-lens"]
        new_archive = archive_root / legacy_keys["new-lens"]
        assert (old_archive / "neural" / "neural.sqlite3").read_bytes() == b"old lens"
        assert not (new_archive / "neural").exists()
    finally:
        await instance.terminate()


async def test_legacy_dirs_without_platform_id_are_merged_by_adapter(host):
    """Degenerate adapters without an ID are consolidated by adapter type."""
    data_root = (
        Path(host.module.get_astrbot_plugin_data_path()) / "astrbot_plugin_jitrl"
    )
    robots_root = data_root / "robots"
    for index, self_id in enumerate(("s1", "s2")):
        legacy_key = hashlib.sha256(f":{self_id}".encode()).hexdigest()
        directory = robots_root / legacy_key
        directory.mkdir(parents=True)
        (directory / "robot.json").write_text(
            json.dumps(
                {
                    "platform_id": "",
                    "platform_name": "test",
                    "self_id": self_id,
                    "label": f"test · {self_id}",
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        bank = MemoryBank(
            directory / "memory.sqlite3",
            checkpoint_dir=directory / "checkpoints",
            checkpoint_interval=0,
        )
        await bank.initialize()
        await _seed(bank, f"task {index}", f"tool-{index}", 1.0, scope=self_id)
        await bank.close()

    instance = host.module.JitRLLearner(
        SimpleNamespace(register_web_api=lambda *args: None),
        {"memory": {"checkpoint_interval": 0}},
    )
    try:
        canonical_key = instance._robot_key("adapter:test")
        assert (robots_root / canonical_key).is_dir()
        bank = await instance._ensure_ready(Event(platform_id="", self_id="s3"))
        assert await bank.count() == 2
        robots = instance._known_robots()
        assert [row["id"] for row in robots] == [canonical_key]
        assert robots[0]["identity"] == "adapter:test"
        assert robots[0]["label"] == "test · s3"

        archive_root = robots_root / ".legacy" / canonical_key
        archived = [path for path in archive_root.iterdir() if path.is_dir()]
        assert len(archived) == 1
        assert (archived[0] / "memory.sqlite3.merged").is_file()
    finally:
        await instance.terminate()


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
        extra_user_content_parts=[],
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


async def test_experience_block_rides_on_the_user_message(plugin):
    await plugin._ensure_ready(Event())
    for tool, reward in (("web_search", 2.0), ("bad_tool", -2.0)):
        await _seed(plugin._bank, "trusted task", tool, reward)
    event = Event()
    request = SimpleNamespace(
        prompt="trusted task",
        system_prompt="base",
        func_tool=SimpleNamespace(names=lambda: ["web_search", "bad_tool"]),
        extra_user_content_parts=[],
    )
    await plugin.on_llm_request(event, request)
    assert request.system_prompt == "base"
    assert len(request.extra_user_content_parts) == 1
    part = request.extra_user_content_parts[0]
    assert "<jitrl_experience>" in part.text
    assert part.temporary, "the injected part must never be persisted"


def _tool_request(*names: str) -> SimpleNamespace:
    """Build an outgoing request exposing the given tool names.

    Args:
        *names: Tool names available for this request.

    Returns:
        A minimal ProviderRequest stand-in.
    """
    return SimpleNamespace(
        prompt="trusted task",
        system_prompt="base",
        func_tool=SimpleNamespace(names=lambda: list(names)),
        extra_user_content_parts=[],
    )


async def test_decision_cache_reuses_a_state_and_drops_on_new_evidence(plugin):
    for tool, reward in (("web_search", 2.0), ("bad_tool", -2.0)):
        await _seed(plugin._bank, "trusted task", tool, reward)
    first = _tool_request("web_search", "bad_tool")
    await plugin.on_llm_request(Event(), first)
    assert len(plugin._decision_cache) == 1
    snapshot = next(iter(plugin._decision_cache.values()))

    # A second request for the same state must reuse the decision instead of
    # re-running retrieval and re-sampling the exploration branch.
    second = _tool_request("web_search", "bad_tool")
    await plugin.on_llm_request(Event(), second)
    assert next(iter(plugin._decision_cache.values())) is snapshot
    assert len(plugin._decision_cache) == 1
    assert second.extra_user_content_parts

    # New evidence invalidates the cache.
    await learn(plugin)
    assert plugin._decision_cache == {}


async def test_two_pass_shortlist_keeps_the_most_informative_actions(plugin):
    for tool, reward in (
        ("tool_a", 2.0),
        ("tool_b", -2.0),
        ("tool_c", 1.0),
        ("tool_d", -1.0),
    ):
        await _seed(plugin._bank, "trusted task", tool, reward)
    request = _tool_request("tool_a", "tool_b", "tool_c", "tool_d")
    await plugin.on_llm_request(Event(), request)
    snapshot = next(iter(plugin._decision_cache.values()))

    # The probe covers every retrieved action, while the decision is restricted
    # to the |C| = 3 actions with the strongest evidence.
    assert len(snapshot.probe.advantage) == 4
    assert len(snapshot.estimate.advantage) == 3
    assert not any("tool_d" in key for key in snapshot.estimate.advantage)


async def test_single_neighbour_never_injects_evidence(plugin):
    await _seed(plugin._bank, "trusted task", "web_search", 2.0)
    request = _tool_request("web_search")
    await plugin.on_llm_request(Event(), request)
    assert request.extra_user_content_parts == []
    snapshot = next(iter(plugin._decision_cache.values()))
    assert snapshot.block == ""
    assert snapshot.decision.applied


async def test_web_forget_requires_explicit_confirmation(host, plugin):
    await learn(plugin)
    response = await web_call(host, plugin, "_web_forget", {"scope": "all"})
    assert response.status_code == 400
    assert await plugin._bank.count() == 1
    response = await web_call(
        host, plugin, "_web_forget", {"scope": "all", "confirm": True}
    )
    assert response.status_code == 200
    assert await plugin._bank.count() == 0


async def test_export_command_hides_the_server_path(plugin):
    await learn(plugin)
    text = await command(plugin.jitrl_export(Event()))
    assert "export-" in text
    assert "\\\\" not in text and "/" not in text.replace("http://", "")
    files = list(plugin._db_path.parent.glob("export-*.json"))
    assert len(files) == 1
