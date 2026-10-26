"""Recovery, isolation, durability and concurrent-write checks for snapshots."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from contextlib import closing
from pathlib import Path

import pytest
import pytest_asyncio

from jitrl.checkpoints import CheckpointStore
from jitrl.config import JitRLConfig
from jitrl.memory import MemoryBank
from jitrl.models import Episode, StepRecord
from jitrl.state import abstract_state, normalize_action


def episode(identifier: str, scope: str = "s1", reward: float = 1.0) -> Episode:
    return Episode(
        episode_id=identifier,
        scope=scope,
        created_at=1000.0,
        user_text="trusted task",
        success=True,
        meta={"gamma": 0.1, "note": "preserve metadata"},
        steps=[
            StepRecord(
                index=0,
                state=abstract_state(task="trusted task"),
                action=normalize_action(kind="tool", name="search", args={"q": "x"}),
                reward=reward,
                discounted_return=reward,
                outcome="ok",
                reward_source="test",
            )
        ],
    )


@pytest_asyncio.fixture
async def bank(tmp_path: Path):
    instance = MemoryBank(tmp_path / "memory.sqlite3", checkpoint_interval=0)
    await instance.initialize()
    try:
        yield instance
    finally:
        await instance.close()


async def contents(bank: MemoryBank) -> dict:
    payload = await bank.export_json()
    payload.pop("exported_at")
    return payload


@pytest.mark.asyncio
async def test_restore_wal_snapshot_and_undo_after_reopen(bank: MemoryBank) -> None:
    await bank.add_episode(episode("trusted"))
    before = await contents(bank)
    saved = await bank.create_checkpoint("known good")
    assert saved["experiences"] == saved["episodes"] == 1
    await bank.apply_user_feedback("trusted", -5)
    await bank.add_episode(episode("polluted", reward=-3))
    polluted = await contents(bank)

    await bank.close()
    await bank.initialize()
    assert (await bank.list_checkpoints())[0]["id"] == saved["id"]
    result = await bank.restore_checkpoint(saved["id"])
    assert await contents(bank) == before
    assert result["safety_checkpoint"]["kind"] == "before_restore"
    assert result["safety_checkpoint"]["experiences"] == 2
    await bank.restore_checkpoint(result["safety_checkpoint"]["id"])
    assert await contents(bank) == polluted


@pytest.mark.asyncio
async def test_session_restore_preserves_other_scopes_and_avoids_id_collisions(bank):
    await bank.add_episode(episode("a", "s1"))
    await bank.add_episode(episode("b", "s2"))
    saved = await bank.create_checkpoint()
    await bank.add_episode(episode("a", "s1", -3))
    await bank.add_episode(episode("b", "s2", 3))
    await bank.add_episode(episode("new", "s1", -3))
    other = (await bank.export_json(scope="s2"))["experiences"]
    result = await bank.restore_checkpoint(saved["id"], scope="s1")
    assert result["experiences"] == result["episodes"] == 1
    restored = (await bank.export_json(scope="s1"))["experiences"]
    assert [row["episode_id"] for row in restored] == ["a"]
    assert restored[0]["reward"] == 1.0
    assert (await bank.export_json(scope="s2"))["experiences"] == other
    await bank.add_episode(episode("after-restore"))
    assert await bank.count() == 3


@pytest.mark.asyncio
async def test_session_restore_rejects_cross_scope_episode_conflict(bank):
    await bank.add_episode(episode("same", "s1"))
    saved = await bank.create_checkpoint()
    await bank.add_episode(episode("same", "s2"))
    before = await contents(bank)
    with pytest.raises(ValueError, match="冲突"):
        await bank.restore_checkpoint(saved["id"], scope="s1")
    assert await contents(bank) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "s1"])
async def test_delete_creates_recoverable_safety_backup(bank, scope):
    await bank.add_episode(episode("a", "s1"))
    await bank.add_episode(episode("b", "s2"))
    before = await contents(bank)
    if scope is None:
        assert await bank.clear() == 2
    else:
        assert await bank.forget(scope=scope) == 1
        assert await bank.count("s2") == 1
    backup = (await bank.list_checkpoints())[0]
    assert backup["kind"] == "before_forget"
    await bank.restore_checkpoint(backup["id"])
    assert await contents(bank) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["clear", "forget", "restore"])
async def test_failed_safety_backup_aborts_mutation(bank, monkeypatch, operation):
    await bank.add_episode(episode("a"))
    saved = await bank.create_checkpoint()
    before = await contents(bank)
    generation = bank.generation

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(CheckpointStore, "create", fail)
    with pytest.raises(OSError, match="disk full"):
        if operation == "clear":
            await bank.clear()
        elif operation == "forget":
            await bank.forget(scope="s1")
        else:
            await bank.restore_checkpoint(saved["id"])
    assert bank.generation == generation
    assert await contents(bank) == before


@pytest.mark.asyncio
async def test_restore_failure_rolls_back_both_tables(bank):
    await bank.add_episode(episode("a"))
    saved = await bank.create_checkpoint()
    await bank.add_episode(episode("b"))
    before = await contents(bank)
    generation = bank.generation
    # A real SQLite failure after DELETE exercises the transaction boundary.
    bank._conn.execute(
        "CREATE TRIGGER reject_restore BEFORE INSERT ON experiences "
        "BEGIN SELECT RAISE(ABORT, 'injected failure'); END"
    )
    bank._conn.commit()
    with pytest.raises(sqlite3.IntegrityError, match="injected failure"):
        await bank.restore_checkpoint(saved["id"])
    assert await contents(bank) == before
    assert bank.generation == generation
    assert (await bank.list_checkpoints())[0]["kind"] == "before_restore"


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["bytes", "version", "schema", "count", "metadata"])
async def test_damaged_snapshot_cannot_modify_memory(bank, damage):
    await bank.add_episode(episode("a"))
    saved = await bank.create_checkpoint()
    path = bank._checkpoints.path(saved["id"])
    if damage == "bytes":
        path.write_bytes(b"not a sqlite database")
    else:
        with closing(sqlite3.connect(path)) as conn:
            if damage == "schema":
                conn.execute("ALTER TABLE experiences ADD COLUMN unexpected TEXT")
            elif damage == "count":
                conn.execute("DELETE FROM experiences")
            elif damage == "metadata":
                conn.execute("UPDATE jitrl_checkpoint SET payload = 'null'")
            else:
                payload = {**saved, "version": 999}
                conn.execute(
                    "UPDATE jitrl_checkpoint SET payload = ?", (json.dumps(payload),)
                )
            conn.commit()
    before = await contents(bank)
    generation = bank.generation
    with pytest.raises((sqlite3.DatabaseError, ValueError)):
        await bank.restore_checkpoint(saved["id"])
    assert await contents(bank) == before
    assert bank.generation == generation
    assert not any(
        row["kind"] == "before_restore" for row in await bank.list_checkpoints()
    )
    await bank.delete_checkpoint(saved["id"])
    assert not path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["../memory", "", "/tmp/file", "A" * 32, 12, None])
async def test_invalid_id_never_opens_or_deletes_another_file(bank, bad_id):
    await bank.add_episode(episode("a"))
    for action in (bank.restore_checkpoint, bank.delete_checkpoint):
        with pytest.raises(ValueError, match="ID"):
            await action(bad_id)
    assert await bank.count() == 1


@pytest.mark.asyncio
async def test_missing_checkpoint_and_noop_forget(bank):
    with pytest.raises(FileNotFoundError):
        await bank.restore_checkpoint("0" * 32)
    with pytest.raises(FileNotFoundError):
        await bank.delete_checkpoint("0" * 32)
    assert await bank.forget() == 0
    assert await bank.list_checkpoints() == []


@pytest.mark.asyncio
async def test_auto_rotation_preserves_manual_and_separate_safety_history(tmp_path):
    bank = MemoryBank(
        tmp_path / "memory.sqlite3", checkpoint_interval=2, keep_checkpoints=2
    )
    try:
        await bank.initialize()
        startup = (await bank.list_checkpoints())[0]
        assert startup["kind"] == "startup" and startup["experiences"] == 0
        manual = await bank.create_checkpoint("stable")
        for i in range(8):
            await bank.add_episode(episode(str(i)))
        rows = await bank.list_checkpoints()
        assert [row["experiences"] for row in rows if row["kind"] == "auto"] == [8, 6]
        for _ in range(4):
            await bank.clear()
        rows = await bank.list_checkpoints()
        assert manual["id"] in {row["id"] for row in rows}
        assert len([row for row in rows if row["kind"] == "before_forget"]) == 2
        assert len([row for row in rows if row["kind"] == "auto"]) == 2
        await bank.delete_checkpoint(manual["id"])
        assert len(await bank.list_checkpoints()) == 4
    finally:
        await bank.close()


@pytest.mark.asyncio
async def test_auto_backup_failure_retries_without_losing_committed_episode(
    tmp_path, monkeypatch
):
    bank = MemoryBank(tmp_path / "memory.sqlite3", checkpoint_interval=1)
    try:
        await bank.initialize()
        original = CheckpointStore.create

        def fail(*args, **kwargs):
            raise OSError("disk full")

        with monkeypatch.context() as patch:
            patch.setattr(CheckpointStore, "create", fail)
            assert await bank.add_episode(episode("a")) == 1
            assert bank._episodes_since_checkpoint == 1
        assert CheckpointStore.create == original
        await bank.add_episode(episode("b"))
        assert (await bank.list_checkpoints())[0]["experiences"] == 2
        assert bank._episodes_since_checkpoint == 0
    finally:
        await bank.close()


@pytest.mark.asyncio
async def test_interrupted_backup_is_not_published(bank, monkeypatch):
    def fail_check(*args, **kwargs):
        raise OSError("interrupted publication")

    monkeypatch.setattr(Path, "replace", fail_check)
    with pytest.raises(OSError):
        await bank.create_checkpoint()
    assert await bank.list_checkpoints() == []
    assert not list(bank._checkpoints.directory.iterdir())


@pytest.mark.asyncio
async def test_generation_blocks_worker_queued_during_restore(bank, monkeypatch):
    await bank.add_episode(episode("trusted"))
    saved = await bank.create_checkpoint()
    generation = bank.generation
    entered, release = threading.Event(), threading.Event()
    restore = CheckpointStore.restore

    def slow_restore(*args, **kwargs):
        entered.set()
        assert release.wait(5), "test did not release restore"
        return restore(*args, **kwargs)

    monkeypatch.setattr(CheckpointStore, "restore", slow_restore)
    restoring = asyncio.create_task(bank.restore_checkpoint(saved["id"]))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        writing = asyncio.create_task(
            bank.add_episode(episode("stale"), expected_generation=generation)
        )
        feedback = asyncio.create_task(
            bank.apply_user_feedback("trusted", -5, expected_generation=generation)
        )
    finally:
        release.set()
    await restoring
    assert await writing == await feedback == 0
    rows = (await bank.export_json())["experiences"]
    assert len(rows) == 1 and rows[0]["reward"] == 1
    assert (
        await bank.add_episode(episode("new"), expected_generation=bank.generation) == 1
    )


@pytest.mark.asyncio
async def test_manual_checkpoint_name_is_validated_and_redacted(bank):
    with pytest.raises(ValueError):
        await bank.create_checkpoint("a" * 81)
    with pytest.raises(ValueError):
        await bank.create_checkpoint({})
    info = await bank.create_checkpoint("stable alice@example.com")
    assert "alice@example.com" not in info["name"]


@pytest.mark.asyncio
async def test_uncommitted_transaction_does_not_hang_backup(bank):
    bank._conn.execute("BEGIN")
    try:
        with pytest.raises(RuntimeError, match="尚未提交"):
            await asyncio.wait_for(bank.create_checkpoint(), 3)
    finally:
        bank._conn.rollback()
    assert await bank.list_checkpoints() == []


@pytest.mark.asyncio
async def test_restore_protects_selected_safety_snapshot_during_rotation(tmp_path):
    bank = MemoryBank(
        tmp_path / "memory.sqlite3", checkpoint_interval=0, keep_checkpoints=1
    )
    try:
        await bank.initialize()
        await bank.add_episode(episode("trusted"))
        await bank.clear()
        saved = (await bank.list_checkpoints())[0]
        await bank.add_episode(episode("polluted"))
        result = await bank.restore_checkpoint(saved["id"])
        assert bank._checkpoints.path(saved["id"]).is_file()
        assert bank._checkpoints.path(result["safety_checkpoint"]["id"]).is_file()
        rows = (await bank.export_json())["experiences"]
        assert [row["episode_id"] for row in rows] == ["trusted"]
    finally:
        await bank.close()


@pytest.mark.asyncio
async def test_transient_bank_requires_explicit_checkpoint_directory(tmp_path):
    bank = MemoryBank(":memory:", checkpoint_interval=0)
    try:
        await bank.initialize()
        with pytest.raises(ValueError, match="checkpoint_dir"):
            await bank.create_checkpoint()
    finally:
        await bank.close()
    bank = MemoryBank(":memory:", checkpoint_dir=tmp_path / "snapshots")
    try:
        await bank.initialize()
        await bank.add_episode(episode("a"))
        saved = await bank.create_checkpoint()
        await bank.clear()
        await bank.restore_checkpoint(saved["id"])
        assert await bank.count() == 1
    finally:
        await bank.close()


def test_checkpoint_config_defaults_and_bounds():
    cfg = JitRLConfig.from_raw({})
    assert cfg.memory.checkpoint_interval == 50
    assert cfg.memory.keep_checkpoints == 20
    cfg = JitRLConfig.from_raw(
        {"memory": {"checkpoint_interval": -2, "keep_checkpoints": 0}}
    )
    assert cfg.memory.checkpoint_interval == 0
    assert cfg.memory.keep_checkpoints == 1
