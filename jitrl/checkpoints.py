"""Independent SQLite snapshots, published atomically outside the live bank."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)
_FORMAT_VERSION = 1
_KINDS = {"manual", "auto", "startup", "before_forget", "before_restore"}
_ID_PATTERN = re.compile(r"[0-9a-f]{32}")


class CheckpointStore:
    """Manage snapshots while the caller holds the memory bank's thread lock.

    Manual snapshots are never pruned. ``keep`` limits each of the automatic
    and safety histories separately, so repeated resets cannot evict the
    entire periodic history. No SQLite connection is shared with this class.
    """

    def __init__(self, directory: Path, *, keep: int = 20) -> None:
        self.directory = directory
        self.keep = max(1, int(keep))

    def path(self, checkpoint_id: str) -> Path:
        """Resolve an opaque ID; filenames and arbitrary paths are rejected."""
        if not isinstance(checkpoint_id, str) or not _ID_PATTERN.fullmatch(
            checkpoint_id
        ):
            raise ValueError("检查点 ID 无效，请从检查点列表复制完整 ID。")
        path = self.directory / f"{checkpoint_id}.sqlite3"
        if path.is_symlink():
            raise ValueError("检查点不能是符号链接。")
        if not path.is_file():
            raise FileNotFoundError("检查点不存在，可能已被删除或轮换。")
        return path

    @staticmethod
    def _open(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _info(conn: sqlite3.Connection, path: Path) -> dict[str, Any]:
        row = conn.execute("SELECT payload FROM jitrl_checkpoint").fetchone()
        if row is None:
            raise ValueError("检查点缺少元数据。")
        info = json.loads(row[0])
        if (
            not isinstance(info, dict)
            or info.get("version") != _FORMAT_VERSION
            or info.get("id") != path.stem
            or info.get("kind") not in _KINDS
            or not isinstance(info.get("name"), str)
            or not isinstance(info.get("created_at"), int | float)
            or not isinstance(info.get("experiences"), int)
            or not isinstance(info.get("episodes"), int)
        ):
            raise ValueError("检查点格式不受支持或元数据损坏。")
        return {**info, "size_bytes": path.stat().st_size}

    def list(self) -> list[dict[str, Any]]:
        """Return newest first, ignoring incomplete or damaged snapshots."""
        result = []
        for path in self.directory.glob("*.sqlite3"):
            if not _ID_PATTERN.fullmatch(path.stem) or path.is_symlink():
                continue
            try:
                with closing(self._open(path)) as conn:
                    result.append(self._info(conn, path))
            except (OSError, sqlite3.Error, ValueError, TypeError):
                _LOG.warning("[JitRL] unreadable checkpoint: %s", path, exc_info=True)
        return sorted(
            result, key=lambda row: (row["created_at"], row["id"]), reverse=True
        )

    def create(
        self,
        conn: sqlite3.Connection,
        *,
        name: str = "",
        kind: str = "manual",
        protect: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Back up committed WAL data, then atomically publish one complete file."""
        if kind not in _KINDS:
            raise ValueError("未知的检查点类型。")
        if not isinstance(name, str) or len(name) > 80:
            raise ValueError("检查点名称必须是最多 80 个字符的文本。")
        if conn.in_transaction:
            raise RuntimeError("无法在尚未提交的事务中创建检查点。")
        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint_id = uuid.uuid4().hex
        target = self.directory / f"{checkpoint_id}.sqlite3"
        temporary = target.with_suffix(".tmp")
        info = {
            "version": _FORMAT_VERSION,
            "id": checkpoint_id,
            "name": name.strip(),
            "kind": kind,
            "created_at": time.time(),
        }
        try:
            with closing(sqlite3.connect(temporary)) as snapshot:
                conn.backup(snapshot)
                snapshot.execute("PRAGMA journal_mode=DELETE")
                snapshot.execute("PRAGMA synchronous=FULL")
                # Count the backed-up view, including committed WAL pages.
                for table in ("experiences", "episodes"):
                    info[table] = snapshot.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0]
                snapshot.execute(
                    "CREATE TABLE jitrl_checkpoint (payload TEXT NOT NULL)"
                )
                snapshot.execute(
                    "INSERT INTO jitrl_checkpoint VALUES (?)",
                    (json.dumps(info, ensure_ascii=False),),
                )
                snapshot.commit()
                if snapshot.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                    raise ValueError("检查点完整性校验失败。")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        self.prune(protect=(*protect, checkpoint_id))
        return {**info, "size_bytes": target.stat().st_size}

    def prune(self, *, protect: tuple[str, ...] = ()) -> None:
        """Rotate automatic/safety snapshots without ever pruning manual ones."""
        try:
            rows = self.list()
            for kinds in ({"auto", "startup"}, {"before_forget", "before_restore"}):
                history = [row for row in rows if row["kind"] in kinds]
                for row in history[self.keep :]:
                    if row["id"] not in protect:
                        self.path(row["id"]).unlink()
        except (OSError, sqlite3.Error, ValueError):
            # A rotation failure must not invalidate an already published backup.
            _LOG.warning("[JitRL] checkpoint rotation failed", exc_info=True)

    def delete(self, checkpoint_id: str) -> None:
        """Delete exactly one selected snapshot, including a damaged file."""
        self.path(checkpoint_id).unlink()

    def restore(
        self,
        conn: sqlite3.Connection,
        checkpoint_id: str,
        *,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Validate first, make an undo snapshot, then replace rows transactionally.

        Only the two memory tables are restored; snapshots, configuration and
        process statistics remain independent. Scoped restores allocate new row
        IDs to avoid collisions with other sessions.
        """
        path = self.path(checkpoint_id)
        with closing(self._open(path)) as snapshot:
            info = self._info(snapshot, path)
            if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("检查点完整性校验失败，未修改当前记忆。")
            columns = {}
            for table in ("episodes", "experiences"):
                current = conn.execute(f"PRAGMA table_info({table})").fetchall()
                stored = snapshot.execute(f"PRAGMA table_info({table})").fetchall()
                if [tuple(row) for row in current] != [tuple(row) for row in stored]:
                    raise ValueError("检查点数据库结构不兼容，未修改当前记忆。")
                count = snapshot.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                if count != info[table]:
                    raise ValueError("检查点记录数不匹配，未修改当前记忆。")
                columns[table] = [
                    row["name"]
                    for row in current
                    if scope is None or row["name"] != "id"
                ]
            if scope is not None:
                other_ids = {
                    row[0]
                    for row in conn.execute(
                        "SELECT episode_id FROM episodes WHERE scope != ? "
                        "UNION SELECT episode_id FROM experiences WHERE scope != ?",
                        (scope, scope),
                    )
                }
                restored_ids = {
                    row[0]
                    for row in snapshot.execute(
                        "SELECT episode_id FROM episodes WHERE scope = ? "
                        "UNION SELECT episode_id FROM experiences WHERE scope = ?",
                        (scope, scope),
                    )
                }
                if other_ids & restored_ids:
                    raise ValueError(
                        "检查点的 episode ID 与其他会话冲突，未修改当前记忆。"
                    )

            safety = self.create(conn, kind="before_restore", protect=(checkpoint_id,))
            condition = "" if scope is None else " WHERE scope = ?"
            params = () if scope is None else (scope,)
            counts = {}
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM experiences" + condition, params)
                conn.execute("DELETE FROM episodes" + condition, params)
                for table, names in columns.items():
                    fields = ", ".join(f'"{name}"' for name in names)
                    placeholders = ", ".join("?" for _ in names)
                    cursor = snapshot.execute(
                        f"SELECT {fields} FROM {table}" + condition, params
                    )
                    count = 0
                    while batch := cursor.fetchmany(500):
                        conn.executemany(
                            f"INSERT INTO {table} ({fields}) VALUES ({placeholders})",
                            batch,
                        )
                        count += len(batch)
                    counts[table] = count
                conn.commit()
            except BaseException:
                conn.rollback()
                raise
        return {
            "checkpoint": info,
            "safety_checkpoint": safety,
            "scope": scope,
            **counts,
        }
