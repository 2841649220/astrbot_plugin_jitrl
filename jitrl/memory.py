"""SQLite-backed non-parametric memory M = {(s_i, a_i, G_i)}.

This module implements the storage and retrieval half of Algorithm 1: finished
episodes are written transactionally as (state, action, return) rows, and
topk() performs the Eq. 24 retrieval (Jaccard similarity over state token
sets) behind a cheap SQL prefilter, an optional surface gate, scope isolation
and a mild recency weighting.

Every SQLite call is pushed onto a worker thread so the AstrBot event loop is
never blocked, and each multi-row write runs inside one transaction.

Diagnostics go through ``astrbot.api.logger``; the standard library ``logging``
module is not used anywhere in this package.
"""

from __future__ import annotations

import asyncio
import heapq
import json
import sqlite3
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .checkpoints import CheckpointStore
from .index import ExperienceIndex, IndexHit
from .models import Episode, Neighbor, StateView
from .state import jaccard, redact

_CROSS_SCOPE_PENALTY: float = 0.85
"""Similarity multiplier applied to neighbours coming from another scope."""

_DEFAULT_GAMMA: float = 0.5
"""Discount factor used when an episode does not record its own."""

_RECENCY_FLOOR: float = 0.5
"""Lower bound of the recency factor; old experience is damped, not erased."""

_SECONDS_PER_DAY: float = 86400.0
_EXPORT_VERSION: int = 1

_INDEX_MAX_ENTRIES: int = 100000
"""Row cap above which the in-memory retrieval index is not used.

The index costs roughly 0.55 KB per 60-token experience, so 100k rows is about
55 MB -- already generous for a chat deployment. Beyond that the bank falls back
to the exact SQL scan so a misconfigured cap can never exhaust memory.
"""

_EAGER_WARM_ROWS: int = 4000
"""Build the index during initialize() up to this row count.

Small banks get a ready index before the first request; larger ones are warmed
in the background by the plugin so startup never blocks on a slow build.
"""

_SCHEMA_STATEMENTS: tuple[str, ...] = (
    "CREATE TABLE IF NOT EXISTS episodes ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "episode_id TEXT UNIQUE NOT NULL, "
    "scope TEXT NOT NULL DEFAULT '', "
    "task_family TEXT NOT NULL DEFAULT '', "
    "user_text TEXT NOT NULL DEFAULT '', "
    "success INTEGER, "
    "created_at REAL NOT NULL DEFAULT 0, "
    "gamma REAL NOT NULL DEFAULT 0.5, "
    "reward_source TEXT NOT NULL DEFAULT '', "
    "meta TEXT NOT NULL DEFAULT '{}'"
    ")",
    "CREATE TABLE IF NOT EXISTS experiences ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "episode_id TEXT NOT NULL DEFAULT '', "
    "scope TEXT NOT NULL DEFAULT '', "
    "step_index INTEGER NOT NULL DEFAULT 0, "
    "surface TEXT NOT NULL DEFAULT '', "
    "state_signature TEXT NOT NULL DEFAULT '', "
    "state_tokens TEXT NOT NULL DEFAULT '[]', "
    "state_text TEXT NOT NULL DEFAULT '', "
    "action_key TEXT NOT NULL DEFAULT '', "
    "action_name TEXT NOT NULL DEFAULT '', "
    "action_kind TEXT NOT NULL DEFAULT '', "
    "action_text TEXT NOT NULL DEFAULT '', "
    "action_args TEXT NOT NULL DEFAULT '{}', "
    "reward REAL NOT NULL DEFAULT 0, "
    "discounted_return REAL NOT NULL DEFAULT 0, "
    "outcome TEXT NOT NULL DEFAULT '', "
    "reward_source TEXT NOT NULL DEFAULT '', "
    "created_at REAL NOT NULL DEFAULT 0"
    ")",
    # Composite indexes: the retrieval scan filters on surface/scope and orders
    # by created_at, and every episode insert deletes by episode_id first. The
    # old single-column indexes could not serve either access pattern.
    "CREATE INDEX IF NOT EXISTS idx_experiences_episode ON experiences(episode_id)",
    "CREATE INDEX IF NOT EXISTS idx_experiences_surface_created "
    "ON experiences(surface, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_experiences_scope_created "
    "ON experiences(scope, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_experiences_action ON experiences(action_key)",
    "CREATE INDEX IF NOT EXISTS idx_experiences_created ON experiences(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_scope ON episodes(scope)",
    "CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at)",
)

_EPISODE_COLUMNS = (
    "episode_id, scope, task_family, user_text, success, created_at, gamma, "
    "reward_source, meta"
)
_INSERT_EPISODE_SQL = (
    f"INSERT INTO episodes ({_EPISODE_COLUMNS}) VALUES ({', '.join(['?'] * 9)})"
)

_EXPERIENCE_COLUMNS = (
    "episode_id, scope, step_index, surface, state_signature, state_tokens, "
    "state_text, action_key, action_name, action_kind, action_text, "
    "action_args, reward, discounted_return, outcome, reward_source, created_at"
)
_INSERT_EXPERIENCE_SQL = (
    f"INSERT INTO experiences ({_EXPERIENCE_COLUMNS}) VALUES ({', '.join(['?'] * 17)})"
)
_SELECT_EXPERIENCE_SQL = (
    "SELECT id AS entry_id, " + _EXPERIENCE_COLUMNS + " FROM experiences"
)


def _loads(raw: Any, default: Any) -> Any:
    """Decode a JSON column without letting corrupt data escape.

    Args:
        raw: Raw column value.
        default: Value returned when the payload is absent or not valid JSON.

    Returns:
        The decoded value, or default.
    """
    if isinstance(raw, str | bytes | bytearray):
        try:
            return json.loads(raw)
        except (TypeError, ValueError):
            return default
    return default


def _as_float(raw: Any, default: float = 0.0) -> float:
    """Coerce a JSON value to float.

    Args:
        raw: Raw value, possibly None or a string.
        default: Value returned when the input cannot be converted.

    Returns:
        The converted float, or default.
    """
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _as_int(raw: Any, default: int = 0) -> int:
    """Coerce a JSON value to int.

    Args:
        raw: Raw value, possibly None or a string.
        default: Value returned when the input cannot be converted.

    Returns:
        The converted int, or default.
    """
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


class MemoryBank:
    """Persistent store of (state, action, discounted return) experiences.

    Attributes:
        db_path: Location of the SQLite file.
        max_entries: Hard cap on stored experience rows; a value of zero or
            less disables pruning.
        redact_secrets: Whether free text is scrubbed before it is stored.
        keep_episodes: Number of episode headers retained during pruning.
        prune_strategy: One of "low_value", "oldest" or "none".
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        max_entries: int = 20000,
        redact: bool = True,
        keep_episodes: int = 500,
        prune_strategy: str = "low_value",
        checkpoint_dir: Path | str | None = None,
        checkpoint_interval: int = 50,
        keep_checkpoints: int = 20,
    ) -> None:
        """Initialize the bank without touching the filesystem.

        Args:
            db_path: SQLite file path, or ":memory:" for a transient bank.
            max_entries: Hard cap on stored experience rows.
            redact: Scrub secrets from free text before storing it.
            keep_episodes: Number of episode headers retained during pruning.
            prune_strategy: Eviction policy: "low_value", "oldest" or "none".
            checkpoint_dir: Snapshot directory; defaults to a sibling directory.
                Transient banks require an explicit directory for snapshots.
            checkpoint_interval: Episodes between automatic snapshots; 0 disables
                automatic snapshots. Safety backups still protect deletion.
            keep_checkpoints: Retained snapshots per automatic/safety history.
        """
        self.db_path = Path(db_path)
        self.max_entries = _as_int(max_entries, 20000)
        self.redact_secrets = bool(redact)
        self.keep_episodes = _as_int(keep_episodes, 500)
        strategy = str(prune_strategy or "low_value").strip().lower()
        self.prune_strategy = (
            strategy if strategy in {"low_value", "oldest", "none"} else "low_value"
        )
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        self._generation = 0
        self._index: ExperienceIndex | None = None
        self._row_count: int | None = None
        self._episodes_since_checkpoint = 0
        self.checkpoint_interval = max(0, _as_int(checkpoint_interval, 50))
        directory = (
            Path(checkpoint_dir)
            if checkpoint_dir is not None
            else self.db_path.parent / f"{self.db_path.stem}-checkpoints"
            if str(self.db_path) != ":memory:"
            else None
        )
        self._checkpoints = (
            CheckpointStore(directory, keep=keep_checkpoints)
            if directory is not None
            else None
        )

    # -- lifecycle -----------------------------------------------------------
    async def initialize(self) -> None:
        """Open the database and create the schema. Idempotent.

        Raises:
            sqlite3.Error: When the database cannot be opened or created.
            OSError: When the parent directory cannot be created.
        """
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        """Open the connection and create the schema (worker thread)."""
        with self._lock:
            if self._conn is not None:
                return
            target = str(self.db_path)
            if target != ":memory:":
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(target, check_same_thread=False, timeout=10.0)
            try:
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA busy_timeout=5000")
                # Retrieval reads the newest rows of a surface/scope; keeping the
                # working set in memory keeps those scans off the disk on a
                # plugin that queries on every single LLM request.
                try:
                    conn.execute("PRAGMA cache_size=-32000")
                    conn.execute("PRAGMA temp_store=MEMORY")
                except sqlite3.Error:
                    pass
                for statement in _SCHEMA_STATEMENTS:
                    conn.execute(statement)
                conn.commit()
                if (
                    self._checkpoints is not None
                    and self.checkpoint_interval > 0
                    and not self._checkpoints.list()
                ):
                    self._checkpoints.create(conn, kind="startup")
            except BaseException:
                conn.close()
                raise
            self._conn = conn
            try:
                self._row_count = self._count_locked(scope=None)
            except sqlite3.Error:
                self._row_count = None
            if self._row_count is not None and 0 < self._row_count <= _EAGER_WARM_ROWS:
                self._ensure_index_locked()

    async def warm_index(self) -> bool:
        """Build the retrieval index in the background.

        Returns:
            True when an index is available afterwards, False when the bank is
            too large, closed, or the build failed.
        """
        return await asyncio.to_thread(self._warm_index_sync)

    def _warm_index_sync(self) -> bool:
        """Build the index on a worker thread.

        Returns:
            True when the index is ready.
        """
        with self._lock:
            if self._conn is None:
                return False
            try:
                return self._ensure_index_locked() is not None
            except (sqlite3.Error, ValueError):
                logger.warning("[JitRL] retrieval index build failed", exc_info=True)
                self._index = None
                return False

    def _ensure_index_locked(self) -> ExperienceIndex | None:
        """Return the retrieval index, building it from SQLite when needed.

        The lock must already be held. A bank larger than the index cap keeps
        returning None, which makes topk() use the exact SQL scan.

        Returns:
            The ready index, or None when the bank is too large / unreadable.
        """
        if self._index is not None:
            return self._index
        conn = self._require_conn()
        total = self._row_count
        if total is None:
            total = self._count_locked(scope=None)
        if total > _INDEX_MAX_ENTRIES:
            return None
        index = ExperienceIndex(max_entries=_INDEX_MAX_ENTRIES)
        rows = conn.execute(
            "SELECT id, scope, surface, state_tokens, action_key, "
            "discounted_return, reward, created_at FROM experiences ORDER BY id"
        )
        for row in rows:
            tokens = _loads(row["state_tokens"], None)
            if not isinstance(tokens, list | tuple):
                continue
            appended = index.append(
                entry_id=_as_int(row["id"]),
                tokens=(str(token) for token in tokens),
                scope=str(row["scope"] or ""),
                surface=str(row["surface"] or ""),
                action_key=str(row["action_key"] or ""),
                discounted_return=_as_float(row["discounted_return"]),
                reward=_as_float(row["reward"]),
                created_at=_as_float(row["created_at"]),
            )
            if not appended and index.is_full:
                # Keep what fits; the caller still gets exact results for the
                # indexed prefix and the scan fallback is not needed.
                break
        self._index = index
        return index

    def _invalidate_index_locked(self) -> None:
        """Drop the in-memory index so the next query rebuilds it."""
        self._index = None

    def _purge_rows_locked(self, entry_ids: list[int]) -> None:
        """Remove rows from the index after they were deleted from SQLite.

        Args:
            entry_ids: Database primary keys of the deleted rows.
        """
        if self._index is not None and entry_ids:
            self._index.remove_many(entry_ids)

    async def close(self) -> None:
        """Close the connection. Idempotent and never raises."""
        await asyncio.to_thread(self._close_sync)

    def _close_sync(self) -> None:
        """Close the connection (worker thread)."""
        with self._lock:
            conn, self._conn = self._conn, None
            if conn is None:
                return
            try:
                conn.commit()
                conn.close()
            except sqlite3.Error:
                pass

    def _require_conn(self) -> sqlite3.Connection:
        """Return the open connection.

        Returns:
            The live SQLite connection.

        Raises:
            RuntimeError: When initialize() has not been awaited yet.
        """
        conn = self._conn
        if conn is None:
            raise RuntimeError(
                "MemoryBank is not initialized; await initialize() first"
            )
        return conn

    @property
    def generation(self) -> int:
        """Epoch captured by requests to reject writes after a reset/restore."""
        return self._generation

    def _require_checkpoints(self) -> CheckpointStore:
        if self._checkpoints is None:
            raise ValueError("内存数据库需要设置 checkpoint_dir 才能使用检查点。")
        return self._checkpoints

    async def create_checkpoint(self, name: str = "") -> dict[str, Any]:
        """Save a manual checkpoint, retained until explicitly deleted."""
        return await asyncio.to_thread(self._create_checkpoint_sync, name)

    def _create_checkpoint_sync(self, name: str) -> dict[str, Any]:
        with self._lock:
            conn = self._require_conn()
            if not isinstance(name, str) or len(name) > 80:
                raise ValueError("检查点名称必须是最多 80 个字符的文本。")
            if self.redact_secrets:
                name = redact(name)
            return self._require_checkpoints().create(conn, name=name)

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        """List durable snapshots from newest to oldest."""
        return await asyncio.to_thread(self._list_checkpoints_sync)

    def _list_checkpoints_sync(self) -> list[dict[str, Any]]:
        with self._lock:
            self._require_conn()
            return self._require_checkpoints().list()

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Permanently remove one snapshot by its complete opaque ID."""
        await asyncio.to_thread(self._delete_checkpoint_sync, checkpoint_id)

    def _delete_checkpoint_sync(self, checkpoint_id: str) -> None:
        with self._lock:
            self._require_conn()
            self._require_checkpoints().delete(checkpoint_id)

    async def restore_checkpoint(
        self, checkpoint_id: str, *, scope: str | None = None
    ) -> dict[str, Any]:
        """Restore all memory or one session, saving an undo snapshot first."""
        return await asyncio.to_thread(
            self._restore_checkpoint_sync, checkpoint_id, scope
        )

    def _restore_checkpoint_sync(
        self, checkpoint_id: str, scope: str | None
    ) -> dict[str, Any]:
        with self._lock:
            result = self._require_checkpoints().restore(
                self._require_conn(), checkpoint_id, scope=scope
            )
            self._generation += 1
            self._episodes_since_checkpoint = 0
            self._row_count = None
            self._invalidate_index_locked()
            return result

    def _backup_before_forget_locked(self) -> None:
        if self._checkpoints is not None:
            self._checkpoints.create(self._require_conn(), kind="before_forget")

    # -- writes --------------------------------------------------------------
    async def add_episode(
        self, episode: Episode, *, expected_generation: int | None = None
    ) -> int:
        """Store one finished episode and its transitions atomically.

        Args:
            episode: Episode to persist. Re-adding the same episode_id replaces
                the previous rows.
            expected_generation: Reject results from before a reset/restore.

        Returns:
            The number of experience rows written.

        Raises:
            RuntimeError: When the bank is not initialized.
            sqlite3.Error: When the transaction fails; it is rolled back first.
        """
        return await asyncio.to_thread(
            self._add_episode_sync, episode, expected_generation
        )

    def _add_episode_sync(
        self, episode: Episode, expected_generation: int | None = None
    ) -> int:
        """Persist one episode inside a single transaction (worker thread).

        Args:
            episode: Episode to persist.

        Returns:
            The number of experience rows written.
        """
        meta = episode.meta if isinstance(episode.meta, Mapping) else {}
        gamma = _as_float(meta.get("gamma", _DEFAULT_GAMMA), _DEFAULT_GAMMA)
        episode_id = str(episode.episode_id or "")
        scope = str(episode.scope or "")
        created_at = _as_float(episode.created_at, time.time())
        user_text = str(episode.user_text or "")
        success = None if episode.success is None else int(bool(episode.success))
        reward_source = str(meta.get("reward_source") or "")

        rows: list[tuple[Any, ...]] = []
        for position, step in enumerate(episode.steps or []):
            state = step.state
            action = step.action
            try:
                tokens = json.dumps(sorted(state.tokens), ensure_ascii=False)
            except TypeError:
                tokens = "[]"
            state_text = str(state.text or "")
            action_text = str(action.text or "")
            try:
                args_json = json.dumps(
                    dict(action.args), ensure_ascii=False, sort_keys=True, default=str
                )
            except (TypeError, ValueError):
                args_json = "{}"
            if self.redact_secrets:
                state_text = redact(state_text)
                action_text = redact(action_text)
                args_json = redact(args_json)
            rows.append(
                (
                    episode_id,
                    scope,
                    _as_int(step.index, position),
                    str(state.surface or ""),
                    str(state.signature or ""),
                    tokens,
                    state_text,
                    str(action.key or ""),
                    str(action.name or ""),
                    str(action.kind or ""),
                    action_text,
                    args_json,
                    _as_float(step.reward),
                    _as_float(step.discounted_return),
                    str(step.outcome or ""),
                    str(step.reward_source or ""),
                    created_at,
                )
            )
        if self.redact_secrets:
            user_text = redact(user_text)

        with self._lock:
            conn = self._require_conn()
            if (
                expected_generation is not None
                and expected_generation != self._generation
            ):
                return 0
            try:
                conn.execute("DELETE FROM episodes WHERE episode_id = ?", (episode_id,))
                replaced = conn.execute(
                    "DELETE FROM experiences WHERE episode_id = ?", (episode_id,)
                )
                replaced_rows = max(0, int(replaced.rowcount or 0))
                if replaced_rows and self._index is not None:
                    # The episode was rewritten under a fresh set of row ids;
                    # drop the stale entries before indexing the new ones.
                    self._invalidate_index_locked()
                conn.execute(
                    _INSERT_EPISODE_SQL,
                    (
                        episode_id,
                        scope,
                        str(episode.task_family or ""),
                        user_text,
                        success,
                        created_at,
                        gamma,
                        reward_source,
                        json.dumps(
                            dict(meta), ensure_ascii=False, sort_keys=True, default=str
                        ),
                    ),
                )
                if rows:
                    conn.executemany(_INSERT_EXPERIENCE_SQL, rows)
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            if self._row_count is not None:
                self._row_count += len(rows) - replaced_rows
            if self._index is not None and rows and self._row_count is not None:
                self._append_episode_to_index_locked(conn, episode_id)
            self._episodes_since_checkpoint += 1
            if self.max_entries > 0:
                total = self._count_locked(scope=None)
                if total > self.max_entries:
                    self._prune_locked()
            if (
                self._checkpoints is not None
                and self.checkpoint_interval > 0
                and self._episodes_since_checkpoint >= self.checkpoint_interval
            ):
                try:
                    self._checkpoints.create(conn, kind="auto")
                    self._episodes_since_checkpoint = 0
                except (OSError, sqlite3.Error, ValueError):
                    logger.warning(
                        "[JitRL] automatic checkpoint failed; will retry next episode",
                        exc_info=True,
                    )
        return len(rows)

    def _append_episode_to_index_locked(
        self, conn: sqlite3.Connection, episode_id: str
    ) -> None:
        """Index the rows of an episode that was just committed.

        Args:
            conn: Live connection (lock already held).
            episode_id: Episode whose rows should be indexed.
        """
        index = self._index
        if index is None:
            return
        rows = conn.execute(
            "SELECT id, scope, surface, state_tokens, action_key, "
            "discounted_return, reward, created_at FROM experiences "
            "WHERE episode_id = ? ORDER BY id",
            (episode_id,),
        ).fetchall()
        for row in rows:
            tokens = _loads(row["state_tokens"], None)
            if not isinstance(tokens, list | tuple):
                continue
            index.append(
                entry_id=_as_int(row["id"]),
                tokens=(str(token) for token in tokens),
                scope=str(row["scope"] or ""),
                surface=str(row["surface"] or ""),
                action_key=str(row["action_key"] or ""),
                discounted_return=_as_float(row["discounted_return"]),
                reward=_as_float(row["reward"]),
                created_at=_as_float(row["created_at"]),
            )

    async def forget(
        self,
        *,
        scope: str | None = None,
        action_key: str | None = None,
        older_than: float | None = None,
    ) -> int:
        """Delete experiences matching the given filters.

        Args:
            scope: Only delete rows from this scope.
            action_key: Only delete rows with this normalised action key.
            older_than: Only delete rows created before this Unix timestamp.

        Returns:
            The number of experience rows deleted. When no filter is supplied
            nothing is deleted, so a caller bug cannot wipe the memory.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._forget_sync, scope, action_key, older_than)

    def _forget_sync(
        self, scope: str | None, action_key: str | None, older_than: float | None
    ) -> int:
        """Delete matching experiences inside a transaction (worker thread).

        Args:
            scope: Scope filter.
            action_key: Action key filter.
            older_than: Creation-time upper bound.

        Returns:
            The number of deleted experience rows.
        """
        conditions: list[str] = []
        params: list[Any] = []
        if scope is not None:
            conditions.append("scope = ?")
            params.append(str(scope))
        if action_key is not None:
            conditions.append("action_key = ?")
            params.append(str(action_key))
        if older_than is not None:
            conditions.append("created_at < ?")
            params.append(_as_float(older_than))
        if not conditions:
            return 0
        where = " WHERE " + " AND ".join(conditions)
        with self._lock:
            conn = self._require_conn()
            self._backup_before_forget_locked()
            deleted_ids: list[int] = []
            if self._index is not None:
                deleted_ids = [
                    _as_int(row[0])
                    for row in conn.execute(
                        "SELECT id FROM experiences" + where, params
                    ).fetchall()
                ]
            try:
                cursor = conn.execute("DELETE FROM experiences" + where, params)
                removed = max(0, int(cursor.rowcount or 0))
                self._drop_empty_episode_headers_locked()
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            self._generation += 1
            if self._row_count is not None:
                self._row_count = max(0, self._row_count - removed)
            self._purge_rows_locked(deleted_ids)
        return removed

    async def clear(self) -> int:
        """Delete every stored experience and episode header.

        Returns:
            The number of experience rows removed.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._clear_sync)

    def _clear_sync(self) -> int:
        """Wipe both tables inside one transaction (worker thread).

        Returns:
            The number of deleted experience rows.
        """
        with self._lock:
            conn = self._require_conn()
            self._backup_before_forget_locked()
            try:
                cursor = conn.execute("DELETE FROM experiences")
                removed = max(0, int(cursor.rowcount or 0))
                conn.execute("DELETE FROM episodes")
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            self._generation += 1
            self._row_count = 0
            if self._index is not None:
                self._index.clear()
        return removed

    async def prune(self) -> int:
        """Evict experiences until the bank is within its configured cap.

        Returns:
            The number of experience rows deleted.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._prune_sync)

    def _prune_sync(self) -> int:
        """Evict over-quota experiences (worker thread).

        Returns:
            The number of deleted experience rows.
        """
        with self._lock:
            return self._prune_locked()

    def _prune_locked(self) -> int:
        """Evict whole episodes until the row count fits the cap.

        Eviction happens per episode so the memory never keeps a fragment of a
        trajectory whose siblings were removed.

        Returns:
            The number of deleted experience rows.
        """
        conn = self._require_conn()
        if self.max_entries <= 0:
            return 0
        total = self._count_locked(scope=None)
        removed = 0
        if total > self.max_entries:
            if self.prune_strategy == "low_value":
                order_by = "MIN(ABS(discounted_return)) ASC, MAX(created_at) ASC"
            else:
                order_by = "MAX(created_at) ASC"
            sql = (
                "SELECT episode_id AS eid, COUNT(*) AS n FROM experiences "
                f"GROUP BY episode_id ORDER BY {order_by}"
            )
            groups = conn.execute(sql).fetchall()
            shortfall = total - self.max_entries
            purged: list[int] = []
            for row in groups:
                if shortfall <= 0:
                    break
                episode_id = str(row["eid"] or "")
                if self._index is not None:
                    purged.extend(
                        _as_int(item[0])
                        for item in conn.execute(
                            "SELECT id FROM experiences WHERE episode_id = ?",
                            (episode_id,),
                        ).fetchall()
                    )
                cursor = conn.execute(
                    "DELETE FROM experiences WHERE episode_id = ?", (episode_id,)
                )
                removed += max(0, int(cursor.rowcount or 0))
                shortfall -= _as_int(row["n"])
            conn.commit()
            if self._row_count is not None:
                self._row_count = max(0, self._row_count - removed)
            self._purge_rows_locked(purged)
        if self.keep_episodes > 0:
            conn.execute(
                "DELETE FROM episodes WHERE id NOT IN ("
                "SELECT id FROM episodes ORDER BY created_at DESC, id DESC LIMIT ?"
                ") AND episode_id NOT IN (SELECT DISTINCT episode_id FROM experiences)",
                (self.keep_episodes,),
            )
            conn.commit()
        self._drop_empty_episode_headers_locked()
        conn.commit()
        return removed

    def _drop_empty_episode_headers_locked(self) -> int:
        """Delete episode headers that no longer have any experience row.

        Returns:
            The number of episode headers deleted.
        """
        conn = self._require_conn()
        cursor = conn.execute(
            "DELETE FROM episodes WHERE episode_id NOT IN "
            "(SELECT DISTINCT episode_id FROM experiences)"
        )
        return max(0, int(cursor.rowcount or 0))

    async def apply_user_feedback(
        self, episode_id: str, delta: float, *, expected_generation: int | None = None
    ) -> int:
        """Shift one episode's rewards and recompute its discounted returns.

        Args:
            episode_id: Episode that received the feedback.
            delta: Reward delta added to every step of the episode.

        Returns:
            The number of experience rows updated.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(
            self._apply_feedback_sync, episode_id, delta, expected_generation
        )

    def _apply_feedback_sync(
        self, episode_id: str, delta: float, expected_generation: int | None = None
    ) -> int:
        """Apply a reward delta and rebuild G_t (worker thread).

        The discount factor is read back from the stored episode header, so
        the reconstruction uses the same gamma that produced the original
        returns.

        Args:
            episode_id: Episode that received the feedback.
            delta: Reward delta added to every step of the episode.

        Returns:
            The number of updated rows, or zero when the episode is unknown.
        """
        identifier = str(episode_id or "")
        shift = _as_float(delta)
        with self._lock:
            conn = self._require_conn()
            if (
                expected_generation is not None
                and expected_generation != self._generation
            ):
                return 0
            header = conn.execute(
                "SELECT gamma FROM episodes WHERE episode_id = ?", (identifier,)
            ).fetchone()
            gamma = _DEFAULT_GAMMA
            if header is not None:
                gamma = _as_float(header["gamma"], _DEFAULT_GAMMA)
            gamma = max(0.0, min(1.0, gamma))
            rows = conn.execute(
                "SELECT id, reward FROM experiences WHERE episode_id = ? "
                "ORDER BY step_index ASC, id ASC",
                (identifier,),
            ).fetchall()
            if not rows:
                return 0
            updates: list[tuple[float, float, int]] = []
            running = 0.0
            for row in reversed(rows):
                reward = _as_float(row["reward"]) + shift
                running = reward + gamma * running
                updates.append((reward, running, _as_int(row["id"])))
            try:
                conn.executemany(
                    "UPDATE experiences SET reward = ?, discounted_return = ? WHERE id = ?",
                    updates,
                )
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            index = self._index
            if index is not None:
                for reward, discounted_return, entry_id in updates:
                    index.update_return(
                        entry_id,
                        discounted_return=discounted_return,
                        reward=reward,
                    )
        return len(updates)

    async def import_json(self, data: Mapping[str, Any]) -> int:
        """Import a payload produced by export_json().

        Importing appends experience rows and upserts episode headers; it never
        deletes existing memory.

        Args:
            data: Mapping with "episodes" and "experiences" lists.

        Returns:
            The number of experience rows imported.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._import_sync, data)

    def _import_sync(self, data: Mapping[str, Any]) -> int:
        """Import a JSON payload inside one transaction (worker thread).

        Args:
            data: Mapping with "episodes" and "experiences" lists.

        Returns:
            The number of imported experience rows.
        """
        if not isinstance(data, Mapping):
            return 0
        raw_episodes = data.get("episodes")
        raw_experiences = data.get("experiences")
        episodes = raw_episodes if isinstance(raw_episodes, list) else []
        experiences = raw_experiences if isinstance(raw_experiences, list) else []
        if not episodes and not experiences:
            return 0
        with self._lock:
            conn = self._require_conn()
            imported = 0
            try:
                for item in episodes:
                    if not isinstance(item, Mapping):
                        continue
                    row = self._episode_row_from_json(item)
                    if row is None:
                        continue
                    conn.execute("DELETE FROM episodes WHERE episode_id = ?", (row[0],))
                    conn.execute(_INSERT_EPISODE_SQL, row)
                for item in experiences:
                    if not isinstance(item, Mapping):
                        continue
                    row = self._experience_row_from_json(item)
                    if row is None:
                        continue
                    conn.execute(_INSERT_EXPERIENCE_SQL, row)
                    imported += 1
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            if imported:
                self._row_count = None
                self._invalidate_index_locked()
        return imported

    def _episode_row_from_json(self, item: Mapping[str, Any]) -> tuple[Any, ...] | None:
        """Flatten one exported episode mapping into an insert tuple.

        Args:
            item: Mapping with the exported episode fields.

        Returns:
            A 9-element row tuple, or None when the row has no episode ID.
        """
        episode_id = str(item.get("episode_id") or "")
        if not episode_id:
            return None
        meta = item.get("meta")
        meta_json = (
            json.dumps(meta, ensure_ascii=False, sort_keys=True, default=str)
            if isinstance(meta, Mapping)
            else "{}"
        )
        success = item.get("success")
        user_text = str(item.get("user_text") or "")
        if self.redact_secrets:
            user_text = redact(user_text)
        return (
            episode_id,
            str(item.get("scope") or ""),
            str(item.get("task_family") or ""),
            user_text,
            None if success is None else int(bool(success)),
            _as_float(item.get("created_at"), time.time()),
            _as_float(item.get("gamma"), _DEFAULT_GAMMA),
            str(item.get("reward_source") or ""),
            meta_json,
        )

    async def merge_json(self, data: Mapping[str, Any]) -> int:
        """Merge an export payload without duplicating known episodes.

        Unlike import_json(), which appends every exported experience row, this
        method treats episode_id as the unit of identity: episodes already
        present in the bank are left untouched and their experience rows are
        skipped. It is therefore safe to retry a partially completed migration
        (for example after a crash between the import and its bookkeeping).

        Args:
            data: Mapping with "episodes" and "experiences" lists.

        Returns:
            The number of experience rows added.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._merge_sync, data)

    def _merge_sync(self, data: Mapping[str, Any]) -> int:
        """Merge an export payload inside one transaction (worker thread).

        Args:
            data: Mapping with "episodes" and "experiences" lists.

        Returns:
            The number of experience rows added.
        """
        if not isinstance(data, Mapping):
            return 0
        raw_episodes = data.get("episodes")
        raw_experiences = data.get("experiences")
        episodes = raw_episodes if isinstance(raw_episodes, list) else []
        experiences = raw_experiences if isinstance(raw_experiences, list) else []
        if not episodes and not experiences:
            return 0
        with self._lock:
            conn = self._require_conn()
            existing = {
                str(row[0]) for row in conn.execute("SELECT episode_id FROM episodes")
            }
            added: set[str] = set()
            imported = 0
            try:
                for item in episodes:
                    if not isinstance(item, Mapping):
                        continue
                    row = self._episode_row_from_json(item)
                    if row is None:
                        continue
                    episode_id = str(row[0])
                    if episode_id in existing:
                        continue
                    conn.execute(_INSERT_EPISODE_SQL, row)
                    existing.add(episode_id)
                    added.add(episode_id)
                for item in experiences:
                    if not isinstance(item, Mapping):
                        continue
                    row = self._experience_row_from_json(item)
                    if row is None:
                        continue
                    episode_id = str(row[0])
                    if episode_id in existing and episode_id not in added:
                        continue
                    conn.execute(_INSERT_EXPERIENCE_SQL, row)
                    imported += 1
                conn.commit()
            except Exception:
                try:
                    conn.rollback()
                except sqlite3.Error:
                    pass
                raise
            if imported or added:
                self._row_count = None
                self._invalidate_index_locked()
        return imported

    def _experience_row_from_json(
        self, item: Mapping[str, Any]
    ) -> tuple[Any, ...] | None:
        """Flatten one exported experience mapping into an insert tuple.

        Args:
            item: Mapping with the exported experience fields.

        Returns:
            A 17-element row tuple, or None when the row has no usable
            episode identifier.
        """
        episode_id = str(item.get("episode_id") or "")
        if not episode_id:
            return None
        tokens = item.get("state_tokens")
        if isinstance(tokens, list | tuple):
            token_json = json.dumps(
                [str(token) for token in tokens], ensure_ascii=False
            )
        elif isinstance(tokens, str):
            token_json = tokens
        else:
            token_json = "[]"
        args = item.get("action_args")
        if isinstance(args, Mapping):
            args_json = json.dumps(
                dict(args), ensure_ascii=False, sort_keys=True, default=str
            )
        elif isinstance(args, str):
            args_json = args
        else:
            args_json = "{}"
        state_text = str(item.get("state_text") or "")
        action_text = str(item.get("action_text") or "")
        if self.redact_secrets:
            state_text = redact(state_text)
            action_text = redact(action_text)
            args_json = redact(args_json)
        return (
            episode_id,
            str(item.get("scope") or ""),
            _as_int(item.get("step_index")),
            str(item.get("surface") or ""),
            str(item.get("state_signature") or ""),
            token_json,
            state_text,
            str(item.get("action_key") or ""),
            str(item.get("action_name") or ""),
            str(item.get("action_kind") or ""),
            action_text,
            args_json,
            _as_float(item.get("reward")),
            _as_float(item.get("discounted_return")),
            str(item.get("outcome") or ""),
            str(item.get("reward_source") or ""),
            _as_float(item.get("created_at"), time.time()),
        )

    def _count_locked(self, scope: str | None) -> int:
        """Count experience rows (lock already held).

        Args:
            scope: Optional scope filter.

        Returns:
            The number of matching rows.
        """
        conn = self._require_conn()
        if scope is None:
            if self._row_count is not None:
                return self._row_count
            row = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()
            self._row_count = _as_int(row[0]) if row is not None else 0
            return self._row_count
        row = conn.execute(
            "SELECT COUNT(*) FROM experiences WHERE scope = ?", (str(scope),)
        ).fetchone()
        return _as_int(row[0]) if row is not None else 0

    # -- reads ---------------------------------------------------------------
    async def count(self, scope: str | None = None) -> int:
        """Count stored experiences.

        Args:
            scope: Optional scope filter.

        Returns:
            The number of matching experience rows.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._count_sync, scope)

    def _count_sync(self, scope: str | None) -> int:
        """Count experience rows (worker thread).

        Args:
            scope: Optional scope filter.

        Returns:
            The number of matching rows.
        """
        with self._lock:
            return self._count_locked(scope)

    async def counts(self, scope: str | None = None) -> dict[str, int]:
        """Count currently stored experiences and episode headers together.

        Args:
            scope: Optional scope filter.

        Returns:
            Current database counts, independent of process learning counters.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._counts_sync, scope)

    def _counts_sync(self, scope: str | None) -> dict[str, int]:
        """Read both counts under one lock so mutations cannot split the view."""
        with self._lock:
            conn = self._require_conn()
            sql = "SELECT COUNT(*) FROM episodes"
            params = ()
            if scope is not None:
                sql += " WHERE scope = ?"
                params = (str(scope),)
            row = conn.execute(sql, params).fetchone()
            return {
                "experiences": self._count_locked(scope),
                "episodes": _as_int(row[0]) if row is not None else 0,
            }

    async def topk(
        self,
        query: StateView,
        *,
        scope: str | None = None,
        k: int = 10,
        threshold: float = 0.8,
        surface_gate: bool = False,
        prefilter_limit: int = 4000,
        recency_half_life_days: float = 30.0,
        scope_mode: str = "hybrid",
        now: float | None = None,
    ) -> list[Neighbor]:
        """Retrieve the k most similar experiences for a query state.

        Candidate generation runs through the inverted index (jitrl.index), so
        the cost is proportional to the rows that can clear the threshold rather
        than to the size of the bank; the result is identical to an exhaustive
        Eq. 24 scan. Rows below the threshold are dropped, cross-scope
        neighbours are down-weighted in hybrid mode, and the survivors are
        ranked by similarity, newest first.

        Args:
            query: Abstracted current state.
            scope: Scope of the current session.
            k: Maximum number of neighbours to return.
            threshold: Minimum Jaccard similarity.
            surface_gate: Restrict results to the query surface.
            prefilter_limit: SQL scan batch size and compatibility cap on the
                number of returned rows. Candidates are never truncated.
            recency_half_life_days: Half-life of the recency weight; a value of
                zero or less disables the recency factor.
            scope_mode: "session", "global" or "hybrid".
            now: Reference timestamp for the recency weight; defaults to the
                current time.

        Returns:
            Neighbours sorted by (-similarity, -created_at). An empty query
            token set yields an empty list.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        if not isinstance(query, StateView) or not query.tokens:
            return []
        return await asyncio.to_thread(
            self._topk_sync,
            query,
            scope=scope,
            k=k,
            threshold=threshold,
            surface_gate=surface_gate,
            prefilter_limit=prefilter_limit,
            recency_half_life_days=recency_half_life_days,
            scope_mode=scope_mode,
            now=now,
        )

    def _neighbours_from_hits_locked(self, hits: list[IndexHit]) -> list[Neighbor]:
        """Attach the display fields of the ranked rows (lock already held).

        The index only stores what retrieval needs; action and state text are
        read back from SQLite for the k winners so the returned Neighbor always
        reflects the committed row (and a row deleted in the meantime is simply
        dropped).

        Args:
            hits: Ranked index hits.

        Returns:
            The neighbours, in the order of hits.
        """
        conn = self._require_conn()
        placeholders = ",".join("?" * len(hits))
        rows = conn.execute(
            "SELECT id, action_key, action_text, discounted_return, reward, "
            "state_text, scope FROM experiences WHERE id IN (" + placeholders + ")",
            [hit.entry_id for hit in hits],
        ).fetchall()
        by_id = {_as_int(row["id"]): row for row in rows}
        neighbours: list[Neighbor] = []
        for hit in hits:
            row = by_id.get(hit.entry_id)
            if row is None:
                continue
            neighbours.append(
                Neighbor(
                    entry_id=hit.entry_id,
                    similarity=hit.similarity,
                    action_key=str(row["action_key"] or ""),
                    action_text=str(row["action_text"] or ""),
                    discounted_return=_as_float(row["discounted_return"]),
                    similarity_weight=hit.weight,
                    reward=_as_float(row["reward"]),
                    state_text=str(row["state_text"] or ""),
                    scope=str(row["scope"] or ""),
                    created_at=hit.created_at,
                )
            )
        return neighbours

    def _topk_sync(
        self,
        query: StateView,
        *,
        scope: str | None,
        k: int,
        threshold: float,
        surface_gate: bool,
        prefilter_limit: int,
        recency_half_life_days: float,
        scope_mode: str,
        now: float | None,
    ) -> list[Neighbor]:
        """Retrieve the exact top-k neighbours (worker thread).

        Uses the in-memory inverted index when one is available and falls back
        to a streaming SQL scan for a bank above the index cap.

        Args:
            query: Abstracted current state.
            scope: Scope of the current session.
            k: Maximum number of neighbours.
            threshold: Minimum Jaccard similarity.
            surface_gate: Restrict results to the query surface.
            prefilter_limit: SQL batch size and compatibility result cap.
            recency_half_life_days: Half-life of the recency weight.
            scope_mode: "session", "global" or "hybrid".
            now: Reference timestamp for the recency weight.

        Returns:
            The ranked neighbours.
        """
        limit = max(1, _as_int(prefilter_limit, 4000))
        neighbours_wanted = min(max(1, _as_int(k, 10)), limit)
        minimum = max(0.0, min(1.0, _as_float(threshold, 0.8)))
        mode = str(scope_mode or "hybrid").strip().lower()
        if mode not in {"session", "global", "hybrid"}:
            mode = "hybrid"
        query_scope = str(scope or "")
        half_life = _as_float(recency_half_life_days, 0.0)
        reference_now = _as_float(now, time.time())

        # Fast path: the inverted index returns exactly the rows the scan below
        # would return (the test suite checks it against a brute-force reference)
        # but only intersects the handful of rows that can possibly clear the
        # threshold. At k = 10 and threshold = 0.8 a 20k-row bank costs a few
        # hundred intersections instead of 20,000 full Jaccard computations.
        with self._lock:
            try:
                index = self._ensure_index_locked()
            except (sqlite3.Error, ValueError):
                logger.warning(
                    "[JitRL] retrieval index unavailable; using the SQL scan",
                    exc_info=True,
                )
                index = None
            if index is not None:
                hits = index.search(
                    query.tokens,
                    k=neighbours_wanted,
                    threshold=minimum,
                    scope=query_scope,
                    surface=str(query.surface or ""),
                    surface_gate=bool(surface_gate),
                    scope_mode=mode,
                    prefilter_limit=limit,
                    recency_half_life_days=half_life,
                    cross_scope_penalty=_CROSS_SCOPE_PENALTY,
                    recency_floor=_RECENCY_FLOOR,
                    now=reference_now,
                )
                if not hits:
                    return []
                return self._neighbours_from_hits_locked(hits)

        # A full scan is slower but remains exact when no index is available.
        # Bound transient memory by the batch size plus a k-row winner heap,
        # rather than hiding old matches behind ORDER BY ... LIMIT.
        sql = (
            "SELECT id, scope, state_tokens, action_key, action_text, "
            "discounted_return, reward, state_text, created_at "
            "FROM experiences WHERE 1 = 1"
        )
        params: list[Any] = []
        if surface_gate:
            sql += " AND surface = ?"
            params.append(str(query.surface or ""))
        if mode == "session" and query_scope:
            sql += " AND scope = ?"
            params.append(query_scope)
        scored: list[tuple[float, float, int, sqlite3.Row]] = []
        with self._lock:
            conn = self._require_conn()
            cursor = conn.execute(sql, params)
            try:
                while batch := cursor.fetchmany(limit):
                    for row in batch:
                        decoded = _loads(row["state_tokens"], None)
                        if not isinstance(decoded, list | tuple):
                            continue
                        row_tokens = frozenset(str(token) for token in decoded)
                        if not row_tokens:
                            continue
                        similarity = jaccard(query.tokens, row_tokens)
                        if similarity < minimum:
                            continue
                        if (
                            mode == "hybrid"
                            and query_scope
                            and str(row["scope"] or "") != query_scope
                        ):
                            similarity *= _CROSS_SCOPE_PENALTY
                        item = (
                            similarity,
                            _as_float(row["created_at"]),
                            _as_int(row["id"]),
                            row,
                        )
                        if len(scored) < neighbours_wanted:
                            heapq.heappush(scored, item)
                        elif item[:3] > scored[0][:3]:
                            heapq.heapreplace(scored, item)
            finally:
                cursor.close()

        neighbours: list[Neighbor] = []
        for similarity, created_at, entry_id, row in sorted(scored, reverse=True):
            if half_life <= 0:
                recency = 1.0
            else:
                age_days = max(0.0, (reference_now - created_at) / _SECONDS_PER_DAY)
                recency = 0.5 ** (age_days / half_life)
            weight = similarity * (_RECENCY_FLOOR + (1.0 - _RECENCY_FLOOR) * recency)
            neighbours.append(
                Neighbor(
                    entry_id=entry_id,
                    similarity=similarity,
                    action_key=str(row["action_key"] or ""),
                    action_text=str(row["action_text"] or ""),
                    discounted_return=_as_float(row["discounted_return"]),
                    similarity_weight=weight,
                    reward=_as_float(row["reward"]),
                    state_text=str(row["state_text"] or ""),
                    scope=str(row["scope"] or ""),
                    created_at=created_at,
                )
            )
        return neighbours

    async def action_stats(
        self, *, scope: str | None = None, limit: int = 20, min_count: int = 1
    ) -> list[dict]:
        """Aggregate per-action outcomes.

        Args:
            scope: Optional scope filter.
            limit: Maximum number of action groups returned.
            min_count: Minimum number of samples for a group to be reported.

        Returns:
            One dict per action key with the keys "action_key",
            "action_name", "action_text", "count", "mean_return",
            "mean_reward" and "positive_rate".

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._action_stats_sync, scope, limit, min_count)

    def _action_stats_sync(
        self, scope: str | None, limit: int, min_count: int
    ) -> list[dict]:
        """Aggregate per-action outcomes (worker thread).

        Args:
            scope: Optional scope filter.
            limit: Maximum number of groups.
            min_count: Minimum sample count.

        Returns:
            The aggregated rows as plain dicts.
        """
        sql = (
            "SELECT action_key, MAX(action_name) AS action_name, "
            "MAX(action_text) AS action_text, COUNT(*) AS n, "
            "AVG(discounted_return) AS mean_return, AVG(reward) AS mean_reward, "
            "AVG(CASE WHEN discounted_return > 0 THEN 1.0 ELSE 0.0 END) AS positive_rate "
            "FROM experiences"
        )
        params: list[Any] = []
        if scope is not None:
            sql += " WHERE scope = ?"
            params.append(str(scope))
        sql += (
            " GROUP BY action_key HAVING COUNT(*) >= ?"
            " ORDER BY n DESC, mean_return DESC, action_key ASC LIMIT ?"
        )
        params.append(max(1, _as_int(min_count, 1)))
        params.append(max(1, _as_int(limit, 20)))
        with self._lock:
            rows = self._require_conn().execute(sql, params).fetchall()
        return [
            {
                "action_key": str(row["action_key"] or ""),
                "action_name": str(row["action_name"] or ""),
                "action_text": str(row["action_text"] or ""),
                "count": _as_int(row["n"]),
                "mean_return": _as_float(row["mean_return"]),
                "mean_reward": _as_float(row["mean_reward"]),
                "positive_rate": _as_float(row["positive_rate"]),
            }
            for row in rows
        ]

    async def learning_curve(
        self, *, scope: str | None = None, buckets: int = 10
    ) -> list[dict]:
        """Bucket stored experiences chronologically to show learning progress.

        Args:
            scope: Optional scope filter.
            buckets: Maximum number of chronological buckets.

        Returns:
            One dict per bucket with the keys "bucket" (zero-based index),
            "count", "mean_return" and "mean_reward".

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._learning_curve_sync, scope, buckets)

    def _learning_curve_sync(self, scope: str | None, buckets: int) -> list[dict]:
        """Bucket experiences by creation order (worker thread).

        Args:
            scope: Optional scope filter.
            buckets: Maximum number of buckets.

        Returns:
            The aggregated buckets as plain dicts.
        """
        sql = "SELECT discounted_return, reward FROM experiences"
        params: list[Any] = []
        if scope is not None:
            sql += " WHERE scope = ?"
            params.append(str(scope))
        sql += " ORDER BY created_at ASC, id ASC"
        with self._lock:
            rows = self._require_conn().execute(sql, params).fetchall()
        total = len(rows)
        if not total:
            return []
        bucket_count = max(1, min(_as_int(buckets, 10), total))
        curve: list[dict] = []
        for index in range(bucket_count):
            chunk = rows[
                index * total // bucket_count : (index + 1) * total // bucket_count
            ]
            if not chunk:
                continue
            returns = [_as_float(row["discounted_return"]) for row in chunk]
            rewards = [_as_float(row["reward"]) for row in chunk]
            curve.append(
                {
                    "bucket": index,
                    "count": len(chunk),
                    "mean_return": sum(returns) / len(returns),
                    "mean_reward": sum(rewards) / len(rewards),
                }
            )
        return curve

    async def recent_episodes(
        self, *, scope: str | None = None, limit: int = 10
    ) -> list[dict]:
        """List recently stored episode headers.

        Args:
            scope: Optional scope filter.
            limit: Maximum number of episodes.

        Returns:
            One dict per episode with the columns of the episodes table, a
            decoded "meta" mapping, a boolean-or-None "success" flag and a
            "step_count" field.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._recent_episodes_sync, scope, limit)

    def _recent_episodes_sync(self, scope: str | None, limit: int) -> list[dict]:
        """Read the newest episode headers (worker thread).

        Args:
            scope: Optional scope filter.
            limit: Maximum number of episodes.

        Returns:
            The episode headers as plain dicts.
        """
        sql = (
            "SELECT episode_id, scope, task_family, user_text, success, "
            "created_at, gamma, reward_source, meta FROM episodes"
        )
        params: list[Any] = []
        if scope is not None:
            sql += " WHERE scope = ?"
            params.append(str(scope))
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(max(1, _as_int(limit, 10)))
        with self._lock:
            conn = self._require_conn()
            rows = conn.execute(sql, params).fetchall()
            episodes: list[dict] = []
            for row in rows:
                episode_id = str(row["episode_id"] or "")
                counts = conn.execute(
                    "SELECT COUNT(*) FROM experiences WHERE episode_id = ?",
                    (episode_id,),
                ).fetchone()
                meta = _loads(row["meta"], {})
                success = row["success"]
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "scope": str(row["scope"] or ""),
                        "task_family": str(row["task_family"] or ""),
                        "user_text": str(row["user_text"] or ""),
                        "success": None if success is None else bool(success),
                        "created_at": _as_float(row["created_at"]),
                        "gamma": _as_float(row["gamma"], _DEFAULT_GAMMA),
                        "reward_source": str(row["reward_source"] or ""),
                        "meta": meta if isinstance(meta, dict) else {},
                        "step_count": _as_int(counts[0]) if counts is not None else 0,
                    }
                )
        return episodes

    async def export_json(
        self, *, scope: str | None = None, limit: int | None = 100000
    ) -> dict:
        """Export stored memory as a JSON-safe mapping.

        Args:
            scope: Optional scope filter.
            limit: Maximum number of rows exported per table; None exports
                every row. Migration uses None so a large archive can never be
                silently truncated before it is marked as merged.

        Returns:
            A mapping with "version", "exported_at", "episodes" and
            "experiences" keys, ready for json.dump or import_json.

        Raises:
            RuntimeError: When the bank is not initialized.
        """
        return await asyncio.to_thread(self._export_sync, scope, limit)

    def _export_sync(self, scope: str | None, limit: int | None) -> dict:
        """Read both tables for export (worker thread).

        Args:
            scope: Optional scope filter.
            limit: Maximum number of rows per table; None means unlimited.

        Returns:
            The export payload.
        """
        max_rows = None if limit is None else max(1, _as_int(limit, 100000))
        where = ""
        params: list[Any] = []
        if scope is not None:
            where = " WHERE scope = ?"
            params.append(str(scope))
        row_limit = "" if max_rows is None else " LIMIT ?"
        row_params: list[Any] = [] if max_rows is None else [max_rows]
        with self._lock:
            conn = self._require_conn()
            episode_rows = conn.execute(
                f"SELECT {_EPISODE_COLUMNS} FROM episodes{where}"
                f" ORDER BY created_at ASC, id ASC{row_limit}",
                [*params, *row_params],
            ).fetchall()
            experience_rows = conn.execute(
                f"{_SELECT_EXPERIENCE_SQL}{where} ORDER BY created_at ASC, id ASC"
                f"{row_limit}",
                [*params, *row_params],
            ).fetchall()

        episodes: list[dict] = []
        for row in episode_rows:
            meta = _loads(row["meta"], {})
            success = row["success"]
            episodes.append(
                {
                    "episode_id": str(row["episode_id"] or ""),
                    "scope": str(row["scope"] or ""),
                    "task_family": str(row["task_family"] or ""),
                    "user_text": str(row["user_text"] or ""),
                    "success": None if success is None else bool(success),
                    "created_at": _as_float(row["created_at"]),
                    "gamma": _as_float(row["gamma"], _DEFAULT_GAMMA),
                    "reward_source": str(row["reward_source"] or ""),
                    "meta": meta if isinstance(meta, dict) else {},
                }
            )
        experiences: list[dict] = []
        for row in experience_rows:
            tokens = _loads(row["state_tokens"], [])
            args = _loads(row["action_args"], {})
            experiences.append(
                {
                    "entry_id": _as_int(row["entry_id"]),
                    "episode_id": str(row["episode_id"] or ""),
                    "scope": str(row["scope"] or ""),
                    "step_index": _as_int(row["step_index"]),
                    "surface": str(row["surface"] or ""),
                    "state_signature": str(row["state_signature"] or ""),
                    "state_tokens": tokens if isinstance(tokens, list) else [],
                    "state_text": str(row["state_text"] or ""),
                    "action_key": str(row["action_key"] or ""),
                    "action_name": str(row["action_name"] or ""),
                    "action_kind": str(row["action_kind"] or ""),
                    "action_text": str(row["action_text"] or ""),
                    "action_args": args if isinstance(args, dict) else {},
                    "reward": _as_float(row["reward"]),
                    "discounted_return": _as_float(row["discounted_return"]),
                    "outcome": str(row["outcome"] or ""),
                    "reward_source": str(row["reward_source"] or ""),
                    "created_at": _as_float(row["created_at"]),
                }
            )
        return {
            "version": _EXPORT_VERSION,
            "exported_at": time.time(),
            "episodes": episodes,
            "experiences": experiences,
        }
