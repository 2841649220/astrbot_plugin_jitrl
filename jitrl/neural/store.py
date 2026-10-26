"""Independent SQLite facts, append-only rewards and exact decision snapshots.

All methods are synchronous and serialized. The async runtime calls them on
worker threads. V0.1.x memory.sqlite3 is never migrated or read for training.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .config import LensConfig
from .contracts import (
    VERSIONS,
    Intervention,
    LensDecision,
    MemoryRecord,
    ReplaySample,
    RetrievalSnapshot,
    StateRecord,
)
from .encoding import digest, state_embedding

SOURCE_RELIABILITY = {
    "environment": 1.0,
    "tool": 0.8,
    "user": 0.9,
    "llm": 0.5,
    "fused": 0.4,
    "heuristic": 0.25,
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def validate_snapshot(state: StateRecord, snapshot: RetrievalSnapshot) -> None:
    if (
        not state.scope
        or not math.isfinite(state.observed_at)
        or not math.isfinite(snapshot.cutoff)
        or state.versions != VERSIONS
        or snapshot.versions != VERSIONS
        or snapshot.scope != state.scope
        or snapshot.cutoff > state.observed_at
    ):
        raise ValueError("incompatible or foreign retrieval snapshot")
    seen: set[str] = set()
    for memory in snapshot.memories:
        if (
            memory.state.scope != state.scope
            or memory.state.environment != state.environment
            or memory.state.versions != state.versions
            or memory.state.episode_id == state.episode_id
            or memory.state.decision_id == state.decision_id
            or memory.state.observed_at >= state.observed_at
            or memory.state.observed_at > memory.settled_at
            or memory.settled_at >= snapshot.cutoff
            or not 0 <= memory.reliability <= 1
            or not 0 <= memory.similarity <= 1
            or memory.reward_source not in SOURCE_RELIABILITY
            or memory.experience_id in seen
            or not all(
                math.isfinite(x)
                for x in (
                    memory.return_g,
                    memory.reliability,
                    memory.similarity,
                    memory.settled_at,
                )
            )
        ):
            raise ValueError("noncausal or duplicate memory in snapshot")
        seen.add(memory.experience_id)


class LensStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS store_meta(id INTEGER PRIMARY KEY CHECK(id=1), global_epoch INTEGER NOT NULL);
            INSERT OR IGNORE INTO store_meta VALUES(1,0);
            CREATE TABLE IF NOT EXISTS scope_epochs(scope TEXT PRIMARY KEY, epoch INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS decisions(
                decision_id TEXT PRIMARY KEY, scope TEXT NOT NULL, episode_id TEXT NOT NULL,
                created_at REAL NOT NULL, epoch INTEGER NOT NULL, state_json TEXT NOT NULL,
                snapshot_json TEXT NOT NULL, intervention_json TEXT NOT NULL,
                rendered TEXT NOT NULL, checkpoint_id TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS lens_decision_scope ON decisions(scope,created_at);
            CREATE TABLE IF NOT EXISTS executions(
                experience_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL REFERENCES decisions(decision_id) ON DELETE CASCADE,
                execution_index INTEGER NOT NULL, action_key TEXT NOT NULL, action_name TEXT NOT NULL,
                executed_at REAL NOT NULL, UNIQUE(decision_id,execution_index));
            CREATE TABLE IF NOT EXISTS settlements(
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE NOT NULL,
                experience_id TEXT NOT NULL REFERENCES executions(experience_id) ON DELETE CASCADE,
                reward REAL NOT NULL, return_g REAL NOT NULL, source TEXT NOT NULL,
                reliability REAL NOT NULL, settled_at REAL NOT NULL, valid INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS lens_settlement_experience ON settlements(experience_id,sequence);
            CREATE TABLE IF NOT EXISTS checkpoints(
                checkpoint_id TEXT PRIMARY KEY, scope TEXT NOT NULL, epoch INTEGER NOT NULL,
                created_at REAL NOT NULL, filename TEXT NOT NULL, sha256 TEXT NOT NULL,
                metrics_json TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS lens_checkpoint_scope ON checkpoints(scope,created_at);
            CREATE TABLE IF NOT EXISTS post_events(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT NOT NULL REFERENCES decisions(decision_id) ON DELETE CASCADE,
                created_at REAL NOT NULL, payload TEXT NOT NULL);
        """)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def epoch(self, scope: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT epoch FROM scope_epochs WHERE scope=?", (scope,)
            ).fetchone()
            global_epoch = self._conn.execute(
                "SELECT global_epoch FROM store_meta WHERE id=1"
            ).fetchone()[0]
            return (int(global_epoch) << 32) + (int(row[0]) if row else 0)

    def record_decision(self, decision: LensDecision, *, expected_epoch: int) -> bool:
        validate_snapshot(decision.state, decision.snapshot)
        state = decision.state
        payload = (
            state.decision_id,
            state.scope,
            state.episode_id,
            state.observed_at,
            expected_epoch,
            _json(state.as_dict()),
            _json(decision.snapshot.as_dict()),
            _json(asdict(decision.intervention)),
            decision.rendered,
            decision.checkpoint_id,
        )
        with self._lock, self._conn:
            if self.epoch(state.scope) != expected_epoch:
                return False
            old = self._conn.execute(
                "SELECT * FROM decisions WHERE decision_id=?", (state.decision_id,)
            ).fetchone()
            if old is not None:
                if tuple(old) != payload:
                    raise ValueError("decision_id already binds a different decision")
                return True
            self._conn.execute(
                "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?)", payload
            )
            return True

    def append_execution(
        self,
        decision_id: str,
        *,
        index: int,
        action_key: str,
        action_name: str,
        expected_epoch: int,
        executed_at: float | None = None,
    ) -> str | None:
        timestamp = time.time() if executed_at is None else executed_at
        if (
            not math.isfinite(timestamp)
            or index < 0
            or not action_key
            or not action_name
        ):
            raise ValueError("invalid executed action")
        experience_id = digest((decision_id, index))
        with self._lock, self._conn:
            decision = self._conn.execute(
                "SELECT scope,epoch,created_at FROM decisions WHERE decision_id=?",
                (decision_id,),
            ).fetchone()
            if (
                not decision
                or decision["epoch"] != expected_epoch
                or self.epoch(decision["scope"]) != expected_epoch
            ):
                return None
            if timestamp < decision["created_at"]:
                raise ValueError("execution precedes decision")
            old = self._conn.execute(
                "SELECT action_key,action_name FROM executions WHERE experience_id=?",
                (experience_id,),
            ).fetchone()
            if old:
                if tuple(old) != (action_key, action_name):
                    raise ValueError("an executed action is immutable")
                return experience_id
            self._conn.execute(
                "INSERT INTO executions VALUES (?,?,?,?,?,?)",
                (experience_id, decision_id, index, action_key, action_name, timestamp),
            )
        return experience_id

    def settle(
        self,
        experience_id: str,
        *,
        reward: float,
        return_g: float,
        source: str,
        expected_epoch: int,
        settled_at: float | None = None,
        valid: bool = True,
        event_id: str | None = None,
    ) -> bool:
        timestamp = time.time() if settled_at is None else settled_at
        if source not in SOURCE_RELIABILITY or not all(
            math.isfinite(v) for v in (reward, return_g, timestamp)
        ):
            raise ValueError("finite rewards with known provenance are required")
        reward, return_g = float(reward), float(return_g)
        event_id = event_id or digest((experience_id, reward, return_g, source, valid))
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT d.scope,d.epoch,x.executed_at FROM executions x JOIN decisions d USING(decision_id) WHERE x.experience_id=?",
                (experience_id,),
            ).fetchone()
            if (
                not row
                or row["epoch"] != expected_epoch
                or self.epoch(row["scope"]) != expected_epoch
            ):
                return False
            if timestamp < row["executed_at"]:
                raise ValueError("reward precedes execution")
            old = self._conn.execute(
                "SELECT experience_id,reward,return_g,source,valid FROM settlements WHERE event_id=?",
                (event_id,),
            ).fetchone()
            if old:
                if tuple(old) != (experience_id, reward, return_g, source, int(valid)):
                    raise ValueError("settlement event_id binds a different payload")
                return False
            latest = self._conn.execute(
                "SELECT settled_at FROM settlements WHERE experience_id=? ORDER BY sequence DESC LIMIT 1",
                (experience_id,),
            ).fetchone()
            if latest and timestamp < latest[0]:
                raise ValueError("settlement time must be monotonic")
            self._conn.execute(
                "INSERT INTO settlements(event_id,experience_id,reward,return_g,source,reliability,settled_at,valid) VALUES(?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    experience_id,
                    reward,
                    return_g,
                    source,
                    SOURCE_RELIABILITY[source],
                    timestamp,
                    int(valid),
                ),
            )
        return True

    def _settled_rows(self, scope: str, cutoff: float, limit: int) -> list[sqlite3.Row]:
        return self._conn.execute(
            """
            SELECT x.experience_id,x.action_key,x.action_name,d.*,s.return_g,s.source,s.reliability,s.settled_at
            FROM executions x JOIN decisions d USING(decision_id)
            JOIN settlements s ON s.experience_id=x.experience_id
            WHERE d.scope=? AND d.epoch=? AND d.created_at<? AND x.executed_at<?
            AND s.settled_at<? AND s.valid=1 AND s.sequence=(
                SELECT MAX(s2.sequence) FROM settlements s2
                WHERE s2.experience_id=x.experience_id AND s2.settled_at<?)
            ORDER BY s.settled_at DESC,x.experience_id LIMIT ?
        """,
            (scope, self.epoch(scope), cutoff, cutoff, cutoff, cutoff, limit),
        ).fetchall()

    def retrieve(self, state: StateRecord, cfg: LensConfig) -> RetrievalSnapshot:
        """Exact bounded lexical/structural coarse search; no ANN claim."""
        query = state_embedding(state, cfg.embedding_dim)
        with self._lock:
            rows = self._settled_rows(state.scope, state.observed_at, cfg.replay_limit)
        candidates: list[tuple[float, float, MemoryRecord]] = []
        for row in rows:
            past = StateRecord.from_dict(json.loads(row["state_json"]))
            if (
                past.environment != state.environment
                or past.versions != state.versions
                or past.episode_id == state.episode_id
                or past.decision_id == state.decision_id
            ):
                continue
            vector = state_embedding(past, cfg.embedding_dim)
            lexical = max(
                0.0, min(1.0, sum(a * b for a, b in zip(query, vector, strict=True)))
            )
            a, b = set(state.events), set(past.events)
            structural = len(a & b) / max(1, len(a | b))
            similarity = 0.8 * lexical + 0.2 * structural
            if similarity < cfg.similarity_threshold:
                continue
            memory = MemoryRecord(
                row["experience_id"],
                past,
                row["action_key"],
                row["action_name"],
                json.loads(row["intervention_json"])["kind"],
                row["return_g"],
                row["source"],
                row["reliability"],
                row["settled_at"],
                similarity,
            )
            candidates.append((lexical, structural, memory))
        half = max(1, cfg.coarse_limit // 2)
        union = {
            m.experience_id: m
            for _, _, m in sorted(
                candidates, key=lambda x: (-x[0], x[2].experience_id)
            )[:half]
        }
        union.update(
            {
                m.experience_id: m
                for _, _, m in sorted(
                    candidates, key=lambda x: (-x[1], x[2].experience_id)
                )[:half]
            }
        )
        memories = tuple(
            sorted(union.values(), key=lambda m: (-m.similarity, m.experience_id))[
                : cfg.coarse_limit
            ]
        )
        snapshot = RetrievalSnapshot(
            digest((state.decision_id, [m.as_dict() for m in memories])),
            state.scope,
            state.observed_at,
            memories,
        )
        validate_snapshot(state, snapshot)
        return snapshot

    def replay(
        self, scope: str, limit: int = 2048, cutoff: float | None = None
    ) -> list[ReplaySample]:
        with self._lock:
            rows = self._settled_rows(
                scope, time.time() if cutoff is None else cutoff, limit
            )
        result = []
        for row in reversed(rows):
            state = StateRecord.from_dict(json.loads(row["state_json"]))
            snapshot = RetrievalSnapshot.from_dict(json.loads(row["snapshot_json"]))
            validate_snapshot(state, snapshot)
            pre = json.loads(row["intervention_json"])
            intervention = Intervention(
                pre["kind"], tuple(pre["memory_ids"]), pre["reason"]
            )
            decision = LensDecision(
                state, snapshot, intervention, row["rendered"], row["checkpoint_id"]
            )
            result.append(
                ReplaySample(
                    row["experience_id"],
                    decision,
                    row["action_key"],
                    row["return_g"],
                    row["reliability"],
                    row["settled_at"],
                )
            )
        return result

    def record_post(self, decision_id: str, payload: dict[str, Any]) -> bool:
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT scope,epoch FROM decisions WHERE decision_id=?", (decision_id,)
            ).fetchone()
            if not row or row["epoch"] != self.epoch(row["scope"]):
                return False
            self._conn.execute(
                "INSERT INTO post_events(decision_id,created_at,payload) VALUES(?,?,?)",
                (decision_id, time.time(), _json(payload)),
            )
        return True

    def decision_matches(self, decision: LensDecision) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM decisions WHERE decision_id=?",
                (decision.state.decision_id,),
            ).fetchone()
            return bool(
                row
                and row["epoch"] == self.epoch(decision.state.scope)
                and row["scope"] == decision.state.scope
                and row["state_json"] == _json(decision.state.as_dict())
                and row["snapshot_json"] == _json(decision.snapshot.as_dict())
                and row["intervention_json"] == _json(asdict(decision.intervention))
                and row["rendered"] == decision.rendered
                and row["checkpoint_id"] == decision.checkpoint_id
            )

    def apply_user_feedback(
        self,
        *,
        scope: str,
        episode_id: str,
        delta: float,
        gamma: float,
        expected_epoch: int,
    ) -> int:
        """Append the legacy episode-feedback credit rule, retaining history.

        A whole-episode verdict becomes an increment to each observed reward;
        this is heuristic credit assignment, not environment ground truth.
        Pending/incomplete episodes do not acquire manufactured step labels.
        """
        if not math.isfinite(delta) or not math.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("invalid feedback parameters")
        with self._lock, self._conn:
            if self.epoch(scope) != expected_epoch:
                return 0
            rows = self._conn.execute(
                """
                SELECT x.experience_id,s.reward,s.valid FROM executions x
                JOIN decisions d USING(decision_id)
                LEFT JOIN settlements s ON s.sequence=(SELECT MAX(s2.sequence) FROM settlements s2 WHERE s2.experience_id=x.experience_id)
                WHERE d.scope=? AND d.episode_id=? AND d.epoch=?
                ORDER BY x.execution_index,x.experience_id
            """,
                (scope, episode_id, expected_epoch),
            ).fetchall()
            if not rows or any(
                row["reward"] is None or row["valid"] != 1 for row in rows
            ):
                return 0
            feedback_id = digest(("episode-feedback", scope, episode_id, float(delta)))
            events = []
            running = 0.0
            timestamp = time.time()
            for row in reversed(rows):
                event_id = digest((feedback_id, row["experience_id"]))
                if self._conn.execute(
                    "SELECT 1 FROM settlements WHERE event_id=?", (event_id,)
                ).fetchone():
                    return 0
                reward = max(-3.0, min(3.0, row["reward"] + delta))
                running = reward + gamma * running
                events.append(
                    (
                        event_id,
                        row["experience_id"],
                        reward,
                        running,
                        "user",
                        SOURCE_RELIABILITY["user"],
                        timestamp,
                        1,
                    )
                )
            self._conn.executemany(
                "INSERT INTO settlements(event_id,experience_id,reward,return_g,source,reliability,settled_at,valid) VALUES(?,?,?,?,?,?,?,?)",
                events,
            )
            return len(events)

    def publish_checkpoint(
        self,
        *,
        scope: str,
        expected_epoch: int,
        checkpoint_id: str,
        filename: str,
        sha256: str,
        metrics: dict[str, Any],
    ) -> bool:
        with self._lock, self._conn:
            if self.epoch(scope) != expected_epoch:
                return False
            self._conn.execute(
                "INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?)",
                (
                    checkpoint_id,
                    scope,
                    expected_epoch,
                    time.time(),
                    filename,
                    sha256,
                    _json(metrics),
                ),
            )
        return True

    def latest_checkpoint(self, scope: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE scope=? AND epoch=? ORDER BY created_at DESC LIMIT 1",
                (scope, self.epoch(scope)),
            ).fetchone()
        return dict(row) if row else None

    def checkpoint(self, scope: str, checkpoint_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE scope=? AND epoch=? AND checkpoint_id=?",
                (scope, self.epoch(scope), checkpoint_id),
            ).fetchone()
        return dict(row) if row else None

    def reset(self, scope: str | None = None) -> list[str]:
        """Invalidate in-flight writers and remove facts and model references."""
        with self._lock, self._conn:
            if scope is None:
                self._conn.execute(
                    "UPDATE store_meta SET global_epoch=global_epoch+1 WHERE id=1"
                )
            scopes = (
                [scope]
                if scope
                else [
                    r[0]
                    for r in self._conn.execute(
                        "SELECT scope FROM decisions UNION SELECT scope FROM scope_epochs UNION SELECT scope FROM checkpoints"
                    )
                ]
            )
            filenames = []
            for item in scopes:
                filenames.extend(
                    r[0]
                    for r in self._conn.execute(
                        "SELECT filename FROM checkpoints WHERE scope=?", (item,)
                    )
                )
                self._conn.execute(
                    "INSERT INTO scope_epochs VALUES(?,1) ON CONFLICT(scope) DO UPDATE SET epoch=epoch+1",
                    (item,),
                )
                self._conn.execute("DELETE FROM decisions WHERE scope=?", (item,))
                self._conn.execute("DELETE FROM checkpoints WHERE scope=?", (item,))
        return filenames

    def stats(self, scope: str | None = None) -> dict[str, int]:
        with self._lock:
            where, args = (" WHERE d.scope=?", (scope,)) if scope else ("", ())
            result = {}
            for name, sql in (
                ("decisions", "SELECT COUNT(*) FROM decisions d"),
                (
                    "executions",
                    "SELECT COUNT(*) FROM executions x JOIN decisions d USING(decision_id)",
                ),
                (
                    "settlements",
                    "SELECT COUNT(*) FROM settlements s JOIN executions x USING(experience_id) JOIN decisions d USING(decision_id)",
                ),
            ):
                result[name] = self._conn.execute(sql + where, args).fetchone()[0]
            return result


def retained_snapshot(
    snapshot: RetrievalSnapshot, indices: list[int]
) -> RetrievalSnapshot:
    memories = tuple(snapshot.memories[i] for i in indices)
    return replace(
        snapshot,
        snapshot_id=digest((snapshot.snapshot_id, [m.experience_id for m in memories])),
        memories=memories,
    )
