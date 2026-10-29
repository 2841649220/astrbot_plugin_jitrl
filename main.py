"""JitRL legacy policy and an optional, locally trained Neural Lens.

V0.2.1 (re-released 2026-10-27) isolates both the legacy learner and the optional
neural controller in jitrl.neural by AstrBot platform instance. The adapter-provided self_id is
recorded as metadata only: several adapters (notably qq_official) derive it
from the message itself, so the legacy 0.1.3 and original 0.2.1
platform_id:self_id key created a new robot for every message shape. The remote LLM remains frozen; only the local
optional PyTorch model is trained. The legacy path described below is selected
when neural.mode is off. All four local token-attention blocks use genuine MiTA
by default.

This plugin implements *Just-In-Time Reinforcement Learning* (Li et al.,
ICML 2026) on top of AstrBot's agent pipeline. The paper keeps the model
frozen and instead performs policy optimisation at inference time:

1. after every episode an LLM evaluator turns the trajectory into step-wise
   rewards `r_t`, which are folded into discounted returns
   `G_t = sum_{u>=t} gamma^(u-t) r_u`;
2. every `(s_t, a_t, G_t)` triplet is appended to a non-parametric memory;
3. before the next request the plugin retrieves the top-k neighbours of the
   current abstracted state with Jaccard similarity, estimates `V(s)` and
   `Q(s, a)` from the stored returns, and derives the advantage
   `A(s, a) = Q(s, a) - V(s)`;
4. the advantage is applied through the closed-form solution of the
   KL-constrained objective, `z'(s, a) = z(s, a) + beta * A(s, a)`.

Because AstrBot talks to black-box chat providers, the logit vector is
obtained with the paper's *Verbalized Logit* variant (Section 5.1): the model
is asked for a 0-100 confidence over the candidate set, that confidence is
turned into `z(s, a)`, and the updated distribution is re-sampled. The
plugin never trains weights and never needs a GPU.

Hook wiring:

* `on_llm_request`  -> retrieve, estimate, decide, and bias the request;
* `on_agent_done`   -> evaluate the finished trajectory and store it;
* `/jitrl ...`      -> inspect, prune and steer the learned memory;
* `jitrl_recall`    -> let the agent query its own memory mid-run.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import random
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.utils.astrbot_path import get_astrbot_plugin_data_path

from .jitrl.config import JitRLConfig
from .jitrl.estimator import MIN_EVIDENCE_SIGNAL, estimate_values
from .jitrl.evaluator import StepRewardEvaluator, discounted_returns
from .jitrl.hint import render_preference_block, render_recall_report
from .jitrl.memory import MemoryBank
from .jitrl.models import Neighbor, PolicyDecision, StateView, ValueEstimate
from .jitrl.neural.adapter import (
    NeuralRun,
    RequestFrame,
    executed_transitions,
    observed_context,
    provider_messages,
)
from .jitrl.neural.contracts import Intervention
from .jitrl.neural.encoding import canonicalize
from .jitrl.policy import build_candidates, select_action, verbalized_to_logit
from .jitrl.reranker import VerbalizedReranker
from .jitrl.state import abstract_state, redact
from .jitrl.trajectory import build_episode, extract_transitions, infer_success

_PLUGIN_DATA_DIR = "astrbot_plugin_jitrl"
"""Subdirectory of AstrBot's plugin data root owned by this plugin."""

_ROBOT_ARCHIVE_DIR = ".legacy"
"""Subdirectory of ``robots/`` holding superseded per-``self_id`` data.

A dot-prefixed directory is never listed as a robot and never reused as a
robot directory, so archived data stays available for inspection while the
dashboard only shows the consolidated platform-instance robots.
"""

_EVENT_EXTRA_KEY = "_jitrl_context"
"""Key under which the decision context travels from request to agent done."""

_INJECTED_KEY = "_jitrl_injected"
"""Guards against injecting the evidence block twice for one event."""

_GENERATION_KEY = "_jitrl_memory_generation"
"""Memory epoch in which an agent request started."""

_NEURAL_KEY = "_jitrl_neural_run"

_STATE_CHAR_LIMIT = 800
"""Character budget of the abstracted state used for retrieval."""

_TASK_CHAR_LIMIT = 600
"""Character budget of the redacted user request kept in memory."""

_LAST_EPISODE_TTL = 1800.0
"""Seconds before a cached `(episode_id, timestamp)` pair is dropped."""

_FEEDBACK_DELTA = 2.0
"""Reward delta applied when the user confirms or rejects the last reply."""

_STATE_CACHE_MAX = 64
"""Abstracted states memoised per process.

An agent run issues several LLM requests with the same user message, so the
redaction/regularisation/tokenisation pipeline would otherwise run again on
every step of the loop.
"""

_DECISION_CACHE_MAX = 512
"""Policy decisions memoised per process.

Retrieval, estimation and hint rendering only depend on (state, scope, memory
epoch, tool set), so a multi-step agent run reuses one decision instead of
recomputing it per request -- and, more importantly, sees stable advice instead
of the sampled exploration branch changing its mind mid-run. The cache is
dropped whenever memory changes.
"""

_EXPORT_KEEP = 5
"""Number of JSON export files kept on disk."""


def _make_temporary_part(text: str) -> Any | None:
    """Wrap the evidence block into a provider-only content part.

    TextPart lives outside the public astrbot.api surface, so it is imported
    lazily: if a future host moves it, the plugin must keep loading and simply
    stop injecting instead of failing at import time.

    Args:
        text: Rendered evidence block.

    Returns:
        A temporary content part, or None when the host API is unavailable.
    """
    try:
        from astrbot.core.agent.message import TextPart

        return TextPart(text=text).mark_as_temp()
    except Exception:
        logger.warning(
            "[JitRL] host TextPart is unavailable; skipping the evidence block"
        )
        return None


def _event_text(event: Any, accessor: str) -> str:
    """Read a string-returning AstrBot event accessor defensively.

    Older host builds and third-party adapters do not all implement every
    accessor, and an adapter may return None. Identity building must never
    raise from inside a message hook: a failure there would break the
    conversation instead of merely mis-grouping learning data.

    Args:
        event: AstrBot message event.
        accessor: Name of the accessor method, for example get_platform_id.

    Returns:
        The stripped string value, or an empty string.
    """
    method = getattr(event, accessor, None)
    if method is None:
        return ""
    try:
        value = method()
    except Exception:
        return ""
    if value is None:
        return ""
    return str(value).strip()


@dataclass(slots=True)
class _DecisionSnapshot:
    """Everything on_llm_request needs for one (state, scope, tools) triple.

    Attributes:
        decision: Sampled policy decision.
        estimate: Value estimate over the final candidate set.
        probe: Value estimate over every retrieved action, used by the tool gate
            and to shortlist candidates.
        neighbors: Retrieved neighbourhood, kept for the hint renderer.
        block: Rendered evidence block (empty when the evidence is too weak).
        task: Redacted user request that produced the state.
        state: Abstracted query state.
    """

    decision: PolicyDecision
    estimate: ValueEstimate
    probe: ValueEstimate
    neighbors: list[Neighbor]
    block: str
    task: str
    state: StateView


class JitRLLearner(Star):
    """Legacy episodic learning and optional isolated local neural learning.

    The class owns the memory bank, the evaluator and the reranker, and wires
    them into AstrBot's event hooks. Every hook is defensive: a failure inside
    the learning layer must never break a conversation, so all entry points
    swallow and log their own exceptions.
    """

    def __init__(self, context: Context, config: dict | None = None) -> None:
        """Initialise the plugin.

        Args:
            context: AstrBot plugin context.
            config: Raw plugin configuration produced from `_conf_schema.json`.
        """
        super().__init__(context)
        self.cfg = JitRLConfig.from_raw(config)
        data_root = Path(get_astrbot_plugin_data_path()) / _PLUGIN_DATA_DIR
        self._data_root = data_root
        self._legacy_db_path = data_root / "memory.sqlite3"
        self._banks: dict[str, MemoryBank] = {}
        self._bank_paths: dict[str, Path] = {}
        self._robot_dirs: dict[str, Path] = {}
        self._active_bank: ContextVar[MemoryBank | None] = ContextVar(
            f"jitrl_bank_{id(self)}", default=None
        )
        self._active_robot_key: ContextVar[str | None] = ContextVar(
            f"jitrl_robot_key_{id(self)}", default=None
        )
        self._last_bank: MemoryBank | None = None
        self._last_db_path = self._legacy_db_path
        self._neural_by_robot: dict[str, Any] = {}
        self._neural_roots: dict[str, Path] = {}
        self._active_neural_root: ContextVar[Path | None] = ContextVar(
            f"jitrl_neural_root_{id(self)}", default=None
        )
        self._last_neural: Any | None = None
        self._legacy_neural_root = data_root / "neural"
        self._neural_ready_lock = asyncio.Lock()
        self._neural_training: set[str] = set()
        self._neural_last_episode: dict[str, tuple[str, float, float, int]] = {}
        self._ready_lock = asyncio.Lock()
        self._background: set[asyncio.Task] = set()
        self._background_coroutines: dict[asyncio.Task, Any] = {}
        self._rng_by_robot: dict[str, random.Random] = {}
        self._unscoped_rng = random.Random()
        self._last_episode: dict[str, tuple[str, float]] = {}
        self._last_decision: dict[str, tuple[float, PolicyDecision]] = {}
        self._state_cache: OrderedDict[
            tuple[str, str, str, str], tuple[str, StateView]
        ] = OrderedDict()
        self._decision_cache: OrderedDict[tuple[Any, ...], _DecisionSnapshot] = (
            OrderedDict()
        )
        self._stats_by_robot: dict[str, dict[str, int]] = {}
        self._unscoped_stats: dict[str, int] = {
            "requests": 0,
            "biased": 0,
            "episodes": 0,
            "experiences": 0,
            "failures": 0,
        }
        self._layout_ready = False
        self._layout_lock = threading.Lock()
        self._ensure_layout()
        self._register_web_apis()

    @property
    def _bank(self) -> MemoryBank | None:
        robot_key = self._active_robot_key.get()
        return self._banks.get(robot_key) if robot_key is not None else self._last_bank

    @property
    def _db_path(self) -> Path:
        robot_key = self._active_robot_key.get()
        if robot_key is not None:
            return self._bank_paths.get(
                robot_key, self._robot_dir(robot_key) / "memory.sqlite3"
            )
        return self._last_db_path

    @property
    def _neural(self) -> Any | None:
        robot_key = self._active_robot_key.get()
        return (
            self._neural_by_robot.get(robot_key)
            if robot_key is not None
            else self._last_neural
        )

    @property
    def _neural_root(self) -> Path:
        return self._active_neural_root.get() or self._legacy_neural_root

    @property
    def _stats(self) -> dict[str, int]:
        robot_key = self._active_robot_key.get()
        if robot_key is None:
            return self._unscoped_stats
        if robot_key not in self._stats_by_robot:
            self._stats_by_robot[robot_key] = {
                "requests": 0,
                "biased": 0,
                "episodes": 0,
                "experiences": 0,
                "failures": 0,
            }
        return self._stats_by_robot[robot_key]

    @property
    def _rng(self) -> random.Random:
        robot_key = self._active_robot_key.get()
        if robot_key is None:
            return self._unscoped_rng
        if robot_key not in self._rng_by_robot:
            self._rng_by_robot[robot_key] = random.Random()
        return self._rng_by_robot[robot_key]

    @staticmethod
    def _robot_key(identity: str) -> str:
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    @staticmethod
    def _robot_identity(event: AstrMessageEvent) -> str:
        """Return the stable isolation identity for the bot behind an event.

        AstrBot's platform-instance ID is unique per configured adapter and is
        stored in the platform configuration, so it identifies one bot across
        restarts. The self ID is deliberately not part of the key: several
        adapters derive it from the message itself (qq_official group messages
        use the first bot mention, C2C messages use the literal unknown_selfid,
        and guild messages use the first mention), so including it created a new
        robot directory for every message shape. The self ID is still persisted
        as informational dashboard metadata.

        Args:
            event: AstrBot message event.

        Returns:
            A stable identity string for hashing into the robot directory name.
        """
        platform_id = _event_text(event, "get_platform_id")
        if platform_id:
            return f"platform:{platform_id}"
        # Degenerate fallback for a third-party adapter that exposes no
        # platform-instance ID. Use the adapter type alone rather than self_id:
        # self_id is exactly the value that flaps per message on some adapters,
        # so including it here would reintroduce the bug at lower volume.
        platform_name = _event_text(event, "get_platform_name")
        return f"adapter:{platform_name or 'unknown'}"

    def _robot_key_for_event(self, event: AstrMessageEvent) -> str:
        return self._robot_key(self._robot_identity(event))

    def _cache_scope(self, event: AstrMessageEvent) -> str:
        return f"{self._robot_key_for_event(event)}:{event.unified_msg_origin}"

    def _robot_dir(self, robot_key: str) -> Path:
        return self._data_root / "robots" / robot_key

    def _robot_metadata(self, event: AstrMessageEvent) -> dict[str, str]:
        """Build the robot.json payload for one event.

        The identity is derived from the platform instance only; the self ID
        and the adapter type are recorded for operators and the dashboard.
        Adapters that return an empty self ID keep the last known value (see
        _with_previous_metadata) so the displayed label does not flap.

        Args:
            event: AstrBot message event.

        Returns:
            Identity fields shared by robot.json, logs and the WebUI.
        """
        platform_id = _event_text(event, "get_platform_id")
        platform_name = _event_text(event, "get_platform_name")
        self_id = _event_text(event, "get_self_id")
        if platform_id:
            label = f"{platform_name} · {platform_id}" if platform_name else platform_id
        else:
            label = f"{platform_name or 'AstrBot'} · {self_id or 'unknown'}"
        return {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "self_id": self_id,
            "identity": self._robot_identity(event),
            "label": label,
        }

    @staticmethod
    def _read_robot_metadata(robot_dir: Path) -> dict[str, str] | None:
        """Read robot.json without ever raising on corrupt or missing data.

        Args:
            robot_dir: Candidate robot directory.

        Returns:
            The normalized metadata mapping, or None when the file is absent,
            unreadable, not JSON or not a mapping.
        """
        try:
            data = json.loads((robot_dir / "robot.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return {
            str(key): "" if value is None else str(value)
            for key, value in data.items()
            if isinstance(value, str | int | float | bool) or value is None
        }

    @staticmethod
    def _write_robot_metadata(robot_dir: Path, metadata: dict[str, str]) -> None:
        """Atomically replace robot.json so a crash cannot corrupt it.

        Args:
            robot_dir: Robot directory to write into.
            metadata: Normalized identity fields.
        """
        robot_dir.mkdir(parents=True, exist_ok=True)
        if JitRLLearner._read_robot_metadata(robot_dir) == metadata:
            return
        # A per-write name matters: several chat sessions can activate the
        # same robot concurrently, and a fixed temporary path would make one
        # writer rename the other writer's file out from under it.
        temporary = robot_dir / f".robot.json.{uuid.uuid4().hex}.tmp"
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(robot_dir / "robot.json")

    @staticmethod
    def _canonical_metadata(
        metadata: dict[str, str], platform_id: str
    ) -> dict[str, str]:
        """Normalize legacy metadata into the platform-instance format.

        Args:
            metadata: Metadata read from a legacy or canonical directory.
            platform_id: Non-empty AstrBot platform-instance ID.

        Returns:
            Metadata carrying the canonical identity and a stable label.
        """
        platform_name = metadata.get("platform_name", "").strip()
        self_id = metadata.get("self_id", "").strip()
        if platform_id:
            identity = f"platform:{platform_id}"
            label = f"{platform_name} · {platform_id}" if platform_name else platform_id
        else:
            platform_name = platform_name or "unknown"
            identity = f"adapter:{platform_name}"
            label = f"{platform_name} · {self_id or 'unknown'}"
        return {
            "platform_id": platform_id,
            "platform_name": platform_name,
            "self_id": self_id,
            "identity": identity,
            "label": label,
        }

    @staticmethod
    def _canonical_identity(metadata: dict[str, str]) -> str | None:
        """Return the canonical identity that a metadata file maps to.

        Args:
            metadata: Metadata read from a robot directory.

        Returns:
            The platform identity, an adapter-type fallback for a degenerate
            adapter without a platform ID, or None when the metadata carries
            neither field (such a directory is left untouched).
        """
        platform_id = metadata.get("platform_id", "").strip()
        if platform_id:
            return f"platform:{platform_id}"
        platform_name = metadata.get("platform_name", "").strip()
        return f"adapter:{platform_name}" if platform_name else None

    def _known_robots(self) -> list[dict[str, str]]:
        """List robot instances that own data, in a stable display order.

        Hidden directories (including the migration archive) and directories
        without robot.json are not robots. A single unreadable metadata file
        must not take the whole dashboard down.

        Returns:
            One mapping per robot instance, including its directory name as id.
        """
        robots: list[dict[str, str]] = []
        root = self._data_root / "robots"
        try:
            entries = sorted(root.iterdir())
        except OSError:
            return robots
        for directory in entries:
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            metadata = self._read_robot_metadata(directory)
            if metadata is None:
                continue
            if not metadata.get("label") and not self._canonical_identity(metadata):
                metadata["label"] = f"unrecognized · {directory.name[:12]}"
            robots.append({"id": directory.name, **metadata})
        robots.sort(key=lambda row: (row.get("label") or row["id"], row["id"]))
        return robots

    @staticmethod
    def _robot_richness(robot_dir: Path) -> tuple[int, int, int, float]:
        """Rank legacy robot directories by how much data they hold.

        The richest directory is promoted to the canonical platform-instance
        directory, so its checkpoints, exports and Neural Lens weights stay
        attached to the consolidated history. Memory from every other legacy
        directory is merged into the same bank; this ranking only decides which
        local artifacts remain directly active.

        Args:
            robot_dir: Legacy robot directory.

        Returns:
            A sortable tuple: experience rows (-1 when unreadable), database
            bytes, whether Neural Lens data exists, database mtime.
        """
        db_path = robot_dir / "memory.sqlite3"
        rows = -1
        size = 0
        modified = 0.0
        if db_path.is_file():
            try:
                stat = db_path.stat()
                size = stat.st_size
                modified = stat.st_mtime
            except OSError:
                size = 0
            try:
                # Percent-encode characters SQLite treats as URI syntax;
                # a literal '#' in the data path would otherwise truncate it.
                uri = f"file:{quote(db_path.as_posix(), safe='/:')}?mode=ro"
                conn = sqlite3.connect(uri, uri=True)
                try:
                    row = conn.execute("SELECT COUNT(*) FROM experiences").fetchone()
                    rows = int(row[0]) if row is not None else 0
                finally:
                    conn.close()
            except sqlite3.Error:
                rows = -1
        return rows, size, int((robot_dir / "neural").is_dir()), modified

    def _archive_robot_dir(self, robot_dir: Path, canonical_key: str) -> None:
        """Move a superseded robot directory out of the dashboard scan path.

        The directory is kept intact under robots/.legacy/<canonical-key>/ so
        no operator data is destroyed; the next bank initialization merges its
        episodes and then marks the database as merged.

        Args:
            robot_dir: Legacy directory to archive.
            canonical_key: Canonical platform-instance robot key.
        """
        archive_root = self._data_root / "robots" / _ROBOT_ARCHIVE_DIR / canonical_key
        try:
            archive_root.mkdir(parents=True, exist_ok=True)
            target = archive_root / robot_dir.name
            if target.exists():
                target = archive_root / f"{robot_dir.name}-{int(time.time() * 1000)}"
            robot_dir.replace(target)
            note = target / "migration.json"
            note.write_text(
                json.dumps(
                    {
                        "merged_into": canonical_key,
                        "archived_at": time.time(),
                        "source_layout": "0.1.3/0.2.1 platform_id:self_id",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            logger.info("[JitRL] archived legacy robot directory %s", robot_dir.name)
        except OSError:
            logger.warning(
                "[JitRL] cannot archive legacy robot directory %s",
                robot_dir,
                exc_info=True,
            )

    def _heal_robot_layout(self) -> None:
        """Consolidate legacy (0.1.3/0.2.1) robot dirs into platform-instance dirs.

        Those releases hashed platform_id:self_id. Adapters such as qq_official vary
        the self ID per message, so one bot could own many directories. The
        canonical directory is keyed by the platform instance only. This sweep
        promotes the richest legacy directory, archives the others, and lets a
        later bank initialization merge their episodes. It is idempotent, never
        deletes data and never touches directories whose platform ID is unknown.
        """
        root = self._data_root / "robots"
        if not root.is_dir():
            return
        try:
            entries = sorted(root.iterdir())
        except OSError:
            logger.warning("[JitRL] cannot scan robot directories", exc_info=True)
            return
        groups: dict[str, list[tuple[Path, dict[str, str]]]] = {}
        for directory in entries:
            if not directory.is_dir() or directory.name.startswith("."):
                continue
            metadata = self._read_robot_metadata(directory)
            if metadata is None:
                logger.warning(
                    "[JitRL] robot directory %s has no readable robot.json; left untouched",
                    directory.name,
                )
                continue
            identity = self._canonical_identity(metadata)
            if identity is None:
                logger.warning(
                    "[JitRL] robot directory %s has no platform id or adapter name; left untouched",
                    directory.name,
                )
                continue
            key = self._robot_key(identity)
            if directory.name == key:
                continue
            groups.setdefault(key, []).append((directory, metadata))
        for key, legacy in groups.items():
            canonical_dir = root / key
            remaining = legacy
            source: dict[str, str] = {}
            if not canonical_dir.exists():
                primary_dir, primary_metadata = max(
                    legacy, key=lambda item: self._robot_richness(item[0])
                )
                try:
                    primary_dir.replace(canonical_dir)
                except OSError:
                    logger.warning(
                        "[JitRL] cannot promote legacy robot directory %s",
                        primary_dir,
                        exc_info=True,
                    )
                    continue
                source = primary_metadata
                remaining = [item for item in legacy if item[0] != primary_dir]
            canonical_metadata = self._read_robot_metadata(canonical_dir) or source
            fallback_metadata = remaining[0][1] if remaining else {}
            for field in ("platform_id", "platform_name", "self_id"):
                if not canonical_metadata.get(field):
                    canonical_metadata[field] = fallback_metadata.get(field, "")
            if self._canonical_identity(canonical_metadata) is None:
                continue
            try:
                self._write_robot_metadata(
                    canonical_dir,
                    self._canonical_metadata(
                        canonical_metadata,
                        canonical_metadata.get("platform_id", "").strip(),
                    ),
                )
            except OSError:
                logger.warning(
                    "[JitRL] cannot write robot metadata in %s",
                    canonical_dir,
                    exc_info=True,
                )
            for legacy_dir, _metadata in remaining:
                self._archive_robot_dir(legacy_dir, key)
            if remaining:
                logger.info(
                    "[JitRL] consolidated %d legacy robot directories into %s",
                    len(remaining),
                    key,
                )

    def _ensure_layout(self) -> None:
        """Run the robot-directory consolidation at most once per process.

        __init__ calls this before any robot directory is created, so the
        richest legacy directory can still be promoted intact. The lazy check
        covers tests and embedders that create data directories after start-up.
        """
        if self._layout_ready:
            return
        with self._layout_lock:
            if self._layout_ready:
                return
            try:
                self._heal_robot_layout()
            except Exception:
                logger.warning(
                    "[JitRL] robot directory consolidation failed", exc_info=True
                )
            finally:
                self._layout_ready = True

    def _activate_robot(
        self, robot_key: str, bank: MemoryBank, db_path: Path, robot_dir: Path
    ) -> None:
        self._active_bank.set(bank)
        self._active_robot_key.set(robot_key)
        self._active_neural_root.set(
            self._neural_roots.get(robot_key, robot_dir / "neural")
        )
        self._last_bank = bank
        self._last_db_path = db_path

    def _migrate_legacy_data(self, robot_dir: Path, db_path: Path) -> None:
        migration_marker = self._data_root / ".legacy-migration.json"
        if migration_marker.exists():
            return

        old_checkpoints = self._data_root / "checkpoints"
        old_neural = self._data_root / "neural"
        old_exports = [
            *self._data_root.glob("export-*.json"),
            self._data_root / "jitrl-export.json",
        ]
        has_legacy_data = (
            any(
                Path(f"{self._legacy_db_path}{suffix}").exists()
                for suffix in ("", "-wal", "-shm")
            )
            or old_checkpoints.exists()
            or old_neural.exists()
            or any(source.exists() for source in old_exports)
        )
        if not has_legacy_data:
            return

        database_files = (
            (self._legacy_db_path, db_path),
            (Path(f"{self._legacy_db_path}-wal"), Path(f"{db_path}-wal")),
            (Path(f"{self._legacy_db_path}-shm"), Path(f"{db_path}-shm")),
        )
        database_conflict = any(
            source.exists() and target.exists() for source, target in database_files
        )
        if not database_conflict:
            for source, target in (*database_files[1:], database_files[0]):
                if source.exists():
                    source.replace(target)

        if (
            old_checkpoints.exists()
            and not database_conflict
            and not (robot_dir / "checkpoints").exists()
        ):
            old_checkpoints.replace(robot_dir / "checkpoints")
        if old_neural.exists() and not (robot_dir / "neural").exists():
            old_neural.replace(robot_dir / "neural")
        if not database_conflict:
            for source in old_exports:
                if source.exists():
                    target = robot_dir / source.name
                    if not target.exists():
                        source.replace(target)

        remaining = [
            source.name
            for source in (
                self._legacy_db_path,
                Path(f"{self._legacy_db_path}-wal"),
                Path(f"{self._legacy_db_path}-shm"),
                old_checkpoints,
                old_neural,
                *old_exports,
            )
            if source.exists()
        ]
        if remaining:
            logger.warning(
                "[JitRL] legacy data preserved at the shared path because the first robot already has conflicting files: %s",
                ", ".join(remaining),
            )

        migration_marker.write_text(
            json.dumps({"robot_id": robot_dir.name}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _with_previous_metadata(
        self, robot_dir: Path, metadata: dict[str, str]
    ) -> dict[str, str]:
        """Keep earlier identity fields when an event omits them.

        Some adapters report an empty self ID for certain event types. Writing
        that empty value over the last known one would make the dashboard label
        flap and lose the only human-readable account hint.

        Args:
            robot_dir: Canonical robot directory.
            metadata: Metadata built from the current event.

        Returns:
            Metadata with previous non-empty values restored where needed.
        """
        previous = self._read_robot_metadata(robot_dir) or {}
        for field in ("platform_id", "platform_name", "self_id"):
            if not metadata.get(field) and previous.get(field):
                metadata[field] = previous[field]
        if not metadata.get("platform_id"):
            metadata["label"] = (
                f"{metadata.get('platform_name') or 'AstrBot'} · "
                f"{metadata.get('self_id') or 'unknown'}"
            )
        return metadata

    def _activate_robot_event(
        self, event: AstrMessageEvent
    ) -> tuple[str, Path, dict[str, str]]:
        """Resolve, create and activate the canonical robot for one event.

        Args:
            event: AstrBot message event.

        Returns:
            The robot key, directory and normalized metadata.
        """
        self._ensure_layout()
        metadata = self._robot_metadata(event)
        robot_key = self._robot_key(metadata["identity"])
        robot_dir = self._robot_dir(robot_key)
        robot_dir.mkdir(parents=True, exist_ok=True)
        metadata = self._with_previous_metadata(robot_dir, metadata)
        try:
            self._write_robot_metadata(robot_dir, metadata)
        except OSError:
            logger.warning(
                "[JitRL] cannot persist robot metadata in %s",
                robot_dir,
                exc_info=True,
            )
        self._robot_dirs[robot_key] = robot_dir
        self._active_robot_key.set(robot_key)
        self._active_neural_root.set(robot_dir / "neural")
        return robot_key, robot_dir, metadata

    @staticmethod
    def _mark_archive_merged(archive: Path, db_path: Path) -> bool:
        """Rename an archived database after its rows were merged.

        Args:
            archive: Archived robot directory.
            db_path: Archived memory database that was merged.

        Returns:
            True when the database and any WAL sidecars were renamed.
        """
        try:
            db_path.replace(archive / "memory.sqlite3.merged")
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{db_path}{suffix}")
                if sidecar.is_file():
                    sidecar.replace(archive / f"memory.sqlite3.merged{suffix}")
        except OSError:
            return False
        return True

    def _adopt_archived_neural(self, robot_key: str, robot_dir: Path) -> None:
        """Move one archived Neural Lens store into the canonical robot.

        Consolidation archives whole legacy directories. A trained lens may
        live in a directory that was not the richest by episode count, so the
        canonical robot could otherwise start untrained. Adopt the archived
        store with the most data and newest write when the canonical robot has
        none; an existing canonical store always wins and is never overwritten.

        Args:
            robot_key: Canonical platform-instance robot key.
            robot_dir: Canonical robot directory.
        """
        target = robot_dir / "neural"
        if (target / "neural.sqlite3").is_file():
            return
        if target.exists():
            # An empty directory can be left behind by a failed runtime start;
            # it must not shadow a usable archived store.
            try:
                if any(target.iterdir()):
                    return
                target.rmdir()
            except OSError:
                return
        archive_root = self._data_root / "robots" / _ROBOT_ARCHIVE_DIR / robot_key
        try:
            archives = sorted(archive_root.iterdir())
        except OSError:
            return
        candidates: list[tuple[int, float, Path]] = []
        for archive in archives:
            store = archive / "neural" / "neural.sqlite3"
            if not store.is_file():
                continue
            try:
                stat = store.stat()
            except OSError:
                continue
            candidates.append((stat.st_size, stat.st_mtime, archive))
        if not candidates:
            return
        # If several legacy directories trained a lens, keep the one with the
        # most data and then the newest write; name order says nothing about
        # which store is fresher.
        _size, _mtime, archive = max(candidates, key=lambda item: (item[0], item[1]))
        try:
            (archive / "neural").replace(target)
        except OSError:
            logger.warning(
                "[JitRL] cannot adopt archived Neural Lens store from %s",
                archive.name,
                exc_info=True,
            )
            return
        logger.info("[JitRL] adopted archived Neural Lens store from %s", archive.name)

    async def _merge_archived_robot_banks(
        self, bank: MemoryBank, robot_key: str
    ) -> None:
        """Merge archived legacy memory into the canonical bank.

        Each archived directory is merged once: after a successful import the
        SQLite file is renamed to memory.sqlite3.merged. MemoryBank.merge_json
        deduplicates by episode ID, so a crash between the import and the
        rename cannot duplicate rows on the next attempt.

        Args:
            bank: Initialized canonical memory bank.
            robot_key: Canonical platform-instance robot key.
        """
        archive_root = self._data_root / "robots" / _ROBOT_ARCHIVE_DIR / robot_key
        if not archive_root.is_dir():
            return
        await asyncio.to_thread(
            self._adopt_archived_neural, robot_key, self._robot_dir(robot_key)
        )
        try:
            archives = sorted(archive_root.iterdir())
        except OSError:
            logger.warning(
                "[JitRL] cannot scan archived robot directories", exc_info=True
            )
            return
        for archive in archives:
            if not archive.is_dir():
                continue
            db_path = archive / "memory.sqlite3"
            if not db_path.is_file():
                continue
            legacy = MemoryBank(
                db_path,
                checkpoint_dir=archive / "checkpoints",
                checkpoint_interval=0,
            )
            try:
                await legacy.initialize()
                try:
                    # No row cap: a truncated archive must never be marked as
                    # merged, because a retry would skip its already imported
                    # episodes and could not recover the remaining rows.
                    payload = await legacy.export_json(limit=None)
                finally:
                    await legacy.close()
            except Exception:
                logger.warning(
                    "[JitRL] cannot read archived memory in %s",
                    archive,
                    exc_info=True,
                )
                continue
            try:
                imported = await bank.merge_json(payload)
            except Exception:
                logger.warning(
                    "[JitRL] cannot merge archived memory from %s",
                    archive,
                    exc_info=True,
                )
                continue
            marked = await asyncio.to_thread(
                self._mark_archive_merged, archive, db_path
            )
            if not marked:
                logger.warning(
                    "[JitRL] merged archived memory but cannot mark it in %s",
                    archive,
                )
                continue
            logger.info(
                "[JitRL] merged %d archived experiences from %s",
                imported,
                archive.name,
            )

    async def _ensure_ready(
        self, event: AstrMessageEvent | None = None, *, robot_id: str | None = None
    ) -> MemoryBank:
        self._ensure_layout()
        if event is not None:
            robot_key, robot_dir, metadata = self._activate_robot_event(event)
        elif robot_id is not None:
            selected = next(
                (robot for robot in self._known_robots() if robot["id"] == robot_id),
                None,
            )
            if selected is None:
                raise ValueError("机器人实例不存在")
            metadata = {key: value for key, value in selected.items() if key != "id"}
            robot_key = robot_id
        else:
            active = self._active_bank.get()
            if active is not None:
                return active
            raise RuntimeError("an AstrBot robot instance is required")

        bank = self._banks.get(robot_key)
        if bank is not None:
            self._activate_robot(
                robot_key,
                bank,
                self._bank_paths[robot_key],
                self._robot_dirs[robot_key],
            )
            return bank
        async with self._ready_lock:
            bank = self._banks.get(robot_key)
            if bank is not None:
                self._activate_robot(
                    robot_key,
                    bank,
                    self._bank_paths[robot_key],
                    self._robot_dirs[robot_key],
                )
                return bank
            robot_dir = self._robot_dirs.get(robot_key, self._robot_dir(robot_key))
            robot_dir.mkdir(parents=True, exist_ok=True)
            db_path = robot_dir / "memory.sqlite3"
            await asyncio.to_thread(self._migrate_legacy_data, robot_dir, db_path)
            try:
                self._write_robot_metadata(robot_dir, metadata)
            except OSError:
                logger.warning(
                    "[JitRL] cannot persist robot metadata in %s",
                    robot_dir,
                    exc_info=True,
                )
            bank = MemoryBank(
                db_path,
                max_entries=self.cfg.memory.max_entries,
                redact=self.cfg.memory.redact_secrets,
                keep_episodes=self.cfg.memory.keep_episodes,
                prune_strategy=self.cfg.memory.prune_strategy,
                checkpoint_dir=robot_dir / "checkpoints",
                checkpoint_interval=self.cfg.memory.checkpoint_interval,
                keep_checkpoints=self.cfg.memory.keep_checkpoints,
            )
            await bank.initialize()
            await self._merge_archived_robot_banks(bank, robot_key)
            self._banks[robot_key] = bank
            self._bank_paths[robot_key] = db_path
            self._robot_dirs[robot_key] = robot_dir
            self._activate_robot(robot_key, bank, db_path, robot_dir)
            self._spawn(bank.warm_index())
            logger.info(
                "[JitRL] memory bank ready for %s",
                metadata.get("label")
                or metadata.get("platform_name")
                or metadata.get("platform_id")
                or robot_key,
            )
            return bank

    async def terminate(self) -> None:
        """Release background work and close the database on unload."""
        tasks = list(self._background)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for task in tasks:
            self._background_done(task)
        self._background.clear()
        self._background_coroutines.clear()
        await asyncio.gather(
            *(runtime.close() for runtime in self._neural_by_robot.values())
        )
        await asyncio.gather(*(bank.close() for bank in self._banks.values()))
        self._neural_by_robot.clear()
        self._neural_roots.clear()
        self._banks.clear()
        self._bank_paths.clear()
        self._last_neural = None
        self._last_bank = None
        self._active_bank.set(None)
        self._active_neural_root.set(None)
        self._active_robot_key.set(None)
        logger.info("[JitRL] plugin terminated")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    async def _ensure_neural(
        self,
        event: AstrMessageEvent | None = None,
        *,
        robot_id: str | None = None,
    ) -> Any:
        self._ensure_layout()
        if event is not None:
            robot_key, robot_dir, _metadata = self._activate_robot_event(event)
        elif robot_id is not None:
            if not any(robot["id"] == robot_id for robot in self._known_robots()):
                raise ValueError("机器人实例不存在")
            robot_key = robot_id
            robot_dir = self._robot_dir(robot_key)
            self._robot_dirs[robot_key] = robot_dir
            self._active_robot_key.set(robot_key)
        else:
            robot_key = self._active_robot_key.get()
            if robot_key is None:
                raise RuntimeError("an AstrBot robot instance is required")
            robot_dir = self._robot_dirs.get(robot_key, self._robot_dir(robot_key))

        runtime = self._neural_by_robot.get(robot_key)
        if runtime is not None:
            self._active_neural_root.set(self._neural_roots[robot_key])
            self._last_neural = runtime
            return runtime

        async with self._neural_ready_lock:
            runtime = self._neural_by_robot.get(robot_key)
            if runtime is not None:
                self._active_neural_root.set(self._neural_roots[robot_key])
                self._last_neural = runtime
                return runtime
            from .jitrl.neural.runtime import LensRuntime

            neural_root = robot_dir / "neural"
            async with self._ready_lock:
                await asyncio.to_thread(
                    self._migrate_legacy_data,
                    robot_dir,
                    robot_dir / "memory.sqlite3",
                )
            await asyncio.to_thread(self._adopt_archived_neural, robot_key, robot_dir)
            if self._legacy_neural_root.exists() and not neural_root.exists():
                await asyncio.to_thread(self._legacy_neural_root.replace, neural_root)
            runtime = await asyncio.to_thread(LensRuntime, neural_root, self.cfg.neural)
            self._neural_by_robot[robot_key] = runtime
            self._neural_roots[robot_key] = neural_root
            self._active_neural_root.set(neural_root)
            self._last_neural = runtime
            return runtime

    async def _neural_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        try:
            runtime = await self._ensure_neural(event)
            messages = provider_messages(req)
            calls, observation, events = observed_context(messages)
            run = event.get_extra(_NEURAL_KEY)
            if not isinstance(run, NeuralRun) or run.completed:
                run = NeuralRun(baseline_calls=calls)
                event.set_extra(_NEURAL_KEY, run)
                cache_scope = self._cache_scope(event)
                await self._maybe_apply_feedback(
                    cache_scope, event.get_message_str() or ""
                )
            new_calls = calls - run.baseline_calls
            if len(run.frames) >= 40:
                return
            tools = set(req.func_tool.names()) if req.func_tool else set()
            goal = (event.get_message_str() or req.prompt or "").strip()
            state = canonicalize(
                scope=event.unified_msg_origin,
                goal=goal,
                observation=observation,
                events=events,
                allowed_actions=tuple(tools | {"reply"}),
                environment="astrbot-v4:"
                + self._surface(event)
                + ":model="
                + str(getattr(req, "model", None) or "unknown"),
                episode_id=run.episode_id,
                step_index=len(new_calls),
            )
            decision, epoch = await runtime.prepare(state)
            if self.cfg.apply_mode == "off":
                decision = replace(
                    decision,
                    intervention=Intervention(reason="MASTER_APPLY_OFF"),
                    rendered="",
                )
            part = (
                _make_temporary_part(decision.rendered) if decision.rendered else None
            )
            if decision.rendered and part is None:
                decision = replace(
                    decision,
                    intervention=Intervention(reason="HOST_TEXT_PART_UNAVAILABLE"),
                    rendered="",
                )
            if not await runtime.commit(decision, epoch):
                return
            if part is not None:
                req.extra_user_content_parts.append(part)
            run.frames.append(RequestFrame(decision, epoch, new_calls))
            self._stats["requests"] += 1
            self._stats["biased"] += int(bool(decision.rendered))
        except Exception:
            self._stats["failures"] += 1
            logger.warning("[JitRL Lens] request skipped; pure NO_OP", exc_info=True)

    async def _neural_done(
        self, event: AstrMessageEvent, run_context: Any, response: Any
    ) -> None:
        run = event.get_extra(_NEURAL_KEY)
        if not isinstance(run, NeuralRun) or run.completed or not run.frames:
            return
        run.completed = True
        messages = list(getattr(run_context, "messages", None) or [])
        final_text = (getattr(response, "completion_text", "") or "").strip()
        transitions, call_ids = executed_transitions(
            messages,
            goal=run.frames[0].decision.state.goal,
            surface=self._surface(event),
            baseline_calls=run.baseline_calls,
            final_text=final_text,
        )
        if not transitions:
            return
        payload = {
            "run": run,
            "transitions": transitions,
            "call_ids": call_ids,
            "umo": event.unified_msg_origin,
            "robot_id": self._robot_key_for_event(event),
            "cache_scope": self._cache_scope(event),
            "final_text": final_text,
        }
        if self.cfg.learning.async_learn:
            self._spawn(self._learn_neural(**payload))
        else:
            await self._learn_neural(**payload)

    async def _learn_neural(
        self,
        *,
        run: NeuralRun,
        transitions: list[Any],
        call_ids: list[str],
        umo: str,
        robot_id: str,
        cache_scope: str,
        final_text: str,
    ) -> None:
        runtime = await self._ensure_neural(robot_id=robot_id)
        preceding: set[str] = set()
        executions = []
        for index, (transition, call_id) in enumerate(
            zip(transitions, call_ids, strict=True)
        ):
            frame = run.frame_for(frozenset(preceding))
            if (
                frame is not None
                and transition.action.name in frame.decision.state.allowed_actions
            ):
                experience_id = await asyncio.to_thread(
                    runtime.store.append_execution,
                    frame.decision.state.decision_id,
                    index=index,
                    action_key=transition.action.key,
                    action_name=transition.action.name,
                    expected_epoch=frame.epoch,
                )
                if experience_id:
                    executions.append((experience_id, index, frame.epoch))
            if call_id != "__final_reply__":
                preceding.add(call_id)
        # PENDING executions survive evaluator failure. An absent label never
        # becomes a manufactured zero-return training target.
        if not executions or self.cfg.learning.evaluator_mode == "off":
            return
        if len(transitions) < self.cfg.learning.min_steps:
            return

        async def generate(*, prompt: str, system_prompt: str | None = None) -> str:
            return await self._llm_text(
                prompt=prompt,
                system_prompt=system_prompt,
                provider_id=self.cfg.learning.evaluator_provider_id,
                umo=umo,
            )

        evaluator = StepRewardEvaluator(
            generate if self.cfg.learning.evaluator_mode in {"auto", "llm"} else None,
            mode=self.cfg.learning.evaluator_mode,
            timeout=self.cfg.learning.evaluator_timeout,
            max_reward=3.0,
            heuristic_weight=self.cfg.learning.heuristic_weight,
        )
        result = await evaluator.score(
            task=run.frames[0].decision.state.goal,
            actions=[redact(t.action.text) for t in transitions],
            observations=[redact(t.observation) for t in transitions],
            outcomes=[t.outcome for t in transitions],
            success=infer_success(transitions=transitions, final_text=final_text)
            if self.cfg.learning.success_detection != "off"
            else None,
        )
        if len(result.rewards) != len(transitions):
            return
        returns = discounted_returns(result.rewards, self.cfg.learning.gamma)
        settled = 0
        for experience_id, index, epoch in executions:
            settled += await asyncio.to_thread(
                runtime.store.settle,
                experience_id,
                reward=result.rewards[index],
                return_g=returns[index],
                source=result.source,
                expected_epoch=epoch,
            )
        self._stats["episodes"] += int(settled > 0)
        self._stats["experiences"] += settled
        if settled:
            self._neural_last_episode[cache_scope] = (
                run.episode_id,
                time.time(),
                self.cfg.learning.gamma,
                executions[0][2],
            )
            if len(self._neural_last_episode) > 512:
                self._neural_last_episode.pop(next(iter(self._neural_last_episode)))
        if (
            self.cfg.neural.auto_train
            and settled
            and cache_scope not in self._neural_training
        ):
            stats = await asyncio.to_thread(runtime.store.stats, umo)
            if (
                stats["executions"] >= self.cfg.neural.min_samples
                and stats["executions"] % self.cfg.neural.train_every < settled
            ):
                self._neural_training.add(cache_scope)

                async def consolidate() -> None:
                    try:
                        await runtime.train(umo)
                    finally:
                        self._neural_training.discard(cache_scope)

                self._spawn(consolidate())

    def _spawn(self, coroutine: Any) -> None:
        """Run a coroutine in the background, logging failures.

        Args:
            coroutine: The coroutine to schedule.
        """

        async def _runner() -> None:
            try:
                await coroutine
            except asyncio.CancelledError:
                raise
            except Exception:
                self._stats["failures"] += 1
                logger.warning("[JitRL] background learning failed", exc_info=True)

        task = asyncio.create_task(_runner())
        self._background.add(task)
        self._background_coroutines[task] = coroutine
        task.add_done_callback(self._background_done)

    def _background_done(self, task: asyncio.Task) -> None:
        """Forget a finished background task and close an unstarted coroutine.

        A task cancelled before its first step never awaits the coroutine it
        was given; Python then warns about a never-awaited coroutine when that
        object is collected. Closing it here is the only safe point: the task
        is already done, so nothing else can start it.

        Args:
            task: Finished background task.
        """
        self._background.discard(task)
        coroutine = self._background_coroutines.pop(task, None)
        if (
            coroutine is not None
            and inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED
        ):
            coroutine.close()

    async def _llm_text(
        self,
        *,
        prompt: str,
        system_prompt: str | None = None,
        provider_id: str = "",
        umo: str | None = None,
    ) -> str:
        """Run one auxiliary chat completion.

        Args:
            prompt: User prompt for the auxiliary call.
            system_prompt: Optional system prompt.
            provider_id: Explicit provider to use; falls back to the session
                provider when empty or unresolvable.
            umo: Unified message origin used to resolve the session provider.

        Returns:
            The completion text, stripped of surrounding whitespace.

        Raises:
            RuntimeError: When no chat provider could be resolved.
        """
        provider = None
        if provider_id:
            try:
                provider = await self.context.provider_manager.get_provider_by_id(
                    provider_id
                )
            except Exception:
                logger.warning(
                    "[JitRL] evaluator provider %s unavailable, falling back",
                    provider_id,
                    exc_info=True,
                )
        if provider is None:
            provider = await self.context.get_using_provider_async(umo)
        if provider is None:
            raise RuntimeError("no chat provider is available for JitRL")

        response = await provider.text_chat(
            prompt=prompt,
            system_prompt=system_prompt or "",
        )
        return (response.completion_text or "").strip()

    def _surface(self, event: AstrMessageEvent) -> str:
        """Return the coarse retrieval partition for an event.

        Args:
            event: The message event being processed.

        Returns:
            The platform name when the first-stage gate is enabled, otherwise
            an empty string so retrieval stays fully cross-platform.
        """
        if not self.cfg.retrieval.surface_gate:
            return ""
        try:
            return event.get_platform_name() or ""
        except Exception:
            return ""

    @staticmethod
    def _action_kind(action_key: str) -> str:
        """Extract the kind prefix of an action key.

        Args:
            action_key: Normalised action identity such as
                `"tool:web_search(query=<str>)"`.

        Returns:
            The kind prefix, e.g. `"tool"`; empty when malformed.
        """
        return action_key.split(":", 1)[0] if ":" in action_key else ""

    @staticmethod
    def _action_name(action_key: str) -> str:
        """Extract the bare tool name of an action key.

        Args:
            action_key: Normalised action identity.

        Returns:
            The tool name between the kind prefix and the first parenthesis.
        """
        body = action_key.split(":", 1)[1] if ":" in action_key else action_key
        return body.split("(", 1)[0]

    def _available_memory_actions(
        self,
        neighbors: list[Neighbor],
        tool_names: set[str],
    ) -> dict[str, tuple[str, int]]:
        """Group neighbours into available candidate actions.

        Actions learned in the past are only offered again when their tool is
        still enabled for this request; otherwise the plugin would recommend
        tools the deployment no longer exposes.

        Args:
            neighbors: Retrieved neighbours.
            tool_names: Names of the tools available to the current request.

        Returns:
            A mapping `action_key -> (action_text, support_count)`.
        """
        grouped: dict[str, tuple[str, int]] = {}
        for neighbor in neighbors:
            key = neighbor.action_key
            if not key:
                continue
            if (
                self._action_kind(key) == "tool"
                and self._action_name(key) not in tool_names
            ):
                continue
            text, count = grouped.get(key, (neighbor.action_text or key, 0))
            grouped[key] = (text, count + 1)
        return grouped

    def _apply_tool_gate(
        self,
        request: ProviderRequest,
        estimate: ValueEstimate,
    ) -> None:
        """Mask tools whose advantage is confidently negative.

        Setting a tool's probability to zero is the `-inf` limit of the logit
        update, so this is the plugin's only *hard* constraint on the action
        space. It is applied only when there is enough evidence and at least
        one tool would remain. The gate reads the probe estimate, which covers
        every retrieved action rather than only the shortlisted candidates, so a
        tool with strong negative evidence cannot escape by losing the
        shortlist ranking.

        Args:
            request: The outgoing provider request whose tool set is mutated.
            estimate: Value estimate over every retrieved action.
        """
        tool_set = request.func_tool
        if tool_set is None or estimate.max_abs_advantage < MIN_EVIDENCE_SIGNAL:
            return
        names = set(tool_set.names())
        variants: dict[str, list[str]] = {}
        for key in estimate.advantage_norm:
            if self._action_kind(key) != "tool":
                continue
            name = self._action_name(key)
            variants.setdefault(name, []).append(key)
        # The host can only mask an entire tool. One bad argument signature
        # must not hide other signatures with positive or insufficient evidence.
        blocked = {
            name
            for name, keys in variants.items()
            if name in names
            and all(
                estimate.action_counts.get(key, 0)
                >= self.cfg.policy.tool_gate_min_count
                and estimate.advantage_norm[key] <= self.cfg.policy.tool_gate_threshold
                and estimate.advantage.get(key, 0.0) <= -MIN_EVIDENCE_SIGNAL
                for key in keys
            )
        }
        if not blocked or not (names - blocked):
            return
        for name in sorted(blocked):
            tool_set.remove_tool(name)
        logger.debug("[JitRL] tool gate masked %s", sorted(blocked))

    async def _rerank(
        self,
        *,
        umo: str,
        state: StateView,
        task: str,
        options: list[tuple[str, str]],
    ) -> dict[str, float]:
        """Ask the model for a verbalised distribution over candidates.

        Args:
            umo: Unified message origin used to resolve the provider.
            state: Abstracted state used for retrieval.
            task: Redacted user request.
            options: `(action_key, action_text)` pairs to score.

        Returns:
            A mapping `action_key -> confidence` in `[0, 1]`; empty when the
            reranker failed, which degrades the update to a zero prior.
        """
        if not options or self.cfg.apply_mode != "logit":
            return {}

        async def _generate(*, prompt: str, system_prompt: str | None = None) -> str:
            return await self._llm_text(
                prompt=prompt, system_prompt=system_prompt, umo=umo
            )

        reranker = VerbalizedReranker(
            _generate,
            timeout=max(5.0, self.cfg.learning.evaluator_timeout),
            max_candidates=max(2, self.cfg.policy.candidate_limit + 1),
        )
        result = await reranker.score(
            task=task,
            state_text=state.text,
            candidates=options,
        )
        if not result.ok:
            logger.debug("[JitRL] reranker degraded: %s", result.error)
        return result.scores if result.ok else {}

    # ------------------------------------------------------------------
    # decision pipeline
    # ------------------------------------------------------------------
    def _state_for(
        self, robot_key: str, umo: str, user_text: str, surface: str
    ) -> tuple[str, StateView]:
        """Return the redacted task and its abstracted state, memoised.

        A single user turn usually produces several LLM requests (one per agent
        step) with the same message, so the redaction/regularisation/tokenisation
        pipeline only has to run once per turn.

        Args:
            robot_key: Robot instance ID, part of the cache key.
            umo: Unified message origin, part of the cache key.
            user_text: Raw user message.
            surface: Coarse retrieval partition.

        Returns:
            A (task, state) pair: the redacted request and its retrieval state.
        """
        key = (robot_key, umo, surface, user_text)
        cached = self._state_cache.get(key)
        if cached is not None:
            self._state_cache.move_to_end(key)
            return cached
        task = redact(user_text)[:_TASK_CHAR_LIMIT]
        state = abstract_state(task=task, surface=surface, limit=_STATE_CHAR_LIMIT)
        self._state_cache[key] = (task, state)
        if len(self._state_cache) > _STATE_CACHE_MAX:
            self._state_cache.popitem(last=False)
        return task, state

    async def _decide(
        self,
        *,
        umo: str,
        task: str,
        state: StateView,
        tool_names: set[str],
    ) -> _DecisionSnapshot | None:
        """Run Algorithm 1 steps 1-6 for one query state.

        Two passes over the neighbourhood make the candidate set better than the
        single-pass version. The first pass estimates the advantage of *every*
        retrieved action, the shortlist keeps the |C| actions whose evidence is
        strongest (largest absolute advantage, then support), and the second
        pass renormalises Eq. 25 over exactly that set -- so a tool with strongly
        negative evidence is shortlisted and penalised instead of being dropped
        before it could influence the decision, while the temperature beta keeps
        operating on a properly normalised advantage range.

        Args:
            umo: Unified message origin acting as the memory scope.
            task: Redacted user request.
            state: Abstracted query state.
            tool_names: Tools available to the current request.

        Returns:
            The decision snapshot, or None when there is no usable evidence.
        """
        cfg = self.cfg
        neighbors = await self._bank.topk(
            state,
            scope=umo,
            k=cfg.retrieval.k,
            threshold=cfg.retrieval.similarity_threshold,
            surface_gate=cfg.retrieval.surface_gate,
            prefilter_limit=cfg.retrieval.prefilter_limit,
            recency_half_life_days=cfg.retrieval.recency_half_life_days,
            scope_mode=cfg.retrieval.scope_isolation,
        )
        if not neighbors:
            return None
        memory_actions = self._available_memory_actions(neighbors, tool_names)
        if not memory_actions:
            return None

        estimate_kwargs = {
            "k": cfg.retrieval.k,
            "ucb_alpha": cfg.policy.ucb_alpha,
            "exploration_rate": cfg.policy.exploration_rate,
            "min_count": cfg.retrieval.min_count,
            "rng": self._rng,
            "weighting": cfg.retrieval.estimator_weighting,
            "exploration_estimate": cfg.policy.exploration_estimate,
        }
        all_keys = list(memory_actions)
        probe = estimate_values(
            neighbors,
            all_keys,
            normalize=cfg.policy.normalize_advantage,
            **estimate_kwargs,
        )
        limit = max(1, cfg.policy.candidate_limit)
        shortlist = sorted(
            all_keys,
            key=lambda key: (
                -abs(probe.advantage.get(key, 0.0)),
                -probe.action_counts.get(key, 0),
                key,
            ),
        )[:limit]
        chosen_actions = {key: memory_actions[key] for key in shortlist}

        llm_actions: tuple[tuple[str, str, float], ...] = ()
        if cfg.apply_mode == "logit":
            options = [(key, chosen_actions[key][0]) for key in shortlist]
            scores = await self._rerank(
                umo=umo, state=state, task=task, options=options
            )
            if scores:
                llm_actions = tuple(
                    (key, text, verbalized_to_logit(scores.get(key, 0.5)))
                    for key, text in options
                )

        candidates = build_candidates(
            llm_actions=llm_actions,
            memory_actions=chosen_actions,
            limit=limit,
            memory_order=shortlist,
        )
        if not candidates:
            return None
        estimate = estimate_values(
            neighbors,
            [candidate.key for candidate in candidates],
            normalize=cfg.policy.normalize_advantage,
            **estimate_kwargs,
        )
        decision = select_action(
            candidates,
            estimate,
            beta=cfg.policy.beta,
            temperature=cfg.policy.sample_temperature,
            normalize=cfg.policy.normalize_advantage,
            mode=cfg.apply_mode,
            rng=self._rng,
        )
        block = render_preference_block(
            decision,
            neighbors=neighbors,
            max_chars=cfg.policy.max_hint_chars,
        )
        return _DecisionSnapshot(
            decision=decision,
            estimate=estimate,
            probe=probe,
            neighbors=neighbors,
            block=block,
            task=task,
            state=state,
        )

    # ------------------------------------------------------------------
    # user feedback
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_feedback(text: str) -> str:
        """Strip punctuation and whitespace from a feedback reply.

        Args:
            text: Raw user message.

        Returns:
            A lowercase, punctuation-free form used for keyword matching.
        """
        cleaned = "".join(
            char
            for char in text.strip().lower()
            if char.isalnum() or "\u4e00" <= char <= "\u9fff"
        )
        return cleaned

    async def _maybe_apply_feedback(self, cache_scope: str, user_text: str) -> None:
        """Fold an explicit user verdict into the previous episode.

        Args:
            cache_scope: Robot-and-session key of the current request.
            user_text: Raw text of the current user message.
        """
        cfg = self.cfg.feedback
        if not cfg.enabled or not user_text or len(user_text) > 16:
            return
        cache = (
            self._neural_last_episode
            if self.cfg.neural.mode != "off"
            else self._last_episode
        )
        cached = cache.get(cache_scope)
        if not cached:
            return
        episode_id, created_at = cached[:2]
        if time.time() - created_at > cfg.window_seconds:
            cache.pop(cache_scope, None)
            return
        normalised = self._normalise_feedback(user_text)
        if not normalised:
            return
        delta = 0.0
        if any(
            normalised == word or normalised.startswith(word)
            for word in cfg.negative_keywords
        ):
            delta = -_FEEDBACK_DELTA
        elif any(
            normalised == word or normalised.startswith(word)
            for word in cfg.positive_keywords
        ):
            delta = _FEEDBACK_DELTA
        if delta == 0.0:
            return
        if self.cfg.neural.mode != "off":
            await self._apply_neural_feedback(cache_scope, delta)
            return
        if self._bank is None:
            return
        updated = await self._bank.apply_user_feedback(
            episode_id, delta, expected_generation=self._bank.generation
        )
        self._last_episode.pop(cache_scope, None)
        self._invalidate_decisions()
        logger.info(
            "[JitRL] user feedback applied: episode=%s delta=%+.1f rows=%d",
            episode_id,
            delta,
            updated,
        )

    async def _apply_neural_feedback(self, cache_scope: str, delta: float) -> int:
        cached = self._neural_last_episode.pop(cache_scope, None)
        if not cached:
            return 0
        robot_id, _, scope = cache_scope.partition(":")
        runtime = await self._ensure_neural(robot_id=robot_id)
        episode_id, created_at, gamma, epoch = cached
        if time.time() - created_at > self.cfg.feedback.window_seconds:
            return 0
        return await runtime.apply_user_feedback(
            scope=scope,
            episode_id=episode_id,
            delta=delta,
            gamma=gamma,
            expected_epoch=epoch,
        )

    # ------------------------------------------------------------------
    # hooks
    # ------------------------------------------------------------------
    @filter.on_llm_request()
    async def on_llm_request(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        """Retrieve relevant experience and bias the outgoing request.

        This is Algorithm 1 steps 1-6 executed for the current state. The
        resulting text block is advisory: the KL term of Eq. 8 keeps the
        update close to the frozen reference policy, so the model is told to
        ignore the block when the evidence does not match the request.

        The block is attached to the current user message as a temporary
        content part (`req.extra_user_content_parts`) instead of the system
        prompt. The system prompt is the cached context prefix, so rewriting it
        on every request would destroy the provider's prompt-cache hit rate.

        Args:
            event: The message event that triggered the request.
            req: The outgoing provider request, mutated in place.
        """
        if not self.cfg.enabled:
            return
        if self.cfg.neural.mode != "off":
            await self._neural_request(event, req)
            return
        if event.get_extra(_INJECTED_KEY):
            return
        try:
            await self._ensure_ready(event)
        except Exception:
            self._stats["failures"] += 1
            logger.warning("[JitRL] memory bank unavailable", exc_info=True)
            return

        umo = event.unified_msg_origin
        cache_scope = self._cache_scope(event)
        generation = self._bank.generation
        event.set_extra(_GENERATION_KEY, generation)
        user_text = (event.get_message_str() or req.prompt or "").strip()

        try:
            await self._maybe_apply_feedback(cache_scope, user_text)
        except Exception:
            logger.warning("[JitRL] feedback application failed", exc_info=True)

        if self.cfg.apply_mode == "off" or not user_text:
            return

        try:
            surface = self._surface(event)
            robot_id = self._robot_key_for_event(event)
            task, state = self._state_for(robot_id, umo, user_text, surface)
            tool_names = set(req.func_tool.names()) if req.func_tool else set()
            cache_key = (
                cache_scope,
                generation,
                self.cfg.apply_mode,
                tuple(sorted(tool_names)),
                state.signature,
            )
            snapshot = self._decision_cache.get(cache_key)
            if snapshot is None:
                snapshot = await self._decide(
                    umo=umo, task=task, state=state, tool_names=tool_names
                )
                if snapshot is None:
                    return
                self._decision_cache[cache_key] = snapshot
                if len(self._decision_cache) > _DECISION_CACHE_MAX:
                    self._decision_cache.popitem(last=False)
            else:
                # Same state, same memory epoch, same tool set: an agent loop
                # reuses one policy decision instead of recomputing (and
                # re-sampling) it on every step.
                self._decision_cache.move_to_end(cache_key)

            if generation != self._bank.generation:
                return
            if snapshot.block:
                # The block is appended to the *user* message as a temporary
                # content part, never to req.system_prompt: the system prompt is
                # the cached context prefix, so rewriting it on every request
                # would wipe out prompt-cache hits. mark_as_temp keeps the part
                # out of persisted history - it is valid for this turn only.
                part = _make_temporary_part(snapshot.block)
                if part is not None:
                    req.extra_user_content_parts.append(part)
                    event.set_extra(_INJECTED_KEY, True)
            if self.cfg.policy.tool_gate:
                self._apply_tool_gate(req, snapshot.probe)

            decision = snapshot.decision
            self._last_decision[cache_scope] = (time.time(), decision)
            if len(self._last_decision) > 512:
                cutoff = time.time() - _LAST_EPISODE_TTL
                self._last_decision = {
                    key: value
                    for key, value in self._last_decision.items()
                    if value[0] >= cutoff
                }
            event.set_extra(
                _EVENT_EXTRA_KEY,
                {
                    "state": snapshot.state,
                    "decision": decision,
                    "surface": surface,
                    "user_text": snapshot.task,
                },
            )
            self._stats["requests"] += 1
            if decision.applied:
                self._stats["biased"] += 1
            if self.cfg.debug.log_decisions:
                logger.info(
                    "[JitRL] %s | neighbours=%d v=%+.3f",
                    decision.reason,
                    snapshot.estimate.neighbor_count,
                    snapshot.estimate.v_hat,
                )
        except Exception:
            self._stats["failures"] += 1
            logger.warning("[JitRL] policy update skipped", exc_info=True)

    @filter.on_agent_done()
    async def on_agent_done(
        self,
        event: AstrMessageEvent,
        run_context: Any,
        response: Any,
    ) -> None:
        """Evaluate the finished trajectory and append it to memory.

        This is Algorithm 1 step 7. Evaluation is dispatched to a background
        task by default so the user never waits for the learning loop.

        Args:
            event: The message event that started the run.
            run_context: Agent run context holding the message history.
            response: Final LLM response of the run.
        """
        if not self.cfg.enabled or not self.cfg.learning.enabled:
            return
        if self.cfg.neural.mode != "off":
            try:
                await self._neural_done(event, run_context, response)
            except Exception:
                self._stats["failures"] += 1
                logger.warning("[JitRL Lens] capture failed", exc_info=True)
            return
        try:
            await self._ensure_ready(event)
        except Exception:
            self._stats["failures"] += 1
            logger.warning("[JitRL] memory bank unavailable", exc_info=True)
            return

        try:
            context = event.get_extra(_EVENT_EXTRA_KEY) or {}
            umo = event.unified_msg_origin
            user_text = context.get("user_text") or (event.get_message_str() or "")
            surface = context.get("surface") or self._surface(event)
            messages = getattr(run_context, "messages", None) or []
            transitions = extract_transitions(
                messages=messages,
                user_text=user_text,
                surface=surface,
            )
            if len(transitions) < max(1, self.cfg.learning.min_steps):
                return
            reply_text = (getattr(response, "completion_text", "") or "").strip()
            generation = event.get_extra(_GENERATION_KEY)
            payload = {
                "umo": umo,
                "robot_id": self._robot_key_for_event(event),
                "cache_scope": self._cache_scope(event),
                "user_text": user_text,
                "surface": surface,
                "transitions": transitions,
                "reply_text": reply_text,
                "generation": self._bank.generation
                if generation is None
                else generation,
            }
            if self.cfg.learning.async_learn:
                self._spawn(self._learn(**payload))
            else:
                await self._learn(**payload)
        except Exception:
            self._stats["failures"] += 1
            logger.warning("[JitRL] episode capture skipped", exc_info=True)

    async def _learn(
        self,
        *,
        umo: str,
        user_text: str,
        surface: str,
        transitions: list[Any],
        reply_text: str,
        generation: int | None = None,
        cache_scope: str = "",
        robot_id: str = "",
    ) -> None:
        """Score a trajectory, build an episode and persist it.

        Args:
            umo: Unified message origin acting as the memory scope.
            user_text: Redacted user request.
            surface: Coarse retrieval partition.
            transitions: Extracted `(state, action, observation)` sequence.
            reply_text: Final assistant reply, used for success detection.
            generation: Memory epoch in which the request started.
        """
        if robot_id:
            await self._ensure_ready(robot_id=robot_id)
        if self._bank is None:
            return
        if generation is None:
            generation = self._bank.generation
        if generation != self._bank.generation:
            return
        actions = [transition.action.text for transition in transitions]
        observations = [transition.observation for transition in transitions]
        outcomes = [transition.outcome for transition in transitions]

        success: bool | None = None
        if self.cfg.learning.success_detection != "off":
            success = infer_success(
                transitions=transitions,
                final_text=reply_text,
                user_text=user_text,
            )

        async def _generate(*, prompt: str, system_prompt: str | None = None) -> str:
            return await self._llm_text(
                prompt=prompt,
                system_prompt=system_prompt,
                provider_id=self.cfg.learning.evaluator_provider_id,
                umo=umo,
            )

        evaluator = StepRewardEvaluator(
            _generate if self.cfg.learning.evaluator_mode in {"auto", "llm"} else None,
            mode=self.cfg.learning.evaluator_mode,
            timeout=self.cfg.learning.evaluator_timeout,
            max_reward=3.0,
            heuristic_weight=self.cfg.learning.heuristic_weight,
        )
        result = await evaluator.score(
            task=user_text,
            actions=actions,
            observations=observations,
            outcomes=outcomes,
            success=success,
        )
        episode = build_episode(
            episode_id=uuid.uuid4().hex,
            scope=umo,
            user_text=user_text,
            transitions=transitions,
            rewards=result.rewards,
            gamma=self.cfg.learning.gamma,
            success=success,
            reward_source=result.source,
            task_family=surface,
            meta={
                "gamma": self.cfg.learning.gamma,
                "apply_mode": self.cfg.apply_mode,
                "evaluator_ok": result.ok,
                "overall": result.overall[:400],
                "reply_chars": len(reply_text),
            },
        )
        written = await self._bank.add_episode(episode, expected_generation=generation)
        if not written or generation != self._bank.generation:
            return
        cache_scope = cache_scope or f"{self._active_robot_key.get()}:{umo}"
        self._stats["episodes"] += 1
        self._stats["experiences"] += written
        # Fresh evidence invalidates every cached decision (the state cache is a
        # pure function of the message, so it stays).
        self._invalidate_decisions()
        self._last_episode[cache_scope or umo] = (episode.episode_id, time.time())
        if len(self._last_episode) > 512:
            cutoff = time.time() - _LAST_EPISODE_TTL
            self._last_episode = {
                key: value
                for key, value in self._last_episode.items()
                if value[1] >= cutoff
            }
        logger.debug(
            "[JitRL] learned episode %s: %d experiences, source=%s",
            episode.episode_id,
            written,
            result.source,
        )

    # ------------------------------------------------------------------
    # agent tools
    # ------------------------------------------------------------------
    @filter.llm_tool(name="jitrl_recall")
    async def jitrl_recall(self, event: AstrMessageEvent, query: str) -> str:
        """Look up how this bot handled comparable situations in the past.

        Use this when the user asks something you may have handled before, or
        when you are unsure which tool or strategy worked well in this
        deployment. The report ranks past actions by their estimated
        advantage; treat it as weak evidence rather than an instruction.

        Args:
            query(string): The situation to look up, in natural language. Pass
                the user's request or a short description of the current state.
        """
        if not self.cfg.enabled or not self.cfg.commands.enable_agent_tools:
            return "Experience memory is disabled."
        if self.cfg.neural.mode != "off":
            return "Neural Lens uses private decision snapshots; legacy recall is disabled in neural mode."
        try:
            await self._ensure_ready(event)
            umo = event.unified_msg_origin
            state = abstract_state(
                task=redact(query or event.get_message_str() or "")[:_TASK_CHAR_LIMIT],
                surface=self._surface(event),
                limit=_STATE_CHAR_LIMIT,
            )
            neighbors = await self._bank.topk(
                state,
                scope=umo,
                k=self.cfg.retrieval.k,
                threshold=self.cfg.retrieval.similarity_threshold,
                surface_gate=self.cfg.retrieval.surface_gate,
                prefilter_limit=self.cfg.retrieval.prefilter_limit,
                recency_half_life_days=self.cfg.retrieval.recency_half_life_days,
                scope_mode=self.cfg.retrieval.scope_isolation,
            )
            estimate = estimate_values(
                neighbors,
                sorted({neighbor.action_key for neighbor in neighbors}),
                k=self.cfg.retrieval.k,
                ucb_alpha=self.cfg.policy.ucb_alpha,
                exploration_rate=self.cfg.policy.exploration_rate,
                normalize=self.cfg.policy.normalize_advantage,
                min_count=self.cfg.retrieval.min_count,
                rng=self._rng,
                weighting=self.cfg.retrieval.estimator_weighting,
                exploration_estimate=self.cfg.policy.exploration_estimate,
            )
            return render_recall_report(state.text, estimate, neighbors=neighbors)
        except Exception:
            logger.warning("[JitRL] recall tool failed", exc_info=True)
            return "Experience memory is temporarily unavailable."

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------
    def _command_denied(self, event: AstrMessageEvent) -> str | None:
        """Return a refusal message when the sender may not use commands.

        Args:
            event: The command event.

        Returns:
            A user-facing refusal string, or `None` when access is granted.
        """
        if not self.cfg.commands.enabled:
            return "JitRL 指令已被配置禁用。"
        if self.cfg.commands.admin_only and not event.is_admin():
            return "JitRL 指令仅管理员可用。"
        return None

    @filter.command_group("jitrl")
    def jitrl_group(self) -> None:
        """JitRL 持续学习：查看与管理 Agent 的经验记忆。"""
        pass

    @jitrl_group.command("status")
    async def jitrl_status(self, event: AstrMessageEvent) -> Any:
        """查看 JitRL 运行状态与记忆概况。"""
        denied = self._command_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        try:
            await self._ensure_ready(event)
            umo = event.unified_msg_origin
            counts = await self._bank.counts()
            scoped = await self._bank.count(umo)
            stats = self._stats
            cached = self._last_decision.get(self._cache_scope(event))
            lines = [
                "JitRL 持续学习状态",
                f"· 开关：{('开' if self.cfg.enabled else '关')} / 应用模式：{self.cfg.apply_mode}",
                f"· 记忆：全局 {counts['experiences']} 条 / {counts['episodes']} 个 Episode，本会话 {scoped} 条",
                f"· 本次运行学习累计：episodes={stats['episodes']}，experiences={stats['experiences']}",
                f"· 干预：requests={stats['requests']}，biased={stats['biased']}，failures={stats['failures']}",
                f"· 超参：k={self.cfg.retrieval.k} r={self.cfg.retrieval.similarity_threshold}"
                f" gamma={self.cfg.learning.gamma} beta={self.cfg.policy.beta}"
                f" lambda={self.cfg.policy.exploration_rate} alpha={self.cfg.policy.ucb_alpha}",
            ]
            if cached:
                decision = cached[1]
                lines.append(f"· 最近决策：{decision.reason or '（无）'}")
            lines.append(f"· Neural Lens：{self.cfg.neural.mode}（/jitrl lens status）")
            if self.cfg.neural.mode != "off":
                lines.append(
                    "· 上述记忆与检查点统计属于 V0.1 经验库；神经库与权重独立管理。"
                )
            yield event.plain_result("\n".join(lines))
        except Exception:
            logger.warning("[JitRL] status command failed", exc_info=True)
            yield event.plain_result("读取状态失败，请查看 AstrBot 日志。")

    @jitrl_group.command("lens")
    async def jitrl_lens(
        self, event: AstrMessageEvent, action: str = "status", confirm: str = ""
    ) -> Any:
        """神经滤镜：status / train / reset confirm；均限当前会话。"""
        denied = self._command_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        action = action.strip().lower()
        if action not in {"status", "train", "reset"}:
            yield event.plain_result(
                "用法：/jitrl lens <status|train|reset>；重置需追加 confirm。"
            )
            return
        if action != "status" and not event.is_admin():
            yield event.plain_result("神经模型训练与重置仅管理员可用。")
            return
        if action == "reset" and confirm.strip().lower() != "confirm":
            yield event.plain_result(
                "将删除当前会话的神经经验、审计快照及全部本地模型权重。确认：/jitrl lens reset confirm"
            )
            return
        try:
            runtime = await self._ensure_neural(event)
            scope = event.unified_msg_origin
            cache_scope = self._cache_scope(event)
            if action == "train":
                result = await runtime.train(scope)
                if result["status"] == "trained":
                    yield event.plain_result(
                        f"本地训练完成：{result['checkpoint_id']}；样本 {result['samples']}，loss {result['before_loss']:.4f} → {result['after_loss']:.4f}。该训练损失不代表真实任务提升。"
                    )
                else:
                    yield event.plain_result(f"本地训练未发布模型：{result}")
            elif action == "reset":
                await runtime.reset(scope)
                self._neural_last_episode.pop(cache_scope, None)
                yield event.plain_result(
                    "已重置当前会话的神经经验和权重；旧后台写入已失效。V0.1 经验库独立保留。"
                )
            else:
                status = await runtime.status(scope)
                yield event.plain_result(
                    f"Neural Lens：{status['mode']}；PyTorch：{status['torch_installed']}\n"
                    f"主干配置：{status.get('attention_backend', self.cfg.neural.attention_backend)}；MiTA {status.get('mita_layers', 0)}/{status.get('attention_layers', 0)} 层；执行 {status.get('mita_execution', self.cfg.neural.mita_execution)}；设备 {status.get('device', self.cfg.neural.device)}\n"
                    f"决策 {status['decisions']} / 执行事实 {status['executions']} / 回报事件 {status['settlements']}\n"
                    f"检查点：{status['checkpoint_id']}；错误：{status['last_error'] or '无'}\n"
                    "AstrBot 已接入 PRE；POST 为独立候选评分 API。\n"
                    "训练：/jitrl lens train；神经库重置：/jitrl lens reset confirm"
                )
        except Exception:
            logger.warning("[JitRL Lens] command failed", exc_info=True)
            yield event.plain_result("神经滤镜操作失败，请查看 AstrBot 日志。")

    @jitrl_group.command("stats")
    async def jitrl_stats(self, event: AstrMessageEvent, limit: int = 10) -> Any:
        """查看各动作的平均回报与正回报比例。

        Args:
            limit(int): 展示条数，默认 10。
        """
        denied = self._command_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        try:
            await self._ensure_ready(event)
            limit = max(1, min(limit, 50))
            rows = await self._bank.action_stats(limit=limit, min_count=1)
            if not rows:
                yield event.plain_result("还没有任何经验记录。先聊几轮吧。")
                return
            lines = ["JitRL 动作统计（按平均回报降序）"]
            for row in rows:
                lines.append(
                    f"· {row['action_text'][:70]} | n={row['count']}"
                    f" | 平均回报={row['mean_return']:+.2f}"
                    f" | 正回报率={row['positive_rate'] * 100:.0f}%"
                )
            yield event.plain_result("\n".join(lines))
        except Exception:
            logger.warning("[JitRL] stats command failed", exc_info=True)
            yield event.plain_result("读取统计失败，请查看 AstrBot 日志。")

    @jitrl_group.command("curve")
    async def jitrl_curve(self, event: AstrMessageEvent, buckets: int = 8) -> Any:
        """查看学习曲线（按时间分桶的平均回报）。

        Args:
            buckets(int): 时间桶数量，默认 8。
        """
        denied = self._command_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        try:
            await self._ensure_ready(event)
            buckets = max(2, min(buckets, 30))
            rows = await self._bank.learning_curve(buckets=buckets)
            if not rows:
                yield event.plain_result("还没有足够的经验绘制学习曲线。")
                return
            peak = max((abs(row["mean_return"]) for row in rows), default=1.0) or 1.0
            lines = ["JitRL 学习曲线（平均回报，越往右越新）"]
            for row in rows:
                width = int(round(abs(row["mean_return"]) / peak * 20))
                bar = ("+" if row["mean_return"] >= 0 else "-") * max(1, width)
                lines.append(
                    f"{row['bucket']:>3} | {bar:<20} {row['mean_return']:+.2f}"
                )
            yield event.plain_result("\n".join(lines))
        except Exception:
            logger.warning("[JitRL] curve command failed", exc_info=True)
            yield event.plain_result("读取学习曲线失败，请查看 AstrBot 日志。")

    @jitrl_group.command("recall")
    async def jitrl_recall_cmd(self, event: AstrMessageEvent, query: str = "") -> Any:
        """按当前（或指定）情景检索相关的历史经验。

        Args:
            query(string): 要检索的情景描述，留空则使用当前消息。
        """
        denied = self._command_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        try:
            report = await self.jitrl_recall(event, query)
            yield event.plain_result(report[:1800])
        except Exception:
            logger.warning("[JitRL] recall command failed", exc_info=True)
            yield event.plain_result("检索失败，请查看 AstrBot 日志。")

    @jitrl_group.command("forget")
    async def jitrl_forget(
        self, event: AstrMessageEvent, scope: str = "session"
    ) -> Any:
        """删除记忆。scope 可选 session（本会话）/ all（全部）。

        Args:
            scope(string): session 表示仅删除本会话记忆，all 表示清空全部。
        """
        denied = self._command_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        try:
            await self._ensure_ready(event)
            scope = scope.strip().lower()
            if scope not in {"all", "session"}:
                yield event.plain_result(
                    "scope 必须是 session 或 all，未删除任何记忆。"
                )
                return
            if scope == "all":
                removed = await self._bank.clear()
            else:
                removed = await self._bank.forget(scope=event.unified_msg_origin)
            self._invalidate_memory_cache()
            yield event.plain_result(
                f"已删除 {removed} 条 V0.1 经验。删除前已自动备份，可用 /jitrl checkpoints 查看并回退。\n"
                "神经经验与权重独立管理：/jitrl lens reset confirm。"
            )
        except Exception:
            logger.warning("[JitRL] forget command failed", exc_info=True)
            yield event.plain_result("删除失败，请查看 AstrBot 日志。")

    @jitrl_group.command("feedback")
    async def jitrl_feedback(
        self, event: AstrMessageEvent, verdict: str = "good"
    ) -> Any:
        """手动回填对上一轮回复的评价：good 或 bad。

        Args:
            verdict(string): good 表示肯定，bad 表示否定。
        """
        denied = self._command_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        try:
            if self.cfg.neural.mode != "off":
                cache_scope = self._cache_scope(event)
                await self._ensure_neural(event)
                delta = (
                    -_FEEDBACK_DELTA
                    if verdict.strip().lower() == "bad"
                    else _FEEDBACK_DELTA
                )
                updated = await self._apply_neural_feedback(cache_scope, delta)
                yield event.plain_result(
                    f"已回填 {delta:+.1f} 用户反馈到上一轮神经经验的 {updated} 个步骤（追加结算，保留原始记录）。"
                )
                return
            await self._ensure_ready(event)
            cache_scope = self._cache_scope(event)
            cached = self._last_episode.get(cache_scope)
            if not cached:
                yield event.plain_result("没有可回填的上一轮经验。")
                return
            delta = (
                _FEEDBACK_DELTA
                if verdict.strip().lower() != "bad"
                else -_FEEDBACK_DELTA
            )
            updated = await self._bank.apply_user_feedback(
                cached[0], delta, expected_generation=self._bank.generation
            )
            self._last_episode.pop(cache_scope, None)
            yield event.plain_result(
                f"已回填 {delta:+.1f} 奖励到上一轮经验的 {updated} 个步骤。"
            )
        except Exception:
            logger.warning("[JitRL] feedback command failed", exc_info=True)
            yield event.plain_result("回填失败，请查看 AstrBot 日志。")

    @jitrl_group.command("export")
    async def jitrl_export(self, event: AstrMessageEvent) -> Any:
        """导出记忆为 JSON 文件（路径会回复给管理员）。"""
        denied = self._command_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        try:
            await self._ensure_ready(event)
            payload = await self._bank.export_json()
            target = await asyncio.to_thread(
                self._write_export, payload, f"export-{int(time.time())}.json"
            )
            logger.info("[JitRL] export written to %s", target)
            yield event.plain_result(
                f"已导出 {len(payload.get('experiences', []))} 条经验到插件数据目录"
                f"（文件名 {target.name}，保留最近 {_EXPORT_KEEP} 份）。"
            )
        except Exception:
            logger.warning("[JitRL] export command failed", exc_info=True)
            yield event.plain_result("导出失败，请查看 AstrBot 日志。")

    @jitrl_group.command("episodes")
    async def jitrl_episodes(self, event: AstrMessageEvent, limit: int = 5) -> Any:
        """查看最近的 episode 摘要。

        Args:
            limit(int): 展示条数，默认 5。
        """
        denied = self._command_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        try:
            await self._ensure_ready(event)
            limit = max(1, min(limit, 20))
            rows = await self._bank.recent_episodes(limit=limit)
            if not rows:
                yield event.plain_result("还没有 episode 记录。")
                return
            lines = ["最近的 JitRL episodes"]
            for row in rows:
                ts = time.strftime(
                    "%m-%d %H:%M", time.localtime(row.get("created_at", 0))
                )
                lines.append(
                    f"· {ts} | steps={row.get('step_count', 0)}"
                    f" | 平均回报={row.get('mean_return', 0.0):+.2f}"
                    f" | {str(row.get('user_text', ''))[:48]}"
                )
            yield event.plain_result("\n".join(lines))
        except Exception:
            logger.warning("[JitRL] episodes command failed", exc_info=True)
            yield event.plain_result("读取 episode 失败，请查看 AstrBot 日志。")

    # ------------------------------------------------------------------
    # exports
    # ------------------------------------------------------------------
    def _write_export(self, payload: dict[str, Any], name: str) -> Path:
        """Write one export file and enforce the retention window.

        Args:
            payload: Serialised memory bank.
            name: File name inside the plugin data directory.

        Returns:
            The path of the written file.
        """
        import json

        target = self._db_path.parent / name
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)
        exports = sorted(
            self._db_path.parent.glob("export-*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for stale in exports[_EXPORT_KEEP:]:
            try:
                stale.unlink()
            except OSError:
                logger.debug("[JitRL] cannot remove stale export %s", stale)
        return target

    # ------------------------------------------------------------------
    # checkpoints
    # ------------------------------------------------------------------
    def _invalidate_memory_cache(self) -> None:
        robot_key = self._active_robot_key.get()
        if robot_key is None:
            self._last_episode.clear()
            self._last_decision.clear()
            self._invalidate_decisions()
            return
        prefix = f"{robot_key}:"
        self._last_episode = {
            key: value
            for key, value in self._last_episode.items()
            if not key.startswith(prefix)
        }
        self._last_decision = {
            key: value
            for key, value in self._last_decision.items()
            if not key.startswith(prefix)
        }
        # Decisions embed advantage estimates and the rendered block, so they
        # must never outlive the memory they were computed from.
        self._invalidate_decisions()

    def _invalidate_decisions(self) -> None:
        robot_key = self._active_robot_key.get()
        if robot_key is None:
            self._decision_cache.clear()
            return
        prefix = f"{robot_key}:"
        self._decision_cache = OrderedDict(
            (key, value)
            for key, value in self._decision_cache.items()
            if not str(key[0]).startswith(prefix)
        )

    def _checkpoint_denied(self, event: AstrMessageEvent) -> str | None:
        denied = self._command_denied(event)
        if denied:
            return denied
        # A snapshot contains every session, even when rolling back one scope.
        return None if event.is_admin() else "检查点管理仅管理员可用。"

    @jitrl_group.command("checkpoint")
    async def jitrl_checkpoint(self, event: AstrMessageEvent, name: str = "") -> Any:
        """创建手动检查点（可选名称，最多 80 字），不会被自动轮换。"""
        denied = self._checkpoint_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        try:
            await self._ensure_ready(event)
            info = await self._bank.create_checkpoint(name)
            yield event.plain_result(
                f"已创建检查点：{info['id']}\n"
                f"名称：{info['name'] or '未命名'} · {info['experiences']} 条经验 / {info['episodes']} 个 episode"
            )
        except ValueError as exc:
            yield event.plain_result(str(exc))
        except Exception:
            logger.warning("[JitRL] checkpoint command failed", exc_info=True)
            yield event.plain_result("创建检查点失败，请查看 AstrBot 日志。")

    @jitrl_group.command("checkpoints")
    async def jitrl_checkpoints(self, event: AstrMessageEvent, limit: int = 10) -> Any:
        """查看最近的检查点及其完整 ID。"""
        denied = self._checkpoint_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        try:
            await self._ensure_ready(event)
            rows = await self._bank.list_checkpoints()
            lines = ["JitRL 检查点（从新到旧；回退前会自动备份当前数据）"]
            for row in rows[: max(1, min(limit, 50))]:
                ts = time.strftime("%m-%d %H:%M", time.localtime(row["created_at"]))
                lines.append(
                    f"· {row['id']}\n  {ts} | {row['kind']} | {row['name'] or '未命名'} | {row['experiences']} 条经验"
                )
            if not rows:
                lines.append("还没有检查点，可用 /jitrl checkpoint 名称 创建。")
            yield event.plain_result("\n".join(lines))
        except Exception:
            logger.warning("[JitRL] checkpoints command failed", exc_info=True)
            yield event.plain_result("读取检查点失败，请查看 AstrBot 日志。")

    @jitrl_group.command("rollback")
    async def jitrl_rollback(
        self,
        event: AstrMessageEvent,
        checkpoint_id: str = "",
        scope: str = "session",
        confirm: str = "",
    ) -> Any:
        """回退检查点：ID、session/all、confirm；默认只回退当前会话。"""
        denied = self._checkpoint_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        scope = scope.strip().lower()
        if not checkpoint_id or scope not in {"session", "all"}:
            yield event.plain_result(
                "用法：/jitrl rollback <完整检查点ID> <session|all> confirm"
            )
            return
        if confirm.strip().lower() != "confirm":
            yield event.plain_result(
                f"将替换{'全部会话' if scope == 'all' else '当前会话'}的记忆，回退前会自动备份。确认请发送：\n"
                f"/jitrl rollback {checkpoint_id} {scope} confirm"
            )
            return
        try:
            await self._ensure_ready(event)
            result = await self._bank.restore_checkpoint(
                checkpoint_id,
                scope=None if scope == "all" else event.unified_msg_origin,
            )
            self._invalidate_memory_cache()
            yield event.plain_result(
                f"已回退{'全部会话' if scope == 'all' else '当前会话'}：{result['experiences']} 条经验。\n"
                f"撤销本次回退可恢复检查点：{result['safety_checkpoint']['id']}\n"
                "此检查点只含 V0.1 经验库，不含神经经验或模型权重。"
            )
        except (ValueError, FileNotFoundError) as exc:
            yield event.plain_result(str(exc))
        except Exception:
            logger.warning("[JitRL] rollback command failed", exc_info=True)
            yield event.plain_result(
                "回退失败，当前记忆保持不变，请查看 AstrBot 日志。"
            )

    @jitrl_group.command("checkpoint_delete")
    async def jitrl_checkpoint_delete(
        self, event: AstrMessageEvent, checkpoint_id: str = "", confirm: str = ""
    ) -> Any:
        """永久删除指定检查点：完整 ID、confirm。"""
        denied = self._checkpoint_denied(event)
        if denied:
            yield event.plain_result(denied)
            return
        if not checkpoint_id or confirm.strip().lower() != "confirm":
            yield event.plain_result(
                "删除检查点不可撤销。用法：/jitrl checkpoint_delete <完整检查点ID> confirm"
            )
            return
        try:
            await self._ensure_ready(event)
            await self._bank.delete_checkpoint(checkpoint_id)
            yield event.plain_result("已删除指定检查点，当前记忆不受影响。")
        except (ValueError, FileNotFoundError) as exc:
            yield event.plain_result(str(exc))
        except Exception:
            logger.warning("[JitRL] checkpoint delete failed", exc_info=True)
            yield event.plain_result("删除检查点失败，请查看 AstrBot 日志。")

    # ------------------------------------------------------------------
    # WebUI view API
    # ------------------------------------------------------------------
    def _register_web_apis(self) -> None:
        """Expose the dashboard endpoints used by the plugin view.

        Registration is best-effort: an older AstrBot build without plugin
        web APIs must still load the plugin, so failures are only logged.
        """
        routes = (
            ("/astrbot_plugin_jitrl/robots", self._web_robots, ["GET"], "robots"),
            ("/astrbot_plugin_jitrl/overview", self._web_overview, ["GET"], "overview"),
            ("/astrbot_plugin_jitrl/curve", self._web_curve, ["GET"], "curve"),
            ("/astrbot_plugin_jitrl/actions", self._web_actions, ["GET"], "actions"),
            ("/astrbot_plugin_jitrl/episodes", self._web_episodes, ["GET"], "episodes"),
            ("/astrbot_plugin_jitrl/forget", self._web_forget, ["POST"], "forget"),
            ("/astrbot_plugin_jitrl/export", self._web_export, ["GET"], "export"),
            (
                "/astrbot_plugin_jitrl/checkpoints",
                self._web_checkpoints,
                ["GET"],
                "checkpoints",
            ),
            (
                "/astrbot_plugin_jitrl/checkpoint",
                self._web_checkpoint,
                ["POST"],
                "checkpoint",
            ),
            (
                "/astrbot_plugin_jitrl/rollback",
                self._web_rollback,
                ["POST"],
                "rollback",
            ),
            (
                "/astrbot_plugin_jitrl/checkpoint_delete",
                self._web_checkpoint_delete,
                ["POST"],
                "checkpoint_delete",
            ),
        )
        register = getattr(self.context, "register_web_api", None)
        if register is None:
            logger.info("[JitRL] plugin web APIs are unsupported in this build")
            return
        for route, handler, methods, desc in routes:
            try:
                register(route, handler, methods, desc)
            except Exception:
                logger.warning("[JitRL] cannot register %s", route, exc_info=True)

    async def _web_robots(self) -> Any:
        from astrbot.api.web import json_response

        return json_response({"robots": self._known_robots()})

    def _select_web_robot(self, requested: str | None) -> str | None:
        robots = self._known_robots()
        if not robots:
            return None
        if requested:
            if any(robot["id"] == requested for robot in robots):
                return requested
            raise ValueError("机器人实例不存在，请刷新面板后重试。")
        if len(robots) == 1:
            return robots[0]["id"]
        raise ValueError("请选择要查看或管理的机器人实例。")

    async def _ensure_web_bank(self, requested: str | None) -> MemoryBank | None:
        robot_id = self._select_web_robot(requested)
        return await self._ensure_ready(robot_id=robot_id) if robot_id else None

    async def _web_overview(self) -> Any:
        """Return the dashboard overview payload.

        Returns:
            A JSON response with counters, hyperparameters and the last
            decision taken by this process.
        """
        from astrbot.api.web import error_response, json_response, request

        try:
            robot_id = self._select_web_robot(request.query.get("robot"))
            bank = await self._ensure_ready(robot_id=robot_id) if robot_id else None
            counts = await bank.counts() if bank else {"experiences": 0, "episodes": 0}
            runtime = self._neural_by_robot.get(robot_id) if robot_id else None
            if runtime is not None:
                neural_status = await runtime.status()
            elif self.cfg.neural.mode == "off":
                neural_status = {"mode": "off"}
            else:
                neural_status = {
                    "mode": self.cfg.neural.mode,
                    "attention_backend": self.cfg.neural.attention_backend,
                    "device": self.cfg.neural.device,
                    "mita_execution": self.cfg.neural.mita_execution,
                    "mita_layers": 0
                    if self.cfg.neural.attention_backend in {"legacy", "gqa"}
                    else self.cfg.neural.blocks
                    - int(self.cfg.neural.attention_backend == "hybrid"),
                    "attention_layers": 1
                    if self.cfg.neural.attention_backend == "legacy"
                    else self.cfg.neural.blocks,
                }
            last = None
            decisions = [
                item
                for key, item in self._last_decision.items()
                if robot_id and key.startswith(f"{robot_id}:")
            ]
            if decisions:
                _, decision = max(decisions, key=lambda item: item[0])
                last = {
                    "reason": decision.reason,
                    "chosen": decision.chosen.text if decision.chosen else "",
                    "v_hat": decision.estimate.v_hat,
                    "neighbors": decision.estimate.neighbor_count,
                    "probabilities": decision.probabilities,
                    "advantages": decision.estimate.advantage_norm,
                }
            return json_response(
                {
                    "enabled": self.cfg.enabled,
                    "neural": neural_status,
                    "apply_mode": self.cfg.apply_mode,
                    "total": counts["experiences"],
                    "episodes": counts["episodes"],
                    "stats": dict(self._stats),
                    "robot_id": robot_id or "",
                    "robots": self._known_robots(),
                    "hyper": {
                        "k": self.cfg.retrieval.k,
                        "threshold": self.cfg.retrieval.similarity_threshold,
                        "gamma": self.cfg.learning.gamma,
                        "beta": self.cfg.policy.beta,
                        "exploration_rate": self.cfg.policy.exploration_rate,
                        "ucb_alpha": self.cfg.policy.ucb_alpha,
                        "scope": self.cfg.retrieval.scope_isolation,
                        "evaluator": self.cfg.learning.evaluator_mode,
                        "weighting": self.cfg.retrieval.estimator_weighting,
                        "exploration": self.cfg.policy.exploration_estimate,
                    },
                    "last_decision": last,
                    "db_path": str(self._db_path) if bank else "",
                }
            )
        except ValueError as exc:
            return error_response(str(exc))
        except Exception:
            logger.warning("[JitRL] overview endpoint failed", exc_info=True)
            return error_response("JitRL overview unavailable", status_code=500)

    async def _web_curve(self) -> Any:
        """Return the learning-curve buckets.

        Returns:
            A JSON response whose "buckets" list holds time-ordered averages.
        """
        from astrbot.api.web import error_response, json_response, request

        try:
            bank = await self._ensure_web_bank(request.query.get("robot"))
            buckets = request.query.get("buckets", 12, type=int)
            rows = (
                await bank.learning_curve(buckets=max(2, min(int(buckets), 60)))
                if bank
                else []
            )
            return json_response({"buckets": rows})
        except ValueError as exc:
            return error_response(str(exc))
        except Exception:
            logger.warning("[JitRL] curve endpoint failed", exc_info=True)
            return error_response("JitRL curve unavailable", status_code=500)

    async def _web_actions(self) -> Any:
        """Return per-action aggregates.

        Returns:
            A JSON response whose "actions" list holds action statistics.
        """
        from astrbot.api.web import error_response, json_response, request

        try:
            bank = await self._ensure_web_bank(request.query.get("robot"))
            limit = request.query.get("limit", 20, type=int)
            rows = (
                await bank.action_stats(limit=max(1, min(int(limit), 200)))
                if bank
                else []
            )
            return json_response({"actions": rows})
        except ValueError as exc:
            return error_response(str(exc))
        except Exception:
            logger.warning("[JitRL] actions endpoint failed", exc_info=True)
            return error_response("JitRL actions unavailable", status_code=500)

    async def _web_episodes(self) -> Any:
        """Return recent episode headers.

        Returns:
            A JSON response whose "episodes" list holds episode summaries.
        """
        from astrbot.api.web import error_response, json_response, request

        try:
            bank = await self._ensure_web_bank(request.query.get("robot"))
            limit = request.query.get("limit", 10, type=int)
            rows = (
                await bank.recent_episodes(limit=max(1, min(int(limit), 100)))
                if bank
                else []
            )
            return json_response({"episodes": rows})
        except ValueError as exc:
            return error_response(str(exc))
        except Exception:
            logger.warning("[JitRL] episodes endpoint failed", exc_info=True)
            return error_response("JitRL episodes unavailable", status_code=500)

    async def _web_checkpoints(self) -> Any:
        """Return snapshot inventory and rotation settings for the dashboard."""
        from astrbot.api.web import error_response, json_response, request

        try:
            bank = await self._ensure_web_bank(request.query.get("robot"))
            return json_response(
                {
                    "checkpoints": await bank.list_checkpoints() if bank else [],
                    "interval": self.cfg.memory.checkpoint_interval,
                    "keep": self.cfg.memory.keep_checkpoints,
                }
            )
        except ValueError as exc:
            return error_response(str(exc))
        except Exception:
            logger.warning("[JitRL] checkpoints endpoint failed", exc_info=True)
            return error_response("读取检查点失败。", status_code=500)

    async def _web_checkpoint(self) -> Any:
        return await self._web_checkpoint_action("create")

    async def _web_rollback(self) -> Any:
        return await self._web_checkpoint_action("restore")

    async def _web_checkpoint_delete(self) -> Any:
        return await self._web_checkpoint_action("delete")

    async def _web_checkpoint_action(self, action: str) -> Any:
        """Handle checkpoint mutations behind AstrBot's authenticated WebUI."""
        from astrbot.api.web import error_response, json_response, request

        try:
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                return error_response("请求体必须是 JSON 对象。")
            if action != "create" and payload.get("confirm") is not True:
                return error_response("回退或删除检查点需要 confirm: true。")
            scope = payload.get("scope", "all")
            if action == "restore":
                if scope not in ("all", "session"):
                    return error_response("scope 必须是 all 或 session。")
                if scope == "session" and (
                    not isinstance(payload.get("umo"), str)
                    or not payload["umo"].strip()
                ):
                    return error_response("会话回退必须提供非空的 umo。")
            bank = await self._ensure_web_bank(payload.get("robot"))
            if bank is None:
                return error_response("当前没有机器人实例可管理。")
            if action == "create":
                return json_response(
                    {
                        "checkpoint": await bank.create_checkpoint(
                            payload.get("name", "")
                        )
                    }
                )
            checkpoint_id = payload.get("id", "")
            if action == "restore":
                result = await bank.restore_checkpoint(
                    checkpoint_id, scope=None if scope == "all" else payload["umo"]
                )
                self._invalidate_memory_cache()
                return json_response(result)
            await bank.delete_checkpoint(checkpoint_id)
            return json_response({"deleted": checkpoint_id})
        except FileNotFoundError as exc:
            return error_response(str(exc), status_code=404)
        except ValueError as exc:
            return error_response(str(exc))
        except Exception:
            logger.warning("[JitRL] checkpoint %s failed", action, exc_info=True)
            return error_response(
                "检查点操作失败，请查看 AstrBot 日志。", status_code=500
            )

    async def _web_forget(self) -> Any:
        """Delete memory from the dashboard.

        Returns:
            A JSON response reporting how many experiences were removed.
        """
        from astrbot.api.web import error_response, json_response, request

        try:
            payload = await request.json(default={})
            if not isinstance(payload, dict):
                return error_response("请求体必须是 JSON 对象。")
            if payload.get("confirm") is not True:
                # Same rule as the checkpoint routes: a destructive call must
                # carry an explicit confirmation, never rely on the UI alone.
                return error_response("清空记忆需要 confirm: true。")
            bank = await self._ensure_web_bank(payload.get("robot"))
            if bank is None:
                return error_response("当前没有机器人实例可管理。")
            scope = str(payload.get("scope", "all")).strip().lower()
            if scope == "all":
                removed = await bank.clear()
            elif scope == "session" and payload.get("umo"):
                removed = await bank.forget(scope=str(payload["umo"]))
            else:
                return error_response("scope must be all, or session with umo")
            self._invalidate_memory_cache()
            return json_response({"removed": removed})
        except ValueError as exc:
            return error_response(str(exc))
        except Exception:
            logger.warning("[JitRL] forget endpoint failed", exc_info=True)
            return error_response("JitRL forget failed", status_code=500)

    async def _web_export(self) -> Any:
        """Export the memory bank as a downloadable JSON file.

        Returns:
            A file response holding the serialized memory bank.
        """
        import json

        from astrbot.api.web import error_response, file_response, request

        try:
            bank = await self._ensure_web_bank(request.query.get("robot"))
            if bank is None:
                return error_response("当前没有机器人实例可导出。")
            payload = await bank.export_json()
            target = self._db_path.parent / "jitrl-export.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return file_response(
                target,
                filename="jitrl-export.json",
                content_type="application/json",
            )
        except ValueError as exc:
            return error_response(str(exc))
        except Exception:
            logger.warning("[JitRL] export endpoint failed", exc_info=True)
            return error_response("JitRL export failed", status_code=500)
