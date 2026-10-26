#!/usr/bin/env python
"""Reproducible benchmark harness for the JitRL retrieval and estimation path.

The plugin's hot path runs on every LLM request: `on_llm_request` abstracts the
current state, retrieves the top-k neighbours from SQLite (Eq. 24 Jaccard
similarity), estimates V/Q/advantages and renders the preference block. This
script replays exactly that path against a deterministic synthetic workload, so
a retrieval/estimator change can be compared against a recorded baseline
without a live AstrBot deployment.

Design constraints:

* standard library plus this repository's own modules - no new dependencies.
  AstrBot itself is not required: a minimal `astrbot.api` logger stub is
  installed before the plugin package is imported, exactly like
  `tests/conftest.py` does for plain CI.
* the workload is generated from a fixed seed, so two runs of the same harness
  version produce the same vocabulary, episodes, queries and timestamps.
* every scenario runs against a throwaway SQLite database in the system temp
  directory, which is removed afterwards. Nothing is written inside the repo
  except the `--json` artefact.
* the measured API is the real async one (`MemoryBank.topk` and
  `add_episode`), so the `asyncio.to_thread` hop is part of the number.

Usage::

    cd <repo root>
    python scripts/bench_jitrl.py --json benchmarks/baseline.json
    python scripts/bench_jitrl.py --quick

The scenario parameters are frozen constants in this file (`_K`, `_THRESHOLD`,
...), not read from `jitrl.config`: the benchmark definition must not drift when
the plugin defaults are tuned, otherwise "before" and "after" runs could not be
compared. The shipped defaults in effect at run time are still recorded under
`meta.shipped_defaults` for reference.

Progress goes to stderr; the summary table goes to stdout when `--json` is
given, otherwise the JSON document is printed to stdout for piping.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import math
import os
import platform
import random
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_astrbot_stub() -> None:
    """Provide `astrbot.api.logger` when the host is not installed.

    `jitrl.memory` and `jitrl.checkpoints` import their logger from
    `astrbot.api` because AstrBot's plugin review forbids the standard library
    `logging` module in plugin code. Plain CI has no AstrBot, so the same stub
    the test bootstrap installs keeps the package importable.
    """
    import importlib.util
    import logging

    if importlib.util.find_spec("astrbot") is not None:
        return
    host = ModuleType("astrbot")
    host.__path__ = []  # type: ignore[attr-defined]
    host.__jitrl_stub__ = True  # type: ignore[attr-defined]
    api = ModuleType("astrbot.api")
    api.logger = logging.getLogger("astrbot")
    host.api = api
    sys.modules["astrbot"] = host
    sys.modules["astrbot.api"] = api


_install_astrbot_stub()

from jitrl.estimator import estimate_values  # noqa: E402
from jitrl.hint import render_preference_block  # noqa: E402
from jitrl.memory import MemoryBank  # noqa: E402
from jitrl.models import Episode, Neighbor, StateView, StepRecord  # noqa: E402
from jitrl.policy import build_candidates, select_action  # noqa: E402
from jitrl.state import (  # noqa: E402
    MAX_STATE_CHARS,
    abstract_state,
    normalize_action,
    tokenize,
)

_HARNESS_VERSION = 1
"""Bump when the workload or the scenario set changes; comparable runs share it."""

# --- frozen scenario parameters (never read from jitrl.config) --------------
_K = 10
_THRESHOLD = 0.8
_THRESHOLD_LOOSE = 0.5
_SURFACE_GATE = False
_SCOPE_MODE = "hybrid"
_RECENCY_HALF_LIFE_DAYS = 30.0
_PREFILTER_LIMIT = 4000
_CANDIDATE_LIMIT = 3
_UCB_ALPHA = 5.0
_EXPLORATION_RATE = 0.3
_BETA = 1.5
_SAMPLE_TEMPERATURE = 1.0
_MAX_HINT_CHARS = 900

_MAX_ENTRIES = 20000
"""Configured cap of the experience bank (jitrl.config.MemoryConfig default)."""

_CHECKPOINT_INTERVAL = 50
"""Production snapshot interval, used by the durability scenario."""

_KEEP_CHECKPOINTS = 20
_PRUNE_STRATEGY = "low_value"
_PRUNE_KEEP_EPISODES = 500

# --- workload parameters ----------------------------------------------------
_SEED = 20260227
_TOKENS_LO = 30
_TOKENS_HI = 120
_REFERENCE_NOW = 1_782_000_000.0
"""Fixed reference instant for every recency computation (2026-06-23 UTC)."""

_TIMESTAMP_SPAN_DAYS = 180.0
_TIMESTAMP_JITTER_DAYS = 0.5
_GAMMA = 0.5
_SINGLE_STEP_SHARE = 0.55
_MAX_EPISODE_STEPS = 4
_VOCAB_SIZE = 6000
_CJK_POOL_SIZE = 900
_CJK_BASE = 0x4E00
_CJK_RANGE = 0x5000
_ZIPF_EXPONENT = 1.05
_CJK_RATIO = 0.35
_QUERY_DROP_RATE = 0.06
_WARMUP_FRACTION = 0.1

_SCOPES = (
    "umo:aiocqhttp:GroupMessage:10001",
    "umo:aiocqhttp:GroupMessage:10002",
    "umo:telegram:PrivateMessage:2001",
    "umo:discord:GroupMessage:3001",
    "umo:lark:GroupMessage:4001",
    "umo:webchat:PrivateMessage:5001",
)
_SURFACES = ("qq", "telegram", "discord", "lark")

_REAL_WORDS = (
    "search",
    "answer",
    "deploy",
    "config",
    "token",
    "query",
    "browser",
    "click",
    "submit",
    "retry",
    "schedule",
    "parse",
    "render",
    "cache",
    "metric",
    "alert",
    "server",
    "client",
    "widget",
    "calendar",
    "invoice",
    "ticket",
    "thread",
    "message",
    "upload",
    "download",
    "profile",
    "session",
    "cookie",
    "schema",
    "migrate",
    "docker",
    "commit",
    "branch",
    "review",
    "merge",
    "release",
    "patch",
    "debug",
    "trace",
    "latency",
    "budget",
    "policy",
    "reward",
    "memory",
    "action",
    "agent",
    "tool",
    "plugin",
    "sandbox",
    "rollback",
    "snapshot",
    "backup",
    "restore",
    "index",
    "shard",
    "queue",
    "worker",
    "cron",
    "webhook",
    "payload",
    "header",
    "cookie",
    "credential",
    "tenant",
    "billing",
    "refund",
    "order",
    "cart",
    "inventory",
    "warehouse",
    "shipment",
    "address",
    "verify",
    "expire",
    "rotate",
    "encrypt",
    "decrypt",
    "digest",
    "checksum",
    "benchmark",
    "profil",
    "garbage",
    "collect",
    "threadpool",
    "stream",
    "buffer",
    "offset",
    "cursor",
    "partition",
    "cluster",
    "replica",
    "failover",
    "throttle",
    "timeout",
    "handshake",
    "upgrade",
    "install",
    "vendor",
    "manifest",
    "directory",
    "symlink",
    "inode",
    "daemon",
    "socket",
    "protocol",
    "gateway",
    "tunnel",
    "proxy",
    "certificate",
    "fingerprint",
    "signature",
)

_SYLLABLES = (
    "ka",
    "re",
    "mon",
    "ta",
    "shi",
    "vor",
    "lin",
    "dex",
    "qua",
    "bex",
    "po",
    "tor",
    "mun",
    "sa",
    "vel",
    "dri",
    "fen",
    "gal",
    "hu",
    "is",
    "jun",
    "kel",
    "lo",
    "mar",
    "nin",
    "or",
    "pil",
    "ran",
    "sol",
    "tun",
)

_REAL_TOOLS = (
    "web_search",
    "browser_open",
    "file_read",
    "file_write",
    "shell_exec",
    "send_email",
    "calendar_create",
    "db_query",
    "http_get",
    "translate",
    "summarize",
    "image_gen",
    "code_run",
    "git_status",
    "issue_create",
    "repo_clone",
    "doc_render",
    "pdf_extract",
    "chart_plot",
    "notify_push",
)

_TOOL_PREFIXES = (
    "plugin",
    "agent",
    "repo",
    "db",
    "mail",
    "cal",
    "img",
    "doc",
    "net",
    "ops",
)
_TOOL_SUFFIXES = (
    "_sync",
    "_query",
    "_apply",
    "_fetch",
    "_render",
    "_publish",
    "_audit",
)
_COMMANDS = (
    "help",
    "status",
    "recall",
    "reset",
    "forget",
    "export",
    "import",
    "checkpoint",
    "stats",
    "curve",
)
_ARG_KEYS = (
    "query",
    "target",
    "path",
    "mode",
    "lang",
    "format",
    "channel",
    "limit",
    "filter",
    "source",
    "dest",
    "scope",
)
_ARG_VALUES = (
    "zh",
    "en",
    "json",
    "fast",
    "full",
    "csv",
    "retry",
    "draft",
    "pg",
    "redis",
    "open",
    "closed",
)
_TASK_CUES = (
    "请检查 ",
    "帮我看下 ",
    "麻烦处理 ",
    "please diagnose ",
    "how do I ",
    "please fix ",
    "再试一次 ",
    "解释一下 ",
)

_GEN_BATCH = 256
_QUERY_INDEX_SALT = 10_000
"""Measured queries use a disjoint seed range from the warmup queries."""

_ESTIMATOR_RNG_SEED = 1234


# ---------------------------------------------------------------------------
# synthetic workload
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _ZipfPool:
    """Zipf-distributed sampler over a fixed, seeded item list.

    Attributes:
        items: Candidate strings, most frequent first.
        cumulative: Running sum of 1/(rank+1)**exponent, used by the sampler.
    """

    items: list[str]
    cumulative: list[float]

    @classmethod
    def build(cls, items: list[str], exponent: float) -> _ZipfPool:
        """Build a sampler whose weights follow a Zipf law.

        Args:
            items: Candidate strings ordered from most to least frequent.
            exponent: Zipf exponent; larger values concentrate mass on the head.

        Returns:
            The ready-to-sample pool.
        """
        cumulative: list[float] = []
        total = 0.0
        for rank in range(len(items)):
            total += 1.0 / (rank + 1.0) ** exponent
            cumulative.append(total)
        return cls(items=items, cumulative=cumulative)

    def sample(self, rng: random.Random) -> str:
        """Draw one item, favouring the head of the list.

        Args:
            rng: Seeded random source.

        Returns:
            The sampled string, or an empty string for an empty pool.
        """
        if not self.items:
            return ""
        target = rng.random() * self.cumulative[-1]
        index = bisect.bisect_right(self.cumulative, target)
        return self.items[min(index, len(self.items) - 1)]


def _vocabulary(rng: random.Random) -> list[str]:
    """Build the mixed technical vocabulary of the synthetic workload.

    Common English technical words occupy the head of the Zipf ranking, so the
    corpus has the realistic shape the paper's retrieval sees: a handful of
    tokens dominate and the long tail is almost never sampled.

    Args:
        rng: Seeded random source for the generated tail.

    Returns:
        The vocabulary, most frequent word first.
    """
    words = sorted(set(_REAL_WORDS))
    seen = set(words)
    # Bounded rejection sampling: the 30 syllables already provide 27900
    # distinct 2-3 syllable combinations, but a hard attempt cap keeps a future
    # vocabulary/collision change from turning this into an infinite loop.
    for _ in range(_VOCAB_SIZE * 40):
        if len(words) >= _VOCAB_SIZE:
            break
        candidate = "".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(2, 3)))
        if len(candidate) < 4 or candidate in seen:
            continue
        seen.add(candidate)
        words.append(candidate)
    index = 0
    while len(words) < _VOCAB_SIZE:
        candidate = f"{rng.choice(_SYLLABLES)}{index}"
        index += 1
        if candidate in seen:
            continue
        seen.add(candidate)
        words.append(candidate)
    return words


def _cjk_vocabulary(rng: random.Random) -> list[str]:
    """Build the CJK character pool used by the Chinese half of the workload.

    Args:
        rng: Seeded random source.

    Returns:
        Distinct CJK characters, most frequent first.
    """
    chars: list[str] = []
    seen: set[str] = set()
    while len(chars) < _CJK_POOL_SIZE:
        char = chr(_CJK_BASE + rng.randrange(_CJK_RANGE))
        if char in seen:
            continue
        seen.add(char)
        chars.append(char)
    return chars


def _tool_names() -> list[str]:
    """Build the tool-name pool that shapes the action key space.

    The pool is built exhaustively instead of by rejection sampling: the
    prefix/suffix grid is finite (10 * 7 one-part names, 10 * 7 * 7 two-suffix
    names), and an unbounded "draw until full" loop would hang once the grid is
    exhausted.

    Returns:
        Distinct tool names, most frequent first.
    """
    names = list(_REAL_TOOLS)
    seen = set(names)
    for prefix in _TOOL_PREFIXES:
        for suffix in _TOOL_SUFFIXES:
            candidate = prefix + suffix
            if candidate not in seen:
                seen.add(candidate)
                names.append(candidate)
    for prefix in _TOOL_PREFIXES:
        for first in _TOOL_SUFFIXES:
            for second in _TOOL_SUFFIXES:
                candidate = prefix + first + second
                if candidate not in seen:
                    seen.add(candidate)
                    names.append(candidate)
    return names


@dataclass(slots=True, frozen=True)
class WorkloadState:
    """One synthetic agent observation before abstraction.

    Attributes:
        task: User request framing the step.
        observation: Raw observation text, mixed English/CJK.
        recent: Bounded local action history.
        surface: Coarse partition key.
    """

    task: str
    observation: str
    recent: tuple[str, ...]
    surface: str


@dataclass(slots=True, frozen=True)
class Query:
    """A retrieval request, i.e. the abstracted current state plus its session.

    Attributes:
        state: Abstracted query state.
        scope: Session scope the request belongs to.
    """

    state: StateView
    scope: str


class Workload:
    """Deterministic generator of realistic agent states and episodes.

    Every draw is derived from `(seed, index)`, so an episode can be rebuilt
    exactly without keeping it in memory. That keeps peak memory flat even at
    N=20000 experiences and makes the workload reproducible regardless of how
    the harness batches its work.

    Attributes:
        words: Zipf pool of English vocabulary (head = common technical words).
        cjk: Zipf pool of CJK characters.
        scopes: Zipf pool of session scopes; one session dominates, as in a
            real deployment.
        surfaces: Zipf pool of coarse partitions.
        tools, commands, arg_keys, arg_values: Pools shaping the action keys.
        cues: Task prefixes.
        tool_names: Every name the tool pool can produce, used by the candidate
            filter that mirrors `main._available_memory_actions`.
    """

    def __init__(self, seed: int = _SEED) -> None:
        """Initialize the vocabulary.

        Args:
            seed: Master seed; all pools and every per-index generator derive
                from it.
        """
        self.seed = seed
        vocab_rng = random.Random(seed ^ 0x517A3D11)
        self.words = _ZipfPool.build(_vocabulary(vocab_rng), _ZIPF_EXPONENT)
        self.cjk = _ZipfPool.build(_cjk_vocabulary(vocab_rng), _ZIPF_EXPONENT * 0.8)
        self.scopes = _ZipfPool.build(list(_SCOPES), 0.9)
        self.surfaces = _ZipfPool.build(list(_SURFACES), 0.9)
        self.tools = _ZipfPool.build(_tool_names(), 0.8)
        self.commands = _ZipfPool.build(list(_COMMANDS), 0.8)
        self.arg_keys = _ZipfPool.build(list(_ARG_KEYS), 0.7)
        self.arg_values = _ZipfPool.build(list(_ARG_VALUES), 0.7)
        self.cues = _ZipfPool.build(list(_TASK_CUES), 0.9)
        self.tool_names = frozenset(self.tools.items)

    # -- deterministic per-index generators ----------------------------------
    def episode_rng(self, index: int) -> random.Random:
        """Return the generator that owns one episode.

        Args:
            index: Zero-based episode index.

        Returns:
            A generator seeded from the master seed and the index only.
        """
        return random.Random((self.seed * 1_000_003 + index) ^ 0x2F5B17)

    def query_rng(self, index: int) -> random.Random:
        """Return the generator that owns one query.

        Args:
            index: Zero-based query index.

        Returns:
            A generator seeded from the master seed and the index only.
        """
        return random.Random((self.seed * 7_919 + index) ^ 0x6B1D2A)

    # -- text primitives -----------------------------------------------------
    def word(self, rng: random.Random) -> str:
        """Draw one English vocabulary word.

        Args:
            rng: Seeded random source.

        Returns:
            The sampled word.
        """
        return self.words.sample(rng)

    def _cjk_run(self, rng: random.Random) -> str:
        chars = [self.cjk.sample(rng) for _ in range(rng.randint(2, 6))]
        return "".join(chars)

    def _noise(self, rng: random.Random) -> str:
        """Emit a volatile identifier that redact()/regularize() must collapse.

        Args:
            rng: Seeded random source.

        Returns:
            A numeric id, URL, timestamp, credential pair or hex digest.
        """
        roll = rng.random()
        if roll < 0.3:
            return f"id={rng.randrange(10**6, 10**9)}"
        if roll < 0.55:
            host = rng.randrange(1, 40)
            item = rng.randrange(10**5)
            return f"https://svc{host}.example.internal/v1/items/{item}"
        if roll < 0.75:
            day = rng.randrange(1, 28)
            hour = rng.randrange(0, 24)
            minute = rng.randrange(0, 60)
            return f"2026-03-{day:02d}T{hour:02d}:{minute:02d}:00Z"
        if roll < 0.9:
            secret = "".join(
                rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(12)
            )
            return f"password={secret}"
        return f"{rng.getrandbits(64):016x}"

    def _observation(self, rng: random.Random, target_tokens: int) -> str:
        """Build one observation of roughly the requested token count.

        Args:
            rng: Seeded random source.
            target_tokens: Approximate number of distinct tokens wanted.

        Returns:
            Mixed English/CJK observation text.
        """
        units: list[str] = []
        budget = int(max(0, target_tokens) * 1.35) + 1
        while budget > 0:
            roll = rng.random()
            if roll < _CJK_RATIO:
                run = self._cjk_run(rng)
                units.append(run)
                budget -= max(1, len(run) - 1)
            elif roll < _CJK_RATIO + 0.1:
                units.append(self._noise(rng))
                budget -= 1
            else:
                count = rng.randint(1, 3)
                for _ in range(count):
                    units.append(self.word(rng))
                budget -= count
        return " ".join(units)

    def task(self, rng: random.Random) -> str:
        """Build one short mixed-language user request.

        Args:
            rng: Seeded random source.

        Returns:
            The task text.
        """
        cue = self.cues.sample(rng)
        words = " ".join(self.word(rng) for _ in range(rng.randint(1, 3)))
        tail = "".join(self.cjk.sample(rng) for _ in range(rng.randint(2, 4)))
        return f"{cue}{words}{tail}"

    def observation_of_size(self, rng: random.Random, size: int) -> str:
        """Build one observation of at least the requested character length.

        Args:
            rng: Seeded random source.
            size: Target character count (a word boundary is respected).

        Returns:
            The observation text, truncated near `size` characters.
        """
        parts: list[str] = []
        length = 0
        while length < size:
            chunk = self._observation(rng, 24)
            parts.append(chunk)
            length += len(chunk) + 1
        text = " ".join(parts)
        if len(text) <= size:
            return text
        return text[:size].rsplit(" ", 1)[0]

    def _action_text(self, rng: random.Random) -> str:
        key = self.arg_keys.sample(rng)
        value = self.arg_values.sample(rng)
        return f"{self.tools.sample(rng)}({key}={value})"

    def action_spec(self, rng: random.Random) -> dict[str, Any]:
        """Build the keyword arguments of one normalised action.

        Args:
            rng: Seeded random source.

        Returns:
            Keyword arguments accepted by `jitrl.state.normalize_action`.
        """
        roll = rng.random()
        if roll < 0.75:
            key = self.arg_keys.sample(rng)
            return {
                "kind": "tool",
                "name": self.tools.sample(rng),
                "args": {key: self.arg_values.sample(rng)},
            }
        if roll < 0.92:
            return {
                "kind": "command",
                "name": self.commands.sample(rng),
                "args": {},
            }
        return {"kind": "reply", "name": "reply", "args": {}}

    # -- state construction --------------------------------------------------
    def _state_tokens(self, state: WorkloadState) -> int:
        combined = " ".join([state.task, state.observation, *state.recent])
        return len(tokenize(combined))

    def _fit(self, state: WorkloadState, rng: random.Random) -> WorkloadState:
        """Nudge a state into the configured 30-120 token band.

        Args:
            state: Candidate state.
            rng: Seeded random source used for padding.

        Returns:
            A state whose tokenised size is inside the band when possible.
        """
        for _ in range(8):
            count = self._state_tokens(state)
            if count < _TOKENS_LO:
                extra = self._observation(rng, _TOKENS_LO - count + 8)
                padded = f"{state.observation} {extra}".strip()
                state = replace(state, observation=padded)
            elif count > _TOKENS_HI:
                units = state.observation.split(" ")
                keep = max(1, int(len(units) * 0.75))
                state = replace(state, observation=" ".join(units[:keep]))
            else:
                return state
        return state

    def make_state(self, rng: random.Random, surface: str) -> WorkloadState:
        """Build one raw observation of the synthetic workload.

        Args:
            rng: Seeded random source.
            surface: Coarse partition the state belongs to.

        Returns:
            The state, sized to the configured token band.
        """
        target = rng.randint(_TOKENS_LO + 2, _TOKENS_HI - 10)
        state = WorkloadState(
            task=self.task(rng),
            observation=self._observation(rng, target),
            recent=tuple(
                self._action_text(rng)
                for _ in range(rng.randint(0, _MAX_EPISODE_STEPS - 1))
            ),
            surface=surface,
        )
        return self._fit(state, rng)

    def _perturb(self, rng: random.Random, state: WorkloadState) -> WorkloadState:
        """Rephrase a stored state the way a repeated request would.

        A live session rarely sends byte-identical text, but it usually sends
        the same intent with fresh identifiers. The perturbation drops a few
        units, swaps one or two words and adds new volatile noise, which lands
        the Jaccard similarity in the realistic 0.8-0.95 range instead of the
        degenerate 1.0 of an exact replay.

        Args:
            rng: Seeded random source.
            state: Stored state to rephrase.

        Returns:
            The perturbed state.
        """
        units = [
            unit
            for unit in state.observation.split(" ")
            if rng.random() >= _QUERY_DROP_RATE
        ]
        for _ in range(rng.randint(1, 2)):
            if units:
                units[rng.randrange(len(units))] = self.word(rng)
        units.append(self._noise(rng))
        return replace(state, observation=" ".join(units), recent=state.recent[-2:])

    # -- episodes and queries ------------------------------------------------
    def created_at(self, rng: random.Random, index: int, total: int) -> float:
        """Timestamp one episode inside the 180 day window before the reference.

        Args:
            rng: Seeded random source for the jitter.
            index: Zero-based episode index; timestamps increase with it, like a
                real bank that is appended to chronologically.
            total: Number of episodes in the run.

        Returns:
            A Unix timestamp.
        """
        span = _TIMESTAMP_SPAN_DAYS * 86400.0
        fraction = index / max(1, total - 1)
        jitter = rng.uniform(-1.0, 1.0) * _TIMESTAMP_JITTER_DAYS * 86400.0
        return _REFERENCE_NOW - span + fraction * span + jitter

    def build_episode(
        self, index: int, total: int
    ) -> tuple[Episode, list[WorkloadState]]:
        """Build one complete experience episode.

        Args:
            index: Zero-based episode index.
            total: Number of episodes in the run, used for timestamps.

        Returns:
            The episode and the raw states it was built from.
        """
        rng = self.episode_rng(index)
        scope = self.scopes.sample(rng)
        surface = self.surfaces.sample(rng)
        steps = (
            1
            if rng.random() < _SINGLE_STEP_SHARE
            else rng.randint(2, _MAX_EPISODE_STEPS)
        )
        states = [self.make_state(rng, surface) for _ in range(steps)]
        created_at = self.created_at(rng, index, total)
        rewards = [round(rng.gauss(0.0, 0.7), 3) for _ in range(steps)]
        returns: list[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + _GAMMA * running
            returns.append(running)
        returns.reverse()
        records: list[StepRecord] = []
        for position, state in enumerate(states):
            records.append(
                StepRecord(
                    index=position,
                    state=abstract_state(
                        task=state.task,
                        observation=state.observation,
                        recent_actions=state.recent,
                        surface=state.surface,
                        limit=MAX_STATE_CHARS,
                    ),
                    action=normalize_action(**self.action_spec(rng)),
                    reward=rewards[position],
                    discounted_return=round(returns[position], 4),
                    outcome="ok" if rng.random() < 0.85 else "error",
                    observation="",
                    reward_source="heuristic",
                )
            )
        episode = Episode(
            episode_id=f"bench-{total:06d}-{index:06d}",
            scope=scope,
            created_at=created_at,
            user_text=states[0].task,
            task_family=surface,
            success=rng.random() < 0.6,
            steps=records,
            meta={
                "gamma": _GAMMA,
                "reward_source": "heuristic",
                "apply_mode": "hint",
            },
        )
        return episode, states

    def build_query(self, index: int, total: int) -> Query:
        """Build one retrieval request for a bank of `total` experiences.

        Args:
            index: Zero-based query index; selects both the source episode and
                the perturbation.
            total: Number of experiences in the bank.

        Returns:
            The abstracted query state and its session scope.
        """
        rng = self.query_rng(index)
        # Bias the source toward recent experience, as a live session would.
        bias = rng.random() ** 1.6
        source = max(0, int((total - 1) - bias * (total - 1)))
        episode, states = self.build_episode(source, total)
        state = self._perturb(rng, states[rng.randrange(len(states))])
        return Query(
            state=abstract_state(
                task=state.task,
                observation=state.observation,
                recent_actions=state.recent,
                surface=state.surface,
                limit=MAX_STATE_CHARS,
            ),
            scope=episode.scope,
        )

    def descriptor(self) -> dict[str, Any]:
        """Describe the workload for the JSON metadata block.

        Returns:
            A JSON-safe summary of every workload dimension.
        """
        return {
            "seed": self.seed,
            "tokens_per_state": [_TOKENS_LO, _TOKENS_HI],
            "vocabulary": {
                "distinct_words": len(self.words.items),
                "cjk_pool": len(self.cjk.items),
                "zipf_exponent": _ZIPF_EXPONENT,
                "cjk_share": _CJK_RATIO,
            },
            "scopes": list(_SCOPES),
            "surfaces": list(_SURFACES),
            "action_space": {
                "tools": len(self.tools.items),
                "commands": len(self.commands.items),
                "arg_keys": len(self.arg_keys.items),
                "arg_values": len(self.arg_values.items),
            },
            "timestamps": {
                "span_days": _TIMESTAMP_SPAN_DAYS,
                "reference_now": _REFERENCE_NOW,
                "jitter_days": _TIMESTAMP_JITTER_DAYS,
            },
            "episodes": {
                "single_step_share": _SINGLE_STEP_SHARE,
                "max_steps": _MAX_EPISODE_STEPS,
                "gamma": _GAMMA,
            },
            "query_perturbation": {
                "drop_rate": _QUERY_DROP_RATE,
                "swaps": [1, 2],
                "source_bias": "recent (uniform**1.6)",
            },
        }


# ---------------------------------------------------------------------------
# measurement helpers
# ---------------------------------------------------------------------------
def _percentile(ordered: list[float], quantile: float) -> float:
    """Return the nearest-rank percentile of an already sorted sample.

    Args:
        ordered: Sorted samples.
        quantile: Quantile in [0, 1].

    Returns:
        The sample at the nearest rank, or 0.0 for an empty input.
    """
    if not ordered:
        return 0.0
    index = math.ceil(quantile * len(ordered)) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def _scenario(
    name: str, n: int | None, samples_s: list[float], extra: dict[str, Any]
) -> dict[str, Any]:
    """Serialize one scenario into the report schema.

    Args:
        name: Stable scenario identifier.
        n: Experience count of the scenario, or None when N-independent.
        samples_s: Per-operation latencies in seconds.
        extra: Scenario-specific diagnostics.

    Returns:
        The JSON-ready scenario record.
    """
    ordered = sorted(samples_s)
    entry: dict[str, Any] = {
        "name": name,
        "n": n,
        "count": len(ordered),
        "p50_ms": round(_percentile(ordered, 0.5) * 1000.0, 6),
        "p95_ms": round(_percentile(ordered, 0.95) * 1000.0, 6),
        "mean_ms": (round(statistics.fmean(ordered) * 1000.0, 6) if ordered else 0.0),
        "extra": dict(extra),
    }
    entry["extra"]["min_ms"] = round(ordered[0] * 1000.0, 6) if ordered else 0.0
    entry["extra"]["max_ms"] = round(ordered[-1] * 1000.0, 6) if ordered else 0.0
    return entry


def _allowed_memory_actions(
    neighbors: list[Neighbor], tool_names: set[str] | frozenset[str]
) -> dict[str, tuple[str, int]]:
    """Group neighbours into candidate actions, mirroring main.py.

    This copies `JitRLPlugin._available_memory_actions`: actions learned in the
    past are only offered again when their tool is still exposed to the request.

    Args:
        neighbors: Retrieved neighbours.
        tool_names: Tool names available to the current request.

    Returns:
        Mapping `action_key -> (action_text, support_count)`.
    """
    grouped: dict[str, tuple[str, int]] = {}
    for neighbor in neighbors:
        key = neighbor.action_key
        if not key:
            continue
        kind, _, body = key.partition(":")
        if not body:
            kind, body = key, key
        if kind == "tool" and body.split("(", 1)[0] not in tool_names:
            continue
        text, count = grouped.get(key, (neighbor.action_text or key, 0))
        grouped[key] = (text, count + 1)
    return grouped


def _db_bytes(directory: Path) -> int:
    """Sum the on-disk size of the bank and its WAL/shm sidecars.

    Args:
        directory: Directory that holds the database.

    Returns:
        The total size in bytes.
    """
    total = 0
    for path in directory.glob("bank.sqlite3*"):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _git(*args: str) -> str | None:
    """Run a read-only git query in the repository root.

    Args:
        *args: Git subcommand and its arguments.

    Returns:
        The stripped stdout, or None when git is unavailable or unhappy.
    """
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _dirty_files() -> int:
    """Count the working-tree changes reported by `git status`.

    Returns:
        The number of changed paths, or -1 when git is unavailable.
    """
    status = _git("status", "--porcelain")
    if status is None:
        return -1
    return len([line for line in status.splitlines() if line.strip()])


def _shipped_defaults() -> dict[str, Any]:
    """Snapshot the plugin's current retrieval/memory defaults, best effort.

    The benchmark itself never reads these values: they are recorded only so a
    later run can show whether the shipped defaults moved between the baseline
    and the "after" tree.

    Returns:
        A JSON-safe dict, empty when `jitrl.config` cannot be read.
    """
    try:
        from jitrl import config as jitrl_config

        cfg = jitrl_config.JitRLConfig()
        retrieval = cfg.retrieval
        memory = cfg.memory
        return {
            "k": retrieval.k,
            "similarity_threshold": retrieval.similarity_threshold,
            "surface_gate": retrieval.surface_gate,
            "scope_isolation": retrieval.scope_isolation,
            "prefilter_limit": retrieval.prefilter_limit,
            "recency_half_life_days": retrieval.recency_half_life_days,
            "min_count": retrieval.min_count,
            "max_entries": memory.max_entries,
            "checkpoint_interval": memory.checkpoint_interval,
            "keep_checkpoints": memory.keep_checkpoints,
        }
    except Exception:  # noqa: BLE001 - metadata must never break a benchmark
        return {}


# ---------------------------------------------------------------------------
# scenario runners
# ---------------------------------------------------------------------------
async def _open_bank(
    directory: Path, *, checkpoint_interval: int, total: int = 0
) -> tuple[MemoryBank, float]:
    """Create and initialize a bank in a fresh directory.

    Args:
        directory: Empty directory owned by one scenario block.
        checkpoint_interval: Episodes between automatic snapshots; 0 disables
            snapshots (used to isolate the raw write path).

    Returns:
        The live bank and the `initialize()` duration in milliseconds.
    """
    # A synthetic episode carries more than one step, so a bank capped at
    # _MAX_ENTRIES rows would start evicting (a full GROUP BY over the table)
    # before the insert phase even ends, and every later insert would pay for
    # another eviction. Give the benchmark enough headroom to measure the write
    # path itself; eviction has its own scenario.
    capacity = max(_MAX_ENTRIES, total * 3)
    bank = MemoryBank(
        directory / "bank.sqlite3",
        max_entries=capacity,
        redact=True,
        keep_episodes=_PRUNE_KEEP_EPISODES,
        prune_strategy=_PRUNE_STRATEGY,
        checkpoint_interval=checkpoint_interval,
        keep_checkpoints=_KEEP_CHECKPOINTS,
    )
    begin = time.perf_counter()
    await bank.initialize()
    return bank, (time.perf_counter() - begin) * 1000.0


async def _insert(
    bank: MemoryBank, workload: Workload, total: int, *, init_ms: float
) -> dict[str, Any]:
    """Insert `total` synthetic experiences one episode at a time.

    Episode construction is deliberately outside the timed region: the scenario
    measures the storage path, not the workload generator.

    Args:
        bank: Initialized memory bank.
        workload: Deterministic workload generator.
        total: Number of experiences to insert.
        init_ms: Duration of the preceding `initialize()` call.

    Returns:
        Per-episode latencies plus throughput diagnostics.
    """
    samples: list[float] = []
    rows = 0
    for start in range(0, total, _GEN_BATCH):
        stop = min(start + _GEN_BATCH, total)
        batch = [
            workload.build_episode(index, total)[0] for index in range(start, stop)
        ]
        for episode in batch:
            begin = time.perf_counter()
            written = await bank.add_episode(episode)
            samples.append(time.perf_counter() - begin)
            rows += written
    elapsed = statistics.fsum(samples)
    return {
        "samples": samples,
        "extra": {
            "episodes": total,
            "rows": rows,
            "rows_per_s": round(rows / elapsed, 1) if elapsed > 0.0 else 0.0,
            "insert_s": round(elapsed, 4),
            "init_ms": round(init_ms, 4),
            "steps_per_episode": round(rows / total, 3) if total else 0.0,
            "db_bytes_open": _db_bytes(bank.db_path.parent),
            "max_entries": bank.max_entries,
            "checkpoint_interval": bank.checkpoint_interval,
        },
    }


async def _one_topk(
    bank: MemoryBank,
    query: Query,
    *,
    threshold: float,
    surface_gate: bool,
) -> list[Neighbor]:
    """Run one retrieval with the production hot-path parameters.

    Args:
        bank: Initialized memory bank.
        query: Query state and session scope.
        threshold: Minimum Jaccard similarity.
        surface_gate: Restrict the SQL pass to the query surface.

    Returns:
        The ranked neighbours.
    """
    return await bank.topk(
        query.state,
        scope=query.scope,
        k=_K,
        threshold=threshold,
        surface_gate=surface_gate,
        prefilter_limit=_PREFILTER_LIMIT,
        recency_half_life_days=_RECENCY_HALF_LIFE_DAYS,
        scope_mode=_SCOPE_MODE,
        now=_REFERENCE_NOW,
    )


async def _warmup(bank: MemoryBank, queries: list[Query]) -> None:
    """Warm the SQLite page cache and every scenario configuration.

    Args:
        bank: Initialized memory bank.
        queries: Disjoint warmup queries.
    """
    for query in queries:
        for threshold, gate in (
            (_THRESHOLD, _SURFACE_GATE),
            (_THRESHOLD_LOOSE, False),
            (_THRESHOLD, True),
        ):
            await _one_topk(bank, query, threshold=threshold, surface_gate=gate)


async def _topk_scenario(
    bank: MemoryBank,
    queries: list[Query],
    total: int,
    *,
    name: str,
    threshold: float,
    surface_gate: bool,
) -> dict[str, Any]:
    """Measure `MemoryBank.topk` for one configuration.

    Args:
        bank: Initialized memory bank.
        queries: Measured queries.
        total: Number of experiences in the bank.
        name: Stable scenario identifier.
        threshold: Minimum Jaccard similarity.
        surface_gate: Restrict the SQL pass to the query surface.

    Returns:
        The scenario record.
    """
    samples: list[float] = []
    hits = 0
    returned = 0
    top1: list[float] = []
    for query in queries:
        begin = time.perf_counter()
        neighbours = await _one_topk(
            bank, query, threshold=threshold, surface_gate=surface_gate
        )
        samples.append(time.perf_counter() - begin)
        if neighbours:
            hits += 1
            returned += len(neighbours)
            top1.append(neighbours[0].similarity)
    return _scenario(
        name,
        total,
        samples,
        {
            "threshold": threshold,
            "surface_gate": surface_gate,
            "k": _K,
            "scope_mode": _SCOPE_MODE,
            "prefilter_limit": _PREFILTER_LIMIT,
            "recency_half_life_days": _RECENCY_HALF_LIFE_DAYS,
            "rows_scored": min(total, _PREFILTER_LIMIT),
            "hit_share": round(hits / len(queries), 4) if queries else 0.0,
            "neighbours_mean": round(returned / max(1, hits), 3),
            "top1_similarity_mean": (round(statistics.fmean(top1), 4) if top1 else 0.0),
        },
    )


def _estimator_scenario(
    neighbour_sets: list[list[Neighbor]], total: int, args: argparse.Namespace
) -> dict[str, Any]:
    """Measure estimate_values + build_candidates + select_action over 10 rows.

    Args:
        neighbour_sets: One 10-neighbour retrieval per measured query.
        total: Number of experiences in the bank.
        args: Parsed command-line options.

    Returns:
        The scenario record (all values are microseconds per iteration).
    """
    rng = random.Random(_ESTIMATOR_RNG_SEED)
    combined: list[float] = []
    build_s: list[float] = []
    estimate_s: list[float] = []
    select_s: list[float] = []
    candidate_counts: list[int] = []
    for _ in range(args.micro_reps):
        for neighbours in neighbour_sets:
            if not neighbours:
                continue
            memory_actions = _allowed_memory_actions(neighbours, set())
            begin = time.perf_counter()
            candidates = build_candidates(
                memory_actions=memory_actions, limit=_CANDIDATE_LIMIT
            )
            after_build = time.perf_counter()
            estimate = estimate_values(
                neighbours,
                [candidate.key for candidate in candidates],
                k=_K,
                ucb_alpha=_UCB_ALPHA,
                exploration_rate=_EXPLORATION_RATE,
                normalize=True,
                min_count=1,
                rng=rng,
            )
            after_estimate = time.perf_counter()
            select_action(
                candidates,
                estimate,
                beta=_BETA,
                temperature=_SAMPLE_TEMPERATURE,
                normalize=True,
                mode="hint",
                rng=rng,
            )
            end = time.perf_counter()
            candidate_counts.append(len(candidates))
            build_s.append(after_build - begin)
            estimate_s.append(after_estimate - after_build)
            select_s.append(end - after_estimate)
            combined.append(end - begin)
    ordered = sorted(combined)
    extras: dict[str, Any] = {
        "iterations": len(ordered),
        "reps": args.micro_reps,
        "neighbours_mean": (
            round(statistics.fmean(len(item) for item in neighbour_sets), 3)
            if neighbour_sets
            else 0.0
        ),
        "candidates_mean": (
            round(statistics.fmean(candidate_counts), 3) if candidate_counts else 0.0
        ),
        "combined_p50_us": round(_percentile(ordered, 0.5) * 1e6, 3),
        "build_candidates_p50_us": round(_percentile(sorted(build_s), 0.5) * 1e6, 3),
        "estimate_values_p50_us": round(_percentile(sorted(estimate_s), 0.5) * 1e6, 3),
        "select_action_p50_us": round(_percentile(sorted(select_s), 0.5) * 1e6, 3),
        "beta": _BETA,
        "ucb_alpha": _UCB_ALPHA,
        "exploration_rate": _EXPLORATION_RATE,
    }
    return _scenario("estimate_candidates_select_top10", total, combined, extras)


async def _e2e_scenario(
    bank: MemoryBank,
    queries: list[Query],
    total: int,
    tool_names: frozenset[str],
) -> dict[str, Any]:
    """Measure the whole pre-injection request path of `on_llm_request`.

    Steps: abstract the query state (already done by the caller, mirroring
    `main`), retrieve neighbours, filter candidate actions, build the candidate
    set, estimate advantages, pick an action and render the preference block.

    Args:
        bank: Initialized memory bank.
        queries: Measured queries.
        total: Number of experiences in the bank.
        tool_names: Tool names exposed to the request.

    Returns:
        The scenario record.
    """
    rng = random.Random(_ESTIMATOR_RNG_SEED)
    samples: list[float] = []
    neighbours_seen: list[int] = []
    block_chars: list[int] = []
    early_exits = 0
    for query in queries:
        begin = time.perf_counter()
        neighbours = await _one_topk(
            bank, query, threshold=_THRESHOLD, surface_gate=_SURFACE_GATE
        )
        if neighbours:
            memory_actions = _allowed_memory_actions(neighbours, tool_names)
            candidates = build_candidates(
                memory_actions=memory_actions, limit=_CANDIDATE_LIMIT
            )
            estimate = estimate_values(
                neighbours,
                [candidate.key for candidate in candidates],
                k=_K,
                ucb_alpha=_UCB_ALPHA,
                exploration_rate=_EXPLORATION_RATE,
                normalize=True,
                min_count=1,
                rng=rng,
            )
            decision = select_action(
                candidates,
                estimate,
                beta=_BETA,
                temperature=_SAMPLE_TEMPERATURE,
                normalize=True,
                mode="hint",
                rng=rng,
            )
            block = render_preference_block(
                decision, neighbors=neighbours, max_chars=_MAX_HINT_CHARS
            )
            block_chars.append(len(block))
        else:
            early_exits += 1
        samples.append(time.perf_counter() - begin)
        neighbours_seen.append(len(neighbours))
    return _scenario(
        "e2e_request_path",
        total,
        samples,
        {
            "neighbours_mean": (
                round(statistics.fmean(neighbours_seen), 3) if neighbours_seen else 0.0
            ),
            "early_exit_share": (
                round(early_exits / len(queries), 4) if queries else 0.0
            ),
            "block_chars_mean": (
                round(statistics.fmean(block_chars), 1) if block_chars else 0.0
            ),
            "apply_mode": "hint",
            "candidate_limit": _CANDIDATE_LIMIT,
            "max_hint_chars": _MAX_HINT_CHARS,
        },
    )


def _state_scenario(workload: Workload, args: argparse.Namespace) -> dict[str, Any]:
    """Measure state abstraction for a typical 2 KB observation.

    Args:
        workload: Deterministic workload generator.
        args: Parsed command-line options.

    Returns:
        The scenario record.
    """
    rng = random.Random(_SEED ^ 0x2B1D)
    observation = workload.observation_of_size(rng, 2048)
    task = workload.task(rng)
    recent = tuple(workload._action_text(rng) for _ in range(2))
    samples: list[float] = []
    tokenize_s: list[float] = []
    view: StateView | None = None
    for _ in range(args.abstract_repeats):
        begin = time.perf_counter()
        view = abstract_state(
            task=task,
            observation=observation,
            recent_actions=recent,
            surface="qq",
            limit=MAX_STATE_CHARS,
        )
        samples.append(time.perf_counter() - begin)
        begin = time.perf_counter()
        tokenize(observation)
        tokenize_s.append(time.perf_counter() - begin)
    return _scenario(
        "state_abstract_2kb",
        None,
        samples,
        {
            "observation_chars": len(observation),
            "task_chars": len(task),
            "state_tokens": len(view.tokens) if view is not None else 0,
            "signature_chars": len(view.signature) if view is not None else 0,
            "text_chars": len(view.text) if view is not None else 0,
            "tokenize_p50_us": round(_percentile(sorted(tokenize_s), 0.5) * 1e6, 3),
            "repeats": args.abstract_repeats,
        },
    )


async def _checkpoint_scenario(
    workload: Workload, total: int, args: argparse.Namespace
) -> dict[str, Any]:
    """Measure insertion with the production snapshot interval enabled.

    Args:
        workload: Deterministic workload generator.
        total: Number of experiences to insert.
        args: Parsed command-line options.

    Returns:
        The scenario record.
    """
    directory = Path(tempfile.mkdtemp(prefix=f"jitrl-bench-chk-{total}-"))
    bank: MemoryBank | None = None
    try:
        bank, init_ms = await _open_bank(
            directory, checkpoint_interval=_CHECKPOINT_INTERVAL, total=total
        )
        inserted = await _insert(bank, workload, total, init_ms=init_ms)
        snapshots = await bank.list_checkpoints()
        extra = dict(inserted["extra"])
        extra["checkpoints"] = len(snapshots)
        extra["checkpoint_bytes"] = sum(
            int(item.get("size_bytes", 0)) for item in snapshots
        )
        extra["snapshot_interval"] = _CHECKPOINT_INTERVAL
        return _scenario("insert_with_checkpoints", total, inserted["samples"], extra)
    finally:
        if bank is not None:
            await bank.close()
        shutil.rmtree(directory, ignore_errors=True)


async def _scale_scenarios(
    workload: Workload, total: int, args: argparse.Namespace
) -> list[dict[str, Any]]:
    """Run every N-dependent scenario against one freshly built bank.

    Args:
        workload: Deterministic workload generator.
        total: Number of experiences in the bank.
        args: Parsed command-line options.

    Returns:
        The scenario records, in report order.
    """
    directory = Path(tempfile.mkdtemp(prefix=f"jitrl-bench-{total}-"))
    bank: MemoryBank | None = None
    results: list[dict[str, Any]] = []
    try:
        bank, init_ms = await _open_bank(directory, checkpoint_interval=0, total=total)
        print(
            f"[bench] N={total}: inserting {total} experiences ...",
            file=sys.stderr,
            flush=True,
        )
        inserted = await _insert(bank, workload, total, init_ms=init_ms)
        print(
            f"[bench] N={total}: {inserted['extra']['rows']} rows at "
            f"{inserted['extra']['rows_per_s']} rows/s",
            file=sys.stderr,
            flush=True,
        )
        results.append(
            _scenario(
                "insert_no_checkpoint",
                total,
                inserted["samples"],
                inserted["extra"],
            )
        )
        queries = [
            workload.build_query(_QUERY_INDEX_SALT + index, total)
            for index in range(args.queries)
        ]
        warm = [workload.build_query(index, total) for index in range(args.warmup)]
        await _warmup(bank, warm)
        print(f"[bench] N={total}: topk scenarios ...", file=sys.stderr, flush=True)
        results.append(
            await _topk_scenario(
                bank,
                queries,
                total,
                name="topk_k10_thr0.8_hybrid",
                threshold=_THRESHOLD,
                surface_gate=_SURFACE_GATE,
            )
        )
        results.append(
            await _topk_scenario(
                bank,
                queries,
                total,
                name="topk_k10_thr0.5_hybrid",
                threshold=_THRESHOLD_LOOSE,
                surface_gate=False,
            )
        )
        results.append(
            await _topk_scenario(
                bank,
                queries,
                total,
                name="topk_k10_thr0.8_hybrid_surface_gate",
                threshold=_THRESHOLD,
                surface_gate=True,
            )
        )
        print(f"[bench] N={total}: estimator + e2e ...", file=sys.stderr, flush=True)
        neighbour_sets = [
            await _one_topk(bank, query, threshold=0.0, surface_gate=False)
            for query in queries
        ]
        results.append(_estimator_scenario(neighbour_sets, total, args))
        results.append(await _e2e_scenario(bank, queries, total, workload.tool_names))
    finally:
        if bank is not None:
            await bank.close()
        shutil.rmtree(directory, ignore_errors=True)
    if total in args.checkpoint_ns:
        print(
            f"[bench] N={total}: checkpoint durability scenario ...",
            file=sys.stderr,
            flush=True,
        )
        results.append(await _checkpoint_scenario(workload, total, args))
        # Keep report order stable: durability sits next to the plain insert.
        results.insert(1, results.pop())
    return results


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _meta(args: argparse.Namespace, workload: Workload) -> dict[str, Any]:
    """Build the metadata block of the report.

    Args:
        args: Parsed command-line options.
        workload: Deterministic workload generator.

    Returns:
        The JSON-ready metadata.
    """
    return {
        "harness": "scripts/bench_jitrl.py",
        "harness_version": _HARNESS_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_files": _dirty_files(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "sqlite": sqlite3.sqlite_version,
        "seed": args.seed,
        "mode": "quick" if args.quick else "full",
        "queries": args.queries,
        "warmup": args.warmup,
        "n_values": list(args.ns),
        "run_parameters": {
            "k": _K,
            "threshold": _THRESHOLD,
            "threshold_loose": _THRESHOLD_LOOSE,
            "surface_gate": _SURFACE_GATE,
            "scope_mode": _SCOPE_MODE,
            "recency_half_life_days": _RECENCY_HALF_LIFE_DAYS,
            "prefilter_limit": _PREFILTER_LIMIT,
            "max_entries": _MAX_ENTRIES,
            "max_entries_selector": [200, 1000, 5000, 20000],
            "checkpoint_interval": _CHECKPOINT_INTERVAL,
            "keep_checkpoints": _KEEP_CHECKPOINTS,
            "prune_strategy": _PRUNE_STRATEGY,
            "candidate_limit": _CANDIDATE_LIMIT,
            "beta": _BETA,
            "ucb_alpha": _UCB_ALPHA,
            "exploration_rate": _EXPLORATION_RATE,
            "sample_temperature": _SAMPLE_TEMPERATURE,
            "max_hint_chars": _MAX_HINT_CHARS,
            "micro_reps": args.micro_reps,
            "abstract_repeats": args.abstract_repeats,
        },
        "shipped_defaults": _shipped_defaults(),
        "workload": workload.descriptor(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write the report as UTF-8 JSON.

    Args:
        path: Destination file.
        payload: Report document.
    """
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def _print_table(payload: dict[str, Any]) -> None:
    """Print a compact summary table of every scenario.

    Args:
        payload: Report document.
    """
    name_width = 38
    header = (
        f"{'scenario':<{name_width}}{'N':>7}{'count':>7}"
        f"{'p50 ms':>12}{'p95 ms':>12}{'mean ms':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in payload["scenarios"]:
        name = str(row["name"])
        n_text = "-" if row["n"] is None else str(row["n"])
        print(
            f"{name:<{name_width}}{n_text:>7}{row['count']:>7}"
            f"{row['p50_ms']:>12.4f}{row['p95_ms']:>12.4f}"
            f"{row['mean_ms']:>12.4f}"
        )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse the command line and apply the quick/full defaults.

    Args:
        argv: Argument vector, or None for `sys.argv[1:]`.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark the JitRL retrieval/estimation hot path.",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        help="write the JSON report to PATH (the summary table goes to stdout)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small workload for a fast smoke run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_SEED,
        help=f"workload seed (default: {_SEED})",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=None,
        help="measured queries per scenario (full: 30, quick: 8)",
    )
    parser.add_argument(
        "--n",
        dest="n_values",
        default=None,
        help="comma-separated experience counts, e.g. 200,1000,5000",
    )
    parser.add_argument(
        "--micro-reps",
        type=int,
        default=None,
        help="repeats of the estimator micro-benchmark (full: 10)",
    )
    parser.add_argument(
        "--abstract-repeats",
        type=int,
        default=None,
        help="repeats of the state-abstraction micro-benchmark (full: 300)",
    )
    parser.add_argument(
        "--no-checkpoints",
        action="store_true",
        help="skip the checkpoint durability scenario",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.queries = args.queries or 8
        args.micro_reps = args.micro_reps or 3
        args.abstract_repeats = args.abstract_repeats or 60
        default_ns = (200, 1000)
    else:
        args.queries = args.queries or 30
        args.micro_reps = args.micro_reps or 10
        args.abstract_repeats = args.abstract_repeats or 300
        default_ns = (200, 1000, 5000, 20000)
    if args.n_values:
        parsed = {int(item) for item in str(args.n_values).split(",") if item.strip()}
        args.ns = tuple(sorted(parsed))
    else:
        args.ns = default_ns
    args.warmup = max(1, round(args.queries * _WARMUP_FRACTION))
    args.checkpoint_ns = () if args.no_checkpoints else args.ns
    return args


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Run every scenario and build the report document.

    Args:
        args: Parsed command-line options.

    Returns:
        The report, ready to be serialized.
    """
    workload = Workload(args.seed)
    print(
        f"[bench] mode={'quick' if args.quick else 'full'} "
        f"n={list(args.ns)} queries={args.queries} warmup={args.warmup}",
        file=sys.stderr,
        flush=True,
    )
    scenarios: list[dict[str, Any]] = [_state_scenario(workload, args)]
    for total in args.ns:
        scenarios.extend(await _scale_scenarios(workload, total, args))
    return {"meta": _meta(args, workload), "scenarios": scenarios}


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector, or None for `sys.argv[1:]`.

    Returns:
        The process exit code.
    """
    args = _parse_args(argv)
    started = time.perf_counter()
    payload = asyncio.run(_run(args))
    payload["meta"]["duration_s"] = round(time.perf_counter() - started, 2)
    if args.json:
        _print_table(payload)
        _write_json(Path(args.json), payload)
        print(f"\n[bench] JSON written to {args.json}", file=sys.stderr)
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
