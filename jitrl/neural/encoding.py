"""Deterministic frozen retrieval features and ordered structural event hashes.

This baseline encoder has zero trainable parameters. It is lexical, not a
pretrained semantic embedding model. All addresses are stable across processes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
import uuid

from ..state import redact
from .contracts import HASH_VERSION, StateRecord

_NAME = re.compile(r"^[a-zA-Z0-9_.:-]{1,80}$")


def canonical_text(text: str, limit: int = 2048) -> str:
    return " ".join(unicodedata.normalize("NFKC", redact(str(text))).split())[:limit]


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def safe_name(name: str) -> bool:
    return bool(_NAME.fullmatch(name))


def canonicalize(
    *,
    scope: str,
    goal: str,
    observation: str = "",
    events: tuple[str, ...] = (),
    allowed_actions: tuple[str, ...] = (),
    environment: str = "astrbot-v4",
    episode_id: str = "",
    step_index: int = 0,
    observed_at: float | None = None,
    decision_id: str = "",
) -> StateRecord:
    if not scope:
        raise ValueError("a nonempty private scope is required")
    timestamp = time.time() if observed_at is None else observed_at
    if not math.isfinite(timestamp) or step_index < 0:
        raise ValueError("invalid decision time/index")
    goal = canonical_text(goal)
    observation = canonical_text(observation)
    actions = tuple(sorted({name for name in allowed_actions if safe_name(name)}))
    sequence = (
        "environment:" + canonical_text(environment, 80),
        *tuple(canonical_text(e, 120).casefold() for e in events[-29:]),
        "observation:" + digest(observation)[:16],
    )
    payload = (goal, observation, sequence, actions, environment)
    return StateRecord(
        scope,
        decision_id or uuid.uuid4().hex,
        episode_id or uuid.uuid4().hex,
        step_index,
        goal,
        observation,
        sequence,
        actions,
        environment,
        digest(payload),
        timestamp,
    )


def frozen_embedding(text: str, dimension: int = 256) -> list[float]:
    text = canonical_text(text, 4096).casefold()
    result = [0.0] * dimension
    for n in (2, 3):
        for i in range(max(0, len(text) - n + 1)):
            h = hashlib.blake2b(
                text[i : i + n].encode(), digest_size=8, person=b"jitrl-retrieval"
            ).digest()
            result[int.from_bytes(h[:4], "little") % dimension] += (
                1.0 if h[4] & 1 else -1.0
            )
    norm = math.sqrt(sum(v * v for v in result))
    return [v / norm for v in result] if norm else result


def state_embedding(state: StateRecord, dimension: int = 256) -> list[float]:
    return frozen_embedding(state_feature_text(state), dimension)


def state_feature_text(state: StateRecord) -> str:
    """The versioned embedding input; IDs, times and rewards are excluded."""
    return state.goal + " " + state.observation + " " + " ".join(state.events)


def engram_addresses(
    events: tuple[str, ...],
    *,
    buckets: int = 4093,
    heads: int = 2,
    max_events: int = 32,
    seed: int = 1729,
) -> list[list[int]]:
    """[T, 2*heads] suffix 2/3-grams; address 0 is a padding-only slot."""
    events = events[-max_events:]
    result = []
    for position in range(len(events)):
        row = []
        for order in (2, 3):
            for head in range(heads):
                if position + 1 < order:
                    row.append(0)
                    continue
                key = (
                    HASH_VERSION,
                    seed,
                    order,
                    head,
                    events[position + 1 - order : position + 1],
                )
                address = int(digest(key)[:16], 16) % buckets + 1
                row.append(address)
        result.append(row)
    return result or [[0] * (2 * heads)]
