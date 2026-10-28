"""Immutable causal records. Scores are local proxies, never remote logits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

NORMALIZER_VERSION = "events-nfkc-v1"
EMBEDDING_VERSION = "signed-char-ngram-v1"
HASH_VERSION = "blake2b-suffix-2-3-v1"
RENDERER_VERSION = "catalog-evidence-v1"
MODEL_VERSION = "neural-lens-v1"
VERSIONS = (
    NORMALIZER_VERSION,
    EMBEDDING_VERSION,
    HASH_VERSION,
    RENDERER_VERSION,
    MODEL_VERSION,
)


@dataclass(frozen=True, slots=True)
class StateRecord:
    scope: str
    decision_id: str
    episode_id: str
    step_index: int
    goal: str
    observation: str
    events: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    environment: str
    state_hash: str
    observed_at: float
    versions: tuple[str, ...] = VERSIONS

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateRecord:
        data = dict(data)
        for key in ("events", "allowed_actions", "versions"):
            data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    experience_id: str
    state: StateRecord
    action_key: str
    action_name: str
    intervention: str
    return_g: float
    reward_source: str
    reliability: float
    settled_at: float
    similarity: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        data = dict(data)
        data["state"] = StateRecord.from_dict(data["state"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class RetrievalSnapshot:
    snapshot_id: str
    scope: str
    cutoff: float
    memories: tuple[MemoryRecord, ...]
    versions: tuple[str, ...] = VERSIONS

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalSnapshot:
        data = dict(data)
        data["versions"] = tuple(data["versions"])
        data["memories"] = tuple(MemoryRecord.from_dict(m) for m in data["memories"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class Intervention:
    kind: str = "NO_OP"
    memory_ids: tuple[str, ...] = ()
    reason: str = "NO_SUPPORT"


@dataclass(frozen=True, slots=True)
class LensDecision:
    state: StateRecord
    snapshot: RetrievalSnapshot
    intervention: Intervention
    rendered: str = ""
    checkpoint_id: str = "untrained"
    post_mode: str = "NO_OP_POST"


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    action_key: str
    action_name: str
    base_score: float = 0.0
    provenance: str = "NONE"


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Support counts distinct settled episode IDs with positive kernel mass."""

    candidate: CandidateSpec
    adjustment: float
    local_value: float | None
    uncertainty: float | None
    support: int
    reason: str

    @property
    def score(self) -> float:
        return self.candidate.base_score + self.adjustment


@dataclass(frozen=True, slots=True)
class ReplaySample:
    experience_id: str
    decision: LensDecision
    action_key: str
    return_g: float
    reliability: float
    settled_at: float
