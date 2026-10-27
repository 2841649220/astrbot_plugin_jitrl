"""Bounded configuration shared by the optional runtime and offline learner."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class LensConfig:
    mode: str = "off"
    pre_enabled: bool = True
    post_enabled: bool = False
    auto_train: bool = False
    train_every: int = 32
    min_samples: int = 32
    train_steps: int = 20
    batch_size: int = 16
    learning_rate: float = 0.0003
    min_support: int = 3
    max_uncertainty: float = 0.5
    relevance_threshold: float = 0.6
    similarity_threshold: float = 0.25
    max_adjustment: float = 1.0
    evidence_tokens: int = 128
    max_memory: int = 32
    coarse_limit: int = 128
    replay_limit: int = 2048
    hidden_dim: int = 384
    embedding_dim: int = 256
    summary_dim: int = 128
    blocks: int = 4
    query_heads: int = 6
    kv_heads: int = 2
    bases: int = 8
    rank: int = 8
    engram_buckets: int = 4093
    engram_heads: int = 2
    engram_dim: int = 16
    max_events: int = 32
    return_scale: float = 6.0
    seed: int = 1729
    attention_backend: str = "mita"
    mita_landmarks: int = 8
    mita_topk: int = 16
    mita_query_chunk: int = 64
    memory_tokens: int = 128
    device: str = "cpu"
    mita_execution: str = "grouped"

    def __post_init__(self) -> None:
        if self.mode not in {"off", "shadow", "active"}:
            raise ValueError("unknown neural mode")
        if self.attention_backend not in {"mita", "hybrid", "gqa", "legacy"}:
            raise ValueError("unknown attention backend")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("unknown neural device")
        if self.mita_execution not in {"grouped", "reference"}:
            raise ValueError("unknown MiTA execution kernel")
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{item.name} must be finite")
            if isinstance(value, int) and not isinstance(value, bool) and value < 1:
                raise ValueError(f"{item.name} must be positive")
        if self.hidden_dim % self.query_heads or self.query_heads % self.kv_heads:
            raise ValueError("hidden_dim/query_heads/kv_heads are incompatible")
        for name in ("learning_rate", "max_uncertainty", "return_scale"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("relevance_threshold", "similarity_threshold"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.max_adjustment < 0 or self.max_memory > self.coarse_limit:
            raise ValueError("invalid memory/adjustment budget")
        if self.memory_tokens > self.coarse_limit:
            raise ValueError("memory token budget exceeds coarse retrieval limit")

    @classmethod
    def from_raw(cls, raw: Any) -> LensConfig:
        """Parse operator settings; tensor dimensions are checkpoint architecture."""
        raw = raw if isinstance(raw, dict) else {}
        base = cls()
        values: dict[str, Any] = {}
        mode = str(raw.get("mode", "off")).lower()
        values["mode"] = mode if mode in {"off", "shadow", "active"} else "off"
        backend = str(raw.get("attention_backend", "mita")).lower()
        values["attention_backend"] = (
            backend if backend in {"mita", "hybrid", "gqa", "legacy"} else "mita"
        )
        device = str(raw.get("device", "cpu")).lower()
        values["device"] = device if device in {"cpu", "cuda", "auto"} else "cpu"
        execution = str(raw.get("mita_execution", "grouped")).lower()
        values["mita_execution"] = (
            execution if execution in {"grouped", "reference"} else "grouped"
        )
        for name in ("pre_enabled", "post_enabled", "auto_train"):
            value = raw.get(name, getattr(base, name))
            values[name] = (
                value
                if isinstance(value, bool)
                else str(value).lower() in {"true", "1", "yes"}
            )
        ranges = {
            "train_every": (1, 10000),
            "min_samples": (2, 100000),
            "train_steps": (1, 1000),
            "batch_size": (1, 128),
            "min_support": (1, 32),
            "evidence_tokens": (32, 256),
            "replay_limit": (32, 20000),
            "mita_landmarks": (1, 32),
            "mita_topk": (1, 64),
            "mita_query_chunk": (1, 256),
            "memory_tokens": (32, 128),
        }
        for name, (lo, hi) in ranges.items():
            try:
                values[name] = max(lo, min(hi, int(raw.get(name, getattr(base, name)))))
            except (ValueError, TypeError, OverflowError):
                pass
        for name, lo, hi in (
            ("learning_rate", 0.000001, 0.01),
            ("max_uncertainty", 0.01, 2.0),
            ("relevance_threshold", 0.0, 1.0),
            ("similarity_threshold", 0.0, 1.0),
            ("max_adjustment", 0.0, 2.0),
        ):
            try:
                value = float(raw.get(name, getattr(base, name)))
                if math.isfinite(value):
                    values[name] = max(lo, min(hi, value))
            except (ValueError, TypeError, OverflowError):
                pass
        return cls(**values)
