"""Conservative async serving, isolated learning and atomic local checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .config import LensConfig
from .contracts import (
    VERSIONS,
    CandidateScore,
    CandidateSpec,
    Intervention,
    LensDecision,
    RetrievalSnapshot,
    StateRecord,
)
from .encoding import digest, safe_name
from .store import SOURCE_RELIABILITY, LensStore, retained_snapshot, validate_snapshot

_ARCHITECTURE = (
    "hidden_dim",
    "embedding_dim",
    "summary_dim",
    "blocks",
    "query_heads",
    "kv_heads",
    "bases",
    "rank",
    "engram_buckets",
    "engram_heads",
    "engram_dim",
    "max_events",
    "return_scale",
    "seed",
)


def render_evidence(
    intervention: Intervention, snapshot: RetrievalSnapshot, byte_budget: int
) -> str:
    """Catalog-only rendering: no historical user/tool/web text is copied.

    UTF-8 bytes are a conservative token-budget proxy for byte-token providers;
    a provider-specific tokenizer is required for an exact token guarantee.
    """
    if intervention.kind == "NO_OP":
        return ""
    if intervention.kind != "SELECT_EVIDENCE":
        raise ValueError("unsupported intervention catalog entry")
    memories = {m.experience_id: m for m in snapshot.memories}
    lines = ["[Historical evidence; advisory, permissions apply]"]
    for key in intervention.memory_ids[:2]:
        memory = memories.get(key)
        if (
            memory is None
            or not safe_name(memory.action_name)
            or not safe_name(key)
            or memory.reward_source not in SOURCE_RELIABILITY
        ):
            raise ValueError("evidence is not in the authorized snapshot")
        line = f"id={key[:12]} action={memory.action_name} G={memory.return_g:+.2f} source={memory.reward_source}"
        if len("\n".join([*lines, line]).encode("utf-8")) <= byte_budget:
            lines.append(line)
    return "\n".join(lines) if len(lines) > 1 else ""


class LensRuntime:
    def __init__(self, root: Path | str, cfg: LensConfig) -> None:
        self.root = Path(root)
        self.cfg = cfg
        self.store = LensStore(self.root / "neural.sqlite3")
        self._models: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self._model_lock = threading.Lock()
        self._training_lock = asyncio.Lock()
        self._workers: set[asyncio.Task] = set()
        self._closing = False
        self._closed = False
        self.last_error = ""

    async def close(self) -> None:
        self._closing = True
        async with self._training_lock:
            if self._closed:
                return
            if self._workers:
                await asyncio.gather(*list(self._workers), return_exceptions=True)
            await asyncio.to_thread(self.store.close)
            self._models.clear()
            self._closed = True

    async def _work(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        if self._closing:
            raise RuntimeError("Neural Lens is closing")
        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        self._workers.add(worker)
        worker.add_done_callback(self._workers.discard)
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            await worker
            raise

    def _path(self, name: str) -> Path:
        path = (self.root / name).resolve()
        if self.root.resolve() not in path.parents or path.suffix != ".pt":
            raise ValueError("invalid checkpoint path")
        return path

    def _load_model(
        self,
        scope: str,
        checkpoint_id: str | None = None,
        decision_time: float | None = None,
    ) -> tuple[Any | None, str]:
        record = (
            self.store.checkpoint(scope, checkpoint_id)
            if checkpoint_id
            else self.store.latest_checkpoint(scope)
        )
        if record is None:
            if checkpoint_id and checkpoint_id != "untrained":
                raise ValueError("pinned neural checkpoint no longer exists")
            return None, "untrained"
        training_cutoff = json.loads(record["metrics_json"])["training_cutoff"]
        if decision_time is not None and (
            record["created_at"] > decision_time or training_cutoff > decision_time
        ):
            raise ValueError("checkpoint was published after this decision")
        key = (scope, record["checkpoint_id"])
        with self._model_lock:
            if key in self._models:
                self._models.move_to_end(key)
                return self._models[key], key[1]
        import torch

        from .model import create_model

        path = self._path(record["filename"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("neural checkpoint checksum mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        architecture = {name: getattr(self.cfg, name) for name in _ARCHITECTURE}
        if (
            payload.get("versions") != list(VERSIONS)
            or payload.get("scope_hash") != digest(scope)
            or payload.get("architecture") != architecture
            or payload.get("checkpoint_id") != key[1]
            or payload.get("trained_before") != training_cutoff
        ):
            raise ValueError("incompatible neural checkpoint")
        model = create_model(self.cfg)
        model.load_state_dict(payload["state_dict"], strict=True)
        if any(
            not torch.isfinite(tensor).all() for tensor in model.state_dict().values()
        ):
            raise ValueError("nonfinite neural checkpoint")
        model.eval()
        with self._model_lock:
            self._models[key] = model
            while len(self._models) > 8:
                self._models.popitem(last=False)
        return model, key[1]

    def _forward(
        self,
        model: Any,
        state: StateRecord,
        snapshot: RetrievalSnapshot,
        actions: list[str],
        pre: str = "NO_OP",
    ) -> dict[str, Any]:
        import torch

        from .model import make_batch

        with torch.inference_mode():
            outputs = model(make_batch([state], [snapshot], [actions], [pre], self.cfg))
        if any(not torch.isfinite(v).all() for v in outputs.values()):
            raise ValueError("nonfinite Lens output")
        return outputs

    async def prepare(self, state: StateRecord) -> tuple[LensDecision, int]:
        """Compute a PRE proposal; caller records the actually attached context."""
        return await self._work(self._prepare_sync, state)

    def _prepare_sync(self, state: StateRecord) -> tuple[LensDecision, int]:
        epoch = self.store.epoch(state.scope)
        snapshot = self.store.retrieve(state, self.cfg)
        pre = Intervention()
        checkpoint_id = "untrained"
        try:
            model, checkpoint_id = self._load_model(
                state.scope, decision_time=state.observed_at
            )
            output = None
            if model is not None and snapshot.memories:
                coarse = self._forward(model, state, snapshot, [])
                ranking = (
                    coarse["relevance"][0].sigmoid().tolist()[: len(snapshot.memories)]
                )
                indices = sorted(
                    range(len(ranking)),
                    key=lambda i: (-ranking[i], snapshot.memories[i].experience_id),
                )[: self.cfg.max_memory]
                snapshot = retained_snapshot(snapshot, indices)
                output = self._forward(model, state, snapshot, [])
            else:
                snapshot = retained_snapshot(
                    snapshot,
                    list(range(min(self.cfg.max_memory, len(snapshot.memories)))),
                )
            if self.cfg.mode == "off":
                pre = Intervention(reason="DISABLED_PURE_NO_OP")
            elif self.cfg.mode == "shadow":
                pre = Intervention(reason="SHADOW_PURE_NO_OP")
            elif model is None:
                pre = Intervention(reason="UNTRAINED")
            elif len(snapshot.memories) < self.cfg.min_support:
                pre = Intervention(reason="LOW_SUPPORT")
            elif not self.cfg.pre_enabled:
                pre = Intervention(reason="PRE_DISABLED")
            elif output is not None:
                uncertainty = output["pre_uncertainty"][0].item()
                relevance = output["relevance"][0].sigmoid().tolist()
                selected = tuple(
                    m.experience_id
                    for i, m in enumerate(snapshot.memories)
                    if relevance[i] >= self.cfg.relevance_threshold
                    and m.reliability > 0
                )[:2]
                if selected and uncertainty <= self.cfg.max_uncertainty:
                    pre = Intervention(
                        "SELECT_EVIDENCE", selected, "OBSERVATIONAL_RELEVANCE_PROXY"
                    )
                else:
                    pre = Intervention(reason="UNCERTAINTY_OR_RELEVANCE")
            rendered = render_evidence(pre, snapshot, self.cfg.evidence_tokens)
            if pre.kind != "NO_OP" and not rendered:
                pre = Intervention(reason="EVIDENCE_BUDGET")
            return LensDecision(state, snapshot, pre, rendered, checkpoint_id), epoch
        except Exception as exc:
            self.last_error = type(exc).__name__
            logger.warning(
                "[JitRL Lens] local inference failed; pure NO_OP (%s)", self.last_error
            )
            # Keep the original causal snapshot for later audit/training, but a
            # broken local model never changes the base request or tool set.
            snapshot = retained_snapshot(
                snapshot, list(range(min(self.cfg.max_memory, len(snapshot.memories))))
            )
            return LensDecision(
                state, snapshot, Intervention(reason="LOCAL_FAILURE")
            ), epoch

    async def commit(self, decision: LensDecision, epoch: int) -> bool:
        return await self._work(
            self.store.record_decision, decision, expected_epoch=epoch
        )

    async def score_candidates(
        self, decision: LensDecision, candidates: list[CandidateSpec]
    ) -> list[CandidateScore]:
        """Score provided candidates only. The environment owns hard validation.

        This API does not execute actions and cannot authorize tool arguments.
        PRE and POST bind the same snapshot and immutable checkpoint.
        """
        return await self._work(self._score_sync, decision, candidates)

    def _score_sync(
        self, decision: LensDecision, candidates: list[CandidateSpec]
    ) -> list[CandidateScore]:
        validate_snapshot(decision.state, decision.snapshot)
        if len(candidates) > 64:
            raise ValueError("candidate budget exceeded")
        seen: set[str] = set()
        for candidate in candidates:
            if (
                not math.isfinite(candidate.base_score)
                or candidate.candidate_id in seen
                or not candidate.action_key
                or candidate.action_name not in decision.state.allowed_actions
                or candidate.provenance
                not in {"TOKEN_LOGPROB", "VERBALIZED", "RANK_ONLY", "NONE"}
            ):
                raise ValueError("invalid or unauthorized candidate")
            seen.add(candidate.candidate_id)
        scores = [
            CandidateScore(c, 0.0, None, None, 0, "NO_OP_POST") for c in candidates
        ]
        if not self.store.decision_matches(decision):
            return [
                CandidateScore(c, 0.0, None, None, 0, "STALE_OR_MISMATCHED_DECISION")
                for c in candidates
            ]
        reason = decision.intervention.reason
        if (
            self.cfg.mode == "active"
            and self.cfg.post_enabled
            and candidates
            and reason != "LOCAL_FAILURE"
            and decision.checkpoint_id != "untrained"
        ):
            try:
                model, _ = self._load_model(
                    decision.state.scope,
                    decision.checkpoint_id,
                    decision_time=decision.state.observed_at,
                )
                if model is not None:
                    output = self._forward(
                        model,
                        decision.state,
                        decision.snapshot,
                        [c.action_key for c in candidates],
                        decision.intervention.kind,
                    )
                    all_memories = decision.snapshot.memories
                    baseline = (
                        sum(m.return_g for m in all_memories)
                        / max(1, len(all_memories))
                        / self.cfg.return_scale
                    )
                    local = output["candidate_values"][0].tolist()
                    uncertainty = output["uncertainty"][0].tolist()
                    scores = []
                    for i, candidate in enumerate(candidates):
                        support = [
                            m
                            for m in all_memories
                            if m.action_key == candidate.action_key
                        ]
                        adjustment = 0.0
                        code = "LOW_ACTION_SUPPORT"
                        if (
                            len(support) >= self.cfg.min_support
                            and uncertainty[i] <= self.cfg.max_uncertainty
                            and candidate.provenance != "NONE"
                        ):
                            empirical = (
                                sum(m.return_g for m in support)
                                / len(support)
                                / self.cfg.return_scale
                            )
                            delta = 0.5 * (empirical - baseline) + 0.5 * (
                                local[i] - baseline
                            )
                            adjustment = max(
                                -self.cfg.max_adjustment,
                                min(self.cfg.max_adjustment, delta),
                            )
                            code = "BOUNDED_CANDIDATE_PROXY"
                        scores.append(
                            CandidateScore(
                                candidate,
                                adjustment,
                                local[i],
                                uncertainty[i],
                                len(support),
                                code,
                            )
                        )
            except Exception as exc:
                self.last_error = type(exc).__name__
                logger.warning(
                    "[JitRL Lens] POST failed; NO_OP_POST (%s)", self.last_error
                )
        recorded = self.store.record_post(
            decision.state.decision_id,
            {
                "checkpoint_id": decision.checkpoint_id,
                "snapshot_id": decision.snapshot.snapshot_id,
                "scores": [asdict(score) for score in scores],
            },
        )
        if not recorded:
            return [
                CandidateScore(c, 0.0, None, None, 0, "STALE_DECISION")
                for c in candidates
            ]
        return scores

    async def train(self, scope: str) -> dict[str, Any]:
        async with self._training_lock:
            return await self._work(self._train_sync, scope)

    async def apply_user_feedback(
        self,
        *,
        scope: str,
        episode_id: str,
        delta: float,
        gamma: float,
        expected_epoch: int,
    ) -> int:
        return await self._work(
            self.store.apply_user_feedback,
            scope=scope,
            episode_id=episode_id,
            delta=delta,
            gamma=gamma,
            expected_epoch=expected_epoch,
        )

    def _train_sync(self, scope: str) -> dict[str, Any]:
        epoch = self.store.epoch(scope)
        cutoff = time.time()
        samples = self.store.replay(scope, self.cfg.replay_limit, cutoff)
        if len(samples) < self.cfg.min_samples:
            return {
                "status": "insufficient_samples",
                "samples": len(samples),
                "required": self.cfg.min_samples,
            }
        try:
            import torch

            from .learner import train_model
            from .model import create_model

            model, _ = self._load_model(scope)
            if model is None:
                model = create_model(self.cfg)
            trained, metrics = train_model(model, self.cfg, samples)
            if not all(
                math.isfinite(float(value))
                for value in metrics.values()
                if isinstance(value, int | float)
            ):
                raise ValueError("nonfinite training metrics")
            checkpoint_id = "lens-" + uuid.uuid4().hex
            relative = f"checkpoints/{digest(scope)}/{checkpoint_id}.pt"
            path = self._path(relative)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            payload = {
                "versions": list(VERSIONS),
                "scope_hash": digest(scope),
                "checkpoint_id": checkpoint_id,
                "architecture": {
                    name: getattr(self.cfg, name) for name in _ARCHITECTURE
                },
                "trained_before": cutoff,
                "state_dict": trained.state_dict(),
            }
            published = False
            try:
                with temporary.open("wb") as stream:
                    torch.save(payload, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                published = self.store.publish_checkpoint(
                    scope=scope,
                    expected_epoch=epoch,
                    checkpoint_id=checkpoint_id,
                    filename=relative,
                    sha256=checksum,
                    metrics={**metrics, "training_cutoff": cutoff},
                )
                if not published:
                    path.unlink(missing_ok=True)
                    return {"status": "stale_generation"}
            finally:
                temporary.unlink(missing_ok=True)
                if not published:
                    path.unlink(missing_ok=True)
            # Only a later decision loads the new weights; serving objects are
            # never mutated. POST can still load its older pinned checkpoint.
            self.last_error = ""
            return {
                "status": "trained",
                "checkpoint_id": checkpoint_id,
                **metrics,
                "parameters": trained.parameter_counts(),
            }
        except ImportError:
            self.last_error = "PYTORCH_UNAVAILABLE"
            return {"status": "unavailable", "reason": self.last_error}
        except Exception as exc:
            self.last_error = type(exc).__name__
            logger.warning("[JitRL Lens] training failed (%s)", self.last_error)
            return {"status": "failed", "reason": self.last_error}

    async def reset(self, scope: str | None = None) -> None:
        """Erase scoped facts and weights; epoch protects against late learners."""
        async with self._training_lock:
            filenames = await self._work(self.store.reset, scope)
            with self._model_lock:
                self._models = OrderedDict(
                    (k, v)
                    for k, v in self._models.items()
                    if scope is not None and k[0] != scope
                )
            for filename in filenames:
                await self._work(self._path(filename).unlink, missing_ok=True)
            await self._work(self._remove_orphan_weights, scope)

    def _remove_orphan_weights(self, scope: str | None) -> None:
        parent = (self.root / "checkpoints").resolve()
        if not parent.exists():
            return
        if self.root.resolve() not in parent.parents:
            raise ValueError("checkpoint directory escaped the neural data root")
        directories = [parent / digest(scope)] if scope else list(parent.iterdir())
        for directory in directories:
            resolved = directory.resolve()
            if (
                resolved.parent != parent
                or len(directory.name) != 64
                or not all(c in "0123456789abcdef" for c in directory.name)
            ):
                raise ValueError("invalid private checkpoint directory")
            if not directory.is_dir():
                continue
            for path in directory.iterdir():
                if path.suffix in {".pt", ".tmp"}:
                    if path.resolve().parent != resolved:
                        raise ValueError("weight file escaped its private directory")
                    path.unlink(missing_ok=True)

    async def status(self, scope: str | None = None) -> dict[str, Any]:
        stats = await self._work(self.store.stats, scope)
        checkpoint = (
            await self._work(self.store.latest_checkpoint, scope) if scope else None
        )
        return {
            "mode": self.cfg.mode,
            "pre_enabled": self.cfg.pre_enabled,
            "post_api_enabled": self.cfg.post_enabled,
            "astrbot_post_connected": False,
            "torch_installed": importlib.util.find_spec("torch") is not None,
            "checkpoint_id": checkpoint["checkpoint_id"] if checkpoint else "untrained",
            "last_error": self.last_error,
            **stats,
        }
