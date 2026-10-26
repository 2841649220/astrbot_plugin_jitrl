"""Observed-return learning on exact historical snapshots, with immutable serving.

Uncertainty predicts observed residual magnitude. It is an uncalibrated proxy,
not a probability bound. Relevance uses structural action-match proxy labels;
neither objective supplies a counterfactual return for an unexecuted action.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import LensConfig
from .contracts import ReplaySample
from .model import LensModel, make_batch


def _batch(
    samples: list[ReplaySample], cfg: LensConfig
) -> tuple[dict[str, Tensor], Tensor, Tensor, Tensor]:
    batch = make_batch(
        [sample.decision.state for sample in samples],
        [sample.decision.snapshot for sample in samples],
        [[sample.action_key] for sample in samples],
        [sample.decision.intervention.kind for sample in samples],
        cfg,
    )
    targets = torch.tensor(
        [max(-1.0, min(1.0, sample.return_g / cfg.return_scale)) for sample in samples]
    )
    reliability = torch.tensor([sample.reliability for sample in samples])
    relevance = torch.zeros_like(batch["memory_mask"], dtype=torch.float32)
    for row, sample in enumerate(samples):
        for column, memory in enumerate(sample.decision.snapshot.memories):
            relevance[row, column] = float(memory.action_key == sample.action_key)
    return batch, targets, reliability, relevance


def _loss(
    model: LensModel, encoded: tuple[dict[str, Tensor], Tensor, Tensor, Tensor]
) -> Tensor:
    batch, targets, reliability, relevance_targets = encoded
    output = model(batch)
    action_value = output["candidate_values"][:, 0]
    pre_value = output["pre_value"]
    observed_loss = F.smooth_l1_loss(action_value, targets, reduction="none")
    observed_loss = observed_loss + F.smooth_l1_loss(
        pre_value, targets, reduction="none"
    )
    # The residual target is detached, so it cannot move value estimates to
    # manufacture an easy uncertainty target.
    action_residual = (action_value.detach() - targets).abs()
    pre_residual = (pre_value.detach() - targets).abs()
    proxy_loss = (output["uncertainty"][:, 0] - action_residual).square()
    proxy_loss = proxy_loss + (output["pre_uncertainty"] - pre_residual).square()
    weighted = (
        (observed_loss + 0.1 * proxy_loss) * reliability
    ).sum() / reliability.sum().clamp_min(1e-8)
    mask = batch["memory_mask"].to(torch.float32) * reliability.unsqueeze(-1)
    relevance_loss = F.binary_cross_entropy_with_logits(
        output["relevance"], relevance_targets, reduction="none"
    )
    relevance_loss = (relevance_loss * mask).sum() / mask.sum().clamp_min(1e-8)
    return weighted + 0.1 * relevance_loss + 0.0001 * output["alpha"].square().mean()


def train_model(
    model: LensModel, cfg: LensConfig, samples: list[ReplaySample]
) -> tuple[LensModel, dict[str, float | int | str]]:
    """Fit a private copy; the caller publishes a new checkpoint atomically.

    Every target is an executed action's settled return. This routine accepts
    exactly one private scope, rejects malformed labels, and validates all
    historical retrieval snapshots before starting an optimizer step.
    """
    if len(samples) < cfg.min_samples:
        raise ValueError("not enough settled replay samples")
    scope = samples[0].decision.state.scope
    for sample in samples:
        state = sample.decision.state
        if state.scope != scope or not scope:
            raise ValueError("cross-scope training is forbidden")
        if (
            not math.isfinite(sample.return_g)
            or not math.isfinite(sample.settled_at)
            or sample.settled_at < state.observed_at
        ):
            raise ValueError("invalid/unsettled return label")
        if not math.isfinite(sample.reliability) or not 0 < sample.reliability <= 1:
            raise ValueError("invalid return reliability")
        if not sample.action_key or not sample.experience_id:
            raise ValueError("an executed action and experience ID are required")
    identifiers = [sample.experience_id for sample in samples]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate replay experience")
    # Encoding once also freezes all training inputs and checks every snapshot
    # before any training occurs. Chunks keep peak memory bounded.
    chunks = [
        samples[start : start + cfg.batch_size]
        for start in range(0, len(samples), cfg.batch_size)
    ]
    device = next(model.parameters()).device
    encoded = []
    for chunk in chunks:
        batch, targets, reliability, relevance = _batch(chunk, cfg)
        encoded.append(
            (
                {name: tensor.to(device) for name, tensor in batch.items()},
                targets.to(device),
                reliability.to(device),
                relevance.to(device),
            )
        )
    updated = copy.deepcopy(model).to(device)
    anchor = [parameter.detach().clone() for parameter in updated.parameters()]
    parameter_count = sum(p.numel() for p in updated.parameters())

    def evaluate() -> float:
        updated.eval()
        with torch.no_grad():
            return sum(
                float(_loss(updated, batch)) * len(chunk)
                for chunk, batch in zip(chunks, encoded)
            ) / len(samples)

    before = evaluate()
    if not math.isfinite(before):
        raise ValueError("non-finite model loss")
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed)
        optimizer = torch.optim.AdamW(
            updated.parameters(), lr=cfg.learning_rate, weight_decay=0.001
        )
        updated.train()
        for _ in range(cfg.train_steps):
            index = int(torch.randint(len(encoded), (1,), generator=generator))
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(updated, encoded[index])
            drift = (
                sum(
                    (p - old).square().sum()
                    for p, old in zip(updated.parameters(), anchor)
                )
                / parameter_count
            )
            loss = loss + 0.01 * drift
            if not torch.isfinite(loss):
                raise ValueError("non-finite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                updated.parameters(), 1.0, error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm):
                raise ValueError("non-finite training gradient")
            optimizer.step()
    if any(not torch.isfinite(parameter).all() for parameter in updated.parameters()):
        raise ValueError("non-finite trained parameter")
    after = evaluate()
    if not math.isfinite(after):
        raise ValueError("non-finite final model loss")
    return updated.eval(), {
        "before_loss": before,
        "after_loss": after,
        "steps": cfg.train_steps,
        "samples": len(samples),
        "uncertainty": "observed-residual-proxy-uncalibrated",
        "relevance": "structural-action-match-proxy",
    }
