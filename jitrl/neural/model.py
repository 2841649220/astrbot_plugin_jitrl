"""Local neural controller inspired by Engram and fast-weight memory designs.

No tensor is sent into the remote model. This is a short-input discriminator,
not an implementation of MiTA attention or a remote LoRA adapter. Boolean masks
in this module use ``True`` for valid positions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import LensConfig
from .contracts import VERSIONS, RetrievalSnapshot, StateRecord
from .encoding import engram_addresses, frozen_embedding, state_embedding

_PRE_INDEX = {"NO_OP": 0, "SELECT_EVIDENCE": 1}
_META_DIM = 6


def masked_softmax(scores: Tensor, valid: Tensor, dim: int = -1) -> Tensor:
    """Return exact zeros for an entirely masked row, without NaN softmax."""
    weights = torch.softmax(scores.masked_fill(~valid, -10000.0), dim=dim)
    weights = weights * valid.to(weights.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class RMSNorm(nn.Module):
    """RMS normalization compatible with older optional PyTorch installations."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class SwiGLUBlock(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.norm = RMSNorm(dimension)
        self.up = nn.Linear(dimension, dimension * 4)
        self.down = nn.Linear(dimension * 2, dimension)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.up(self.norm(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * value) * 0.5


class EngramLookup(nn.Module):
    """Event suffix tables, context gate and a true causal sequence convolution.

    The structural event vocabulary and BLAKE2-based addresses are project
    adaptations; these are slowly trained embedding parameters, not episodes.
    """

    def __init__(self, cfg: LensConfig) -> None:
        super().__init__()
        self.tables = nn.ModuleList(
            nn.Embedding(cfg.engram_buckets + 1, cfg.engram_dim, padding_idx=0)
            for _ in range(cfg.engram_heads * 2)
        )
        width = cfg.engram_dim * cfg.engram_heads * 2
        self.key = nn.Linear(width, cfg.hidden_dim, bias=False)
        self.value = nn.Linear(width, cfg.hidden_dim, bias=False)
        self.query_norm = RMSNorm(cfg.hidden_dim)
        self.key_norm = RMSNorm(cfg.hidden_dim)
        self.value_norm = RMSNorm(cfg.hidden_dim)
        self.convolution = nn.Conv1d(
            cfg.hidden_dim,
            cfg.hidden_dim,
            kernel_size=4,
            dilation=3,
            groups=cfg.hidden_dim,
            bias=False,
        )

    def forward(self, state: Tensor, addresses: Tensor, event_mask: Tensor) -> Tensor:
        embedding = torch.cat(
            [table(addresses[..., i]) for i, table in enumerate(self.tables)],
            dim=-1,
        )
        key = self.key(embedding)
        gate = torch.sigmoid(
            (self.query_norm(state).unsqueeze(1) * self.key_norm(key)).sum(-1)
            / math.sqrt(state.shape[-1])
        )
        # Positions without a full suffix contain only padding table entries.
        valid = event_mask & addresses.ne(0).any(-1)
        value = self.value(embedding) * gate.unsqueeze(-1) * valid.unsqueeze(-1)
        # Left-only padding preserves causality, including with dilation=3.
        filtered = self.convolution(
            F.pad(self.value_norm(value).transpose(1, 2), (9, 0))
        )
        return (value + F.silu(filtered.transpose(1, 2))) * valid.unsqueeze(-1)


class GroupedMemoryAttention(nn.Module):
    """One state query, grouped KV heads; attention work is linear in memory K."""

    def __init__(self, cfg: LensConfig) -> None:
        super().__init__()
        self.query_heads = cfg.query_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.hidden_dim // cfg.query_heads
        self.query = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.key = nn.Linear(cfg.hidden_dim, self.kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(
            cfg.hidden_dim, self.kv_heads * self.head_dim, bias=False
        )
        self.output = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def forward(self, state: Tensor, memory: Tensor, valid: Tensor) -> Tensor:
        batch, count, _ = memory.shape
        query = self.query(state).reshape(batch, self.query_heads, 1, self.head_dim)
        key = (
            self.key(memory)
            .reshape(batch, count, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.value(memory)
            .reshape(batch, count, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        copies = self.query_heads // self.kv_heads
        key = key.repeat_interleave(copies, dim=1)
        value = value.repeat_interleave(copies, dim=1)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.head_dim)
        weights = masked_softmax(scores, valid[:, None, None, :])
        context = (weights @ value).reshape(batch, -1)
        return self.output(context)


class FastAdapter(nn.Module):
    """Bounded low-rank basis modulation without materializing a dense delta W."""

    def __init__(self, cfg: LensConfig) -> None:
        super().__init__()
        self.coefficients = nn.Linear(cfg.hidden_dim + cfg.summary_dim, cfg.bases)
        self.gate = nn.Linear(cfg.hidden_dim + cfg.summary_dim, 1)
        self.u = nn.Parameter(torch.randn(cfg.bases, cfg.hidden_dim, cfg.rank) * 0.02)
        self.v = nn.Parameter(torch.randn(cfg.bases, cfg.rank, cfg.hidden_dim) * 0.02)
        self.bases = cfg.bases

    def forward(
        self, state: Tensor, summary: Tensor, has_memory: Tensor
    ) -> tuple[Tensor, Tensor]:
        condition = torch.cat((state, summary), dim=-1)
        alpha = torch.tanh(self.coefficients(condition)) / self.bases
        alpha = alpha * has_memory.unsqueeze(-1)
        gate = torch.sigmoid(self.gate(condition)) * 0.5
        # Each basis factor has Frobenius norm <=1, so the combined operator
        # norm is at most 0.5; coefficients have L1 norm <=1.
        u = self.u / self.u.norm(dim=(1, 2), keepdim=True).clamp_min(1.0)
        v = self.v / self.v.norm(dim=(1, 2), keepdim=True).clamp_min(1.0)
        low = torch.einsum("bd,jrd->bjr", state, v)
        residual = torch.einsum("bjr,jdr->bjd", low, u)
        residual = (residual * alpha.unsqueeze(-1)).sum(1) * gate
        return residual, alpha


class LensModel(nn.Module):
    """Mostly-MLP local Lens with explicit PRE and candidate-conditioned POST."""

    def __init__(self, cfg: LensConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        self.state_encoder = nn.Sequential(nn.Linear(cfg.embedding_dim, d), RMSNorm(d))
        self.engram = EngramLookup(cfg)
        self.memory_encoder = nn.Sequential(
            nn.Linear(cfg.embedding_dim * 2 + _META_DIM, d),
            nn.SiLU(),
            RMSNorm(d),
        )
        self.compiler = nn.Sequential(
            nn.Linear(d * 3 + _META_DIM, d), nn.SiLU(), nn.Linear(d, d)
        )
        self.relevance = nn.Linear(d, 1)
        self.summary = nn.Linear(d, cfg.summary_dim)
        self.empty_summary = nn.Parameter(torch.zeros(cfg.summary_dim))
        self.adapter = FastAdapter(cfg)
        self.attention = GroupedMemoryAttention(cfg)
        self.blocks = nn.ModuleList(SwiGLUBlock(d) for _ in range(cfg.blocks))
        self.output_norm = RMSNorm(d)
        self.pre_embedding = nn.Embedding(len(_PRE_INDEX), 16)
        self.pre_head = nn.Sequential(
            nn.Linear(d + cfg.summary_dim + 16, d), nn.SiLU(), nn.Linear(d, 2)
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(cfg.embedding_dim, d), nn.SiLU(), RMSNorm(d)
        )
        self.candidate_head = nn.Sequential(
            nn.Linear(d * 3 + cfg.summary_dim + 16, d),
            nn.SiLU(),
            nn.Linear(d, 2),
        )

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        state = self.state_encoder(batch["state"])
        events = self.engram(state, batch["events"], batch["event_mask"])
        state = state + events.sum(1) / batch["event_mask"].sum(
            1, keepdim=True
        ).clamp_min(1)
        valid = batch["memory_mask"]
        memory = self.memory_encoder(
            torch.cat(
                (batch["memory_states"], batch["memory_actions"], batch["memory_meta"]),
                -1,
            )
        )
        memory = memory * valid.unsqueeze(-1)
        query = state.unsqueeze(1).expand_as(memory)
        compiled = self.compiler(
            torch.cat((query, memory, query * memory, batch["memory_meta"]), -1)
        )
        relevance = self.relevance(compiled).squeeze(-1)
        weights = masked_softmax(relevance, valid)
        has_memory = valid.any(-1)
        summary = self.summary((compiled * weights.unsqueeze(-1)).sum(1))
        summary = torch.where(has_memory.unsqueeze(-1), summary, self.empty_summary)
        residual, alpha = self.adapter(state, summary, has_memory)
        state = state + residual
        state = state + self.attention(state, memory, valid)
        for block in self.blocks:
            state = block(state)
        state = self.output_norm(state)
        pre = self.pre_embedding(batch["pre_index"])
        pre_output = self.pre_head(torch.cat((state, summary, pre), -1))
        candidates = self.candidate_encoder(batch["candidates"])
        count = candidates.shape[1]
        expanded = state.unsqueeze(1).expand(-1, count, -1)
        candidate_output = self.candidate_head(
            torch.cat(
                (
                    expanded,
                    candidates,
                    expanded * candidates,
                    summary.unsqueeze(1).expand(-1, count, -1),
                    pre.unsqueeze(1).expand(-1, count, -1),
                ),
                -1,
            )
        )
        candidate_mask = batch["candidate_mask"]
        return {
            "pre_value": torch.tanh(pre_output[:, 0]),
            "pre_uncertainty": F.softplus(pre_output[:, 1]) + 1e-4,
            "candidate_values": torch.tanh(candidate_output[..., 0]) * candidate_mask,
            "uncertainty": F.softplus(candidate_output[..., 1]) + 1e-4,
            "relevance": relevance,
            "alpha": alpha,
            "adapter_residual": residual,
            "hidden": state,
        }

    def parameter_counts(self) -> dict[str, int]:
        tables = sum(p.numel() for p in self.engram.tables.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "active_neural": total - tables,
            "engram_tables": tables,
            "total": total,
            "frozen_encoder": 0,
        }


def create_model(cfg: LensConfig) -> LensModel:
    """Initialize repeatably without changing the AstrBot process RNG state."""
    with torch.random.fork_rng(devices=[]):
        # torch.manual_seed also reseeds CUDA generators. This CPU-only
        # controller must leave an independently running host GPU model alone.
        torch.random.default_generator.manual_seed(cfg.seed)
        return LensModel(cfg).eval()


def _validate_snapshot(
    state: StateRecord, snapshot: RetrievalSnapshot, cfg: LensConfig
) -> None:
    if state.versions != VERSIONS or snapshot.versions != state.versions:
        raise ValueError("incompatible state/snapshot versions")
    if not state.scope or snapshot.scope != state.scope:
        raise ValueError("foreign memory scope")
    if (
        not math.isfinite(state.observed_at)
        or not math.isfinite(snapshot.cutoff)
        or snapshot.cutoff > state.observed_at
    ):
        raise ValueError("future retrieval snapshot")
    if len(snapshot.memories) > cfg.coarse_limit:
        raise ValueError("snapshot exceeds coarse memory budget")
    identifiers: set[str] = set()
    for memory in snapshot.memories:
        if memory.experience_id in identifiers:
            raise ValueError("duplicate memory experience")
        identifiers.add(memory.experience_id)
        if (
            memory.state.scope != state.scope
            or memory.state.environment != state.environment
            or memory.state.versions != state.versions
        ):
            raise ValueError("incompatible memory scope/environment/version")
        if (
            memory.state.episode_id == state.episode_id
            or memory.state.decision_id == state.decision_id
        ):
            raise ValueError("same-episode/self retrieval leaks labels")
        if (
            not math.isfinite(memory.state.observed_at)
            or not math.isfinite(memory.settled_at)
            or not memory.state.observed_at <= memory.settled_at < snapshot.cutoff
        ):
            raise ValueError("memory not settled before historical cutoff")
        if (
            not math.isfinite(memory.return_g)
            or not math.isfinite(memory.reliability)
            or not 0 <= memory.reliability <= 1
            or not math.isfinite(memory.similarity)
            or not 0 <= memory.similarity <= 1
        ):
            raise ValueError("invalid memory numeric metadata")


def make_batch(
    states: list[StateRecord],
    snapshots: list[RetrievalSnapshot],
    candidates: list[list[str]],
    pre_kinds: list[str],
    cfg: LensConfig,
) -> dict[str, Tensor]:
    """Encode immutable historical snapshots; no live lookup occurs in replay.

    Memory order is preserved exactly, including when encoding a coarse
    shortlist for learned reranking. No memory IDs are silently truncated.
    """
    size = len(states)
    if not size or not len(snapshots) == len(candidates) == len(pre_kinds) == size:
        raise ValueError("batch fields must have the same nonzero length")
    for state, snapshot in zip(states, snapshots):
        _validate_snapshot(state, snapshot, cfg)
    if any(kind not in _PRE_INDEX for kind in pre_kinds):
        raise ValueError("unsupported PRE intervention")
    memory_count = max(1, max(len(s.memories) for s in snapshots))
    candidate_count = max(1, max(map(len, candidates)))
    if candidate_count > cfg.coarse_limit:
        raise ValueError("candidate set exceeds local budget")
    addresses = [
        engram_addresses(
            s.events,
            buckets=cfg.engram_buckets,
            heads=cfg.engram_heads,
            max_events=cfg.max_events,
            seed=cfg.seed,
        )
        for s in states
    ]
    event_count = max(map(len, addresses))
    batch = {
        "state": torch.tensor(
            [state_embedding(s, cfg.embedding_dim) for s in states], dtype=torch.float32
        ),
        "events": torch.zeros(
            size, event_count, cfg.engram_heads * 2, dtype=torch.long
        ),
        "event_mask": torch.zeros(size, event_count, dtype=torch.bool),
        "memory_states": torch.zeros(size, memory_count, cfg.embedding_dim),
        "memory_actions": torch.zeros(size, memory_count, cfg.embedding_dim),
        "memory_meta": torch.zeros(size, memory_count, _META_DIM),
        "memory_mask": torch.zeros(size, memory_count, dtype=torch.bool),
        "candidates": torch.zeros(size, candidate_count, cfg.embedding_dim),
        "candidate_mask": torch.zeros(size, candidate_count, dtype=torch.bool),
        "pre_index": torch.tensor([_PRE_INDEX[p] for p in pre_kinds], dtype=torch.long),
    }
    for row, (state, snapshot) in enumerate(zip(states, snapshots)):
        batch["events"][row, : len(addresses[row])] = torch.tensor(addresses[row])
        batch["event_mask"][row, : min(len(state.events), cfg.max_events)] = True
        for column, memory in enumerate(snapshot.memories):
            batch["memory_states"][row, column] = torch.tensor(
                state_embedding(memory.state, cfg.embedding_dim)
            )
            batch["memory_actions"][row, column] = torch.tensor(
                frozen_embedding(memory.action_key, cfg.embedding_dim)
            )
            age = math.log1p(max(0.0, state.observed_at - memory.settled_at)) / 10.0
            batch["memory_meta"][row, column] = torch.tensor(
                [
                    max(-1.0, min(1.0, memory.return_g / cfg.return_scale)),
                    memory.reliability,
                    memory.similarity,
                    min(1.0, age),
                    float(memory.intervention == "SELECT_EVIDENCE"),
                    float(memory.intervention == "NO_OP"),
                ]
            )
            batch["memory_mask"][row, column] = True
        for column, action_key in enumerate(candidates[row]):
            batch["candidates"][row, column] = torch.tensor(
                frozen_embedding(action_key, cfg.embedding_dim)
            )
            batch["candidate_mask"][row, column] = True
    return batch
