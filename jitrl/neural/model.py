"""Local token backbone with genuine MiTA compression and sparse expert routing.

No tensor is sent into the remote model. This is a short-input discriminator,
with a four-layer MiTA backbone by default. The legacy one-query GQA model
remains available for weight compatibility and ablation. Boolean masks use
``True`` for valid positions.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import GroupedSelfAttention, MiTAAttention
from .config import LensConfig
from .contracts import VERSIONS, RetrievalSnapshot, StateRecord
from .encoding import engram_addresses, frozen_embedding, state_feature_text

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


class LensTokenBlock(nn.Module):
    """Pre-norm token interaction followed by a SwiGLU residual block."""

    def __init__(self, cfg: LensConfig, backend: str) -> None:
        super().__init__()
        self.backend = backend
        self.norm = RMSNorm(cfg.hidden_dim)
        self.attention = (
            MiTAAttention(cfg) if backend == "mita" else GroupedSelfAttention(cfg)
        )
        self.mlp = SwiGLUBlock(cfg.hidden_dim)

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        tokens = tokens + self.attention(self.norm(tokens), valid) * 0.5
        return self.mlp(tokens).masked_fill(~valid.unsqueeze(-1), 0.0)


class LensModel(nn.Module):
    """Local MiTA backbone with explicit PRE and candidate-conditioned POST."""

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
        if cfg.attention_backend == "legacy":
            # Exact v0.2.0 parameter names and tensor path are retained.
            self.attention = GroupedMemoryAttention(cfg)
            self.blocks = nn.ModuleList(SwiGLUBlock(d) for _ in range(cfg.blocks))
        else:
            self.event_encoder = nn.Sequential(
                nn.Linear(cfg.embedding_dim, d), RMSNorm(d)
            )
            self.token_types = nn.Embedding(3, d)
            self.event_positions = nn.Embedding(cfg.max_events, d)
            backends = ["mita"] * cfg.blocks
            if cfg.attention_backend == "gqa":
                backends = ["gqa"] * cfg.blocks
            elif cfg.attention_backend == "hybrid":
                # One global GQA anchor, all remaining interactions use MiTA.
                backends[-1] = "gqa"
            self.blocks = nn.ModuleList(
                LensTokenBlock(cfg, backend) for backend in backends
            )
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
        memory_states = batch["memory_states"].masked_fill(~valid.unsqueeze(-1), 0.0)
        memory_actions = batch["memory_actions"].masked_fill(~valid.unsqueeze(-1), 0.0)
        memory_meta = batch["memory_meta"].masked_fill(~valid.unsqueeze(-1), 0.0)
        memory = self.memory_encoder(
            torch.cat(
                (memory_states, memory_actions, memory_meta),
                -1,
            )
        )
        memory = memory * valid.unsqueeze(-1)
        if self.cfg.attention_backend != "legacy":
            event_count = events.shape[1]
            position = torch.arange(event_count, device=state.device)
            event_features = batch["event_features"].masked_fill(
                ~batch["event_mask"].unsqueeze(-1), 0.0
            )
            event_tokens = self.event_encoder(event_features) + events
            event_tokens = (
                event_tokens
                + self.event_positions(position).unsqueeze(0)
                + self.token_types.weight[1]
            )
            tokens = torch.cat(
                (
                    state.unsqueeze(1) + self.token_types.weight[0],
                    event_tokens,
                    memory + self.token_types.weight[2],
                ),
                dim=1,
            )
            token_mask = torch.cat(
                (torch.ones_like(valid[:, :1]), batch["event_mask"], valid), dim=1
            )
            tokens = tokens.masked_fill(~token_mask.unsqueeze(-1), 0.0)
            for block in self.blocks:
                tokens = block(tokens, token_mask)
            state = tokens[:, 0]
            memory = tokens[:, 1 + event_count :].masked_fill(~valid.unsqueeze(-1), 0.0)
        query = state.unsqueeze(1).expand_as(memory)
        compiled = self.compiler(
            torch.cat((query, memory, query * memory, memory_meta), -1)
        )
        relevance = self.relevance(compiled).squeeze(-1)
        weights = masked_softmax(relevance, valid)
        has_memory = valid.any(-1)
        summary = self.summary((compiled * weights.unsqueeze(-1)).sum(1))
        summary = torch.where(has_memory.unsqueeze(-1), summary, self.empty_summary)
        residual, alpha = self.adapter(state, summary, has_memory)
        state = state + residual
        if self.cfg.attention_backend == "legacy":
            state = state + self.attention(state, memory, valid)
            for block in self.blocks:
                state = block(state)
        state = self.output_norm(state)
        pre = self.pre_embedding(batch["pre_index"])
        pre_output = self.pre_head(torch.cat((state, summary, pre), -1))
        candidate_features = batch["candidates"].masked_fill(
            ~batch["candidate_mask"].unsqueeze(-1), 0.0
        )
        candidates = self.candidate_encoder(candidate_features)
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

    def attention_inventory(self) -> dict[str, int | str]:
        if self.cfg.attention_backend == "legacy":
            return {
                "backend": "legacy",
                "mita_layers": 0,
                "gqa_layers": 1,
                "total_attention_layers": 1,
            }
        mita = sum(isinstance(block.attention, MiTAAttention) for block in self.blocks)
        return {
            "backend": self.cfg.attention_backend,
            "mita_layers": mita,
            "gqa_layers": len(self.blocks) - mita,
            "total_attention_layers": len(self.blocks),
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
    # Intern immutable features only inside this call. Replayed snapshots often
    # share memories, events and actions; their labels/meta stay row-specific.
    # No process-global text cache or cross-scope feature reuse is introduced.
    features: dict[tuple[str, str], tuple[float, ...]] = {}

    def embedding(scope: str, text: str) -> tuple[float, ...]:
        key = (scope, text)
        if key not in features:
            features[key] = tuple(frozen_embedding(text, cfg.embedding_dim))
        return features[key]

    def state_vector(state: StateRecord) -> tuple[float, ...]:
        return embedding(state.scope, state_feature_text(state))

    address_cache: dict[tuple[str, tuple[str, ...]], list[list[int]]] = {}
    addresses = []
    for state in states:
        key = (state.scope, state.events)
        if key not in address_cache:
            address_cache[key] = engram_addresses(
                state.events,
                buckets=cfg.engram_buckets,
                heads=cfg.engram_heads,
                max_events=cfg.max_events,
                seed=cfg.seed,
            )
        addresses.append(address_cache[key])
    event_count = max(map(len, addresses))
    rows: dict[str, list] = {
        name: []
        for name in (
            "state",
            "events",
            "event_mask",
            "event_features",
            "memory_states",
            "memory_actions",
            "memory_meta",
            "memory_mask",
            "candidates",
            "candidate_mask",
        )
    }
    zero = (0.0,) * cfg.embedding_dim
    zero_address = [0] * (cfg.engram_heads * 2)
    for row, (state, snapshot) in enumerate(zip(states, snapshots)):
        events = state.events[-cfg.max_events :]
        rows["state"].append(state_vector(state))
        rows["events"].append(
            addresses[row] + [zero_address] * (event_count - len(addresses[row]))
        )
        rows["event_mask"].append(
            [True] * len(events) + [False] * (event_count - len(events))
        )
        rows["event_features"].append(
            [embedding(state.scope, event) for event in events]
            + [zero] * (event_count - len(events))
        )
        padding = memory_count - len(snapshot.memories)
        rows["memory_states"].append(
            [state_vector(memory.state) for memory in snapshot.memories]
            + [zero] * padding
        )
        rows["memory_actions"].append(
            [embedding(state.scope, memory.action_key) for memory in snapshot.memories]
            + [zero] * padding
        )
        meta = []
        for memory in snapshot.memories:
            age = math.log1p(max(0.0, state.observed_at - memory.settled_at)) / 10.0
            meta.append(
                [
                    max(-1.0, min(1.0, memory.return_g / cfg.return_scale)),
                    memory.reliability,
                    memory.similarity,
                    min(1.0, age),
                    float(memory.intervention == "SELECT_EVIDENCE"),
                    float(memory.intervention == "NO_OP"),
                ]
            )
        rows["memory_meta"].append(meta + [[0.0] * _META_DIM] * padding)
        rows["memory_mask"].append([True] * len(snapshot.memories) + [False] * padding)
        padding = candidate_count - len(candidates[row])
        rows["candidates"].append(
            [embedding(state.scope, key) for key in candidates[row]] + [zero] * padding
        )
        rows["candidate_mask"].append([True] * len(candidates[row]) + [False] * padding)
    # One allocation/conversion per field replaces hundreds of tiny tensor
    # allocations and indexed writes. The public tensor keys/shapes are stable.
    dtypes = {
        "state": torch.float32,
        "events": torch.long,
        "event_mask": torch.bool,
        "memory_mask": torch.bool,
        "candidate_mask": torch.bool,
    }
    return {
        **{
            name: torch.tensor(
                data,
                dtype=dtypes.get(name, torch.get_default_dtype()),
            )
            for name, data in rows.items()
        },
        "pre_index": torch.tensor([_PRE_INDEX[p] for p in pre_kinds], dtype=torch.long),
    }
