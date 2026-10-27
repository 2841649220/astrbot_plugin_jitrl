"""Masked MiTA attention, equations (5)-(12) of Wen et al. (2026).

This implementation keeps the paper's shared compression, reusable top-k KV
experts, one-expert routing and *joint* softmax. It uses masked 1D pooling for
the local Lens token sequence and grouped KV projections. It does not require
FlashAttention or claim the throughput of the paper's custom GPU kernels.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import LensConfig


def _masked_softmax(scores: Tensor, valid: Tensor) -> Tensor:
    valid = valid.expand_as(scores)
    any_valid = valid.any(-1, keepdim=True)
    masked = scores.masked_fill(~valid, -torch.inf)
    masked = torch.where(any_valid, masked, torch.zeros_like(masked))
    return torch.softmax(masked, dim=-1).masked_fill(~valid, 0.0)


def masked_landmarks(query: Tensor, valid: Tensor, count: int) -> tuple[Tensor, Tensor]:
    """Average equal windows of the *valid* query order, ignoring padding.

    For a sample with L valid positions, valid rank r belongs to window
    floor(r*count/L). Non-divisible lengths differ by at most one position;
    empty windows are masked. Inserting padding does not change any window.
    """
    batch, heads, length, dimension = query.shape
    # Half atomics lose enough bits to change hard top-k/routes between
    # identical forwards. Accumulate pooling in float32 before casting back.
    accumulator_dtype = (
        torch.float32 if query.dtype in {torch.float16, torch.bfloat16} else query.dtype
    )
    clean = query.masked_fill(~valid[:, None, :, None], 0.0).to(accumulator_dtype)
    population = valid.sum(-1).clamp_min(1)
    rank = (valid.long().cumsum(-1) - 1).clamp_min(0)
    window = torch.div(rank * count, population[:, None], rounding_mode="floor")
    window = window.clamp_max(count - 1)
    index = window[:, None, :, None].expand(batch, heads, length, dimension)
    pooled = torch.zeros(
        batch, heads, count, dimension, device=query.device, dtype=accumulator_dtype
    ).scatter_add(2, index, clean)
    sizes = torch.zeros(
        batch, count, device=query.device, dtype=accumulator_dtype
    ).scatter_add(1, window, valid.to(accumulator_dtype))
    pooled = pooled / sizes[:, None, :, None].clamp_min(1)
    return pooled.to(query.dtype), sizes > 0


def mita_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    valid: Tensor,
    landmarks: int = 8,
    topk: int = 16,
    query_chunk: int = 64,
    *,
    return_diagnostics: bool = True,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute self-attention with one routed KV expert per query.

    Q/K/V are [B,H,N,D]; valid is [B,N], True for known tokens. Only
    [B,H,m,N], [B,H,N,m] and chunked [B,H,C,k] score tensors are formed.
    Discrete top-k/argmax choices have their ordinary PyTorch gradients: none
    through the indices, gradients through the selected vectors and softmax.
    """
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("MiTA Q/K/V must share shape [B,H,N,D]")
    batch, heads, length, dimension = query.shape
    if valid.shape != (batch, length) or valid.dtype != torch.bool:
        raise ValueError("MiTA valid mask must be boolean [B,N]")
    if min(landmarks, topk, query_chunk, dimension) < 1:
        raise ValueError("MiTA budgets and head dimension must be positive")
    if not length:
        if not return_diagnostics:
            return torch.zeros_like(query), {}
        return torch.zeros_like(query), {
            "landmark_queries": query.new_zeros(batch, heads, landmarks, dimension),
            "landmark_values": value.new_zeros(batch, heads, landmarks, dimension),
            "landmark_valid": valid.new_zeros(batch, landmarks),
            "expert_indices": torch.empty(
                batch, heads, landmarks, 0, device=query.device, dtype=torch.long
            ),
            "expert_valid": valid.new_zeros(batch, heads, landmarks, 0),
            "routes": torch.empty(
                batch, heads, 0, device=query.device, dtype=torch.long
            ),
            "score_elements": torch.tensor(0, device=query.device),
            "peak_score_elements": torch.tensor(0, device=query.device),
        }
    query = query.masked_fill(~valid[:, None, :, None], 0.0)
    key = key.masked_fill(~valid[:, None, :, None], 0.0)
    value = value.masked_fill(~valid[:, None, :, None], 0.0)
    landmark_query, landmark_valid = masked_landmarks(query, valid, landmarks)
    scale = math.sqrt(dimension)
    # The same landmark-key scores construct exact compressed values and
    # identify each reusable expert's original KV entries (Eq. 7 and 8).
    compression_scores = landmark_query @ key.transpose(-1, -2) / scale
    compression_valid = valid[:, None, None, :] & landmark_valid[:, None, :, None]
    compression_weights = _masked_softmax(compression_scores, compression_valid)
    landmark_value = compression_weights @ value
    width = min(topk, length)
    indices = (
        compression_scores.masked_fill(~compression_valid, -torch.inf)
        .topk(width, dim=-1)
        .indices
    )
    vector_indices = indices.unsqueeze(-1).expand(
        batch, heads, landmarks, width, dimension
    )
    expert_key = (
        key.unsqueeze(2).expand(-1, -1, landmarks, -1, -1).gather(3, vector_indices)
    )
    expert_value = (
        value.unsqueeze(2).expand(-1, -1, landmarks, -1, -1).gather(3, vector_indices)
    )
    expert_valid = (
        valid[:, None, None, :]
        .expand(batch, heads, landmarks, length)
        .gather(3, indices)
    )
    expert_valid = expert_valid & landmark_valid[:, None, :, None]
    # These query-landmark logits are both the router and the shared expert
    # logits (Eq. 9). Reuse them instead of recomputing a query-key matrix.
    routing_scores = query @ landmark_query.transpose(-1, -2) / scale
    routes = routing_scores.masked_fill(
        ~landmark_valid[:, None, None, :], -torch.inf
    ).argmax(-1)
    chunks = []
    for start in range(0, length, query_chunk):
        end = min(length, start + query_chunk)
        count = end - start
        selected = routes[:, :, start:end]
        chosen_vectors = selected[..., None, None].expand(
            batch, heads, count, width, dimension
        )
        routed_key = expert_key.gather(2, chosen_vectors)
        routed_value = expert_value.gather(2, chosen_vectors)
        routed_valid = expert_valid.gather(
            2, selected[..., None].expand(batch, heads, count, width)
        )
        local_scores = (query[:, :, start:end, None, :] * routed_key).sum(-1) / scale
        scores = torch.cat((routing_scores[:, :, start:end], local_scores), dim=-1)
        shared_valid = landmark_valid[:, None, None, :].expand(
            batch, heads, count, landmarks
        )
        support = torch.cat((shared_valid, routed_valid), dim=-1)
        support = support & valid[:, None, start:end, None]
        # A SINGLE denominator covers compressed and original entries. Averaging
        # two independently normalized attentions would not implement Eq. 10.
        weights = _masked_softmax(scores, support)
        shared_output = weights[..., :landmarks] @ landmark_value
        routed_output = (weights[..., landmarks:, None] * routed_value).sum(-2)
        chunks.append(shared_output + routed_output)
    result = torch.cat(chunks, dim=2).masked_fill(~valid[:, None, :, None], 0.0)
    if not return_diagnostics:
        return result, {}
    diagnostics = {
        "landmark_queries": landmark_query.detach(),
        "landmark_values": landmark_value.detach(),
        "landmark_valid": landmark_valid.detach(),
        "expert_indices": indices.detach(),
        "expert_valid": expert_valid.detach(),
        "routes": routes.detach().masked_fill(~valid[:, None, :], -1),
        # Counts describe dot-product score elements, not overall FLOPs, peak
        # allocator bytes or a guarantee of end-to-end speedup.
        "score_elements": torch.tensor(
            batch * heads * length * (2 * landmarks + width), device=query.device
        ),
        "peak_score_elements": torch.tensor(
            batch
            * heads
            * max(landmarks * length, min(query_chunk, length) * (landmarks + width)),
            device=query.device,
        ),
    }
    return result, diagnostics


class _GroupedProjections(nn.Module):
    def __init__(self, cfg: LensConfig) -> None:
        super().__init__()
        self.query_heads = cfg.query_heads
        self.kv_heads = cfg.kv_heads
        self.head_dim = cfg.hidden_dim // cfg.query_heads
        self.query = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
        self.key = nn.Linear(cfg.hidden_dim, cfg.kv_heads * self.head_dim, bias=False)
        self.value = nn.Linear(cfg.hidden_dim, cfg.kv_heads * self.head_dim, bias=False)
        self.output = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)

    def project(self, tokens: Tensor, valid: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if (
            tokens.ndim != 3
            or valid.shape != tokens.shape[:2]
            or valid.dtype != torch.bool
        ):
            raise ValueError("attention tokens/mask must be [B,N,d] / bool[B,N]")
        batch, length, _ = tokens.shape
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.0)
        query = (
            self.query(tokens)
            .reshape(batch, length, self.query_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.key(tokens)
            .reshape(batch, length, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.value(tokens)
            .reshape(batch, length, self.kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        copies = self.query_heads // self.kv_heads
        return (
            query,
            key.repeat_interleave(copies, dim=1),
            value.repeat_interleave(copies, dim=1),
        )

    def merge(self, context: Tensor, valid: Tensor) -> Tensor:
        batch, _, length, _ = context.shape
        context = context.transpose(1, 2).reshape(
            batch, length, self.query_heads * self.head_dim
        )
        return self.output(context).masked_fill(~valid.unsqueeze(-1), 0.0)


class MiTAAttention(_GroupedProjections):
    """Content-routed, compressed and sparse token self-attention for the Lens."""

    def __init__(self, cfg: LensConfig) -> None:
        super().__init__(cfg)
        self.landmarks = cfg.mita_landmarks
        self.topk = cfg.mita_topk
        self.query_chunk = cfg.mita_query_chunk
        self.execution = cfg.mita_execution

    def _context(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        valid: Tensor,
        *,
        diagnostics: bool,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if self.execution == "reference":
            return mita_attention(
                query,
                key,
                value,
                valid,
                self.landmarks,
                self.topk,
                self.query_chunk,
                return_diagnostics=diagnostics,
            )
        # The same MiTA support and joint normalizer, executed as expert query
        # blocks so K/V are reused instead of being copied for every query.
        from .mita_grouped import grouped_mita_attention

        block_size = (
            min(self.query_chunk, 32) if query.shape[2] <= 256 else self.query_chunk
        )
        return grouped_mita_attention(
            query,
            key,
            value,
            valid,
            self.landmarks,
            self.topk,
            block_size,
            return_diagnostics=diagnostics,
            static_pack=query.is_cuda and query.shape[2] >= 4096,
        )

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        query, key, value = self.project(tokens, valid)
        context, _ = self._context(query, key, value, valid, diagnostics=False)
        return self.merge(context, valid)

    def inspect_routing(self, tokens: Tensor, valid: Tensor) -> dict[str, Tensor]:
        query, key, value = self.project(tokens, valid)
        _, diagnostics = self._context(query, key, value, valid, diagnostics=True)
        return diagnostics


class GroupedSelfAttention(_GroupedProjections):
    """Matched-projection full GQA control, with padding-safe empty rows."""

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        query, key, value = self.project(tokens, valid)
        if not tokens.shape[1]:
            return torch.zeros_like(tokens)
        # Some older SDPA backends return NaN for an all-False key mask. A
        # zero-valued dummy key makes that row finite; merge still zeros every
        # invalid query, so the dummy cannot become evidence or an output.
        support = valid.clone()
        support[:, 0] = support[:, 0] | ~valid.any(-1)
        context = F.scaled_dot_product_attention(
            query, key, value, attn_mask=support[:, None, None, :]
        )
        return self.merge(context, valid)
