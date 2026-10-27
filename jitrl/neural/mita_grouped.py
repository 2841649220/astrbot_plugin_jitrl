"""Grouped-query MiTA prototype using one batched SDPA for routed experts.

The MiTA equations are unchanged. Queries are sorted by their chosen expert,
packed into blocks, and restored to their original order. Each block reuses a
single concatenated shared+original KV expert rather than copying K/V for each
query. Dynamic packing requires one device-to-host size synchronization;
optional static packing removes it by padding a proven block-count bound.
Padding and sorting remain measurable costs in both variants.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from .attention import _masked_softmax, masked_landmarks


def grouped_mita_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    valid: Tensor,
    landmarks: int = 8,
    topk: int = 16,
    query_chunk: int = 64,
    *,
    return_diagnostics: bool = True,
    static_pack: bool = False,
) -> tuple[Tensor, dict[str, Tensor]]:
    """MiTA Eq. 5-12 with packed expert query blocks and joint fused SDPA.

    Q/K/V are [B,H,N,D], valid is bool[B,N]. query_chunk is the number of
    query slots in one packed expert block. It controls padding versus KV
    reuse, not the mathematical result. static_pack allocates a proven upper
    bound of blocks and pads a dummy expert, avoiding ragged host size sync.
    No N-by-N full-attention fallback is
    used; original KV experts remain selected by landmark top-k and routing.
    """
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("MiTA Q/K/V must share shape [B,H,N,D]")
    batch, heads, length, dimension = query.shape
    if valid.shape != (batch, length) or valid.dtype != torch.bool:
        raise ValueError("MiTA valid mask must be boolean [B,N]")
    if min(landmarks, topk, query_chunk, dimension) < 1:
        raise ValueError("MiTA budgets and head dimension must be positive")
    # A short sequence cannot fill a longer expert query block. This static
    # shape bound avoids needless 64-slot blocks for e.g. a six-token request.
    query_chunk = min(query_chunk, max(1, length))
    query = query.masked_fill(~valid[:, None, :, None], 0.0)
    key = key.masked_fill(~valid[:, None, :, None], 0.0)
    value = value.masked_fill(~valid[:, None, :, None], 0.0)
    landmark_query, landmark_valid = masked_landmarks(query, valid, landmarks)
    scale = math.sqrt(dimension)
    compression_scores = landmark_query @ key.transpose(-1, -2) / scale
    compression_valid = valid[:, None, None, :] & landmark_valid[:, None, :, None]
    landmark_value = _masked_softmax(compression_scores, compression_valid) @ value
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
    routing_scores = query @ landmark_query.transpose(-1, -2) / scale
    routes = routing_scores.masked_fill(
        ~landmark_valid[:, None, None, :], -torch.inf
    ).argmax(-1)

    head_count = batch * heads
    groups = head_count * landmarks
    query_valid = valid[:, None, :].expand(batch, heads, length).reshape(-1)
    group_id = (
        routes.reshape(head_count, length)
        + torch.arange(head_count, device=query.device)[:, None] * landmarks
    )
    group_id = group_id.reshape(-1)
    counts = torch.zeros(groups, device=query.device, dtype=torch.long).scatter_add(
        0, group_id, query_valid.long()
    )
    blocks_per_group = torch.div(
        counts + query_chunk - 1, query_chunk, rounding_mode="floor"
    )
    # One scalar synchronization determines a ragged allocation. There are no
    # per-expert item()/Python loops, and no routing choices are differentiated.
    if static_pack:
        # Sum_e ceil(n_e/C) <= ceil(N/C)+m-1, and at most N nonempty
        # blocks exist for one head. Allocate the known per-head bound, then
        # route spare block slots to a zero-valued dummy expert.
        per_head_bound = min(
            length, (length + query_chunk - 1) // query_chunk + landmarks - 1
        )
        block_count = head_count * per_head_bound
    else:
        block_count = int(blocks_per_group.sum().item())
    diagnostics = (
        {
            "landmark_queries": landmark_query.detach(),
            "landmark_values": landmark_value.detach(),
            "landmark_valid": landmark_valid.detach(),
            "expert_indices": indices.detach(),
            "expert_valid": expert_valid.detach(),
            "routes": routes.detach().masked_fill(~valid[:, None, :], -1),
            "group_counts": counts.reshape(batch, heads, landmarks).detach(),
            "packed_blocks": torch.tensor(block_count, device=query.device),
            "packed_query_slots": torch.tensor(
                block_count * query_chunk, device=query.device
            ),
            "host_size_syncs": torch.tensor(
                0 if static_pack else 1, device=query.device
            ),
            "active_packed_blocks": blocks_per_group.sum().detach(),
        }
        if return_diagnostics
        else {}
    )
    if not block_count:
        # Preserve an ordinary zero gradient for Q/K/V on an all-empty batch.
        if return_diagnostics:
            diagnostics["score_elements"] = torch.tensor(
                2 * batch * heads * landmarks * length, device=query.device
            )
            diagnostics["peak_score_elements"] = torch.tensor(
                batch * heads * landmarks * length, device=query.device
            )
        return (query + key + value) * 0.0, diagnostics
    # Stable group sorting preserves the query order within every expert. Pad
    # queries get a sentinel group after all valid entries and write only zero.
    sorting_groups = group_id.masked_fill(~query_valid, groups)
    order = torch.argsort(sorting_groups, stable=True)
    starts = counts.cumsum(0) - counts
    sorted_groups = sorting_groups[order].clamp_max(groups - 1)
    sorted_rank = (
        torch.arange(order.numel(), device=query.device) - starts[sorted_groups]
    )
    rank = torch.empty_like(sorted_rank).scatter(0, order, sorted_rank)
    block_offsets = blocks_per_group.cumsum(0) - blocks_per_group
    query_block = block_offsets[group_id] + torch.div(
        rank, query_chunk, rounding_mode="floor"
    )
    query_slot = query_block * query_chunk + rank.remainder(query_chunk)
    query_slot = query_slot.masked_fill(~query_valid, 0)
    slots = query_slot[:, None].expand(-1, dimension)
    packed_query = query.new_zeros(block_count * query_chunk, dimension).scatter_add(
        0, slots, query.reshape(-1, dimension)
    )
    if static_pack:
        repeats = torch.cat(
            (blocks_per_group, (block_count - blocks_per_group.sum()).reshape(1))
        )
        block_group = torch.repeat_interleave(
            torch.arange(groups + 1, device=query.device),
            repeats,
            output_size=block_count,
        )
    else:
        block_group = torch.repeat_interleave(
            torch.arange(groups, device=query.device),
            blocks_per_group,
            output_size=block_count,
        )

    shared_key = landmark_query.unsqueeze(2).expand(-1, -1, landmarks, -1, -1)
    shared_value = landmark_value.unsqueeze(2).expand(-1, -1, landmarks, -1, -1)
    merged_key = torch.cat((shared_key, expert_key), dim=3).reshape(
        groups, landmarks + width, dimension
    )
    merged_value = torch.cat((shared_value, expert_value), dim=3).reshape(
        groups, landmarks + width, dimension
    )
    shared_valid = landmark_valid[:, None, None, :].expand(
        batch, heads, landmarks, landmarks
    )
    merged_valid = torch.cat((shared_valid, expert_valid), dim=-1).reshape(
        groups, landmarks + width
    )
    if static_pack:
        merged_key = torch.cat(
            (merged_key, key.new_zeros(1, landmarks + width, dimension)), dim=0
        )
        merged_value = torch.cat(
            (merged_value, value.new_zeros(1, landmarks + width, dimension)), dim=0
        )
        dummy_valid = torch.zeros(
            1, landmarks + width, dtype=torch.bool, device=query.device
        )
        dummy_valid[:, 0] = True
        merged_valid = torch.cat((merged_valid, dummy_valid), dim=0)
    block_key = merged_key.index_select(0, block_group)
    block_value = merged_value.index_select(0, block_group)
    block_valid = merged_valid.index_select(0, block_group)
    # Every packed block has an actually valid query, hence at least one valid
    # shared key. This keeps SDPA finite even on older all-masked-row backends.
    packed_output = F.scaled_dot_product_attention(
        packed_query.reshape(block_count, 1, query_chunk, dimension),
        block_key.unsqueeze(1),
        block_value.unsqueeze(1),
        attn_mask=block_valid[:, None, None, :],
    ).reshape(-1, dimension)
    result = packed_output.gather(0, slots).reshape(batch, heads, length, dimension)
    result = result.masked_fill(~valid[:, None, :, None], 0.0)
    # Logical score capacities include padding in SDPA; fused kernels need not
    # materialize that score matrix, so these are not allocator peak bytes.
    if return_diagnostics:
        diagnostics["score_elements"] = torch.tensor(
            2 * batch * heads * landmarks * length
            + block_count * query_chunk * (landmarks + width),
            device=query.device,
        )
        diagnostics["peak_score_elements"] = torch.tensor(
            max(
                batch * heads * landmarks * length,
                block_count * query_chunk * (landmarks + width),
            ),
            device=query.device,
        )
    return result, diagnostics
