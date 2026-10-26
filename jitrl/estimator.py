"""Value and advantage estimation for the current state.

This module implements steps 1-5 of Algorithm 1 of *Just-In-Time Reinforcement
Learning* (Li et al., ICML 2026):

* Eq. 4  -- state value V(s) = (1/|N(s)|) * sum G_i
* Eq. 5  -- known-action value Q(s, a) over N(s, a)
* Eq. 6  -- optimistic bonus V(s) + alpha / |N(s)| for unseen actions, taken
             with probability lambda
* Eq. 7  -- advantage A(s, a) = Q(s, a) - V(s)
* Eq. 25 -- normalised advantage A / (max|A| + eps)

Retrieval itself (Jaccard top-k, Eq. 24) lives in jitrl.memory; this module
only consumes the resulting neighbourhood, so it stays pure and deterministic.
Randomness is injected as a random.Random instance: a fixed seed reproduces the
exact estimates, which keeps the optimism branch unit-testable.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from .models import Neighbor, ValueEstimate

EPSILON: float = 1e-8
"""Denominator guard added to max|A| in Eq. 25."""


def estimate_values(
    neighbors: Sequence[Neighbor],
    candidate_keys: Sequence[str],
    *,
    k: int = 10,
    ucb_alpha: float = 5.0,
    exploration_rate: float = 0.3,
    normalize: bool = True,
    min_count: int = 1,
    rng: random.Random | None = None,
    epsilon: float = EPSILON,
) -> ValueEstimate:
    """Estimate V(s), Q(s, a), A(s, a) and A-tilde(s, a) for one state.

    The paper's uniform averages 1/|N| are implemented as normalised similarity
    weights: the Neighbor.similarity_weight field is used when positive and the
    average degrades to an unweighted mean when no positive weight exists.

    Args:
        neighbors: Retrieved neighbourhood N(s), already top-k by Jaccard
            similarity (Eq. 24).
        candidate_keys: Candidate action set C; duplicates are dropped while the
            first-seen order is preserved.
        k: Neighbourhood size k of the retrieval layer. It is accepted for
            call-site symmetry with RetrievalConfig.k; neighbors is already
            truncated by the memory layer, so no row is dropped here.
        ucb_alpha: Optimism scale alpha of Eq. 6.
        exploration_rate: Probability lambda of taking the optimism bonus.
        normalize: Apply Eq. 25 to fill advantage_norm.
        min_count: Minimum number of supporting neighbours before an action
            counts as known; below it the Eq. 6 branch is used. Values below one
            are clamped to one, so an action without any neighbour is always
            treated as unknown.
        rng: Random source for the Eq. 6 draw; a fresh random.Random instance is
            used when omitted.
        epsilon: Denominator guard used by Eq. 25.

    Returns:
        A ValueEstimate whose q_hat, advantage and advantage_norm dicts follow
        the de-duplicated candidate order. When neighbor_count is zero every
        estimate is zero and no optimism bonus is granted, because
        alpha / |N(s)| would be undefined.
    """
    # De-duplicate the candidate set while keeping the caller's order, so the
    # returned dicts line up with the candidate list used downstream.
    unique_keys: list[str] = []
    seen_keys: set[str] = set()
    for key in candidate_keys:
        if key not in seen_keys:
            seen_keys.add(key)
            unique_keys.append(key)

    neighbor_count = len(neighbors)
    mean_similarity = (
        sum(neighbor.similarity for neighbor in neighbors) / neighbor_count
        if neighbor_count
        else 0.0
    )

    # Eq. 4: V(s) is a similarity-weighted mean of the stored returns G_i.
    total_weight = 0.0
    weighted_return = 0.0
    for neighbor in neighbors:
        weight = neighbor.similarity_weight if neighbor.similarity_weight > 0.0 else 0.0
        total_weight += weight
        weighted_return += weight * neighbor.discounted_return
    if total_weight > 0.0:
        v_hat = weighted_return / total_weight
    elif neighbor_count:
        # Every stored weight is non-positive: degrade to an unweighted mean.
        v_hat = (
            sum(neighbor.discounted_return for neighbor in neighbors) / neighbor_count
        )
    else:
        v_hat = 0.0

    grouped = aggregate_by_action(neighbors)
    rng = rng if rng is not None else random.Random()
    min_support = max(min_count, 1)

    q_hat: dict[str, float] = {}
    explored: set[str] = set()
    for key in unique_keys:
        group = grouped.get(key, [])
        if len(group) >= min_support:
            # Eq. 5: Q(s, a) averages only the neighbours that took action a.
            group_weight = 0.0
            group_return = 0.0
            for neighbor in group:
                weight = (
                    neighbor.similarity_weight
                    if neighbor.similarity_weight > 0.0
                    else 0.0
                )
                group_weight += weight
                group_return += weight * neighbor.discounted_return
            if group_weight > 0.0:
                q_hat[key] = group_return / group_weight
            else:
                q_hat[key] = sum(
                    neighbor.discounted_return for neighbor in group
                ) / len(group)
        elif (
            neighbor_count
            and exploration_rate > 0.0
            and rng.random() < exploration_rate
        ):
            # Eq. 6: optimism for an action the neighbourhood has never tried.
            q_hat[key] = v_hat + ucb_alpha / neighbor_count
            explored.add(key)
        else:
            q_hat[key] = 0.0

    # Eq. 7: A(s, a) = Q(s, a) - V(s).
    advantage = {key: q_hat[key] - v_hat for key in unique_keys}

    if normalize:
        # Eq. 25: divide by max|A| so the update is scale-free. An all-tie
        # candidate set must yield zeros instead of 0/0 = NaN.
        max_abs = max((abs(value) for value in advantage.values()), default=0.0)
        if max_abs > 0.0:
            advantage_norm = {
                key: value / (max_abs + epsilon) for key, value in advantage.items()
            }
        else:
            advantage_norm = dict.fromkeys(unique_keys, 0.0)
    else:
        advantage_norm = dict(advantage)

    return ValueEstimate(
        v_hat=v_hat,
        q_hat=q_hat,
        advantage=advantage,
        advantage_norm=advantage_norm,
        action_counts={key: len(grouped.get(key, ())) for key in unique_keys},
        explored=explored,
        neighbor_count=neighbor_count,
        mean_similarity=mean_similarity,
    )


def aggregate_by_action(neighbors: Sequence[Neighbor]) -> dict[str, list[Neighbor]]:
    """Group neighbours by their action key, preserving retrieval order.

    Args:
        neighbors: Retrieved neighbourhood N(s).

    Returns:
        Mapping of action key to the neighbours that took that action, in the
        order the rows were retrieved.
    """
    grouped: dict[str, list[Neighbor]] = {}
    for neighbor in neighbors:
        grouped.setdefault(neighbor.action_key, []).append(neighbor)
    return grouped


def summarize_action(neighbors: Sequence[Neighbor]) -> dict[str, float]:
    """Describe the evidence behind a single action.

    Used for decision traces, hint text and tool gating. All values are plain
    (unweighted) statistics; count is returned as an int even though the mapping
    is annotated dict[str, float].

    Args:
        neighbors: All neighbours that took the same action.

    Returns:
        A dict with count, mean_return, mean_reward, positive_rate (share of
        neighbours with a positive return) and mean_similarity. All numbers are
        zero for an empty input.
    """
    count = len(neighbors)
    if count == 0:
        return {
            "count": 0,
            "mean_return": 0.0,
            "mean_reward": 0.0,
            "positive_rate": 0.0,
            "mean_similarity": 0.0,
        }
    return {
        "count": count,
        "mean_return": sum(neighbor.discounted_return for neighbor in neighbors)
        / count,
        "mean_reward": sum(neighbor.reward for neighbor in neighbors) / count,
        "positive_rate": sum(
            1 for neighbor in neighbors if neighbor.discounted_return > 0.0
        )
        / count,
        "mean_similarity": sum(neighbor.similarity for neighbor in neighbors) / count,
    }
