"""Value and advantage estimation for the current state.

This module implements steps 1-5 of Algorithm 1 of *Just-In-Time Reinforcement
Learning* (Li et al., ICML 2026):

* Eq. 4  -- state value V(s) = (1/|N(s)|) * sum G_i
* Eq. 5  -- known-action value Q(s, a) over N(s, a)
* Eq. 6  -- optimistic bonus V(s) + alpha / |N(s)| for unseen actions, taken
             with probability lambda
* Eq. 7  -- advantage A(s, a) = Q(s, a) - V(s)
* Eq. 25 -- normalised advantage A / (max|A| + eps)

Retrieval itself (Jaccard top-k, Eq. 24) lives in jitrl.index / jitrl.memory;
this module only consumes the resulting neighbourhood, so it stays pure and
deterministic. Randomness is injected as a random.Random instance, which keeps
the paper-exact exploration branch reproducible.

Engineering deviations from the paper, all selectable or documented:

1. Kernel weighting (weighting="kernel", the default). Eq. 4/5 average the
   neighbourhood uniformly. Averaging with the retrieval weight
   (similarity * recency) instead is a Nadaraya-Watson kernel smoother: it is
   still consistent under the paper's kNN regime (Assumption C.3) but has lower
   bias, because a neighbour at the similarity threshold is not trusted as much
   as an exact match -- which matters a lot here, since chat states are noisier
   than WebArena URLs. The important fix over the previous implementation is
   that V and Q now always use the *same* weighting scheme; before, a group
   whose weights summed to zero silently fell back to an unweighted mean while
   V stayed weighted, so A = Q - V compared two different estimators.
   weighting="uniform" restores the paper's exact arithmetic.

2. Empirically-shrunk thin support. The paper's min_count knife-edge decision
   (below it an action is treated as unseen and may receive a random optimism
   bonus) is replaced by a conjugate-style shrinkage towards V(s):
   Q = (n * Q_empirical + kappa * V) / (n + kappa) with kappa = min_count. With
   the default min_count = 1 nothing changes; with min_count > 1 a single lucky
   observation no longer flips the advantage between +1 and -1 from one request
   to the next. Actions with *no* support still take the exploration branch.

3. Expected exploration instead of a sampled one
   (exploration_estimate="expected", the default). The paper draws, per
   candidate and per step, between Q = V + alpha/|N| with probability lambda and
   Q = 0 otherwise. Two problems follow. First, its neutral branch sets Q = 0,
   so A = Q - V = -V: in a state with positive value an untried action is
   punished harder than any observed action, and in a negative-value state it is
   rewarded, which is the opposite of "prevent over-exploration". Second, a
   plugin cannot sample the next action -- it only injects evidence into another
   model's prompt -- so a sampled estimate is pure variance: the same state
   yields +1 on one request and -1 on the next.
   This implementation therefore keeps the optimistic branch, uses the neutral
   baseline Q = V (so A = 0 rather than -V) and takes the expectation,
   A_unseen = lambda * alpha / |N(s)|: monotone, bounded, and constant for a
   fixed neighbourhood. exploration_estimate="sample" restores the paper's
   sampling, including its Q = 0 branch, for ablation studies.

4. Normalisation floor. Eq. 25 divides by max|A|, which amplifies pure noise
   when every advantage is ~0 (identical stored returns). Below
   normalize_floor the advantages are reported as zero instead, so the policy
   update becomes a no-op rather than an arbitrary shake-up. The unnormalised
   scale is exposed as ValueEstimate.max_abs_advantage for evidence gating.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from .models import Neighbor, ValueEstimate

EPSILON: float = 1e-8
"""Denominator guard added to max|A| in Eq. 25."""

DEFAULT_NORMALIZE_FLOOR: float = 1e-3
"""Below this max|A| the normalised advantage is reported as zero."""

MIN_EVIDENCE_SIGNAL: float = 0.05
"""Raw advantage floor for prompt evidence and hard tool masking."""

_WEIGHTING_MODES: frozenset[str] = frozenset({"kernel", "uniform"})
_EXPLORATION_MODES: frozenset[str] = frozenset({"expected", "sample"})


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
    weighting: str = "kernel",
    exploration_estimate: str = "expected",
    normalize_floor: float = DEFAULT_NORMALIZE_FLOOR,
) -> ValueEstimate:
    """Estimate V(s), Q(s, a), A(s, a) and A-tilde(s, a) for one state.

    Args:
        neighbors: Retrieved neighbourhood N(s), already top-k by Jaccard
            similarity (Eq. 24).
        candidate_keys: Candidate action set C; duplicates are dropped while the
            first-seen order is preserved.
        k: Neighbourhood size k of the retrieval layer. It is accepted for
            call-site symmetry with RetrievalConfig.k; neighbors is already
            truncated by the memory layer, so no row is dropped here.
        ucb_alpha: Optimism scale alpha of Eq. 6.
        exploration_rate: Probability lambda of taking the optimism bonus. In
            the expected mode it is the mixing weight of the optimistic value.
        normalize: Apply Eq. 25 to fill advantage_norm.
        min_count: Minimum number of supporting neighbours before an action is
            trusted. Below it the estimate shrinks towards V(s); zero support
            still takes the exploration branch. Values below one are clamped to
            one.
        rng: Random source for the sampled exploration branch; a fresh
            random.Random instance is used when omitted. The expected branch is
            deterministic and ignores it.
        epsilon: Denominator guard used by Eq. 25.
        weighting: "kernel" weights neighbours by similarity_weight (the
            default); "uniform" restores the paper's exact 1/|N| arithmetic.
        exploration_estimate: "expected" computes the closed-form expectation of
            the exploration branch; "sample" restores the paper's Bernoulli
            draw, including its Q = 0 neutral branch.
        normalize_floor: max|A| below which normalisation is skipped and every
            normalised advantage is reported as zero.

    Returns:
        A ValueEstimate whose q_hat, advantage and advantage_norm dicts follow
        the de-duplicated candidate order. When neighbor_count is zero every
        estimate is zero and no optimism bonus is granted, because
        alpha / |N(s)| would be undefined.
    """
    unique_keys: list[str] = []
    seen_keys: set[str] = set()
    for key in candidate_keys:
        if key not in seen_keys:
            seen_keys.add(key)
            unique_keys.append(key)

    neighbor_count = len(neighbors)
    if neighbor_count:
        mean_similarity = (
            sum(neighbor.similarity for neighbor in neighbors) / neighbor_count
        )
    else:
        mean_similarity = 0.0

    mode = weighting if weighting in _WEIGHTING_MODES else "kernel"
    explore = (
        exploration_estimate
        if exploration_estimate in _EXPLORATION_MODES
        else "expected"
    )
    min_support = max(min_count, 1)
    index_count = max(neighbor_count, 1)
    bonus = ucb_alpha / index_count

    # One weight per neighbour, shared by V and every Q group so the two
    # estimates always live on the same scale.
    if mode == "uniform" or neighbor_count == 0:
        weights = [1.0] * neighbor_count
    else:
        weights = [
            neighbor.similarity_weight if neighbor.similarity_weight > 0.0 else 0.0
            for neighbor in neighbors
        ]
        if sum(weights) <= 0.0:
            # Every row reports a non-positive weight: degrade to the paper's
            # uniform mean instead of dividing by zero.
            weights = [1.0] * neighbor_count
    total_weight = sum(weights)

    if neighbor_count and total_weight > 0.0:
        weighted_return = 0.0
        for position, neighbor in enumerate(neighbors):
            weighted_return += weights[position] * neighbor.discounted_return
        v_hat = weighted_return / total_weight
    else:
        v_hat = 0.0

    # Aggregate the neighbourhood once: per action we need the support count,
    # the kernel mass and the kernel-weighted return. Using the same weights as
    # V(s) is what keeps A = Q - V a comparison of like with like.
    support: dict[str, int] = {}
    group_weight: dict[str, float] = {}
    group_return: dict[str, float] = {}
    plain_return: dict[str, float] = {}
    for position, neighbor in enumerate(neighbors):
        key = neighbor.action_key
        weight = weights[position]
        support[key] = support.get(key, 0) + 1
        group_weight[key] = group_weight.get(key, 0.0) + weight
        group_return[key] = group_return.get(key, 0.0) + weight * (
            neighbor.discounted_return
        )
        plain_return[key] = plain_return.get(key, 0.0) + neighbor.discounted_return

    generator = rng if rng is not None else random.Random()
    q_hat: dict[str, float] = {}
    explored: set[str] = set()
    for key in unique_keys:
        count = support.get(key, 0)
        if count == 0:
            rate = max(0.0, exploration_rate)
            if neighbor_count == 0:
                # Divergence guard: alpha / |N(s)| is undefined.
                q_hat[key] = 0.0
            elif explore == "sample":
                if rate > 0.0 and generator.random() < rate:
                    q_hat[key] = v_hat + bonus
                    explored.add(key)
                else:
                    # Paper's lambda = 0 case: the neutral branch sets Q = 0.
                    q_hat[key] = 0.0
            else:
                # Expected optimistic branch with the neutral baseline Q = V, so
                # A = lambda * alpha / |N|: zero at lambda = 0, positive above.
                q_hat[key] = v_hat + rate * bonus
                if rate > 0.0 and ucb_alpha > 0.0:
                    explored.add(key)
            continue

        mass = group_weight.get(key, 0.0)
        if mass > 0.0:
            q_empirical = group_return[key] / mass
        else:
            # A group whose rows all carry zero kernel weight (only reachable
            # with similarity 0) still has real observations: fall back to the
            # paper's plain mean for that group alone.
            q_empirical = plain_return[key] / count
        if count >= min_support:
            q_hat[key] = q_empirical
        else:
            # Empirical-Bayes shrinkage: kappa pseudo-observations at V(s).
            kappa = float(min_support)
            q_hat[key] = (count * q_empirical + kappa * v_hat) / (count + kappa)

    advantage = {key: q_hat[key] - v_hat for key in unique_keys}

    max_abs = max((abs(value) for value in advantage.values()), default=0.0)
    if normalize:
        floor = normalize_floor if normalize_floor > 0.0 else 0.0
        if max_abs > floor:
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
        action_counts={key: support.get(key, 0) for key in unique_keys},
        explored=explored,
        neighbor_count=neighbor_count,
        mean_similarity=mean_similarity,
        max_abs_advantage=max_abs,
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
