"""Closed-form policy update, i.e. Algorithm 1 step 6.

The paper's update (Eq. 9/10) is

    pi*(a|s) proportional to pi_theta(a|s) * exp(beta * A(s, a))

which is exactly a logit shift z'(s, a) = z(s, a) + beta * A-tilde(s, a)
followed by a softmax. For black-box models that expose no log-probabilities the
verbalised-logit variant of Section 5.1 is used: the model rates each candidate
with a 0-100 confidence and verbalized_to_logit turns that rating into a logit,
so the same closed-form update applies.

The candidate set follows Algorithm 1: C = C_LLM union {a_i in N(s)}, and
memory-only actions enter with base logit z(s, a) = 0.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from .models import Candidate, PolicyDecision, ValueEstimate


def verbalized_to_logit(confidence: float, *, floor: float = 1e-4) -> float:
    """Convert a verbalised confidence in [0, 1] into a logit.

    Args:
        confidence: Model-reported confidence that the action is the right one.
            Values below floor (including negatives and NaN) are clamped, values
            at or above one map to log(1) = 0.
        floor: Smallest confidence accepted, keeping the logarithm finite.

    Returns:
        log(max(confidence, floor)), i.e. a non-positive logit.
    """
    if math.isnan(confidence):
        confidence = 0.0
    if confidence >= 1.0:
        return 0.0
    return math.log(max(confidence, floor, 1e-12))


def softmax(
    logits: Mapping[str, float], *, temperature: float = 1.0
) -> dict[str, float]:
    """Numerically stable softmax over a keyed logit mapping.

    Args:
        logits: Logits keyed by action. Key insertion order is preserved in the
            result.
        temperature: Sampling temperature; values at or below zero are treated
            as one.

    Returns:
        A proper distribution. Degenerate inputs (all -inf, all NaN, empty)
        return a uniform distribution over the available keys, or an empty dict
        for empty input, instead of raising or producing NaN.
    """
    if not logits:
        return {}
    scale = temperature if temperature > 0.0 else 1.0
    scaled = {key: float(value) / scale for key, value in logits.items()}
    peak = max(scaled.values())
    if math.isnan(peak) or peak == -math.inf:
        # No usable ordering information: stay neutral instead of raising.
        share = 1.0 / len(scaled)
        return dict.fromkeys(scaled, share)

    weights: dict[str, float] = {}
    for key, value in scaled.items():
        if math.isnan(value):
            weights[key] = 0.0
        elif value == math.inf:
            # Only reachable when the peak is +inf; split the mass among them.
            weights[key] = 1.0
        else:
            weights[key] = math.exp(value - peak)
    total = sum(weights.values())
    if total <= 0.0 or math.isnan(total):
        share = 1.0 / len(weights)
        return dict.fromkeys(weights, share)
    return {key: weight / total for key, weight in weights.items()}


def build_candidates(
    *,
    llm_actions: Sequence[tuple[str, str, float]] = (),
    memory_actions: Mapping[str, tuple[str, int]] | None = None,
    llm_scores: Mapping[str, float] | None = None,
    limit: int = 3,
    memory_order: Sequence[str] | None = None,
) -> list[Candidate]:
    """Merge the LLM proposal set with the memory-derived action set.

    Implements the augmented candidate set of Algorithm 1:
    C = C_LLM union {a_i : (s_i, a_i, G_i) in N(s)}. Memory-only actions are
    initialised with base_logit = 0.0 exactly as in the paper.

    Args:
        llm_actions: (key, text, logit) triples proposed by the model. Duplicate
            keys keep their first occurrence.
        memory_actions: Aggregated memory actions as key -> (text, count).
        llm_scores: Optional verbalised confidences in [0, 1]; when present they
            override the base logit of the matching candidate via
            verbalized_to_logit.
        limit: Maximum size of the candidate set C. Values below one are
            clamped to one so the set cannot be emptied by configuration.
        memory_order: Optional priority order for memory-derived actions, e.g.
            the ranking produced by a first-pass value estimate. Actions absent
            from the list fall back to the support count.

    Returns:
        At most max(limit, 1) candidates. Without LLM proposals every memory-side
        action is returned by descending support (count, then key) truncated to
        the capacity, so the retrieved actions that Algorithm 1 adds to C are
        never silently dropped. With LLM proposals the LLM side is ranked by
        descending base logit and takes the available slots first, remaining
        slots go to memory-only actions by descending support, and when the LLM
        side fills every slot the weakest LLM candidate is replaced by the
        best-supported memory action so the memory side always keeps one seat.
    """
    memory = memory_actions or {}
    scores = llm_scores or {}

    merged: dict[str, Candidate] = {}
    for key, text, logit in llm_actions:
        if key in merged:
            continue
        merged[key] = Candidate(
            key=key,
            text=text or key,
            base_logit=float(logit),
            source="llm",
            count=0,
        )
    for key, (text, count) in memory.items():
        support = max(int(count), 0)
        existing = merged.get(key)
        if existing is None:
            merged[key] = Candidate(
                key=key,
                text=text or key,
                base_logit=0.0,
                source="memory",
                count=support,
            )
        else:
            existing.source = "both"
            existing.count = support
            if not existing.text:
                existing.text = text or key

    # Section 5.1: a verbalised confidence is the black-box substitute for a
    # log-probability, so it replaces the caller-supplied logit.
    for key, confidence in scores.items():
        candidate = merged.get(key)
        if candidate is None:
            continue
        value = float(confidence)
        value = min(max(value, 0.0), 1.0) if math.isfinite(value) else 0.0
        candidate.confidence = value
        candidate.base_logit = verbalized_to_logit(value)

    if not merged:
        return []

    # Clamp the capacity so a configured zero cannot erase the candidate set:
    # Algorithm 1 always keeps at least one action in C.
    capacity = max(limit, 1)

    llm_ranked = sorted(
        (
            candidate
            for candidate in merged.values()
            if candidate.source in {"llm", "both"}
        ),
        key=lambda candidate: candidate.base_logit,
        reverse=True,
    )
    # Memory-only actions, best supported first; the key breaks count ties so
    # the result is stable across runs and dictionary insertion orders. A caller
    # supplied priority order (the two-pass shortlist) takes precedence.
    priority = {key: position for position, key in enumerate(memory_order or ())}
    memory_only = sorted(
        (candidate for candidate in merged.values() if candidate.source == "memory"),
        key=lambda candidate: (
            priority.get(candidate.key, len(priority)),
            -candidate.count,
            candidate.key,
        ),
    )

    selected = llm_ranked[:capacity]
    if memory_only:
        free = capacity - len(selected)
        if free > 0:
            selected.extend(memory_only[:free])
        else:
            # Every slot went to the LLM side, but Algorithm 1 always augments
            # C with the retrieved actions: give the best-supported memory
            # action the seat of the weakest LLM candidate.
            weakest = min(
                range(len(selected)),
                key=lambda index: selected[index].base_logit,
            )
            selected[weakest] = memory_only[0]
    return selected


def update_logits(
    candidates: Sequence[Candidate],
    estimate: ValueEstimate,
    *,
    beta: float,
    normalize: bool = True,
) -> dict[str, float]:
    """Apply the closed-form logit shift z' = z + beta * A-tilde.

    Args:
        candidates: Candidate set C in presentation order.
        estimate: Value estimate carrying the advantages.
        beta: KL temperature beta of Eq. 9/10.
        normalize: Use advantage_norm (Eq. 25) instead of the raw advantage.

    Returns:
        Post-update logits keyed by candidate. Candidates absent from the
        estimate keep their base logit, i.e. their advantage is treated as zero.
    """
    advantages = estimate.advantage_norm if normalize else estimate.advantage
    logits: dict[str, float] = {}
    for candidate in candidates:
        advantage = float(advantages.get(candidate.key, 0.0))
        if not math.isfinite(advantage):
            advantage = 0.0
        logits[candidate.key] = candidate.base_logit + beta * advantage
    return logits


def select_action(
    candidates: Sequence[Candidate],
    estimate: ValueEstimate,
    *,
    beta: float = 1.5,
    temperature: float = 1.0,
    normalize: bool = True,
    mode: str = "hint",
    rng: random.Random | None = None,
) -> PolicyDecision:
    """Sample the next action from the updated policy.

    Args:
        candidates: Candidate set C in presentation order.
        estimate: Value estimate produced by jitrl.estimator.estimate_values.
        beta: KL temperature beta of the logit update.
        temperature: Softmax temperature applied to both distributions.
        normalize: Use advantage_norm (Eq. 25) for the update.
        mode: 'off' disables the update but still reports the distributions;
            'hint' and 'logit' sample from the updated distribution.
        rng: Random source for sampling; a fresh random.Random instance is used
            when omitted.

    Returns:
        A PolicyDecision with the sampled candidate, the prior and post-update
        distributions, both logit vectors and a short reason. With no candidates
        the decision carries chosen = None.
    """
    if not candidates:
        return PolicyDecision(mode=mode, estimate=estimate, beta=beta)

    logits_before = {candidate.key: candidate.base_logit for candidate in candidates}
    applied = mode != "off"
    logits_after = (
        update_logits(candidates, estimate, beta=beta, normalize=normalize)
        if applied
        else dict(logits_before)
    )
    prior = softmax(logits_before, temperature=temperature)
    probabilities = softmax(logits_after, temperature=temperature)

    chosen: Candidate | None = None
    if applied:
        generator = rng if rng is not None else random.Random()
        threshold = generator.random()
        cumulative = 0.0
        for candidate in candidates:
            cumulative += probabilities.get(candidate.key, 0.0)
            if threshold <= cumulative:
                chosen = candidate
                break
        if chosen is None:
            # Floating-point drift can leave the cumulative mass just below the
            # draw; fall back to the last candidate instead of returning None.
            chosen = candidates[-1]

    reason = ""
    if chosen is not None:
        advantages = estimate.advantage_norm if normalize else estimate.advantage
        advantage = float(advantages.get(chosen.key, 0.0))
        if not math.isfinite(advantage):
            advantage = 0.0
        reason = (
            f"picked {chosen.key} via beta={beta:g} advantage={advantage:+.2f} "
            f"(p={probabilities.get(chosen.key, 0.0):.2f} "
            f"vs prior {prior.get(chosen.key, 0.0):.2f})"
        )

    return PolicyDecision(
        chosen=chosen,
        probabilities=probabilities,
        prior=prior,
        beta=beta,
        mode=mode,
        estimate=estimate,
        logits_before=logits_before,
        logits_after=logits_after,
        reason=reason,
    )
