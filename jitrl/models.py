"""Shared data contracts for the JitRL core.

These dataclasses are the wire format between the learning loop
(`trajectory` + `evaluator`), the memory bank (`memory`), the value
estimator (`estimator`) and the policy updater (`policy`).

Everything here is pure data: no I/O, no LLM calls, no AstrBot imports, so the
core can be unit-tested without a running bot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

ActionKind = Literal["tool", "reply", "command", "custom"]
"""What kind of agent action an experience row describes."""

Outcome = Literal["ok", "error", "unknown"]
"""Observed result of executing an action."""

RewardSource = Literal["llm", "heuristic", "user", "fused"]
"""Where a step reward came from; kept for auditing / ablation."""

PolicyMode = Literal["off", "hint", "logit"]
"""How the policy update is expressed towards the downstream agent."""


@dataclass(slots=True, frozen=True)
class StateView:
    """A compact, retrieval-ready abstraction of one agent observation.

    Mirrors the paper's state construction: raw observations are compressed to
    a representation that maps *functionally equivalent* states close together
    so that Jaccard similarity (Eq. 24) retrieves useful neighbours.

    Attributes:
        signature: Canonical, variable-normalised string form of the state.
            Volatile identifiers (numbers, emails, URLs, quoted payloads) are
            replaced by placeholders, exactly like the paper's *Regularized
            URL* trick but for arbitrary text.
        tokens: Token set used for Jaccard similarity.
        text: Short human-readable digest, safe to embed in a prompt.
        surface: Coarse partition key (e.g. the platform or task family) used
            by the optional first-stage retrieval gate.
        raw_chars: Length of the original observation, kept for statistics.
    """

    signature: str
    tokens: frozenset[str]
    text: str
    surface: str = ""
    raw_chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage.

        Returns:
            A JSON-safe dict with the token set flattened to a sorted list.
        """
        return {
            "signature": self.signature,
            "tokens": sorted(self.tokens),
            "text": self.text,
            "surface": self.surface,
            "raw_chars": self.raw_chars,
        }


@dataclass(slots=True, frozen=True)
class ActionView:
    """A semantically normalised agent action.

    Mirrors the paper's *action handling*: instance-specific payloads (element
    ids, free text arguments) are normalised away so that an action learned in
    one episode is recognisable in the next.

    Attributes:
        kind: Category of the action.
        name: Tool name, or `"reply"` for a plain assistant message.
        key: Stable identity used for value aggregation, e.g.
            `"tool:web_search(query=<str>)"`.
        text: Human-readable rendering used in prompts and dashboards.
        args: Argument signature (values may be truncated placeholders).
    """

    kind: ActionKind
    name: str
    key: str
    text: str
    args: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Serialize for JSON storage.

        Returns:
            A JSON-safe dict.
        """
        return {
            "kind": self.kind,
            "name": self.name,
            "key": self.key,
            "text": self.text,
            "args": dict(self.args),
        }


@dataclass(slots=True)
class StepRecord:
    """One `(s_t, a_t, G_t)` transition of a finished episode.

    Attributes:
        index: Zero-based position inside the episode.
        state: Abstracted state observed *before* the action was taken.
        action: The action that was taken.
        reward: Step-wise reward `r_t` produced by the evaluator.
        discounted_return: `G_t = sum_{u>=t} gamma^(u-t) * r_u` (Eq. 3).
        outcome: Whether the action errored out.
        observation: Truncated observation produced by the action.
        reward_source: Provenance of `reward`.
    """

    index: int
    state: StateView
    action: ActionView
    reward: float = 0.0
    discounted_return: float = 0.0
    outcome: Outcome = "unknown"
    observation: str = ""
    reward_source: RewardSource = "heuristic"


@dataclass(slots=True)
class Episode:
    """A completed agent run, i.e. one trajectory `tau`.

    Attributes:
        episode_id: Unique id of the run.
        scope: Isolation key (usually the unified message origin).
        created_at: Unix timestamp of the run.
        user_text: Redacted user request that started the run.
        task_family: Coarse task label, used by the optional retrieval gate.
        success: Whether the task succeeded, when it could be determined.
        steps: Ordered transitions.
        meta: Extra bookkeeping (model, tool count, latency, ...).
    """

    episode_id: str
    scope: str
    created_at: float = field(default_factory=time.time)
    user_text: str = ""
    task_family: str = ""
    success: bool | None = None
    steps: list[StepRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neighbor:
    """A memory row retrieved as a neighbour of the query state.

    Attributes:
        entry_id: Primary key of the stored experience.
        similarity: Jaccard similarity `J(s, s_i)` (Eq. 24).
        action_key: Normalised action identity.
        action_text: Human-readable action.
        discounted_return: Stored `G_i`.
        similarity_weight: `similarity * recency_factor`, used as the sampling
            weight when averaging returns.
        reward: Stored step reward `r_i`.
        state_text: Short digest of the neighbour state.
        scope: Scope the row belongs to.
        created_at: Row timestamp.
    """

    entry_id: int
    similarity: float
    action_key: str
    action_text: str
    discounted_return: float
    similarity_weight: float
    reward: float = 0.0
    state_text: str = ""
    scope: str = ""
    created_at: float = 0.0


@dataclass(slots=True)
class ValueEstimate:
    """Result of Algorithm 1 steps 2-5 for one query state.

    Attributes:
        v_hat: State value estimate (Eq. 4).
        q_hat: Action value estimate per candidate action (Eq. 5/6).
        advantage: Raw advantage `A(s,a) = Q(s,a) - V(s)` (Eq. 7).
        advantage_norm: Normalised advantage `A / (max|A| + eps)` (Eq. 25).
        action_counts: Number of neighbours supporting each action.
        explored: Actions that received the optimism bonus.
        neighbor_count: Size of `N(s)`.
        mean_similarity: Average similarity of the retrieved neighbourhood.
        max_abs_advantage: max|A| before normalisation, i.e. the scale Eq. 25
            divides by. Exposed so downstream code can tell weak evidence
            (a few 0.01-scale advantages) from strong evidence.
    """

    v_hat: float = 0.0
    q_hat: dict[str, float] = field(default_factory=dict)
    advantage: dict[str, float] = field(default_factory=dict)
    advantage_norm: dict[str, float] = field(default_factory=dict)
    action_counts: dict[str, int] = field(default_factory=dict)
    explored: set[str] = field(default_factory=set)
    neighbor_count: int = 0
    mean_similarity: float = 0.0
    max_abs_advantage: float = 0.0

    @property
    def best_action(self) -> str | None:
        """Return the action with the largest normalised advantage.

        Returns:
            The winning action key, or `None` when no candidate exists.
        """
        if not self.advantage_norm:
            return None
        return max(self.advantage_norm, key=lambda key: self.advantage_norm[key])


@dataclass(slots=True)
class Candidate:
    """An action proposed for the current state.

    Attributes:
        key: Normalised action identity.
        text: Human-readable action.
        base_logit: `z(s, a)` of the frozen reference policy. Memory-only
            actions start at `0.0` exactly as in Algorithm 1.
        source: Where the candidate came from: `"llm"`, `"memory"` or
            `"both"`.
        count: Number of supporting neighbours.
        confidence: Verbalised likelihood in `[0, 1]` when the caller
            produced one; `None` otherwise.
    """

    key: str
    text: str
    base_logit: float = 0.0
    source: Literal["llm", "memory", "both"] = "llm"
    count: int = 0
    confidence: float | None = None


@dataclass(slots=True)
class PolicyDecision:
    """Output of Algorithm 1 step 6 plus the bookkeeping needed to explain it.

    Attributes:
        chosen: The sampled candidate, or `None` if there was nothing to do.
        probabilities: Post-update action distribution after the softmax.
        prior: Distribution implied by the base logits alone, for comparison.
        beta: Temperature used for the update.
        mode: Application mode the decision was produced for.
        estimate: Value estimate backing the decision.
        logits_before: `z(s, a)` before the update.
        logits_after: `z'(s, a)` after the update.
        reason: Short human-readable explanation of the choice.
    """

    chosen: Candidate | None = None
    probabilities: dict[str, float] = field(default_factory=dict)
    prior: dict[str, float] = field(default_factory=dict)
    beta: float = 0.0
    mode: PolicyMode = "off"
    estimate: ValueEstimate = field(default_factory=ValueEstimate)
    logits_before: dict[str, float] = field(default_factory=dict)
    logits_after: dict[str, float] = field(default_factory=dict)
    reason: str = ""

    @property
    def applied(self) -> bool:
        """Whether the update actually shifted the distribution.

        Returns:
            `True` when a candidate was chosen and the mode is not `"off"`.
        """
        return self.chosen is not None and self.mode != "off"
