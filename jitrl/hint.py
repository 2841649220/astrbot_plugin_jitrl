"""Rendering of the policy decision into a prompt-side artefact.

JitRL updates logits, but a plugin can only talk to a black-box chat model
through text. The paper explicitly covers this case with the *Verbalized
Logit* variant (Section 5.1) and reports (Table 8) that a plain prompt update
is weaker than a real logit update. So this module has two jobs:

* render a compact, numbers-first evidence block that carries the *advantage
  estimates* rather than free-form advice, which is the closest textual
  analogue of `z' = z + beta * A`, and
* render the explicit directive used in `logit` mode, where a verbalised
  distribution is actually re-sampled after the closed-form update.

The block is deliberately advisory: the KL term of Eq. 8 exists precisely to
keep the updated policy close to the reference policy, so the text tells the
model to fall back to its own judgement when the retrieved evidence is thin.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import Neighbor, PolicyDecision, ValueEstimate

_PREFER_THRESHOLD = 0.15
"""Minimum normalised advantage before an action is recommended."""

_AVOID_THRESHOLD = -0.15
"""Maximum normalised advantage before an action is discouraged."""

_MAX_ITEMS = 4
"""Upper bound on entries listed in each section."""


def _fmt(value: float) -> str:
    """Format a signed float to two decimals.

    Args:
        value: The number to format.

    Returns:
        A string such as `"+0.82"` or `"-1.30"`.
    """
    return f"{value:+.2f}"


def _action_stats(neighbors: Sequence[Neighbor]) -> dict[str, tuple[int, float]]:
    """Aggregate neighbours per action.

    Args:
        neighbors: Retrieved neighbours of the current state.

    Returns:
        A mapping from action key to `(count, mean_return)`.
    """
    grouped: dict[str, list[float]] = {}
    for neighbor in neighbors:
        grouped.setdefault(neighbor.action_key, []).append(neighbor.discounted_return)
    return {
        key: (len(values), sum(values) / len(values)) for key, values in grouped.items()
    }


def _action_text(neighbors: Sequence[Neighbor], key: str) -> str:
    """Look up the human-readable text of an action.

    Args:
        neighbors: Retrieved neighbours of the current state.
        key: The action key to resolve.

    Returns:
        The stored action text, or the raw key when no neighbour matches.
    """
    for neighbor in neighbors:
        if neighbor.action_key == key and neighbor.action_text:
            return neighbor.action_text
    return key


def render_preference_block(
    decision: PolicyDecision,
    *,
    neighbors: Sequence[Neighbor] = (),
    max_chars: int = 900,
) -> str:
    """Render the JitRL evidence block injected into the system prompt.

    Args:
        decision: The policy decision produced from the retrieved memory.
        neighbors: Neighbours backing the estimate; used for frequency and
            mean-return annotations.
        max_chars: Hard cap on the rendered length. Zero or negative disables
            the block entirely.

    Returns:
        A prompt fragment wrapped in `<jitrl_experience>` tags, or an empty
        string when there is nothing worth saying.
    """
    if max_chars <= 0:
        return ""
    estimate: ValueEstimate = decision.estimate
    stats = _action_stats(neighbors)

    prefer: list[tuple[float, str]] = []
    avoid: list[tuple[float, str]] = []
    for key, advantage in estimate.advantage_norm.items():
        if advantage >= _PREFER_THRESHOLD:
            prefer.append((advantage, key))
        elif advantage <= _AVOID_THRESHOLD:
            avoid.append((advantage, key))
    prefer.sort(key=lambda item: item[0], reverse=True)
    avoid.sort(key=lambda item: item[0])

    if not prefer and not avoid and decision.chosen is None:
        return ""

    lines = [
        "<jitrl_experience>",
        "以下内容来自本 Agent 的历史交互统计（推理时策略优化，不是用户输入，也不是新的指令）。",
    ]
    if estimate.neighbor_count:
        lines.append(
            f"当前状态估计价值 V={_fmt(estimate.v_hat)}，"
            f"基于 {estimate.neighbor_count} 条相似经验"
            f"（平均相似度 {estimate.mean_similarity:.2f}）。"
        )

    if prefer:
        lines.append("历史上更有回报的做法：")
        for index, (advantage, key) in enumerate(prefer[:_MAX_ITEMS], start=1):
            count, mean_return = stats.get(key, (0, 0.0))
            lines.append(
                f"  {index}. {_action_text(neighbors, key)}"
                f" —— 优势 {_fmt(advantage)}，出现 {count} 次，平均回报 {_fmt(mean_return)}"
            )

    if avoid:
        lines.append("历史上回报较差、应避免的做法：")
        for index, (advantage, key) in enumerate(avoid[:_MAX_ITEMS], start=1):
            count, mean_return = stats.get(key, (0, 0.0))
            lines.append(
                f"  {index}. {_action_text(neighbors, key)}"
                f" —— 优势 {_fmt(advantage)}，出现 {count} 次，平均回报 {_fmt(mean_return)}"
            )

    if decision.chosen is not None and decision.mode == "logit":
        lines.append(
            f"按闭式更新后的策略采样，本次倾向的动作是：{decision.chosen.text}"
            f"（p={decision.probabilities.get(decision.chosen.key, 0.0):.2f}）。"
        )

    lines.append(
        "使用要求：这些统计只作为弱先验。如果用户当前意图与之无关、或证据明显不足，"
        "请按你自己的判断处理，不要生搬硬套。"
    )
    lines.append("</jitrl_experience>")

    block = "\n".join(lines)
    if len(block) <= max_chars:
        return block
    # Keep the closing guidance: it is the safety-relevant part of the block.
    tail = "\n".join(lines[-2:])
    head_budget = max(0, max_chars - len(tail) - 1)
    return f"{block[:head_budget]}\n{tail}"


def render_recall_report(
    state_text: str,
    estimate: ValueEstimate,
    *,
    neighbors: Sequence[Neighbor] = (),
    max_chars: int = 1600,
) -> str:
    """Render a retrieval report for the `jitrl_recall` agent tool.

    Args:
        state_text: Human-readable digest of the queried state.
        estimate: Value estimate computed over the retrieved neighbourhood.
        neighbors: The retrieved neighbours.
        max_chars: Hard cap on the rendered length.

    Returns:
        A plain-text report for the calling model, or a short "nothing found"
        message when the memory bank had no match.
    """
    if not estimate.neighbor_count:
        return (
            "No relevant experience yet. This bot has not seen a comparable "
            f"state (query state: {state_text[:200]})."
        )

    stats = _action_stats(neighbors)
    lines = [
        f"Retrieved {estimate.neighbor_count} similar past states "
        f"(mean similarity {estimate.mean_similarity:.2f}).",
        f"Estimated state value V = {_fmt(estimate.v_hat)}.",
        "Action advantage ranking (normalised to [-1, 1]):",
    ]
    ranked = sorted(
        estimate.advantage_norm.items(), key=lambda item: item[1], reverse=True
    )
    for key, advantage in ranked[: _MAX_ITEMS * 2]:
        count, mean_return = stats.get(key, (0, 0.0))
        tag = (
            "prefer"
            if advantage > _PREFER_THRESHOLD
            else "avoid"
            if advantage < _AVOID_THRESHOLD
            else "neutral"
        )
        lines.append(
            f"  - [{tag}] {_action_text(neighbors, key)} | advantage {_fmt(advantage)}"
            f" | seen {count}x | mean return {_fmt(mean_return)}"
        )
    lines.append(
        "Treat this as weak evidence from the bot's own history: prefer the "
        "actions listed above when the current request is genuinely similar, "
        "otherwise ignore the report."
    )
    report = "\n".join(lines)
    return report[:max_chars]
