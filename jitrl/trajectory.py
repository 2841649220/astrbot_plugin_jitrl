"""Episode reconstruction from an AstrBot agent run.

The learning loop needs (s_t, a_t, o_t) tuples, but AstrBot exposes a finished
run as the append-only run_context.messages list, mixing the system prompt, the
user turn, assistant tool calls, tool results and intermediate assistant text.
This module folds that list into Transition objects and assembles the Episode
that the evaluator and the memory bank consume.

Two details drive retrieval quality:

* the state observed *before* step i is built from step i-1's observation plus
  the bounded action history, using the same abstract_state call as the
  on_llm_request hook, so a query state at inference time and a stored state at
  learning time live in the same space;
* only the last plain-text assistant message is kept as the terminating reply:
  intermediate text is usually a truncated or interrupted artifact and would
  add noise to the value estimate.

Every input here is untrusted (pydantic objects *or* plain dicts, JSON string
tool arguments that may be malformed, arbitrary tool output), so each message is
parsed behind its own guard and a bad message is skipped instead of aborting the
whole episode.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .evaluator import discounted_returns
from .models import ActionView, Episode, StateView, StepRecord
from .state import abstract_state, describe_action, normalize_action, redact

# Result text carrying any of these markers is treated as a failed action.
_ERROR_MARKERS: tuple[str, ...] = (
    "error",
    "failed",
    "failure",
    "exception",
    "traceback",
    "timeout",
    "timed out",
    "not found",
    "permission denied",
    "unauthorized",
    "refused",
    "invalid",
    "失败",
    "错误",
    "异常",
    "超时",
    "拒绝",
    "无权限",
    "未找到",
)

# Final-answer wording that signals the task was NOT completed.
_FAILURE_MARKERS: tuple[str, ...] = (
    "无法",
    "不能",
    "抱歉",
    "sorry",
    "i cannot",
    "unable to",
    "failed",
)

# Cap on the terminating reply stored as an action text.
_REPLY_TEXT_LIMIT = 120

# Only the newest actions form the paper's bounded local action history.
_RECENT_ACTION_WINDOW = 3

_PLACEHOLDER_OBSERVATION = "(no result)"


@dataclass(slots=True)
class Transition:
    """One (s_t, a_t, o_t) step of a reconstructed episode.

    Attributes:
        state: Abstracted state observed before the action was taken.
        action: Normalised action.
        observation: Truncated observation produced by the action.
        outcome: ok, error or unknown, from detect_outcome.
    """

    state: StateView
    action: ActionView
    observation: str = ""
    outcome: str = "unknown"


def _field(source: Any, key: str) -> Any:
    """Read a field from a dict-like or attribute-like message object.

    AstrBot hands the hook either pydantic Message objects or plain dicts (the
    unit tests use dicts), so both access paths are supported.

    Args:
        source: Message, content part or tool call to read from.
        key: Field name.

    Returns:
        The field value, or None when it is absent or source is not a message.
    """
    if isinstance(source, Mapping):
        return source.get(key)
    return getattr(source, key, None)


def _text_of(content: Any) -> str:
    """Flatten a message content field into plain text.

    Args:
        content: A string, a list of content parts, or None. A content part is
            either a dict such as {"type": "text", "text": "..."} /
            {"type": "think", "think": "..."} or an object exposing .text /
            .think.

    Returns:
        The concatenated readable text; empty string when nothing is readable.
        Never raises.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, Mapping):
        text = content.get("text")
        if isinstance(text, str) and text:
            return text
        think = content.get("think")
        return think if isinstance(think, str) else ""
    if isinstance(content, (list, tuple)):
        return "".join(_text_of(part) for part in content)
    text = getattr(content, "text", None)
    if isinstance(text, str) and text:
        return text
    think = getattr(content, "think", None)
    return think if isinstance(think, str) else ""


def detect_outcome(result_text: str, *, is_error: bool = False) -> str:
    """Classify the result of one action.

    Args:
        result_text: Raw tool output or result text.
        is_error: Explicit error flag supplied by the runtime.

    Returns:
        "error" when the flag is set or the text carries a known failure marker,
        "unknown" when the text is empty, otherwise "ok".
    """
    if is_error:
        return "error"
    text = "" if result_text is None else str(result_text).strip().lower()
    if not text:
        return "unknown"
    # API envelopes are stronger evidence than keywords inside their payload.
    # A successful search about "timeout errors" is not a failed search, while
    # {"success": false, "message": "nothing happened"} really is a failure.
    if text.startswith("{"):
        try:
            payload = json.loads(str(result_text).strip())
        except (TypeError, ValueError):
            payload = None
        if isinstance(payload, Mapping):
            status = str(payload.get("status") or "").strip().lower()
            if (
                payload.get("is_error") is True
                or payload.get("isError") is True
                or payload.get("success") is False
                or payload.get("ok") is False
                or payload.get("error")
                or payload.get("errors")
                or status in {"error", "failed", "failure", "timeout"}
            ):
                return "error"
            if (
                payload.get("success") is True
                or payload.get("ok") is True
                or status in {"ok", "success", "succeeded"}
                or "error" in payload
                or "errors" in payload
            ):
                return "ok"
    if any(marker in text for marker in _ERROR_MARKERS):
        return "error"
    return "ok"


def extract_transitions(
    *,
    messages: Sequence[Any],
    user_text: str,
    surface: str = "",
    task_family: str = "",
    max_steps: int = 40,
    observation_limit: int = 400,
) -> list[Transition]:
    """Rebuild (s_t, a_t, o_t) transitions from a finished agent run.

    Only the suffix after the last user message belongs to this episode; the
    host also supplies earlier conversation history. One transition is produced
    per current tool call plus one terminating assistant reply. Inputs without
    a user marker are accepted as an already isolated run for compatibility.

    Args:
        messages: run_context.messages: a mix of pydantic Message objects and
            plain dicts.
        user_text: The user request that started the run; it frames every state.
        surface: Coarse partition key forwarded to abstract_state.
        task_family: Accepted for call-site symmetry with build_episode; the
            state view only partitions on surface.
        max_steps: Maximum number of transitions kept (the first ones win).
        observation_limit: Maximum length of a stored observation.

    Returns:
        The reconstructed transitions, possibly empty. A malformed message is
        skipped; this function never raises.
    """
    try:
        step_limit = int(max_steps)
    except (TypeError, ValueError):
        step_limit = 40
    if step_limit <= 0:
        return []
    try:
        obs_limit = max(0, int(observation_limit))
    except (TypeError, ValueError):
        obs_limit = 400
    try:
        items = list(messages)
    except TypeError:
        return []
    if not items:
        return []

    for index in range(len(items) - 1, -1, -1):
        try:
            role = _field(items[index], "role")
            if isinstance(role, str) and role.strip().lower() == "user":
                items = items[index + 1 :]
                break
        except Exception:
            continue

    # Pass 1: concatenate tool outputs per tool_call_id so a call finds its
    # result regardless of message order.
    tool_results: dict[str, str] = {}
    tool_errors: set[str] = set()
    for message in items:
        try:
            role = _field(message, "role")
            if not isinstance(role, str) or role.strip().lower() != "tool":
                continue
            call_id = _field(message, "tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                continue
            if (
                _field(message, "is_error") is True
                or _field(message, "isError") is True
            ):
                tool_errors.add(call_id)
            text = _text_of(_field(message, "content")).strip()
            if text:
                previous = tool_results.get(call_id, "")
                tool_results[call_id] = f"{previous}\n{text}".strip()
        except Exception:
            continue

    # Pass 2: one step per tool call plus the last plain-text assistant reply.
    collected: list[tuple[ActionView, str, str]] = []
    reply_text = ""
    for message in items:
        try:
            role = _field(message, "role")
            if not isinstance(role, str) or role.strip().lower() != "assistant":
                continue
            content = _text_of(_field(message, "content")).strip()
            tool_calls = _field(message, "tool_calls")
            calls = list(tool_calls) if isinstance(tool_calls, (list, tuple)) else []
            if not calls:
                if content:
                    reply_text = content[:_REPLY_TEXT_LIMIT]
                continue
            for call in calls:
                function = _field(call, "function")
                name = _field(function, "name")
                if not isinstance(name, str) or not name.strip():
                    continue
                # Any earlier plain text was an intermediate message; a later
                # tool call means it was not the run's terminating answer.
                reply_text = ""
                name = name.strip()
                raw_args = _field(function, "arguments")
                args: dict[str, Any] = {}
                if isinstance(raw_args, Mapping):
                    args = {str(key): value for key, value in raw_args.items()}
                elif isinstance(raw_args, str) and raw_args.strip():
                    try:
                        decoded = json.loads(raw_args)
                    except (TypeError, ValueError):
                        decoded = None
                    if isinstance(decoded, dict):
                        args = {str(key): value for key, value in decoded.items()}
                preview = ", ".join(
                    f"{key}={str(value)[:40]}" for key, value in list(args.items())[:3]
                )[:120]
                action_text = f"{name}({preview})" if preview else f"{name}()"
                action = normalize_action(
                    kind="tool", name=name, args=args, text=action_text
                )
                call_id = _field(call, "id")
                result = (
                    tool_results.get(call_id, "") if isinstance(call_id, str) else ""
                )
                if result:
                    observation = result[:obs_limit] if obs_limit else ""
                    # Classify on the full output so a trailing failure marker is
                    # not lost when the stored observation is truncated.
                    outcome = detect_outcome(result, is_error=call_id in tool_errors)
                else:
                    observation = _PLACEHOLDER_OBSERVATION
                    outcome = "error" if call_id in tool_errors else "unknown"
                collected.append((action, observation, outcome))
        except Exception:
            continue

    if reply_text:
        try:
            reply_action = normalize_action(
                kind="reply", name="reply", args={}, text=reply_text
            )
        except Exception:
            reply_action = None
        if reply_action is not None:
            collected.append((reply_action, "", "unknown"))

    transitions: list[Transition] = []
    recent: list[str] = []
    for index, (action, observation, outcome) in enumerate(collected[:step_limit]):
        previous_observation = collected[index - 1][1] if index > 0 else ""
        try:
            state = abstract_state(
                task=user_text,
                observation=previous_observation,
                recent_actions=tuple(recent[-_RECENT_ACTION_WINDOW:]),
                surface=surface,
            )
        except Exception:
            continue
        transitions.append(
            Transition(
                state=state, action=action, observation=observation, outcome=outcome
            )
        )
        recent.append(action.text)
    return transitions


def build_episode(
    *,
    episode_id: str,
    scope: str,
    user_text: str,
    transitions: Sequence[Transition],
    rewards: Sequence[float],
    gamma: float,
    success: bool | None = None,
    reward_source: str = "heuristic",
    task_family: str = "",
    created_at: float | None = None,
    meta: dict[str, Any] | None = None,
) -> Episode:
    """Assemble a finished episode with its discounted returns (Eq. 3).

    Args:
        episode_id: Unique id of the run.
        scope: Isolation key, usually the unified message origin.
        user_text: Redacted user request that started the run.
        transitions: Reconstructed (s_t, a_t, o_t) steps, in order.
        rewards: Step rewards r_t; shorter lists are padded with 0.0 and extra
            entries are dropped so every step gets exactly one reward.
        gamma: Discount factor.
        success: Optional task-level success signal.
        reward_source: Provenance recorded on every step.
        task_family: Coarse task label used by the retrieval gate.
        created_at: Unix timestamp; defaults to now.
        meta: Extra bookkeeping (model, latency, ...).

    Returns:
        The populated Episode. Never raises on malformed rewards or timestamps.
    """
    step_rewards: list[float] = []
    for index in range(len(transitions)):
        try:
            step_rewards.append(float(rewards[index]))
        except (IndexError, TypeError, ValueError):
            step_rewards.append(0.0)
    returns = discounted_returns(step_rewards, gamma)

    try:
        timestamp = float(created_at) if created_at is not None else time.time()
    except (TypeError, ValueError):
        timestamp = time.time()
    try:
        safe_user_text = redact(str(user_text))
    except Exception:
        safe_user_text = str(user_text)

    steps = [
        StepRecord(
            index=index,
            state=transition.state,
            action=transition.action,
            reward=step_rewards[index],
            discounted_return=returns[index],
            outcome=transition.outcome,
            observation=transition.observation,
            reward_source=reward_source,
        )
        for index, transition in enumerate(transitions)
    ]
    return Episode(
        episode_id=str(episode_id),
        scope=str(scope),
        created_at=timestamp,
        user_text=safe_user_text,
        task_family=task_family,
        success=success,
        steps=steps,
        meta=dict(meta) if isinstance(meta, Mapping) else {},
    )


def infer_success(
    *,
    transitions: Sequence[Transition],
    final_text: str,
    user_text: str = "",
) -> bool | None:
    """Infer task success from the reconstructed episode.

    Args:
        transitions: Reconstructed steps.
        final_text: The agent's final answer.
        user_text: Accepted for call-site symmetry; the current heuristic only
            inspects the transitions and the final answer.

    Returns:
        None when a step errored (too ambiguous to judge), False when the final
        answer is empty or apologises/refuses, otherwise True.
    """
    for transition in transitions:
        if transition.outcome == "error":
            return None
    text = final_text.strip() if isinstance(final_text, str) else ""
    if not text:
        return False
    lowered = text.lower()
    if any(marker in lowered for marker in _FAILURE_MARKERS):
        return False
    return True


def summarize_episode(episode: Episode) -> str:
    """Render a compact human-readable digest of one episode.

    Args:
        episode: Episode to summarise.

    Returns:
        Multi-line text for the /jitrl command output and for logs.
    """
    if episode.success is True:
        outcome = "success"
    elif episode.success is False:
        outcome = "failed"
    else:
        outcome = "unknown"
    lines = [
        f"episode={episode.episode_id} scope={episode.scope} "
        f"outcome={outcome} steps={len(episode.steps)}"
    ]
    if episode.task_family:
        lines.append(f"task_family={episode.task_family}")
    if episode.user_text:
        lines.append(f"task: {episode.user_text[:160]}")
    for step in episode.steps[:20]:
        lines.append(
            f"  {step.index + 1:>2}. {describe_action(step.action)} "
            f"outcome={step.outcome} r={step.reward:+.2f} "
            f"G={step.discounted_return:+.2f} ({step.reward_source})"
        )
    if len(episode.steps) > 20:
        lines.append(f"  ... {len(episode.steps) - 20} more steps")
    if episode.steps:
        mean_reward = sum(step.reward for step in episode.steps) / len(episode.steps)
        lines.append(
            f"mean_reward={mean_reward:+.3f} "
            f"return_G0={episode.steps[0].discounted_return:+.3f}"
        )
    return "\n".join(lines)
