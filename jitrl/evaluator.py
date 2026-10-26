"""Step-wise reward evaluation for the JitRL learning loop.

Algorithm 1, step 7 of *Just-In-Time Reinforcement Learning* (Li et al.,
ICML 2026) turns a finished episode into training signal: an LLM evaluator
scores every action `r_t` in `[-3, +3]`, the discounted returns
`G_t = sum_{u>=t} gamma^(u-t) * r_u` are computed, and `(s_t, a_t, G_t)` is
written to the non-parametric memory.

The prompt mirrors Appendix G.1.2: analyse what happened after the action,
judge the action USEFUL / HARMFUL / NEUTRAL, state how certain that judgement
is, then map the pair onto the reward scale::

    USEFUL  + certain          -> +3     HARMFUL + certain          -> -3
    USEFUL  + somewhat unsure  -> +2     HARMFUL + somewhat unsure  -> -2
    maybe useful, very unsure  -> +1     maybe harmful, very unsure -> -1
    NEUTRAL                    ->  0

Engineering deviation from the paper (documented on purpose): Appendix G.1.2
asks for a prose block (Result / Usefulness / Certainty / Score) plus a
repeat/avoid note, and Appendix G.2.2 scores with the raw Jericho game score.
This plugin needs machine-readable output, so the evaluator requests JSON::

    {"step_analysis": [{"step": 0, "action": "...",
                        "detailed_reasoning": "...", "score": 0}],
     "overall_assessment": "..."}

and constrains every score to [-3, +3] (max_reward can tighten the cap).
The same JSON can therefore be fused with the zero-cost rule-based scorer,
which doubles as the fallback whenever the model is unavailable.

Every parsing path degrades instead of raising: a malformed model response must
never break a chat or lose an episode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

_MODES = ("auto", "llm", "heuristic", "off")

# Action names containing these markers mutate the world, so a successful call
# is a stronger positive signal than a read-only call.
_WRITE_MARKERS = (
    "write",
    "create",
    "delete",
    "remove",
    "update",
    "set",
    "save",
    "commit",
    "send",
    "push",
    "upload",
)

# Actions that end the run; succeeding with one of them is mildly positive.
_TERMINAL_MARKERS = ("finish", "send_message", "reply")

# Observations that carry no information even though the call succeeded.
_EMPTY_OBSERVATION_VALUES = {"[]", "{}", "null", "none", "ok"}

# Below this many characters an observation is treated as low information.
_LOW_INFO_CHARS = 8

_SYSTEM_PROMPT = """You are a rigorous step-level reward evaluator for a \
tool-using AI agent.

You receive one finished agent episode: the user's request, then a numbered list
of the actions the agent took, the observation each action produced, and the
outcome reported by the runtime.

For EVERY step, follow this rubric in order:

1. Analysis: state what happened after the action. Did the observation move the
   agent closer to completing the task? Did it waste effort or cause harm?
2. Usefulness: classify the action as USEFUL, HARMFUL or NEUTRAL.
3. Certainty: state whether your judgement is certain, somewhat uncertain, or
   very uncertain.
4. Score: map the pair onto this integer scale.
   USEFUL and certain                    -> +3
   USEFUL but somewhat uncertain         -> +2
   Possibly useful but very uncertain    -> +1
   HARMFUL and certain                   -> -3
   HARMFUL but somewhat uncertain        -> -2
   Possibly harmful but very uncertain   -> -1
   NEUTRAL                               ->  0

Judge each action by its consequences, not by whether it merely looks
reasonable. Score every step independently: do not skip steps and do not reward
the final answer twice. Every score must be a number inside [-3, +3].

In detailed_reasoning, finish with the follow-up decision: repeat when the same
action should be taken again in a similar situation, avoid when it should not.

Reply with a single JSON object and nothing else, using exactly this shape:
{"step_analysis": [{"step": 0, "action": "...", "detailed_reasoning": "...", "score": 0}], "overall_assessment": "..."}
"""


def _as_float(value: Any, default: float) -> float:
    """Coerce an untrusted value to float without raising.

    Args:
        value: Raw value, usually coming from the plugin configuration.
        default: Value returned when value is not numeric.

    Returns:
        The parsed float, or default.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, limit: float) -> float:
    """Clamp a reward into the symmetric range [-limit, limit].

    Args:
        value: Raw reward value; non-finite values collapse to 0.0.
        limit: Absolute upper bound (3.0 for the paper's scale).

    Returns:
        The clamped, finite reward.
    """
    bound = abs(_as_float(limit, 3.0))
    number = _as_float(value, 0.0)
    if not math.isfinite(number):
        return 0.0
    return max(-bound, min(bound, number))


@dataclass(slots=True)
class RewardResult:
    """Outcome of one step-wise reward evaluation.

    Attributes:
        rewards: One reward r_t per step, aligned with the evaluated actions.
        source: "llm", "heuristic" or "fused"; records what actually produced
            the numbers after any degradation.
        overall: The evaluator's overall_assessment text when one was parsed.
        ok: Whether the requested scoring path completed. False means a
            degraded result (usually the heuristic fallback).
        error: Human-readable degradation reason, empty on success.
    """

    rewards: list[float] = field(default_factory=list)
    source: str = "heuristic"
    overall: str = ""
    ok: bool = False
    error: str = ""


def discounted_returns(rewards: Sequence[float], gamma: float) -> list[float]:
    """Compute discounted returns G_t = sum_{u>=t} gamma^(u-t) * r_u.

    Args:
        rewards: Step rewards r_t in chronological order.
        gamma: Discount factor; clamped to [0, 1].

    Returns:
        A list with the same length as rewards; empty input yields [].
    """
    discount = min(1.0, max(0.0, _as_float(gamma, 0.0)))
    total = 0.0
    returns = [0.0] * len(rewards)
    for index in range(len(rewards) - 1, -1, -1):
        total = _as_float(rewards[index], 0.0) + discount * total
        returns[index] = total
    return returns


class StepRewardEvaluator:
    """Assigns Algorithm 1 step-7 rewards to the steps of an episode.

    One instance is built per plugin from JitRLConfig.learning; score is the
    only entry point the rest of the plugin needs.

    Args:
        generate: Async callable invoked as
            await generate(prompt=..., system_prompt=...) returning the raw
            model text. None disables LLM scoring entirely.
        mode: auto, llm, heuristic or off. auto spends an LLM call only when
            the episode has at least two steps or any step errored; unknown
            values fall back to auto.
        timeout: Hard per-call timeout in seconds.
        max_reward: Absolute cap applied to every returned reward.
        heuristic_weight: Weight w of the rule-based score in the fusion
            (1 - w) * llm + w * heuristic, clamped to [0, 1].
        logger: Logger used for degradation warnings.
    """

    def __init__(
        self,
        generate: Callable[..., Awaitable[str]] | None,
        *,
        mode: str = "auto",
        timeout: float = 30.0,
        max_reward: float = 3.0,
        heuristic_weight: float = 0.5,
        logger: logging.Logger | None = None,
    ) -> None:
        self._generate = generate
        mode_value = mode.strip().lower() if isinstance(mode, str) else ""
        self._mode = mode_value if mode_value in _MODES else "auto"
        self._timeout = max(0.1, _as_float(timeout, 30.0))
        cap = abs(_as_float(max_reward, 3.0))
        self._max_reward = cap if cap > 0.0 else 3.0
        weight = _as_float(heuristic_weight, 0.5)
        self._heuristic_weight = min(1.0, max(0.0, weight))
        self._logger = logger or logging.getLogger(__name__)

    async def score(
        self,
        *,
        task: str,
        actions: Sequence[str],
        observations: Sequence[str],
        outcomes: Sequence[str],
        success: bool | None = None,
    ) -> RewardResult:
        """Score every step of a finished episode.

        Args:
            task: The user request that started the run.
            actions: Rendered actions in execution order.
            observations: Observation produced by each action.
            outcomes: ok, error or unknown per action.
            success: Optional task-level success signal used by the heuristic.

        Returns:
            A RewardResult whose rewards are aligned with the shortest of the
            three inputs. Degradations are reported through ok/error instead of
            raising.
        """
        step_count = min(len(actions), len(observations), len(outcomes))
        if step_count <= 0:
            return RewardResult(rewards=[], source="heuristic", ok=True)
        step_actions = [str(item) for item in actions[:step_count]]
        step_observations = [str(item) for item in observations[:step_count]]
        step_outcomes = [str(item).strip().lower() for item in outcomes[:step_count]]

        if self._mode == "off":
            return RewardResult(rewards=[0.0] * step_count, source="heuristic", ok=True)

        heuristics = self.heuristic_rewards(
            step_actions,
            step_observations,
            step_outcomes,
            success=success,
            max_reward=self._max_reward,
        )
        if self._mode == "heuristic" or self._generate is None:
            return RewardResult(rewards=heuristics, source="heuristic", ok=True)

        # auto keeps cheap one-step runs out of the LLM unless they errored.
        if self._mode == "auto" and step_count < 2 and "error" not in step_outcomes:
            return RewardResult(rewards=heuristics, source="heuristic", ok=True)

        system_prompt, user_prompt = self.build_prompt(
            task, step_actions, step_observations, step_outcomes
        )
        try:
            raw = await asyncio.wait_for(
                self._generate(prompt=user_prompt, system_prompt=system_prompt),
                timeout=self._timeout,
            )
        except (TimeoutError, asyncio.TimeoutError):
            self._logger.warning(
                "JitRL evaluator timed out after %.1fs; falling back to heuristics",
                self._timeout,
            )
            return RewardResult(
                rewards=heuristics,
                source="heuristic",
                ok=False,
                error=f"timeout after {self._timeout:g}s",
            )
        except Exception as exc:
            self._logger.warning(
                "JitRL evaluator call failed (%s); falling back to heuristics", exc
            )
            return RewardResult(
                rewards=heuristics,
                source="heuristic",
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )

        parsed = self.parse_rewards(raw, step_count, self._max_reward)
        if parsed is None:
            self._logger.warning(
                "JitRL evaluator returned unparseable scores; falling back to heuristics"
            )
            return RewardResult(
                rewards=heuristics,
                source="heuristic",
                ok=False,
                error="unparseable evaluator response",
            )

        weight = self._heuristic_weight
        fused = [
            round(_clamp((1.0 - weight) * llm + weight * heur, self._max_reward), 3)
            for llm, heur in zip(parsed, heuristics, strict=True)
        ]
        overall = ""
        if isinstance(raw, str):
            match = re.search(r'"overall_assessment"\s*:\s*"([^"]{0,400})"', raw)
            if match:
                overall = match.group(1)
        return RewardResult(rewards=fused, source="fused", overall=overall, ok=True)

    @staticmethod
    def heuristic_rewards(
        actions: Sequence[str],
        observations: Sequence[str],
        outcomes: Sequence[str],
        *,
        success: bool | None = None,
        max_reward: float = 3.0,
    ) -> list[float]:
        """Score steps with zero-cost rules.

        The rules are the fallback when no LLM is available and one half of the
        fused signal otherwise:

        * error outcome -> -2.0; ok with a non-empty observation -> +1.0; ok
          with an empty observation -> 0.0; anything else keeps 0.0.
        * A successful mutating action (write, send, delete, ...) adds +0.5.
        * A short (fewer than 8 characters) or obviously empty observation
          ([] / {} / null / None / ok) subtracts 0.5; this stacks on top of the
          base score regardless of outcome.
        * The last step adds +1.0 on success=True, -1.0 on success=False, and
          +0.5 when it is a terminal action.

        Args:
            actions: Rendered actions in execution order.
            observations: Observation produced by each action; missing entries
                are treated as empty.
            outcomes: ok, error or unknown per action; missing entries are
                treated as unknown.
            success: Optional task-level success signal.
            max_reward: Absolute cap applied to each score.

        Returns:
            One clamped reward per entry of actions.
        """
        rewards: list[float] = []
        for index, raw_action in enumerate(actions):
            action_text = str(raw_action).lower()
            observation = (
                str(observations[index]) if index < len(observations) else ""
            ).strip()
            outcome = (
                str(outcomes[index]).strip().lower()
                if index < len(outcomes)
                else "unknown"
            )

            score = 0.0
            if outcome == "error":
                score = -2.0
            elif outcome == "ok":
                score = 1.0 if observation else 0.0
                if any(marker in action_text for marker in _WRITE_MARKERS):
                    score += 0.5
            if (
                len(observation) < _LOW_INFO_CHARS
                or observation.lower() in _EMPTY_OBSERVATION_VALUES
            ):
                score -= 0.5
            rewards.append(score)

        if rewards:
            if success is True:
                rewards[-1] += 1.0
            elif success is False:
                rewards[-1] -= 1.0
            if any(marker in str(actions[-1]).lower() for marker in _TERMINAL_MARKERS):
                rewards[-1] += 0.5
        return [round(_clamp(value, max_reward), 3) for value in rewards]

    @staticmethod
    def parse_rewards(
        raw: str, expected: int, max_reward: float = 3.0
    ) -> list[float] | None:
        """Extract step scores from a model response.

        Fallbacks are attempted in order, mirroring what real providers emit:
        strict JSON, a fenced code block, the widest brace/bracket span, a
        loose scan of score pairs, and finally a bare bracket list. Both
        step_analysis[].score objects and [1, 2, 3] lists are accepted.

        Args:
            raw: Raw model text.
            expected: Number of steps that must be scored.
            max_reward: Absolute cap applied to every parsed score.

        Returns:
            A list of expected floats with missing scores padded by 0.0 and
            extra scores dropped, or None when nothing could be parsed.
        """
        if expected <= 0:
            return []
        if not isinstance(raw, str) or not raw.strip():
            return None
        text = raw.strip()
        cap = abs(_as_float(max_reward, 3.0))
        limit = cap if cap > 0.0 else 3.0

        def _scores_from(payload: Any) -> list[float] | None:
            """Collect numeric scores from a decoded JSON payload."""
            if isinstance(payload, dict):
                rows = payload.get("step_analysis")
                if rows is None:
                    rows = payload.get("scores")
                return _scores_from(rows) if rows is not None else None
            if isinstance(payload, (list, tuple)):
                values: list[float] = []
                for row in payload:
                    if isinstance(row, dict):
                        raw_value = row.get("score")
                    else:
                        raw_value = row
                    if isinstance(raw_value, bool) or not isinstance(
                        raw_value, (int, float)
                    ):
                        continue
                    values.append(float(raw_value))
                return values or None
            return None

        def _align(values: Sequence[float]) -> list[float]:
            """Clamp and pad/truncate a raw score list to expected entries."""
            aligned = [round(_clamp(value, limit), 3) for value in values[:expected]]
            aligned.extend([0.0] * (expected - len(aligned)))
            return aligned

        candidates: list[str] = [text]
        candidates.extend(
            block.strip()
            for block in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        )
        span = re.search(r"[\{\[].*[\}\]]", text, flags=re.DOTALL)
        if span:
            candidates.append(span.group(0))
        for candidate in candidates:
            if not candidate:
                continue
            try:
                payload = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            scores = _scores_from(payload)
            if scores:
                return _align(scores)

        loose = re.findall(
            r'score["\s]*[:=]\s*["\s]*(-?\d+(?:\.\d+)?)', text, re.IGNORECASE
        )
        if loose:
            return _align([float(item) for item in loose])

        bare = re.search(r"\[[^\[\]]*\]", text)
        if bare:
            try:
                payload = json.loads(bare.group(0))
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, list):
                scores = _scores_from(payload)
                if scores:
                    return _align(scores)
        return None

    @staticmethod
    def build_prompt(
        task: str,
        actions: Sequence[str],
        observations: Sequence[str],
        outcomes: Sequence[str],
    ) -> tuple[str, str]:
        """Build the step-wise reward-evaluation prompt (Appendix G.1.2).

        The paper asks for a prose Result / Usefulness / Certainty / Score block
        plus a repeat/avoid note. The reasoning stages are kept verbatim, but
        the required answer is strict JSON so the scores can be parsed and fused
        with the rule-based signals.

        Args:
            task: The user request that started the run.
            actions: Rendered actions in execution order.
            observations: Observation produced by each action.
            outcomes: ok, error or unknown per action.

        Returns:
            (system_prompt, user_prompt) ready for a chat-completion call.
        """
        step_count = min(len(actions), len(observations), len(outcomes))
        lines = [
            "TASK:",
            str(task).strip() or "(empty task)",
            "",
            f"STEPS ({step_count} total):",
        ]
        for index in range(step_count):
            action = " ".join(str(actions[index]).split())[:300]
            observation = " ".join(str(observations[index]).split())
            if not observation:
                observation = "(empty)"
            elif len(observation) > 600:
                observation = observation[:600] + " ..."
            outcome = str(outcomes[index]).strip() or "unknown"
            lines.extend(
                [
                    f"Step {index}",
                    f"  action: {action}",
                    f"  outcome: {outcome}",
                    f"  observation: {observation}",
                ]
            )
        lines.extend(
            [
                "",
                "Return the JSON object with exactly "
                f"{step_count} step_analysis entries now.",
            ]
        )
        return _SYSTEM_PROMPT, "\n".join(lines)
