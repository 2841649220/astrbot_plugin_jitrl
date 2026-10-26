"""Verbalised-logit scoring for black-box models (paper Section 5.1).

The frozen LLM that drives the agent is usually a black box: no log-probability
for a candidate action is available, so the closed-form update of Eq. 9/10 has
no base logit to shift. Section 5.1 solves this by asking the model for a 0-100
confidence per candidate and treating its logarithm as a verbalised logit.

This module only builds the prompt, calls the injected coroutine and parses the
answer. It imports no AstrBot *runtime* services beyond the host logger, so it
is unit-testable with a fake generator; the loading of the real provider is the
caller's job. The generator is awaited with keyword arguments only:

    raw = await generate(prompt=..., system_prompt=...)
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from astrbot.api import logger

GenerateFn = Callable[..., Awaitable[str]]
"""Injected model call, awaited as generate(prompt=..., system_prompt=...)."""

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
"""Matches a Markdown code fence, optionally tagged json."""

_ID_RE = re.compile(r'"id"\s*:\s*"?([^",}\s]+)"?')
"""Matches a textual id occurrence, used by the last-resort parser."""

_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*(-?\d+(?:\.\d+)?)')
"""Matches a textual confidence occurrence, used by the last-resort parser."""


@dataclass(slots=True)
class RerankResult:
    """Outcome of one verbalised scoring pass.

    Attributes:
        scores: Confidence in [0, 1] per candidate key; keys the model did not
            answer fall back to 0.5, the neutral prior.
        raw: Raw model output, kept for logging and debugging.
        ok: Whether a usable answer was obtained from the model.
        error: Short English failure description when ok is False.
    """

    scores: dict[str, float] = field(default_factory=dict)
    raw: str = ""
    ok: bool = False
    error: str = ""


class VerbalizedReranker:
    """Scores candidate actions by asking a black-box model for confidences.

    Attributes are private on purpose: the reranker is a thin adapter around the
    injected generator, not a stateful component.
    """

    def __init__(
        self,
        generate: GenerateFn | None,
        *,
        timeout: float = 20.0,
        max_candidates: int = 4,
    ) -> None:
        """Store the generator and the call budget.

        Args:
            generate: Async callable awaited as generate(prompt=..., system_prompt=...).
                None disables scoring, which makes score() return ok=False.
            timeout: Hard timeout in seconds for a single scoring call.
            max_candidates: Maximum number of candidates sent to the model; the
                rest are filled with the neutral 0.5 prior.
        """
        self._generate = generate
        self._timeout = timeout
        self._max_candidates = max(int(max_candidates), 1)

    async def score(
        self,
        *,
        task: str,
        state_text: str,
        candidates: Sequence[tuple[str, str]],
        observation: str = "",
        system_prompt: str | None = None,
    ) -> RerankResult:
        """Ask the model to rate every candidate action for the current state.

        Args:
            task: The user request the agent is serving.
            state_text: Short digest of the current state.
            candidates: (key, text) pairs in presentation order.
            observation: Optional latest observation, appended to the prompt.
            system_prompt: Optional system prompt forwarded to the generator.

        Returns:
            A RerankResult. Timeouts, transport errors and missing generators
            never raise: they come back as ok=False with a short error string,
            and the caller keeps its default policy prior.
        """
        all_keys = [key for key, _ in candidates]
        if self._generate is None:
            return RerankResult(ok=False, error="no generate function configured")
        if not all_keys:
            return RerankResult(ok=False, error="no candidates to score")

        asked = list(candidates)[: self._max_candidates]
        prompt = self.build_prompt(task, state_text, asked, observation)
        try:
            raw = await asyncio.wait_for(
                self._generate(prompt=prompt, system_prompt=system_prompt),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            error = f"timeout after {self._timeout:g}s"
            logger.warning("verbalized reranker timeout: %s", error)
            return RerankResult(ok=False, error=error)
        except Exception as exc:  # noqa: BLE001 - a helper must never break a chat
            error = f"{type(exc).__name__}: {exc}"
            logger.warning("verbalized reranker call failed: %s", error)
            return RerankResult(ok=False, error=error)

        if not isinstance(raw, str) or not raw.strip():
            return RerankResult(ok=False, error="empty model response")
        return RerankResult(scores=self.parse(raw, all_keys), raw=raw, ok=True)

    @staticmethod
    def build_prompt(
        task: str,
        state_text: str,
        candidates: Sequence[tuple[str, str]],
        observation: str = "",
    ) -> str:
        """Render the English scoring prompt for the model.

        Args:
            task: The user request the agent is serving.
            state_text: Short digest of the current state.
            candidates: (key, text) pairs in presentation order; they are
                numbered from one so the reply can reference them by id.
            observation: Optional latest observation, appended when non-empty.

        Returns:
            A prompt that demands a JSON-only reply with per-candidate
            confidences in 0-100 plus the id of the best action.
        """
        lines = [
            "You are the action scorer of a tool-using agent.",
            "Estimate how likely it is that you would pick each candidate",
            "action in the current state.",
            "",
            f"Task: {task.strip() or '(unspecified)'}",
            f"Current state: {state_text.strip() or '(empty)'}",
        ]
        if observation.strip():
            lines.append(f"Recent observation: {observation.strip()}")
        lines.append("")
        lines.append("Candidate actions:")
        for index, (key, text) in enumerate(candidates, start=1):
            lines.append(f"{index}. [{key}] {text.strip() or key}")
        lines.extend(
            [
                "",
                "For every candidate give a confidence between 0 and 100. 0 means",
                "you are certain it is the wrong action, 100 means you are certain",
                "it is the right one. The confidence must estimate the probability",
                "that you would choose that action in this state.",
                "",
                "Answer with JSON only, without prose and without Markdown fences,",
                'using exactly this shape: {"scores": [{"id": 1, "confidence": 0}],',
                '"best": 1}',
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def parse(raw: str, keys: Sequence[str]) -> dict[str, float]:
        """Extract candidate confidences from a free-form model answer.

        Four progressively weaker strategies are tried: strict JSON, fenced JSON,
        the first brace-balanced object embedded in prose, and finally a textual
        pairing of id/confidence occurrences. Values above one are treated as a
        0-100 scale and divided by 100; every value is clamped to [0, 1] and
        missing keys fall back to 0.5 (neutral prior).

        Args:
            raw: Raw model output, possibly wrapped in prose or code fences.
            keys: Candidate keys in presentation order; they also define the
                id-to-key mapping (id 1 is the first key).

        Returns:
            A mapping covering every supplied key, in the order given.
        """
        ordered = list(dict.fromkeys(keys))
        if not ordered:
            return {}
        text = raw or ""

        payload: object | None = None
        attempts = [text.strip()]
        fenced = _FENCE_RE.search(text)
        if fenced is not None:
            attempts.append(fenced.group(1).strip())
        for attempt in attempts:
            if not attempt:
                continue
            try:
                payload = json.loads(attempt)
                break
            except (json.JSONDecodeError, ValueError):
                payload = None

        # Attempt 3: the first brace-balanced object embedded in surrounding
        # prose. Quotes are tracked so braces inside strings do not count.
        cursor = text.find("{")
        while payload is None and cursor != -1:
            depth = 0
            in_string = False
            escaped = False
            for position in range(cursor, len(text)):
                char = text[position]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            payload = json.loads(text[cursor : position + 1])
                        except (json.JSONDecodeError, ValueError):
                            payload = None
                        break
            cursor = text.find("{", cursor + 1)

        if isinstance(payload, dict):
            entries: object = payload.get("scores", payload.get("candidates", payload))
        else:
            entries = payload

        pairs: list[tuple[object, float]] = []
        if isinstance(entries, list):
            for index, item in enumerate(entries):
                if isinstance(item, bool):
                    continue
                if isinstance(item, int | float):
                    number = float(item)
                    if math.isfinite(number):
                        pairs.append((index + 1, number))
                elif isinstance(item, dict):
                    identifier = item.get("id", item.get("key", index + 1))
                    value = item.get("confidence", item.get("score", item.get("value")))
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        number = float(value)
                        if math.isfinite(number):
                            pairs.append((identifier, number))
        elif isinstance(entries, dict):
            for identifier, value in entries.items():
                if isinstance(value, bool):
                    continue
                if isinstance(value, int | float):
                    number = float(value)
                    if math.isfinite(number):
                        pairs.append((identifier, number))
                elif isinstance(value, dict):
                    inner = value.get("confidence", value.get("score"))
                    if isinstance(inner, int | float) and not isinstance(inner, bool):
                        number = float(inner)
                        if math.isfinite(number):
                            pairs.append((identifier, number))

        scores = VerbalizedReranker._map_pairs(pairs, ordered)
        if not scores:
            # Attempt 4: pair up textual id and confidence occurrences. This
            # rescues truncated answers that no JSON parser accepts, and also
            # stray sub-objects picked up by the brace scan above.
            identifiers = _ID_RE.findall(text)
            numbers = _CONFIDENCE_RE.findall(text)
            scores = VerbalizedReranker._map_pairs(
                [
                    (identifier.strip(), float(value))
                    for identifier, value in zip(identifiers, numbers)
                ],
                ordered,
            )

        return {key: scores.get(key, 0.5) for key in ordered}

    @staticmethod
    def _map_pairs(
        pairs: Sequence[tuple[object, float]], ordered: Sequence[str]
    ) -> dict[str, float]:
        """Map identifier/value pairs onto candidate keys and clamp them.

        Identifiers may be 1-based ids, numeric strings or the candidate keys
        themselves. A single value above one means the model answered on a
        0-100 scale, in which case every value is divided by 100.

        Args:
            pairs: Identifier/value pairs in the order the model produced them.
            ordered: Candidate keys in presentation order; they define the
                id-to-key mapping (id 1 is the first key).

        Returns:
            Confidences in [0, 1] for the keys the pairs resolve to. An empty
            dict means no pair could be mapped.
        """
        scale = 100.0 if any(value > 1.0 for _, value in pairs) else 1.0
        scores: dict[str, float] = {}
        for identifier, value in pairs:
            key: str | None = None
            position = -1
            if isinstance(identifier, int | float) and not isinstance(identifier, bool):
                position = int(identifier) - 1
            else:
                stripped = str(identifier).strip()
                if stripped in ordered:
                    key = stripped
                elif stripped.isdigit():
                    position = int(stripped) - 1
            if key is None and 0 <= position < len(ordered):
                key = ordered[position]
            if key is None:
                continue
            scores[key] = min(max(value / scale, 0.0), 1.0)
        return scores
