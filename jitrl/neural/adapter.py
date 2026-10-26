"""AstrBot message adapter with explicit request-to-execution association.

Tool *proposals* without a matching result are never treated as executed facts.
The adapter retains IDs only to associate a completed run, not tool arguments.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from ..trajectory import Transition, extract_transitions
from .contracts import LensDecision
from .encoding import canonical_text, safe_name


def value(obj: Any, key: str, default: Any = None) -> Any:
    return (
        obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)
    )


def text_content(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return " ".join(str(value(item, "text", "")) for item in obj)
    return ""


def provider_messages(request: Any) -> list[Any]:
    """Read both ProviderRequest.contexts and singular/list tool result packets.

    AstrBot may keep the last ToolCallsResult outside contexts. Merge a local
    view without changing either host field or counting duplicate results.
    """
    messages = list(value(request, "contexts", None) or [])
    result_ids = {
        value(m, "tool_call_id") for m in messages if value(m, "role") == "tool"
    }
    call_ids = {
        value(call, "id")
        for m in messages
        for call in value(m, "tool_calls", None) or []
    }
    packets = value(request, "tool_calls_result")
    packets = (
        packets
        if isinstance(packets, list | tuple)
        else [packets]
        if packets is not None
        else []
    )
    for packet in packets:
        assistant = value(packet, "tool_calls_info")
        if assistant is not None:
            calls = [
                call
                for call in value(assistant, "tool_calls", None) or []
                if value(call, "id") not in call_ids
            ]
            if calls:
                messages.append({"role": "assistant", "tool_calls": calls})
                call_ids.update(value(call, "id") for call in calls)
        for result in value(packet, "tool_calls_result", None) or []:
            call_id = value(result, "tool_call_id")
            if call_id not in result_ids:
                messages.append(result)
                result_ids.add(call_id)
    return messages


def observed_context(
    messages: list[Any],
) -> tuple[frozenset[str], str, tuple[str, ...]]:
    completed = set()
    observation = ""
    events = []
    for message in messages:
        if value(message, "role") == "tool":
            call_id = value(message, "tool_call_id")
            if isinstance(call_id, str):
                completed.add(call_id)
            observation = canonical_text(text_content(value(message, "content")))
            events.append("tool_result:observed")
        elif value(message, "role") == "assistant":
            for call in value(message, "tool_calls", None) or []:
                name = value(value(call, "function"), "name", "")
                if isinstance(name, str) and safe_name(name):
                    events.append("tool_call:" + name)
    return frozenset(completed), observation, tuple(events[-29:])


@dataclass(slots=True)
class RequestFrame:
    decision: LensDecision
    epoch: int
    observed_calls: frozenset[str]


@dataclass(slots=True)
class NeuralRun:
    episode_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    baseline_calls: frozenset[str] = frozenset()
    frames: list[RequestFrame] = field(default_factory=list)
    completed: bool = False

    def frame_for(self, preceding_calls: frozenset[str]) -> RequestFrame | None:
        for frame in reversed(self.frames):
            if frame.observed_calls <= preceding_calls:
                return frame
        return None


def executed_transitions(
    messages: list[Any],
    *,
    goal: str,
    surface: str,
    baseline_calls: frozenset[str],
    final_text: str,
) -> tuple[list[Transition], list[str]]:
    """Extract only calls with observed results plus the actual final reply."""
    # Limit to the current user turn when the host retains older conversation.
    last_user = max(
        (i for i, message in enumerate(messages) if value(message, "role") == "user"),
        default=-1,
    )
    messages = messages[last_user + 1 :]
    results = {
        value(message, "tool_call_id")
        for message in messages
        if value(message, "role") == "tool"
    }
    selected = []
    call_ids: list[str] = []
    for message in messages:
        role = value(message, "role")
        if role == "assistant":
            calls = []
            for call in value(message, "tool_calls", None) or []:
                call_id = value(call, "id")
                name = value(value(call, "function"), "name", "")
                if (
                    isinstance(call_id, str)
                    and call_id in results
                    and call_id not in baseline_calls
                    and call_id not in call_ids
                    and isinstance(name, str)
                    and safe_name(name)
                ):
                    calls.append(call)
                    call_ids.append(call_id)
            if calls:
                selected.append({"role": "assistant", "tool_calls": calls})
        elif role == "tool" and value(message, "tool_call_id") not in baseline_calls:
            selected.append(message)
    if final_text.strip():
        selected.append({"role": "assistant", "content": canonical_text(final_text)})
        call_ids.append("__final_reply__")
    transitions = extract_transitions(
        messages=selected, user_text=goal, surface=surface
    )
    if len(transitions) != len(call_ids[:40]):
        # Ambiguous parsing must not assign a return to the wrong decision.
        return [], []
    return transitions, call_ids[: len(transitions)]
