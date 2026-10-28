"""Learning targets must belong to the current turn and observed outcome."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jitrl.state import abstract_state
from jitrl.trajectory import detect_outcome, extract_transitions


def call(name="current", call_id="reused-id"):
    return {
        "role": "assistant",
        "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": "{}"}}],
    }


@pytest.mark.parametrize("object_messages", [False, True])
def test_history_does_not_consume_current_episode_budget(object_messages):
    messages = [
        {"role": "user", "content": "old task"},
        call("historical"),
        {"role": "tool", "tool_call_id": "reused-id", "content": "ERROR from old task"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": [{"type": "text", "text": "current task"}]},
        call(),
        {"role": "tool", "tool_call_id": "reused-id", "content": "current result"},
        {"role": "assistant", "content": "current answer"},
    ]
    if object_messages:
        messages = [SimpleNamespace(**item) for item in messages]
    transitions = extract_transitions(
        messages=messages, user_text="current task", max_steps=2
    )
    assert [item.action.name for item in transitions] == ["current", "reply"]
    assert transitions[0].observation == "current result"
    assert transitions[0].outcome == "ok"
    assert transitions[0].state == abstract_state(task="current task")
    assert transitions[1].action.text == "current answer"


def test_unanswered_current_turn_cannot_replay_old_answer():
    messages = [
        {"role": "user", "content": "same task"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "same task"},
    ]
    assert extract_transitions(messages=messages, user_text="same task") == []


def test_intermediate_text_before_last_tool_is_not_a_final_answer():
    transitions = extract_transitions(
        messages=[{"role": "assistant", "content": "I will check"}, call()],
        user_text="current task",
    )
    assert [item.action.kind for item in transitions] == ["tool"]


@pytest.mark.parametrize("content", ["", "ordinary response"])
@pytest.mark.parametrize("flag", ["isError", "is_error"])
def test_explicit_tool_failure_is_preserved(content, flag):
    transitions = extract_transitions(
        messages=[
            call(),
            {
                "role": "tool",
                "tool_call_id": "reused-id",
                "content": content,
                flag: True,
            },
        ],
        user_text="current task",
    )
    assert transitions[0].outcome == "error"


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ('{"success":true,"data":"how to handle timeout errors"}', "ok"),
        ('{"error":null,"data":"error handling tutorial"}', "ok"),
        ('{"errors":[],"data":"failed request example"}', "ok"),
        ('{"status":"ok","data":"invalid argument documentation"}', "ok"),
        ('{"success":false,"message":"nothing happened"}', "error"),
        ('{"ok":false}', "error"),
        ('{"isError":true}', "error"),
        ('{"status":"failed"}', "error"),
        ('{"success":true,"error":"permission denied"}', "error"),
        ('{"data":"request timeout"}', "error"),
    ],
)
def test_structured_status_precedes_payload_keywords(result, expected):
    assert detect_outcome(result) == expected
