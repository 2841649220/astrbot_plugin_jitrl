"""Unit tests for episode reconstruction from an AstrBot agent run."""

from __future__ import annotations

import types

import pytest

from jitrl.models import Episode
from jitrl.state import abstract_state
from jitrl.trajectory import (
    Transition,
    build_episode,
    detect_outcome,
    extract_transitions,
    infer_success,
    summarize_episode,
)

USER_TEXT = "帮我查下明天天气"
SURFACE = "qq:group:123"

# A realistic AstrBot run: the user asks for the weather, the agent searches,
# emits an intermediate line, calls finish, then answers in plain text.
MESSAGES: list[dict[str, object]] = [
    {"role": "system", "content": "You are AstrBot."},
    {"role": "user", "content": USER_TEXT},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "web_search", "arguments": '{"q": "weather"}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"temp": 21, "sky": "sunny"}',
    },
    {"role": "_checkpoint", "content": "internal checkpoint"},
    {"role": "assistant", "content": "正在为你查询...", "tool_calls": None},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "finish", "arguments": '{"text": "明天 21 度"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_2", "content": "ok"},
    {"role": "assistant", "content": "明天 21 度，建议带件外套。"},
]


def _weather_transitions() -> list[Transition]:
    """Reconstruct the transitions of the shared weather run."""
    return extract_transitions(messages=MESSAGES, user_text=USER_TEXT)


# --------------------------------------------------------------------------- #
# extract_transitions
# --------------------------------------------------------------------------- #


def test_extract_transitions_from_dict_messages() -> None:
    """A dict-based run yields tool steps plus one terminating reply."""
    transitions = _weather_transitions()
    assert len(transitions) == 3
    assert [item.action.name for item in transitions] == [
        "web_search",
        "finish",
        "reply",
    ]
    assert [item.action.kind for item in transitions] == ["tool", "tool", "reply"]
    assert [item.outcome for item in transitions] == ["ok", "ok", "unknown"]
    assert '"temp": 21' in transitions[0].observation
    assert transitions[1].observation == "ok"
    assert transitions[2].observation == ""
    assert transitions[2].action.text == "明天 21 度，建议带件外套。"


def test_extract_transitions_ignores_intermediate_reply_text() -> None:
    """Only the last plain-text assistant message survives as the reply."""
    transitions = _weather_transitions()
    assert all("正在为你查询" not in item.action.text for item in transitions)
    assert transitions[-1].action.kind == "reply"
    assert transitions[-1].action.text.startswith("明天 21 度")


def test_extract_transitions_skips_system_user_and_checkpoint() -> None:
    """System, user and _checkpoint messages never become actions."""
    transitions = _weather_transitions()
    names = {item.action.name for item in transitions}
    assert "system" not in names
    assert "user" not in names
    assert "_checkpoint" not in names
    assert len(transitions) == 3


def test_extract_transitions_matches_tool_call_ids_out_of_order() -> None:
    """Tool results bind to their call id even when delivered in reverse order."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.txt"}'},
                },
                {
                    "id": "call_b",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "b.txt"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_b", "content": "contents of b"},
        {"role": "tool", "tool_call_id": "call_a", "content": "contents of a"},
    ]
    transitions = extract_transitions(messages=messages, user_text="read both files")
    assert len(transitions) == 2
    assert transitions[0].observation == "contents of a"
    assert transitions[1].observation == "contents of b"
    assert transitions[0].action.name == "read_file"
    assert transitions[1].action.args == {"path": "b.txt"}


def test_extract_transitions_missing_result_is_placeholder() -> None:
    """A tool call without a matching tool message degrades gracefully."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_lost",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"q": "x"}'},
                }
            ],
        }
    ]
    transitions = extract_transitions(messages=messages, user_text="search")
    assert len(transitions) == 1
    assert transitions[0].observation == "(no result)"
    assert transitions[0].outcome == "unknown"


@pytest.mark.parametrize("raw_arguments", ["not json at all", "", None, "[1, 2]"])
def test_extract_transitions_tolerates_bad_arguments(raw_arguments: object) -> None:
    """Malformed tool arguments become an empty arg dict instead of raising."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": raw_arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a useful result"},
    ]
    transitions = extract_transitions(messages=messages, user_text="search")
    assert len(transitions) == 1
    assert transitions[0].action.args == {}
    assert transitions[0].action.text.endswith("()")


def test_extract_transitions_ignores_content_next_to_tool_calls() -> None:
    """Preambles attached to a tool call are not treated as the final reply."""
    messages = [
        {
            "role": "assistant",
            "content": "let me check that for you",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": '{"q": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "search result body"},
    ]
    transitions = extract_transitions(messages=messages, user_text="search")
    assert len(transitions) == 1
    assert transitions[0].action.kind == "tool"
    assert all(item.action.kind != "reply" for item in transitions)


def test_extract_transitions_truncates_reply_text() -> None:
    """The terminating reply text is capped at 120 characters."""
    transitions = extract_transitions(
        messages=[{"role": "assistant", "content": "x" * 200}],
        user_text="say something",
    )
    assert len(transitions) == 1
    assert len(transitions[0].action.text) == 120


def test_extract_transitions_max_steps_truncates() -> None:
    """max_steps keeps the first N transitions only."""
    messages: list[dict[str, object]] = []
    for index in range(5):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {
                            "name": f"tool_{index}",
                            "arguments": '{"n": 1}',
                        },
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{index}",
                "content": f"result {index} body",
            }
        )
    transitions = extract_transitions(
        messages=messages, user_text="do five things", max_steps=2
    )
    assert len(transitions) == 2
    assert [item.action.name for item in transitions] == ["tool_0", "tool_1"]


@pytest.mark.parametrize("messages", [[], None, ()])
def test_extract_transitions_handles_empty_input(messages: object) -> None:
    """Empty or non-iterable message lists yield no transitions."""
    assert extract_transitions(messages=messages, user_text="hi") == []


def test_extract_transitions_observation_limit() -> None:
    """Observation truncation keeps the head and classifies on the full text."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "http_get", "arguments": '{"url": "x"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "0123456789 ERROR boom"},
    ]
    transitions = extract_transitions(
        messages=messages, user_text="fetch", observation_limit=10
    )
    assert transitions[0].observation == "0123456789"
    assert transitions[0].outcome == "error"


# --------------------------------------------------------------------------- #
# state construction
# --------------------------------------------------------------------------- #


def test_first_state_matches_the_inference_query_state() -> None:
    """Step 0 must be byte-identical to the state on_llm_request builds."""
    transitions = _weather_transitions()
    expected = abstract_state(task=USER_TEXT)
    assert transitions[0].state.signature == expected.signature
    assert transitions[0].state.text == expected.text
    assert transitions[0].state.tokens == expected.tokens
    assert transitions[0].state.raw_chars == expected.raw_chars


def test_first_state_respects_surface() -> None:
    """The surface partition key is forwarded to abstract_state."""
    transitions = extract_transitions(
        messages=MESSAGES, user_text=USER_TEXT, surface=SURFACE
    )
    expected = abstract_state(task=USER_TEXT, surface=SURFACE)
    assert transitions[0].state.signature == expected.signature
    assert transitions[0].state.surface == SURFACE


def test_later_state_uses_previous_observation_and_action_history() -> None:
    """Step i>=1 is built from step i-1's observation and action history."""
    transitions = _weather_transitions()
    expected = abstract_state(
        task=USER_TEXT,
        observation=transitions[0].observation,
        recent_actions=(transitions[0].action.text,),
    )
    assert transitions[1].state.signature == expected.signature


def test_recent_action_window_is_three() -> None:
    """Only the three newest actions enter the state signature."""
    messages: list[dict[str, object]] = []
    for index in range(5):
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{index}",
                        "type": "function",
                        "function": {"name": f"tool_{index}", "arguments": "{}"},
                    }
                ],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": f"call_{index}",
                "content": f"result {index} body",
            }
        )
    transitions = extract_transitions(messages=messages, user_text="do five things")
    assert len(transitions) == 5
    expected = abstract_state(
        task="do five things",
        observation=transitions[3].observation,
        recent_actions=(
            transitions[1].action.text,
            transitions[2].action.text,
            transitions[3].action.text,
        ),
    )
    assert transitions[4].state.signature == expected.signature


# --------------------------------------------------------------------------- #
# object-shaped messages (pydantic stand-ins)
# --------------------------------------------------------------------------- #


def test_extract_transitions_accepts_object_messages() -> None:
    """Pydantic-style Message/ToolCall objects parse like dicts."""
    messages = [
        types.SimpleNamespace(role="system", content="You are AstrBot."),
        types.SimpleNamespace(role="user", content="read the config"),
        types.SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                types.SimpleNamespace(
                    id="call_obj",
                    type="function",
                    function=types.SimpleNamespace(
                        name="read_file", arguments='{"path": "a.txt"}'
                    ),
                )
            ],
        ),
        types.SimpleNamespace(
            role="tool",
            tool_call_id="call_obj",
            content=[types.SimpleNamespace(type="text", text="file contents of a.txt")],
        ),
        types.SimpleNamespace(
            role="assistant",
            content=[{"type": "text", "text": "读取完成，内容是 a"}],
        ),
    ]
    transitions = extract_transitions(messages=messages, user_text="read the config")
    assert len(transitions) == 2
    assert transitions[0].action.name == "read_file"
    assert transitions[0].observation == "file contents of a.txt"
    assert transitions[0].action.args == {"path": "a.txt"}
    assert transitions[1].action.kind == "reply"
    assert transitions[1].action.text == "读取完成，内容是 a"
    assert (
        transitions[0].state.signature
        == abstract_state(task="read the config").signature
    )


def test_extract_transitions_accepts_think_content_parts() -> None:
    """A think part still yields readable text for the sanity path."""
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "think", "think": "reasoning shown"}],
        }
    ]
    transitions = extract_transitions(messages=messages, user_text="think")
    assert len(transitions) == 1
    assert transitions[0].action.text == "reasoning shown"


# --------------------------------------------------------------------------- #
# detect_outcome
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "Error: boom",
        "FAILED to reach the API",
        "Traceback (most recent call last)",
        "request timeout",
        "connection timed out",
        "file not found",
        "permission denied",
        "Unauthorized",
        "connection refused",
        "invalid argument",
        "operation failed",
        "操作失败",
        "发生错误",
        "出现异常",
        "请求超时",
        "连接被拒绝",
        "无权限访问",
        "未找到页面",
    ],
)
def test_detect_outcome_error_markers(text: str) -> None:
    """Every known failure marker classifies as error, case-insensitively."""
    assert detect_outcome(text) == "error"


def test_detect_outcome_explicit_flag_and_empty_text() -> None:
    """The explicit flag wins; empty text is unknown; normal text is ok."""
    assert detect_outcome("", is_error=True) == "error"
    assert detect_outcome("") == "unknown"
    assert detect_outcome("   ") == "unknown"
    assert detect_outcome(None) == "unknown"
    assert detect_outcome("everything went fine") == "ok"
    assert detect_outcome("21 degrees and sunny") == "ok"


# --------------------------------------------------------------------------- #
# infer_success
# --------------------------------------------------------------------------- #


def test_infer_success_error_step_is_unknown() -> None:
    """An errored step makes the verdict deliberately inconclusive."""
    transitions = _weather_transitions()
    transitions[0].outcome = "error"
    assert infer_success(transitions=transitions, final_text="明天 21 度") is None


def test_infer_success_empty_final_text() -> None:
    """No final answer means the run failed."""
    assert infer_success(transitions=[], final_text="") is False
    assert infer_success(transitions=[], final_text="   ") is False


@pytest.mark.parametrize(
    "final_text",
    [
        "抱歉，我无法完成这个请求",
        "我不能访问这个文件",
        "I cannot access that file",
        "unable to reach the server",
        "sorry, that failed",
    ],
)
def test_infer_success_failure_wording(final_text: str) -> None:
    """Apologies and refusals count as failures."""
    assert infer_success(transitions=[], final_text=final_text) is False


def test_infer_success_positive() -> None:
    """A concrete non-apologetic answer counts as success."""
    assert (
        infer_success(transitions=[], final_text="明天 21 度，建议带件外套。") is True
    )


# --------------------------------------------------------------------------- #
# build_episode
# --------------------------------------------------------------------------- #


def test_build_episode_populates_steps_and_returns() -> None:
    """Step records carry rewards, returns and provenance; user text is scrubbed."""
    transitions = _weather_transitions()
    episode = build_episode(
        episode_id="ep-1",
        scope="qq:group:123",
        user_text="我的邮箱 a@b.com，帮我查下明天天气",
        transitions=transitions,
        rewards=[1.0, -2.0, 3.0],
        gamma=0.5,
        success=True,
        reward_source="fused",
        task_family="weather",
        created_at=1700000000.0,
        meta={"model": "deepseek"},
    )
    assert episode.episode_id == "ep-1"
    assert episode.scope == "qq:group:123"
    assert episode.created_at == 1700000000.0
    assert "<email>" in episode.user_text
    assert "a@b.com" not in episode.user_text
    assert episode.task_family == "weather"
    assert episode.success is True
    assert len(episode.steps) == 3
    assert [step.index for step in episode.steps] == [0, 1, 2]
    assert [step.reward for step in episode.steps] == pytest.approx([1.0, -2.0, 3.0])
    assert [step.discounted_return for step in episode.steps] == pytest.approx(
        [0.75, -0.5, 3.0]
    )
    assert all(step.reward_source == "fused" for step in episode.steps)
    assert episode.steps[0].action.name == "web_search"
    assert episode.steps[0].outcome == "ok"
    assert episode.steps[0].observation == transitions[0].observation
    assert episode.meta == {"model": "deepseek"}


def test_build_episode_pads_short_rewards() -> None:
    """A short rewards list is padded with 0.0 per step."""
    transitions = _weather_transitions()[:2]
    episode = build_episode(
        episode_id="ep-2",
        scope="s",
        user_text=USER_TEXT,
        transitions=transitions,
        rewards=[1.0],
        gamma=0.5,
    )
    assert len(episode.steps) == 2
    assert episode.steps[0].reward == pytest.approx(1.0)
    assert episode.steps[1].reward == pytest.approx(0.0)
    assert episode.steps[0].discounted_return == pytest.approx(1.0)
    assert episode.steps[1].discounted_return == pytest.approx(0.0)
    assert episode.steps[0].reward_source == "heuristic"


def test_build_episode_truncates_extra_rewards() -> None:
    """Surplus rewards are dropped instead of creating phantom steps."""
    transitions = _weather_transitions()[:2]
    episode = build_episode(
        episode_id="ep-3",
        scope="s",
        user_text=USER_TEXT,
        transitions=transitions,
        rewards=[1.0, 2.0, 3.0, 4.0],
        gamma=0.5,
    )
    assert len(episode.steps) == 2
    assert episode.steps[1].reward == pytest.approx(2.0)


def test_build_episode_defaults_timestamp() -> None:
    """created_at defaults to the current time and meta defaults to empty."""
    episode = build_episode(
        episode_id="ep-4",
        scope="s",
        user_text=USER_TEXT,
        transitions=[],
        rewards=[],
        gamma=0.5,
    )
    assert episode.created_at > 0.0
    assert episode.meta == {}
    assert episode.steps == []
    assert episode.success is None


# --------------------------------------------------------------------------- #
# summarize_episode
# --------------------------------------------------------------------------- #


def test_summarize_episode_is_human_readable() -> None:
    """The digest mentions the identity, the verdict, the steps and the totals."""
    episode = build_episode(
        episode_id="ep-5",
        scope="qq:group:123",
        user_text=USER_TEXT,
        transitions=_weather_transitions(),
        rewards=[1.0, -2.0, 3.0],
        gamma=0.5,
        success=True,
        task_family="weather",
    )
    summary = summarize_episode(episode)
    assert isinstance(summary, str)
    assert "ep-5" in summary
    assert "qq:group:123" in summary
    assert "outcome=success" in summary
    assert "steps=3" in summary
    assert "task_family=weather" in summary
    assert USER_TEXT in summary
    assert "web_search" in summary
    assert "r=" in summary
    assert "G=" in summary
    assert "mean_reward" in summary
    assert "return_G0" in summary
    assert len(summary.splitlines()) >= 5


def test_summarize_episode_without_steps() -> None:
    """An empty episode renders a header without totals."""
    summary = summarize_episode(Episode(episode_id="ep-0", scope="s"))
    assert "steps=0" in summary
    assert "mean_reward" not in summary


def test_transition_defaults() -> None:
    """A transition defaults to no observation and an unknown outcome."""
    reference = _weather_transitions()[0]
    transition = Transition(state=reference.state, action=reference.action)
    assert transition.observation == ""
    assert transition.outcome == "unknown"
