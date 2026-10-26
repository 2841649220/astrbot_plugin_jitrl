"""Unit tests for the step-wise reward evaluator (Algorithm 1, step 7)."""

from __future__ import annotations

import asyncio

import pytest

from jitrl.evaluator import RewardResult, StepRewardEvaluator, discounted_returns


class _FakeGenerate:
    """Async stand-in for the plugin's LLM call that records every invocation."""

    def __init__(
        self,
        reply: str = "",
        *,
        delay: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.delay = delay
        self.error = error
        self.calls: list[dict[str, str]] = []

    async def __call__(self, **kwargs: str) -> str:
        self.calls.append(kwargs)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.reply


# --------------------------------------------------------------------------- #
# discounted_returns
# --------------------------------------------------------------------------- #


def test_discounted_returns_matches_closed_form() -> None:
    """G_t is the standard backward recursion for gamma=0.5."""
    returns = discounted_returns([1.0, -2.0, 3.0], 0.5)
    assert len(returns) == 3
    assert returns[2] == pytest.approx(3.0)
    assert returns[1] == pytest.approx(-2.0 + 0.5 * 3.0)
    assert returns[0] == pytest.approx(1.0 + 0.5 * (0.5 * 3.0 - 2.0))
    assert returns == pytest.approx([0.75, -0.5, 3.0])


def test_discounted_returns_empty_input() -> None:
    """Empty input stays empty; the return type is always a list."""
    assert discounted_returns([], 0.5) == []
    assert isinstance(discounted_returns([], 0.5), list)


def test_discounted_returns_clamps_gamma() -> None:
    """gamma above 1 collapses to 1 and gamma below 0 collapses to 0."""
    assert discounted_returns([1.0, 2.0], 2.0) == pytest.approx([3.0, 2.0])
    assert discounted_returns([1.0, 2.0], -1.0) == pytest.approx([1.0, 2.0])


def test_discounted_returns_length_and_zeros() -> None:
    """The output length tracks the input and all-zero rewards stay zero."""
    assert len(discounted_returns([0.0] * 5, 0.9)) == 5
    assert discounted_returns([0.0, 0.0, 0.0], 0.9) == pytest.approx([0.0, 0.0, 0.0])


# --------------------------------------------------------------------------- #
# heuristic_rewards
# --------------------------------------------------------------------------- #


def test_heuristic_error_outcome_is_negative() -> None:
    """An errored action scores -2.0 when its observation is informative."""
    rewards = StepRewardEvaluator.heuristic_rewards(
        ["web_search"], ["total failure: connection reset by peer"], ["error"]
    )
    assert rewards == pytest.approx([-2.0])
    assert isinstance(rewards, list)
    assert all(isinstance(value, float) for value in rewards)


def test_heuristic_ok_with_observation() -> None:
    """A successful action with a real observation scores +1.0."""
    rewards = StepRewardEvaluator.heuristic_rewards(
        ["web_search"], ["tomorrow 21 degrees, bring a jacket"], ["ok"]
    )
    assert rewards == pytest.approx([1.0])


def test_heuristic_ok_with_empty_observation() -> None:
    """ok + empty observation keeps the 0.0 base and loses 0.5 for low signal."""
    rewards = StepRewardEvaluator.heuristic_rewards(["web_search"], [""], ["ok"])
    assert rewards == pytest.approx([-0.5])


def test_heuristic_write_bonus() -> None:
    """A successful mutating action earns the extra +0.5."""
    rewards = StepRewardEvaluator.heuristic_rewards(
        ["save_file"], ["saved the report to disk successfully"], ["ok"]
    )
    assert rewards == pytest.approx([1.5])


def test_heuristic_low_information_penalty() -> None:
    """Short or empty-result observations lose 0.5."""
    short = StepRewardEvaluator.heuristic_rewards(["web_search"], ["ok"], ["ok"])
    empty_list = StepRewardEvaluator.heuristic_rewards(["web_search"], ["[]"], ["ok"])
    empty_object = StepRewardEvaluator.heuristic_rewards(["web_search"], ["{}"], ["ok"])
    assert short == pytest.approx([0.5])
    assert empty_list == pytest.approx([0.5])
    assert empty_object == pytest.approx([0.5])


def test_heuristic_terminal_and_success_bonus() -> None:
    """The last step is boosted by success and by a terminal action."""
    rewards = StepRewardEvaluator.heuristic_rewards(
        ["web_search", "finish"],
        ["a sufficiently long observation", "ok"],
        ["ok", "ok"],
        success=True,
    )
    assert rewards == pytest.approx([1.0, 2.0])


def test_heuristic_success_false_penalty() -> None:
    """A failed run subtracts 1.0 on the last step and stays clamped."""
    rewards = StepRewardEvaluator.heuristic_rewards(
        ["finish"], ["cannot finish the task"], ["error"], success=False
    )
    assert rewards == pytest.approx([-2.5])


def test_heuristic_max_reward_clamp() -> None:
    """Every value respects the configured cap."""
    rewards = StepRewardEvaluator.heuristic_rewards(
        ["save_file"],
        ["saved the report to disk successfully"],
        ["ok"],
        max_reward=1.0,
    )
    assert rewards == pytest.approx([1.0])


def test_heuristic_aligns_missing_inputs() -> None:
    """Missing observations/outcomes are treated as empty/unknown, not errors."""
    rewards = StepRewardEvaluator.heuristic_rewards(["a", "b"], ["only one"], ["ok"])
    assert len(rewards) == 2
    assert rewards[0] == pytest.approx(1.0)
    assert rewards[1] == pytest.approx(-0.5)


# --------------------------------------------------------------------------- #
# parse_rewards
# --------------------------------------------------------------------------- #

_JSON_REPLY = (
    '{"step_analysis": [{"step": 0, "action": "a", "score": 2}, '
    '{"step": 1, "action": "b", "score": -1}], "overall_assessment": "mixed"}'
)


def test_parse_rewards_direct_json() -> None:
    """Strict JSON with step_analysis[].score parses in step order."""
    assert StepRewardEvaluator.parse_rewards(_JSON_REPLY, 2) == pytest.approx(
        [2.0, -1.0]
    )


def test_parse_rewards_code_fence() -> None:
    """A fenced json block surrounded by chatter still parses."""
    fence = chr(96) * 3
    raw = f"Sure:\n{fence}json\n{_JSON_REPLY}\n{fence}\nHope this helps."
    assert StepRewardEvaluator.parse_rewards(raw, 2) == pytest.approx([2.0, -1.0])


def test_parse_rewards_embedded_braces() -> None:
    """A JSON object embedded in prose is recovered via the brace span."""
    raw = 'Here is the result: {"step_analysis": [{"score": 1.5}]} done'
    assert StepRewardEvaluator.parse_rewards(raw, 1) == pytest.approx([1.5])


def test_parse_rewards_loose_score_scan() -> None:
    """Without valid JSON the score pairs are scanned textually."""
    raw = "Step 0 score: 3, Step 1 score: -2"
    assert StepRewardEvaluator.parse_rewards(raw, 2) == pytest.approx([3.0, -2.0])


def test_parse_rewards_bare_list() -> None:
    """A bare [1, -2, 3] list is accepted and padded to the expected length."""
    parsed = StepRewardEvaluator.parse_rewards("[1, -2, 3]", 5)
    assert parsed == pytest.approx([1.0, -2.0, 3.0, 0.0, 0.0])


def test_parse_rewards_truncates_extra_scores() -> None:
    """More scores than steps are truncated to the expected length."""
    parsed = StepRewardEvaluator.parse_rewards("[1, 2, 3, 4]", 2)
    assert parsed == pytest.approx([1.0, 2.0])
    assert len(parsed or []) == 2


def test_parse_rewards_clamps_scores() -> None:
    """Out-of-scale scores are clamped to the cap."""
    raw = '{"step_analysis": [{"score": 9}, {"score": -9}]}'
    assert StepRewardEvaluator.parse_rewards(raw, 2) == pytest.approx([3.0, -3.0])
    assert StepRewardEvaluator.parse_rewards(raw, 2, 2.0) == pytest.approx([2.0, -2.0])


def test_parse_rewards_pads_missing_scores() -> None:
    """Fewer scores than steps are padded with 0.0."""
    raw = '{"step_analysis": [{"score": 2}]}'
    assert StepRewardEvaluator.parse_rewards(raw, 3) == pytest.approx([2.0, 0.0, 0.0])


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "I could not evaluate this episode.",
        '{"step_analysis": "not-a-list"}',
        "Result: USEFUL. Usefulness: useful. Certainty: certain.",
    ],
)
def test_parse_rewards_returns_none_on_failure(raw: str) -> None:
    """Unparseable output returns None instead of raising."""
    assert StepRewardEvaluator.parse_rewards(raw, 2) is None


def test_parse_rewards_expected_zero() -> None:
    """Zero steps means an empty score list, even for empty input."""
    assert StepRewardEvaluator.parse_rewards("", 0) == []
    assert StepRewardEvaluator.parse_rewards(_JSON_REPLY, 0) == []


# --------------------------------------------------------------------------- #
# build_prompt
# --------------------------------------------------------------------------- #


def test_build_prompt_contains_paper_rubric_and_json_shape() -> None:
    """The prompt carries the G.1.2 rubric and demands machine-readable JSON."""
    system_prompt, user_prompt = StepRewardEvaluator.build_prompt(
        "check tomorrow's weather",
        ['web_search({"q": "weather"})', "finish({})"],
        ["21C and sunny", "ok"],
        ["ok", "ok"],
    )
    assert "USEFUL" in system_prompt
    assert "HARMFUL" in system_prompt
    assert "NEUTRAL" in system_prompt
    assert "certain" in system_prompt
    assert "+3" in system_prompt
    assert "-3" in system_prompt
    assert "step_analysis" in system_prompt
    assert "overall_assessment" in system_prompt
    assert "check tomorrow's weather" in user_prompt
    assert "web_search" in user_prompt
    assert "Step 1" in user_prompt
    assert "21C and sunny" in user_prompt
    assert "exactly 2" in user_prompt


def test_build_prompt_truncates_long_observations() -> None:
    """A long observation is clipped inside the user prompt."""
    _, user_prompt = StepRewardEvaluator.build_prompt(
        "task", ["tool()"], ["x" * 900], ["ok"]
    )
    assert "x" * 900 not in user_prompt
    assert "..." in user_prompt


def test_build_prompt_budget_shrinks_per_step_allowance() -> None:
    """Long episodes share one prompt budget instead of 600 chars per step."""
    steps = 40
    actions = [f"tool({index})" for index in range(steps)]
    observations = ["y" * 600 for _ in range(steps)]
    outcomes = ["ok"] * steps
    _, user_prompt = StepRewardEvaluator.build_prompt(
        "task", actions, observations, outcomes
    )
    # 12000 // 40 = 300 characters of observation per step.
    assert user_prompt.count("y" * 300 + " ...") == steps
    assert "y" * 301 not in user_prompt
    assert f"STEPS ({steps} total)" in user_prompt

    # A short episode keeps the full 600-character allowance.
    _, short_prompt = StepRewardEvaluator.build_prompt(
        "task", ["tool(0)"], ["y" * 600], ["ok"]
    )
    assert "y" * 600 in short_prompt
    assert "y" * 601 not in short_prompt


# --------------------------------------------------------------------------- #
# score(): mode routing and degradation
# --------------------------------------------------------------------------- #


def test_reward_result_defaults() -> None:
    """The result dataclass is safe to construct empty."""
    result = RewardResult()
    assert result.rewards == []
    assert result.source == "heuristic"
    assert result.overall == ""
    assert result.ok is False
    assert result.error == ""


@pytest.mark.asyncio
async def test_score_empty_inputs() -> None:
    """No steps means an empty but successful result."""
    evaluator = StepRewardEvaluator(_FakeGenerate(_JSON_REPLY), mode="llm")
    result = await evaluator.score(task="t", actions=[], observations=[], outcomes=[])
    assert result.rewards == []
    assert result.ok is True
    assert result.source == "heuristic"


@pytest.mark.asyncio
async def test_score_mode_off_returns_zeros() -> None:
    """mode=off never calls the model and yields neutral rewards."""
    generate = _FakeGenerate(_JSON_REPLY)
    evaluator = StepRewardEvaluator(generate, mode="off")
    result = await evaluator.score(
        task="t",
        actions=["a", "b"],
        observations=["obs one", "obs two"],
        outcomes=["ok", "ok"],
    )
    assert result.rewards == [0.0, 0.0]
    assert result.source == "heuristic"
    assert result.ok is True
    assert generate.calls == []


@pytest.mark.asyncio
async def test_score_mode_heuristic_skips_llm() -> None:
    """mode=heuristic uses the rules and never calls the model."""
    generate = _FakeGenerate(_JSON_REPLY)
    evaluator = StepRewardEvaluator(generate, mode="heuristic")
    result = await evaluator.score(
        task="t",
        actions=["a", "b"],
        observations=["observation one", "observation two"],
        outcomes=["ok", "ok"],
    )
    assert result.source == "heuristic"
    assert result.ok is True
    assert result.rewards == pytest.approx([1.0, 1.0])
    assert generate.calls == []


@pytest.mark.asyncio
async def test_score_without_generate_degrades_to_heuristic() -> None:
    """A missing model callable short-circuits to the heuristic path."""
    evaluator = StepRewardEvaluator(None, mode="llm")
    result = await evaluator.score(
        task="t",
        actions=["save_file"],
        observations=["wrote the file successfully"],
        outcomes=["ok"],
    )
    assert result.source == "heuristic"
    assert result.rewards == pytest.approx([1.5])
    assert result.ok is True


@pytest.mark.asyncio
async def test_score_auto_skips_llm_for_single_clean_step() -> None:
    """auto must not spend an LLM call on a one-step run without errors."""
    generate = _FakeGenerate(_JSON_REPLY)
    evaluator = StepRewardEvaluator(generate, mode="auto")
    result = await evaluator.score(
        task="t",
        actions=["web_search"],
        observations=["a long enough observation"],
        outcomes=["ok"],
    )
    assert generate.calls == []
    assert result.source == "heuristic"
    assert result.ok is True
    assert result.rewards == pytest.approx([1.0])


@pytest.mark.asyncio
async def test_score_auto_calls_llm_for_errored_single_step() -> None:
    """auto does call the model when the single step errored."""
    generate = _FakeGenerate('{"step_analysis": [{"score": -3}]}')
    evaluator = StepRewardEvaluator(generate, mode="auto")
    result = await evaluator.score(
        task="t",
        actions=["web_search"],
        observations=["connection reset by peer"],
        outcomes=["error"],
    )
    assert len(generate.calls) == 1
    assert result.source == "fused"
    assert result.ok is True
    # 0.5 * (-3) + 0.5 * (-2) == -2.5
    assert result.rewards == pytest.approx([-2.5])


@pytest.mark.asyncio
async def test_score_auto_calls_llm_for_multi_step() -> None:
    """auto calls the model as soon as the episode has two steps."""
    generate = _FakeGenerate('{"step_analysis": [{"score": 1}, {"score": 1}]}')
    evaluator = StepRewardEvaluator(generate, mode="auto")
    result = await evaluator.score(
        task="t",
        actions=["a", "b"],
        observations=["obs one", "obs two"],
        outcomes=["ok", "ok"],
    )
    assert len(generate.calls) == 1
    assert result.source == "fused"


@pytest.mark.asyncio
async def test_score_fuses_llm_and_heuristic() -> None:
    """Fusion is (1 - w) * llm + w * heuristic with exact arithmetic."""
    generate = _FakeGenerate(
        '{"step_analysis": [{"score": 3}, {"score": 0}], '
        '"overall_assessment": "solid run"}'
    )
    evaluator = StepRewardEvaluator(generate, mode="llm", heuristic_weight=0.5)
    result = await evaluator.score(
        task="t",
        actions=["web_search", "finish"],
        observations=["today weather 21 degrees sunny", "ok"],
        outcomes=["ok", "ok"],
        success=True,
    )
    assert result.source == "fused"
    assert result.ok is True
    assert result.error == ""
    assert result.rewards == pytest.approx([2.0, 1.0])
    assert result.overall == "solid run"


@pytest.mark.asyncio
async def test_score_fusion_weight_is_configurable() -> None:
    """heuristic_weight=1.0 ignores the model, 0.0 ignores the rules."""
    reply = '{"step_analysis": [{"score": 3}, {"score": 3}]}'
    args = {
        "task": "t",
        "actions": ["web_search", "finish"],
        "observations": ["today weather 21 degrees sunny", "ok"],
        "outcomes": ["ok", "ok"],
        "success": True,
    }
    full_heuristic = StepRewardEvaluator(
        _FakeGenerate(reply), mode="llm", heuristic_weight=1.0
    )
    full_llm = StepRewardEvaluator(
        _FakeGenerate(reply), mode="llm", heuristic_weight=0.0
    )
    assert (await full_heuristic.score(**args)).rewards == pytest.approx([1.0, 2.0])
    assert (await full_llm.score(**args)).rewards == pytest.approx([3.0, 3.0])


@pytest.mark.asyncio
async def test_score_passes_prompt_and_system_prompt_as_keywords() -> None:
    """The model callable receives both prompts as keyword arguments."""
    generate = _FakeGenerate('{"step_analysis": [{"score": 1}, {"score": 1}]}')
    evaluator = StepRewardEvaluator(generate, mode="llm")
    await evaluator.score(
        task="find the weather",
        actions=["web_search", ""],
        observations=["sunny", ""],
        outcomes=["ok", "unknown"],
    )
    assert len(generate.calls) == 1
    call = generate.calls[0]
    assert set(call) == {"prompt", "system_prompt"}
    assert "find the weather" in call["prompt"]
    assert "step_analysis" in call["system_prompt"]


@pytest.mark.asyncio
async def test_score_clamps_llm_scores_to_max_reward() -> None:
    """LLM scores above the cap are clipped after fusion."""
    generate = _FakeGenerate('{"step_analysis": [{"score": 9}, {"score": -9}]}')
    evaluator = StepRewardEvaluator(
        generate, mode="llm", max_reward=2.0, heuristic_weight=0.0
    )
    result = await evaluator.score(
        task="t",
        actions=["a", "b"],
        observations=["obs one", "obs two"],
        outcomes=["ok", "ok"],
    )
    assert result.rewards == pytest.approx([2.0, -2.0])


@pytest.mark.asyncio
async def test_score_degrades_on_unparseable_reply() -> None:
    """A model answer without scores falls back to the heuristic."""
    generate = _FakeGenerate("I think the run went fine.")
    evaluator = StepRewardEvaluator(generate, mode="llm")
    result = await evaluator.score(
        task="t",
        actions=["web_search", "finish"],
        observations=["today weather 21 degrees sunny", "ok"],
        outcomes=["ok", "ok"],
        success=True,
    )
    assert result.source == "heuristic"
    assert result.ok is False
    assert "unparseable" in result.error
    assert result.rewards == pytest.approx([1.0, 2.0])


@pytest.mark.asyncio
async def test_score_degrades_on_generator_exception() -> None:
    """A raising model callable is caught and downgraded."""
    generate = _FakeGenerate(error=RuntimeError("provider exploded"))
    evaluator = StepRewardEvaluator(generate, mode="llm")
    result = await evaluator.score(
        task="t",
        actions=["web_search", "finish"],
        observations=["today weather 21 degrees sunny", "ok"],
        outcomes=["ok", "ok"],
        success=True,
    )
    assert result.ok is False
    assert result.source == "heuristic"
    assert "RuntimeError" in result.error
    assert result.rewards == pytest.approx([1.0, 2.0])


@pytest.mark.asyncio
async def test_score_degrades_on_timeout() -> None:
    """A slow model callable is cut off by the timeout and downgraded."""
    generate = _FakeGenerate(
        '{"step_analysis": [{"score": 3}, {"score": 3}]}', delay=0.5
    )
    evaluator = StepRewardEvaluator(generate, mode="llm", timeout=0.01)
    result = await evaluator.score(
        task="t",
        actions=["web_search", "finish"],
        observations=["today weather 21 degrees sunny", "ok"],
        outcomes=["ok", "ok"],
        success=True,
    )
    assert result.ok is False
    assert result.source == "heuristic"
    assert "timeout" in result.error.lower()
    assert result.rewards == pytest.approx([1.0, 2.0])


@pytest.mark.asyncio
async def test_score_aligns_to_shortest_input() -> None:
    """The step count is the shortest of actions/observations/outcomes."""
    generate = _FakeGenerate(
        '{"step_analysis": [{"score": 1}, {"score": 1}, {"score": 1}]}'
    )
    evaluator = StepRewardEvaluator(generate, mode="llm")
    result = await evaluator.score(
        task="t",
        actions=["a", "b", "c"],
        observations=["obs one", "obs two"],
        outcomes=["ok", "ok", "ok"],
    )
    assert len(result.rewards) == 2
    assert len(generate.calls) == 1


@pytest.mark.asyncio
async def test_score_unknown_mode_falls_back_to_auto() -> None:
    """An unrecognised mode behaves like auto instead of raising."""
    generate = _FakeGenerate('{"step_analysis": [{"score": 1}]}')
    evaluator = StepRewardEvaluator(generate, mode="banana")
    single = await evaluator.score(
        task="t", actions=["a"], observations=["obs one"], outcomes=["ok"]
    )
    assert generate.calls == []
    assert single.source == "heuristic"
    await evaluator.score(
        task="t", actions=["a", "b"], observations=["o1", "o2"], outcomes=["ok", "ok"]
    )
    assert len(generate.calls) == 1
