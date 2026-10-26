"""Unit tests for jitrl.policy (Algorithm 1 step 6) and jitrl.reranker."""

from __future__ import annotations

import asyncio
import math
import random
from typing import Literal

import pytest

from jitrl.models import Candidate, ValueEstimate
from jitrl.policy import (
    build_candidates,
    select_action,
    softmax,
    update_logits,
    verbalized_to_logit,
)
from jitrl.reranker import VerbalizedReranker


def make_candidate(
    key: str,
    base_logit: float = 0.0,
    *,
    source: Literal["llm", "memory", "both"] = "llm",
    count: int = 0,
    text: str = "",
) -> Candidate:
    """Build a Candidate with sane defaults for the policy tests."""
    return Candidate(
        key=key,
        text=text or key,
        base_logit=base_logit,
        source=source,
        count=count,
    )


def test_verbalized_to_logit_boundaries() -> None:
    assert verbalized_to_logit(1.0) == 0.0
    assert verbalized_to_logit(1.5) == 0.0
    assert verbalized_to_logit(0.5) == pytest.approx(math.log(0.5))
    assert verbalized_to_logit(0.0) == pytest.approx(math.log(1e-4))
    assert verbalized_to_logit(-0.3) == pytest.approx(math.log(1e-4))
    assert math.isfinite(verbalized_to_logit(float("nan")))
    assert verbalized_to_logit(0.0, floor=0.5) == pytest.approx(math.log(0.5))


def test_verbalized_to_logit_is_monotone() -> None:
    assert verbalized_to_logit(0.9) > verbalized_to_logit(0.1)
    assert verbalized_to_logit(0.1) < 0.0
    assert verbalized_to_logit(1.0) > verbalized_to_logit(0.99)


def test_softmax_is_uniform_for_tied_logits() -> None:
    distribution = softmax({"a": 0.0, "b": 0.0})
    assert distribution == pytest.approx({"a": 0.5, "b": 0.5})
    assert list(distribution) == ["a", "b"]
    assert softmax({}) == {}


def test_softmax_normalizes_and_survives_large_logits() -> None:
    distribution = softmax({"a": 1.0, "b": 0.0})
    assert sum(distribution.values()) == pytest.approx(1.0)
    assert distribution["a"] > distribution["b"]
    shifted = softmax({"a": 1000.0, "b": 999.0})
    assert shifted["a"] == pytest.approx(distribution["a"])
    assert all(math.isfinite(value) for value in shifted.values())


def test_softmax_temperature_shapes_the_distribution() -> None:
    sharp = softmax({"a": 1.0, "b": 0.0}, temperature=1.0)
    flat = softmax({"a": 1.0, "b": 0.0}, temperature=2.0)
    assert flat["a"] < sharp["a"]
    assert softmax({"a": 1.0, "b": 0.0}, temperature=0.0) == pytest.approx(sharp)
    assert softmax({"a": 1.0, "b": 0.0}, temperature=-2.0) == pytest.approx(sharp)


def test_softmax_degenerate_inputs_fall_back_to_uniform() -> None:
    neutral = softmax({"a": -math.inf, "b": -math.inf})
    assert neutral == pytest.approx({"a": 0.5, "b": 0.5})
    nan_distribution = softmax({"a": float("nan"), "b": float("nan")})
    assert nan_distribution == pytest.approx({"a": 0.5, "b": 0.5})


def test_build_candidates_merges_llm_and_memory_sources() -> None:
    candidates = build_candidates(
        llm_actions=[("tool:a", "A", 2.0), ("tool:b", "B", 1.0)],
        memory_actions={"tool:b": ("B memory", 3), "tool:c": ("C", 2)},
        limit=10,
    )
    by_key = {candidate.key: candidate for candidate in candidates}
    assert [candidate.key for candidate in candidates] == ["tool:a", "tool:b", "tool:c"]
    assert by_key["tool:a"].source == "llm"
    assert by_key["tool:a"].base_logit == 2.0
    assert by_key["tool:a"].count == 0
    assert by_key["tool:a"].confidence is None
    assert by_key["tool:b"].source == "both"
    assert by_key["tool:b"].base_logit == 1.0
    assert by_key["tool:b"].count == 3
    assert by_key["tool:b"].text == "B"
    assert by_key["tool:c"].source == "memory"
    assert by_key["tool:c"].base_logit == 0.0
    assert by_key["tool:c"].count == 2


def test_build_candidates_keeps_memory_only_logit_at_zero() -> None:
    candidates = build_candidates(memory_actions={"tool:m": ("M", 5)})
    assert len(candidates) == 1
    assert candidates[0].source == "memory"
    assert candidates[0].base_logit == 0.0
    assert candidates[0].count == 5


def test_build_candidates_deduplicates_llm_proposals() -> None:
    candidates = build_candidates(
        llm_actions=[("k", "first", 1.0), ("k", "second", 5.0)]
    )
    assert len(candidates) == 1
    assert candidates[0].text == "first"
    assert candidates[0].base_logit == 1.0


def test_build_candidates_falls_back_to_key_for_blank_text() -> None:
    llm_candidate = build_candidates(llm_actions=[("tool:x", "", 1.0)])[0]
    assert llm_candidate.text == "tool:x"
    memory_candidate = build_candidates(memory_actions={"tool:y": ("", 1)})[0]
    assert memory_candidate.text == "tool:y"


def test_build_candidates_applies_verbalized_confidences() -> None:
    candidates = build_candidates(
        llm_actions=[("a", "A", 3.0), ("b", "B", 2.0)],
        llm_scores={"a": 0.5, "b": 1.0, "unknown": 0.1},
        limit=10,
    )
    by_key = {candidate.key: candidate for candidate in candidates}
    assert len(candidates) == 2
    assert by_key["a"].base_logit == pytest.approx(math.log(0.5))
    assert by_key["a"].confidence == 0.5
    assert by_key["b"].base_logit == 0.0
    assert by_key["b"].confidence == 1.0


def test_build_candidates_reserves_a_memory_action_when_truncated() -> None:
    candidates = build_candidates(
        llm_actions=[("l1", "1", 3.0), ("l2", "2", 2.0), ("l3", "3", 1.0)],
        memory_actions={"m1": ("M", 4)},
        limit=2,
    )
    # The weakest LLM seat (l2) is traded for the memory action, so the set
    # never exceeds the configured capacity.
    assert [candidate.key for candidate in candidates] == ["l1", "m1"]
    assert len(candidates) == 2
    assert candidates[-1].source == "memory"
    assert candidates[-1].base_logit == 0.0


def test_build_candidates_keeps_all_memory_actions_without_llm() -> None:
    """Regression: hint mode passes no LLM proposals, so C is the memory side."""
    candidates = build_candidates(
        memory_actions={
            "tool:web_search(q=<str>)": ("web_search(q=x)", 1),
            "tool:bad_tool(q=<str>)": ("bad_tool(q=x)", 1),
        },
        limit=3,
    )
    assert len(candidates) == 2
    # Equal support, so the key breaks the tie in ascending order.
    assert [candidate.key for candidate in candidates] == [
        "tool:bad_tool(q=<str>)",
        "tool:web_search(q=<str>)",
    ]
    assert all(candidate.source == "memory" for candidate in candidates)
    assert all(candidate.base_logit == 0.0 for candidate in candidates)


def test_build_candidates_returns_single_memory_action_without_llm() -> None:
    candidates = build_candidates(memory_actions={"tool:only": ("only", 2)}, limit=3)
    assert len(candidates) == 1
    assert candidates[0].key == "tool:only"


def test_build_candidates_truncates_memory_only_set_by_support() -> None:
    candidates = build_candidates(
        memory_actions={
            "m1": ("1", 5),
            "m2": ("2", 1),
            "m3": ("3", 4),
            "m4": ("4", 3),
            "m5": ("5", 2),
        },
        limit=3,
    )
    assert [candidate.key for candidate in candidates] == ["m1", "m3", "m4"]
    assert [candidate.count for candidate in candidates] == [5, 4, 3]


def test_build_candidates_keeps_one_memory_seat_when_llm_fills_limit() -> None:
    candidates = build_candidates(
        llm_actions=[("l1", "1", 3.0), ("l2", "2", 2.0), ("l3", "3", 1.0)],
        memory_actions={"m_low": ("L", 1), "m_high": ("H", 9)},
        limit=3,
    )
    assert len(candidates) == 3
    assert [candidate.key for candidate in candidates] == ["l1", "l2", "m_high"]
    memory_side = [c for c in candidates if c.source == "memory"]
    assert len(memory_side) == 1


def test_build_candidates_limit_shrinks_without_memory_actions() -> None:
    candidates = build_candidates(
        llm_actions=[("l1", "1", 3.0), ("l2", "2", 2.0), ("l3", "3", 1.0)],
        limit=2,
    )
    assert [candidate.key for candidate in candidates] == ["l1", "l2"]


def test_build_candidates_capacity_one_keeps_memory_seat() -> None:
    candidates = build_candidates(
        llm_actions=[("l1", "1", 3.0), ("l2", "2", 2.0)],
        memory_actions={"m1": ("M", 1)},
        limit=1,
    )
    assert [candidate.key for candidate in candidates] == ["m1"]
    assert len(candidates) == 1


def test_build_candidates_promotes_the_best_supported_memory_action() -> None:
    candidates = build_candidates(
        llm_actions=[("l1", "1", 3.0)],
        memory_actions={"m_low": ("L", 1), "m_high": ("H", 9)},
        limit=2,
    )
    assert [candidate.key for candidate in candidates] == ["l1", "m_high"]


def test_build_candidates_returns_empty_without_input() -> None:
    assert build_candidates() == []
    assert build_candidates(llm_actions=[], memory_actions={}, limit=3) == []


def test_update_logits_applies_eq_9_shift() -> None:
    estimate = ValueEstimate(advantage_norm={"a": 0.3})
    logits = update_logits([make_candidate("a", base_logit=0.5)], estimate, beta=2.0)
    # z' = 0.5 + 2.0 * 0.3 = 1.1
    assert logits == pytest.approx({"a": 1.1})


def test_update_logits_treats_missing_advantage_as_zero() -> None:
    candidates = [make_candidate("a", 0.5), make_candidate("b", -1.0)]
    estimate = ValueEstimate(advantage_norm={"a": 0.5})
    logits = update_logits(candidates, estimate, beta=1.0)
    assert logits["a"] == pytest.approx(1.0)
    assert logits["b"] == pytest.approx(-1.0)


def test_update_logits_can_use_raw_advantages() -> None:
    estimate = ValueEstimate(advantage={"a": 0.5}, advantage_norm={"a": 0.1})
    candidate = make_candidate("a", base_logit=0.0)
    assert update_logits([candidate], estimate, beta=1.0)["a"] == pytest.approx(0.1)
    assert update_logits([candidate], estimate, beta=1.0, normalize=False)[
        "a"
    ] == pytest.approx(0.5)


def test_select_action_samples_from_the_updated_distribution() -> None:
    candidates = [make_candidate("a"), make_candidate("b")]
    estimate = ValueEstimate(advantage_norm={"a": 0.9, "b": -0.9})
    decision = select_action(candidates, estimate, beta=1.0, rng=random.Random(1))
    expected = math.exp(0.9) / (math.exp(0.9) + math.exp(-0.9))
    assert decision.chosen is not None
    assert decision.chosen.key == "a"
    assert decision.probabilities["a"] == pytest.approx(expected)
    assert decision.probabilities["b"] == pytest.approx(1.0 - expected)
    assert decision.prior == pytest.approx({"a": 0.5, "b": 0.5})
    assert decision.logits_before == pytest.approx({"a": 0.0, "b": 0.0})
    assert decision.logits_after == pytest.approx({"a": 0.9, "b": -0.9})
    assert decision.mode == "hint"
    assert decision.beta == 1.0
    assert decision.applied is True
    assert "picked a" in decision.reason
    assert "beta=1" in decision.reason
    assert "advantage=+0.90" in decision.reason
    assert "p=0.86" in decision.reason
    assert "prior 0.50" in decision.reason


def test_select_action_is_a_sample_not_an_argmax() -> None:
    candidates = [make_candidate("a"), make_candidate("b")]
    estimate = ValueEstimate(advantage_norm={"a": 0.9, "b": -0.9})
    decision = select_action(candidates, estimate, beta=1.0, rng=random.Random(2))
    assert decision.chosen is not None
    assert decision.chosen.key == "b"


def test_select_action_is_deterministic_for_a_fixed_seed() -> None:
    candidates = [make_candidate("a"), make_candidate("b")]
    estimate = ValueEstimate(advantage_norm={"a": 0.4, "b": 0.1})
    first = select_action(candidates, estimate, beta=2.0, rng=random.Random(3))
    second = select_action(candidates, estimate, beta=2.0, rng=random.Random(3))
    assert first.chosen is not None
    assert second.chosen is not None
    assert first.chosen.key == second.chosen.key
    assert first.probabilities == second.probabilities


def test_select_action_off_mode_reports_without_picking() -> None:
    candidates = [make_candidate("a"), make_candidate("b")]
    estimate = ValueEstimate(advantage_norm={"a": 0.9, "b": -0.9})
    decision = select_action(
        candidates, estimate, beta=1.0, mode="off", rng=random.Random(1)
    )
    assert decision.chosen is None
    assert decision.applied is False
    assert decision.mode == "off"
    assert decision.probabilities == decision.prior
    assert decision.logits_after == decision.logits_before
    assert decision.reason == ""


def test_select_action_without_candidates_returns_an_empty_decision() -> None:
    estimate = ValueEstimate()
    decision = select_action([], estimate, mode="logit")
    assert decision.chosen is None
    assert decision.mode == "logit"
    assert decision.estimate is estimate
    assert decision.probabilities == {}
    assert decision.prior == {}
    assert decision.reason == ""


def test_parse_reads_plain_json() -> None:
    keys = ["tool:a", "tool:b"]
    raw = '{"scores": [{"id": 1, "confidence": 80}, {"id": 2, "confidence": 20}]}'
    scores = VerbalizedReranker.parse(raw, keys)
    assert scores == pytest.approx({"tool:a": 0.8, "tool:b": 0.2})
    assert list(scores) == keys


def test_parse_reads_fenced_json() -> None:
    keys = ["tool:a", "tool:b"]
    raw = (
        "Here you go:\n\n"
        "```json\n"
        '{"scores": [{"id": 1, "confidence": 70}, {"id": 2, "confidence": 10}]}\n'
        "```\n"
    )
    scores = VerbalizedReranker.parse(raw, keys)
    assert scores["tool:a"] == pytest.approx(0.7)
    assert scores["tool:b"] == pytest.approx(0.1)


def test_parse_recovers_json_embedded_in_prose() -> None:
    keys = ["tool:a", "tool:b"]
    raw = 'Sure, here it is: {"scores": [{"id": 1, "confidence": 1.0}]} Done.'
    scores = VerbalizedReranker.parse(raw, keys)
    assert scores["tool:a"] == pytest.approx(1.0)
    assert scores["tool:b"] == pytest.approx(0.5)


def test_parse_recovers_truncated_json_by_regex() -> None:
    keys = ["tool:a", "tool:b"]
    raw = '{"scores": [{"id": 1, "confidence": 90}, {"id": 2, "confidence":'
    scores = VerbalizedReranker.parse(raw, keys)
    assert scores["tool:a"] == pytest.approx(0.9)
    assert scores["tool:b"] == pytest.approx(0.5)


def test_parse_accepts_dict_scores_and_positional_ids() -> None:
    keys = ["tool:a", "tool:b"]
    named = VerbalizedReranker.parse('{"scores": {"tool:a": 0.6, "tool:b": 0.4}}', keys)
    assert named == pytest.approx({"tool:a": 0.6, "tool:b": 0.4})
    positional = VerbalizedReranker.parse('{"scores": {"1": 0.2, "2": 0.9}}', keys)
    assert positional == pytest.approx({"tool:a": 0.2, "tool:b": 0.9})


def test_parse_accepts_bare_list_of_numbers() -> None:
    keys = ["tool:a", "tool:b"]
    scores = VerbalizedReranker.parse("[0.25, 0.75]", keys)
    assert scores == pytest.approx({"tool:a": 0.25, "tool:b": 0.75})


def test_parse_scales_zero_to_hundred_values() -> None:
    keys = ["tool:a", "tool:b"]
    scores = VerbalizedReranker.parse(
        '{"scores": [{"id": 1, "confidence": 50}, {"id": 2, "confidence": 100}]}',
        keys,
    )
    assert scores["tool:a"] == pytest.approx(0.5)
    assert scores["tool:b"] == pytest.approx(1.0)


def test_parse_clamps_out_of_range_values() -> None:
    keys = ["tool:a", "tool:b"]
    scores = VerbalizedReranker.parse(
        '{"scores": [{"id": 1, "confidence": -20}, {"id": 2, "confidence": 250}]}',
        keys,
    )
    assert scores["tool:a"] == 0.0
    assert scores["tool:b"] == 1.0


def test_parse_fills_missing_keys_with_neutral_prior() -> None:
    keys = ["tool:a", "tool:b", "tool:c"]
    scores = VerbalizedReranker.parse(
        '{"scores": [{"id": 1, "confidence": 0.9}]}', keys
    )
    assert scores["tool:a"] == pytest.approx(0.9)
    assert scores["tool:b"] == 0.5
    assert scores["tool:c"] == 0.5
    assert list(scores) == keys


def test_parse_handles_garbage_and_empty_input() -> None:
    keys = ["tool:a", "tool:b"]
    assert VerbalizedReranker.parse("I cannot help with that.", keys) == {
        "tool:a": 0.5,
        "tool:b": 0.5,
    }
    assert VerbalizedReranker.parse("", keys) == {"tool:a": 0.5, "tool:b": 0.5}
    assert VerbalizedReranker.parse('{"scores": []}', []) == {}


def test_parse_maps_action_keys_as_identifiers() -> None:
    keys = ["tool:a", "tool:b"]
    scores = VerbalizedReranker.parse('{"scores": {"tool:b": 60}}', keys)
    assert scores["tool:b"] == pytest.approx(0.6)
    assert scores["tool:a"] == 0.5


def test_reranker_score_uses_keyword_generate() -> None:
    calls: list[dict[str, object]] = []

    async def generate(*, prompt: str, system_prompt: str | None = None) -> str:
        calls.append({"prompt": prompt, "system_prompt": system_prompt})
        return '{"scores": [{"id": 1, "confidence": 30}, {"id": 2, "confidence": 90}]}'

    reranker = VerbalizedReranker(generate)
    result = asyncio.run(
        reranker.score(
            task="find the weather",
            state_text="user asks about weather",
            candidates=[
                ("tool:weather", "check weather"),
                ("tool:search", "web search"),
            ],
            observation="no data yet",
            system_prompt="be terse",
        )
    )
    assert result.ok is True
    assert result.error == ""
    assert result.raw.startswith("{")
    assert result.scores == pytest.approx({"tool:weather": 0.3, "tool:search": 0.9})
    assert len(calls) == 1
    assert calls[0]["system_prompt"] == "be terse"
    assert "tool:weather" in str(calls[0]["prompt"])


def test_reranker_truncates_candidates_to_max_candidates() -> None:
    prompts: list[str] = []

    async def generate(*, prompt: str, system_prompt: str | None = None) -> str:
        prompts.append(prompt)
        return '{"scores": [{"id": 1, "confidence": 100}]}'

    candidates = [(f"tool:{index}", f"action {index}") for index in range(6)]
    reranker = VerbalizedReranker(generate, max_candidates=2)
    result = asyncio.run(
        reranker.score(task="t", state_text="s", candidates=candidates)
    )
    assert result.ok is True
    assert len(result.scores) == 6
    assert result.scores["tool:0"] == pytest.approx(1.0)
    assert result.scores["tool:1"] == 0.5
    assert result.scores["tool:5"] == 0.5
    assert "action 0" in prompts[0]
    assert "action 1" in prompts[0]
    assert "action 2" not in prompts[0]


def test_reranker_timeout_returns_not_ok() -> None:
    async def slow(**kwargs: object) -> str:
        await asyncio.sleep(5.0)
        return "{}"

    reranker = VerbalizedReranker(slow, timeout=0.01)
    result = asyncio.run(
        reranker.score(task="t", state_text="s", candidates=[("k", "K")])
    )
    assert result.ok is False
    assert "timeout" in result.error
    assert result.scores == {}


def test_reranker_exception_returns_not_ok() -> None:
    async def broken(**kwargs: object) -> str:
        raise RuntimeError("boom")

    reranker = VerbalizedReranker(broken)
    result = asyncio.run(
        reranker.score(task="t", state_text="s", candidates=[("k", "K")])
    )
    assert result.ok is False
    assert "RuntimeError" in result.error
    assert "boom" in result.error


def test_reranker_without_generate_returns_not_ok() -> None:
    reranker = VerbalizedReranker(None)
    result = asyncio.run(
        reranker.score(task="t", state_text="s", candidates=[("k", "K")])
    )
    assert result.ok is False
    assert "no generate" in result.error


def test_reranker_without_candidates_returns_not_ok() -> None:
    async def generate(**kwargs: object) -> str:
        return "{}"

    reranker = VerbalizedReranker(generate)
    result = asyncio.run(reranker.score(task="t", state_text="s", candidates=[]))
    assert result.ok is False
    assert "no candidates" in result.error


def test_reranker_empty_response_returns_not_ok() -> None:
    async def empty(**kwargs: object) -> str:
        return "   "

    reranker = VerbalizedReranker(empty)
    result = asyncio.run(
        reranker.score(task="t", state_text="s", candidates=[("k", "K")])
    )
    assert result.ok is False
    assert "empty model response" in result.error


def test_build_prompt_contains_json_contract_and_context() -> None:
    prompt = VerbalizedReranker.build_prompt(
        "task text", "state text", [("k", "K action")], "obs text"
    )
    assert "task text" in prompt
    assert "state text" in prompt
    assert "obs text" in prompt
    assert "1. [k] K action" in prompt
    assert '"scores"' in prompt
    assert '"confidence"' in prompt
    assert "JSON only" in prompt

    without_observation = VerbalizedReranker.build_prompt("t", "s", [("k", "K")])
    assert "Recent observation" not in without_observation
