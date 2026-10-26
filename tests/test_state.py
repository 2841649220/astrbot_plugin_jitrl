"""Unit tests for jitrl.state: redaction, regularisation, tokens, Jaccard."""

from __future__ import annotations

import json

import pytest

from jitrl.models import ActionView
from jitrl.state import (
    MAX_STATE_CHARS,
    abstract_state,
    describe_action,
    jaccard,
    normalize_action,
    redact,
    redact_payload,
    regularize,
    tokenize,
)

REDACT_SAMPLES = [
    "write to bob@example.com or call 13800138000",
    'password="hunter2"',
    "api_key: abcdef123456",
    "key=sk-abcdefgh12345678",
    "Authorization: Bearer abcdefghij123456",
    "ghp_aaaaaaaaaaaaaaaaaaaa",
    "AKIAIOSFODNN7EXAMPLE",
    "xoxb-1234567890-abcdefghij",
    "token=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijkl",
    "see https://alice:s3cr3t@example.com/p?token=abc123&lang=zh",
    "plain text with no secrets and an <url> placeholder",
]


@pytest.mark.parametrize("sample", REDACT_SAMPLES)
def test_redact_is_idempotent(sample: str) -> None:
    """Redaction must be a fixed point so re-scrubbing cannot corrupt text."""
    once = redact(sample)
    assert redact(once) == once


def test_redact_email_and_phone() -> None:
    """Personal identifiers are replaced by their dedicated placeholders."""
    assert redact("mail bob@example.com now") == "mail <email> now"
    assert redact("call 13800138000") == "call <phone>"
    assert redact("1380013800") == "1380013800"


def test_redact_provider_tokens() -> None:
    """Well-known credential shapes collapse to a single secret placeholder."""
    assert redact("sk-abcdefgh12345678") == "<secret>"
    assert redact("ghp_aaaaaaaaaaaaaaaaaaaa") == "<secret>"
    assert redact("AKIAIOSFODNN7EXAMPLE") == "<secret>"
    assert redact("xoxb-1234567890-abcdefghij") == "<secret>"
    assert redact("Bearer abcdefghij123456") == "<secret>"


def test_redact_key_value_pairs() -> None:
    """Sensitive keys keep their name while the value is scrubbed."""
    assert redact('password="hunter2"') == "password=<secret>"
    assert redact("api_key=abcdef123456") == "api_key=<secret>"
    assert redact("Authorization: Bearer tokenvalue12345") == "Authorization: <secret>"


def test_redact_url_keeps_structure() -> None:
    """URL userinfo and sensitive query parameters are scrubbed in place."""
    scrubbed = redact(
        "go to https://alice:s3cr3t@example.com/path?token=abc123&lang=zh"
    )
    assert "alice" not in scrubbed
    assert "s3cr3t" not in scrubbed
    assert "abc123" not in scrubbed
    assert "<secret>@example.com" in scrubbed
    assert "token=<secret>" in scrubbed
    assert "lang=zh" in scrubbed


def test_redact_placeholders_survive_regularize() -> None:
    """Regularisation must not rewrite placeholders produced by redaction."""
    normalized = regularize(
        redact(
            "mail bob@example.com call 13800138000 open https://example.com/i/12345678"
        )
    )
    assert "<email>" in normalized
    assert "<phone>" in normalized
    assert "<url>" in normalized
    assert "@example.com" not in normalized


def test_redact_coerces_odd_input() -> None:
    """Non-string input is coerced instead of raising at runtime."""
    assert redact(None) == ""  # type: ignore[arg-type]
    assert redact("") == ""
    assert redact(12345) == "12345"  # type: ignore[arg-type]


def test_regularize_urls() -> None:
    """Any absolute URL collapses to a single placeholder."""
    assert regularize("open https://example.com/a/1?b=2") == "open <url>"
    assert regularize("no link here") == "no link here"


def test_regularize_numbers() -> None:
    """Long integers are volatile identifiers, short ones may be semantic."""
    assert regularize("order 1234") == "order <num>"
    assert regularize("version 2") == "version 2"
    assert regularize("score 42") == "score 42"


def test_regularize_uuid_and_hex() -> None:
    """UUIDs and long hex digests get their own placeholders."""
    assert regularize("550e8400-e29b-41d4-a716-446655440000") == "<uuid>"
    assert regularize("commit deadbeef12") == "commit <hex>"
    assert regularize("short abc") == "short abc"


def test_regularize_dates() -> None:
    """ISO timestamps and clock times collapse onto one date placeholder."""
    assert regularize("at 2024-05-06T12:34:56Z") == "at <date>"
    assert regularize("on 2024-05-06") == "on <date>"
    assert regularize("meet 09:30") == "meet <date>"


def test_regularize_quoted_payload() -> None:
    """Only long quoted payloads are abstracted away."""
    assert regularize('note "' + "z" * 45 + '"') == 'note "<str>"'
    assert regularize('note "' + "z" * 40 + '"') == 'note "' + "z" * 40 + '"'
    assert regularize("标记 \u201c" + "长" * 45 + "\u201d") == '标记 "<str>"'


def test_regularize_hex_boundary() -> None:
    """Hex runs of eight or more characters are placeholders, shorter stay."""
    assert regularize("aaaaaaaa") == "<hex>"
    assert regularize("aaaaaaa") == "aaaaaaa"
    assert regularize("abcdef12") == "<hex>"
    assert regularize("abcdef1") == "abcdef1"


def test_regularize_collapses_whitespace() -> None:
    """Whitespace is flattened so equal states compare equal."""
    assert regularize("  a \n\n b\t c  ") == "a b c"
    assert regularize("") == ""
    assert regularize(None) == ""  # type: ignore[arg-type]


def test_tokenize_chinese_bigram() -> None:
    """CJK runs are cut into adjacent-character bigrams."""
    assert tokenize("天气怎么样") == frozenset({"天气", "气怎", "怎么", "么样"})


def test_tokenize_single_cjk_char() -> None:
    """An isolated CJK character survives as a unigram."""
    assert tokenize("单") == frozenset({"单"})
    assert tokenize("打开 browser") == frozenset({"打开", "browser"})


def test_tokenize_english_and_stopwords() -> None:
    """English words are lower-cased and stopwords are dropped."""
    assert tokenize("Please Search the Weather") == frozenset({"search", "weather"})
    assert tokenize("a b c") == frozenset()


def test_tokenize_placeholders() -> None:
    """Placeholders participate in similarity as single tokens."""
    assert tokenize("<url> and 42") == frozenset({"<url>", "42"})
    assert tokenize("<EMAIL>") == frozenset({"<email>"})


def test_tokenize_drops_long_digits() -> None:
    """Digit runs are kept only when short enough to be meaningful."""
    assert tokenize("123456") == frozenset({"123456"})
    assert tokenize("1234567") == frozenset()


def test_tokenize_empty_and_punctuation() -> None:
    """Blank and punctuation-only input yields an empty token set."""
    assert tokenize("") == frozenset()
    assert tokenize("   ") == frozenset()
    assert tokenize("!!! ,,,") == frozenset()
    assert isinstance(tokenize("weather"), frozenset)


def test_jaccard_both_empty() -> None:
    """An empty query state must not match every stored state."""
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_jaccard_one_empty() -> None:
    """A one-sided empty set has no intersection and scores zero."""
    assert jaccard(frozenset(), frozenset({"a"})) == 0.0
    assert jaccard(frozenset({"a"}), frozenset()) == 0.0


def test_jaccard_disjoint() -> None:
    """Disjoint token sets score zero."""
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_jaccard_identical() -> None:
    """Identical token sets score one."""
    assert jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0


def test_jaccard_partial() -> None:
    """The score is the intersection over the union."""
    assert jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)
    assert 0.0 <= jaccard(frozenset({"a", "b"}), frozenset({"a", "b", "c"})) <= 1.0


def test_abstract_state_deterministic() -> None:
    """Identical inputs always produce identical views."""
    first = abstract_state(
        task="check weather",
        observation="sunny 25",
        recent_actions=["open(app)", "type(city)"],
        surface="web",
    )
    second = abstract_state(
        task="check weather",
        observation="sunny 25",
        recent_actions=["open(app)", "type(city)"],
        surface="web",
    )
    assert first == second
    assert first.signature == second.signature
    assert first.tokens == second.tokens


def test_abstract_state_recent_actions_window() -> None:
    """Only the newest three actions influence the state."""
    view = abstract_state(
        task="t",
        observation="o",
        recent_actions=["alpha", "bravo", "charlie", "delta", "echo"],
    )
    assert "echo" in view.tokens
    assert "delta" in view.tokens
    assert "charlie" in view.tokens
    assert "bravo" not in view.tokens
    assert "alpha" not in view.tokens


def test_abstract_state_truncates() -> None:
    """Both the signature and the prompt digest respect the limit."""
    view = abstract_state(task="x" * 500, observation="y" * 500, limit=40)
    assert len(view.signature) <= 40
    assert len(view.text) <= 40
    assert len(view.signature) <= MAX_STATE_CHARS


def test_abstract_state_sections() -> None:
    """The human-readable digest names the task, history and observation."""
    view = abstract_state(
        task="check weather", observation="sunny 25", recent_actions=["open(app)"]
    )
    assert view.text.startswith("task: check weather")
    assert "| recent: open(app)" in view.text
    assert "| observed: sunny 25" in view.text
    empty = abstract_state()
    assert empty.text == ""
    assert empty.tokens == frozenset()
    assert empty.signature == "surface=|task=|obs=|ctx="


def test_abstract_state_redacts_and_regularizes() -> None:
    """The stored signature never contains raw secrets or volatile ids."""
    view = abstract_state(
        task="mail bob@example.com",
        observation="open https://example.com/items/12345678",
        surface="web",
    )
    assert "<email>" in view.signature
    assert "<url>" in view.signature
    assert "bob@example.com" not in view.signature
    assert view.surface == "web"


def test_abstract_state_metadata() -> None:
    """Surface and raw character count are reported verbatim."""
    view = abstract_state(task="check", observation="ok", surface="cli")
    assert view.surface == "cli"
    assert view.raw_chars == len("check") + len("ok")
    assert abstract_state().raw_chars == 0


def test_normalize_action_kind() -> None:
    """Unknown action kinds fall back to the custom bucket."""
    assert normalize_action(kind=" TOOL ", name="x").kind == "tool"
    assert normalize_action(kind="reply", name="x").kind == "reply"
    assert normalize_action(kind="command", name="x").kind == "command"
    assert normalize_action(kind="weird", name="x").kind == "custom"
    assert normalize_action(kind="", name="x").kind == "custom"


def test_normalize_action_name() -> None:
    """Action names are lower-cased and stripped of punctuation."""
    assert normalize_action(kind="tool", name="  Web Search!  ").name == "web_search_"
    assert normalize_action(kind="reply", name="").name == "reply"


def test_normalize_action_aggregates_free_text() -> None:
    """Different free-text payloads of the same tool share one key."""
    first = normalize_action(
        kind="tool", name="web_search", args={"query": "weather in Beijing"}
    )
    second = normalize_action(
        kind="tool", name="web_search", args={"query": "stock price of AAPL"}
    )
    assert first.key == second.key == "tool:web_search(query=<str>)"
    assert first.key != normalize_action(kind="tool", name="web_search").key


def test_normalize_action_argument_placeholders() -> None:
    """Every argument type maps onto a stable placeholder."""
    action = normalize_action(
        kind="tool",
        name="fetch",
        args={
            "limit": 5,
            "stream": True,
            "lang": "zh",
            "cursor": None,
            "ids": [1, 2],
            "opts": {"a": 1},
        },
    )
    assert action.key == (
        "tool:fetch(cursor=<null>,ids=<list>,lang=zh,limit=<num>,opts=<obj>,stream=true)"
    )
    assert action.args["limit"] == 5


def test_normalize_action_redacts_arguments() -> None:
    """Credentials inside arguments never reach the key."""
    action = normalize_action(
        kind="tool", name="login", args={"token": "sk-abcdefgh12345678"}
    )
    assert action.args["token"] == "<secret>"
    assert "sk-" not in action.key
    assert action.key == "tool:login(token=<str>)"


def test_normalize_action_text_fallback() -> None:
    """Without caller text a compact rendering is built from the arguments."""
    action = normalize_action(kind="tool", name="search", args={"q": "x" * 100})
    assert action.text == "search(q=" + "x" * 40 + ")"
    assert normalize_action(kind="reply", name="reply").text == "reply()"


def test_normalize_action_keeps_caller_text() -> None:
    """Caller-provided text wins and is still scrubbed."""
    action = normalize_action(kind="reply", name="reply", text="  call 13800138000  ")
    assert action.text == "call <phone>"


def test_describe_action() -> None:
    """describe_action renders a prefixed, length-capped line."""
    action = normalize_action(kind="tool", name="search", text="look up weather")
    assert describe_action(action) == "[tool] look up weather"
    long_action = ActionView(
        kind="reply", name="reply", key="reply:reply()", text="y" * 200
    )
    assert describe_action(long_action) == "[reply] " + "y" * 160


def test_redact_payload_nested() -> None:
    """Nested containers are scrubbed recursively without losing types."""
    payload = {
        "owner": "bob@example.com",
        "items": [{"token": "sk-abcdefgh12345678"}],
        "count": 3,
        "flag": True,
        "none": None,
    }
    scrubbed = redact_payload(payload)
    assert scrubbed["owner"] == "<email>"
    assert scrubbed["items"][0]["token"] == "<secret>"
    assert scrubbed["count"] == 3
    assert scrubbed["flag"] is True
    assert scrubbed["none"] is None


def test_redact_payload_depth_guard() -> None:
    """Absurdly deep payloads are cut off instead of exhausting the stack."""
    payload: dict[str, object] = {"leaf": "x"}
    for _ in range(10):
        payload = {"nested": payload}
    dumped = json.dumps(redact_payload(payload))
    assert dumped.count("<deep>") == 1


def test_redact_payload_keeps_tuple() -> None:
    """Tuple payloads keep their concrete type."""
    scrubbed = redact_payload(("bob@example.com",))
    assert isinstance(scrubbed, tuple)
    assert scrubbed == ("<email>",)
