"""State construction and action normalisation for JitRL (Algorithm 1).

Raw agent observations -- DOM snapshots, verbose game transcripts, chat
history -- are far too noisy to compare directly. This module compresses them
into a StateView whose token set lets Eq. 24 (Jaccard similarity) retrieve
functionally equivalent past states, and normalises actions so that an
experience learned in one episode stays recognisable in the next one.

The pipeline is deliberately linear and stateless:

    redact -> regularize -> tokenize

"redact" must run first: running "regularize" beforehand would hide secrets
behind "<num>"/"<url>" placeholders before they can be scrubbed. Every
placeholder produced by "redact" is preserved verbatim by "regularize", so
scrubbing twice yields the same text as scrubbing once.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .models import ActionKind, ActionView, StateView

MAX_STATE_CHARS: int = 1200
"""Upper bound on the textual state representation embedded in a prompt."""

_MAX_ACTION_CONTEXT = 3
"""Only the newest actions belong to the paper's local action history."""

_MAX_TASK_CHARS = 160
_MAX_OBSERVATION_CHARS = 400
_MAX_ACTION_CHARS = 80
_MAX_PAYLOAD_DEPTH = 6
_MAX_ENUM_ARG_CHARS = 24
_MAX_ARG_TEXT_CHARS = 40
_MAX_DESCRIBE_CHARS = 160
_MAX_DIGIT_TOKEN_CHARS = 6

# Placeholder vocabulary shared by "redact", "regularize" and "tokenize".
_PLACEHOLDER_NAMES: tuple[str, ...] = (
    "url",
    "num",
    "hex",
    "uuid",
    "date",
    "str",
    "email",
    "phone",
    "secret",
    "list",
    "obj",
    "null",
    "deep",
)
_PLACEHOLDER_ALT = "|".join(_PLACEHOLDER_NAMES)
_PLACEHOLDER_RE = re.compile(rf"<(?:{_PLACEHOLDER_ALT})>")

# Credential-bearing key names, longest first so the alternation cannot match
# a shorter prefix of a longer key ("auth" versus "authorization").
_SENSITIVE_KEYS: tuple[str, ...] = tuple(
    sorted(
        (
            "password",
            "passwd",
            "pwd",
            "token",
            "secret",
            "api_key",
            "apikey",
            "api-key",
            "access_key",
            "accesskey",
            "access-token",
            "ak",
            "sk",
            "cookie",
            "session",
            "sessionid",
            "auth",
            "authorization",
            "credential",
            "credentials",
            "private_key",
        ),
        key=len,
        reverse=True,
    )
)
_SENSITIVE_KEY_ALT = "|".join(re.escape(key) for key in _SENSITIVE_KEYS)

# --- redaction patterns -----------------------------------------------------
_URL_USERINFO_RE = re.compile(r"(?i)(?<=://)[^/@\s<>]+@")
_URL_QUERY_SECRET_RE = re.compile(
    rf"(?i)([?&](?:{_SENSITIVE_KEY_ALT})=)([^&\s\"'#<>]+)"
)
_SECRET_TOKEN_RE = re.compile(
    r"(?i)(?:"
    r"\bsk-[a-z0-9_\-]{8,}"
    r"|\b(?:ghp|gho|ghu|ghs|ghr)_[a-z0-9]{16,}"
    r"|\bgithub_pat_[a-z0-9_]{16,}"
    r"|\bakia[0-9a-z]{12,}"
    r"|\bxox[abprs]-[a-z0-9\-]{8,}"
    r"|\beyj[a-z0-9_\-]{8,}(?:\.[a-z0-9_\-]{4,}){2}"
    r")"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[a-z0-9._\-+/=]{8,}")
_KEY_VALUE_SECRET_RE = re.compile(
    rf"(?i)\b({_SENSITIVE_KEY_ALT})\b(\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s,;&'\"}\]]+)"
)
_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)+")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")

# --- regularisation patterns ------------------------------------------------
_REG_URL_RE = re.compile(r"(?i)\b(?:https?)://\S+")
_REG_UUID_RE = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)
_REG_DATETIME_RE = re.compile(
    r"(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"(?:[T ]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?"
    r"(?!\d)"
)
_REG_TIME_RE = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)")
_REG_QUOTED_RE = re.compile(
    r"\"[^\"]{41,}\"|'[^']{41,}'|\u201c[^\u201d]{41,}\u201d|\u2018[^\u2019]{41,}\u2019"
)
_REG_NUM_RE = re.compile(r"(?<![\w.])\d{4,}(?![\w.])")
_REG_HEX_RE = re.compile(r"(?i)\b[0-9a-f]{8,}\b")
_REG_WS_RE = re.compile(r"\s+")

# --- tokenisation patterns --------------------------------------------------
_ASCII_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]{1,}")
_DIGIT_TOKEN_RE = re.compile(r"\d{1,}")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_ENUM_ARG_RE = re.compile(r"[A-Za-z0-9_./:\-]+")
_ACTION_NAME_RE = re.compile(r"[^a-z0-9_.:\-]")

# Function words carry no discriminative signal for retrieval. Negations are
# intentionally absent: "not"/"no" flip the meaning of a state.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "and",
        "but",
        "for",
        "with",
        "from",
        "that",
        "this",
        "these",
        "those",
        "then",
        "than",
        "there",
        "here",
        "when",
        "where",
        "which",
        "while",
        "what",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "are",
        "was",
        "were",
        "been",
        "being",
        "have",
        "has",
        "had",
        "does",
        "did",
        "doing",
        "will",
        "would",
        "should",
        "could",
        "can",
        "may",
        "might",
        "must",
        "you",
        "your",
        "yours",
        "they",
        "them",
        "their",
        "theirs",
        "she",
        "her",
        "hers",
        "him",
        "his",
        "its",
        "our",
        "ours",
        "please",
        "just",
        "about",
        "into",
        "over",
        "under",
        "again",
        "further",
        "once",
        "also",
        "very",
        "too",
        "some",
        "any",
        "each",
        "more",
        "most",
        "other",
        "such",
        "only",
        "own",
        "same",
    }
)

_ACTION_KINDS: frozenset[str] = frozenset({"tool", "reply", "command", "custom"})


def redact(text: str) -> str:
    """Scrub secrets and personal data from arbitrary text.

    The function is idempotent: redact(redact(x)) equals redact(x). Every
    replacement is one of the shared angle-bracket placeholders, which
    regularize() leaves untouched.

    Args:
        text: Raw text, typically a user message or an observation.

    Returns:
        The scrubbed text. None and non-string input is coerced to an empty
        string or to str(value) instead of raising.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""
    out = _URL_USERINFO_RE.sub("<secret>@", text)
    out = _URL_QUERY_SECRET_RE.sub(r"\1<secret>", out)
    out = _SECRET_TOKEN_RE.sub("<secret>", out)
    out = _BEARER_RE.sub("<secret>", out)
    out = _KEY_VALUE_SECRET_RE.sub(r"\1\2<secret>", out)
    out = _EMAIL_RE.sub("<email>", out)
    out = _PHONE_RE.sub("<phone>", out)
    return out


def regularize(text: str) -> str:
    """Replace volatile identifiers with stable placeholders.

    This is the text analogue of the paper's Regularized URL: instance
    specific ids, timestamps and long payloads become placeholder tokens so
    that functionally equivalent states collapse onto the same representation.

    Args:
        text: Text to normalise, usually already passed through redact().

    Returns:
        The normalised text with collapsed whitespace and no leading or
        trailing blanks.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""
    out = _REG_URL_RE.sub("<url>", text)
    out = _REG_UUID_RE.sub("<uuid>", out)
    out = _REG_DATETIME_RE.sub("<date>", out)
    out = _REG_TIME_RE.sub("<date>", out)
    out = _REG_QUOTED_RE.sub('"<str>"', out)
    # Long integers before hex: an 8+ digit number is noise either way, but
    # "<num>" is the more faithful placeholder for a purely numeric run.
    out = _REG_NUM_RE.sub("<num>", out)
    out = _REG_HEX_RE.sub("<hex>", out)
    return _REG_WS_RE.sub(" ", out).strip()


def tokenize(text: str) -> frozenset[str]:
    """Split text into the token set used by Jaccard retrieval.

    The tokeniser is mixed Chinese/English aware:

    * ASCII words of at least two characters, lower-cased;
    * standalone digit runs of at most six characters (longer runs are noise);
    * CJK runs cut into adjacent-character bigrams, with an isolated character
      kept as a unigram;
    * angle-bracket placeholders kept whole;
    * English stopwords dropped.

    Args:
        text: Text to tokenise; normalised text yields the best results.

    Returns:
        An immutable token set. Empty or punctuation-only input yields an
        empty set.
    """
    if text is None:
        return frozenset()
    if not isinstance(text, str):
        text = str(text)
    lowered = text.lower()
    if not lowered.strip():
        return frozenset()

    tokens: set[str] = set()
    placeholders = _PLACEHOLDER_RE.findall(lowered)
    if placeholders:
        tokens.update(placeholders)
        lowered = _PLACEHOLDER_RE.sub(" ", lowered)
    for word in _ASCII_TOKEN_RE.findall(lowered):
        if word not in _STOPWORDS:
            tokens.add(word)
    for digits in _DIGIT_TOKEN_RE.findall(lowered):
        if len(digits) <= _MAX_DIGIT_TOKEN_CHARS:
            tokens.add(digits)
    for run in _CJK_RUN_RE.findall(lowered):
        if len(run) == 1:
            tokens.add(run)
            continue
        tokens.update(run[offset : offset + 2] for offset in range(len(run) - 1))
    return frozenset(tokens)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Compute the Jaccard similarity of two token sets (Eq. 24).

    Args:
        a: First token set.
        b: Second token set.

    Returns:
        The size of the intersection divided by the size of the union, in
        [0, 1]. Two empty sets score 0.0 on purpose: an empty query state must
        not match everything in memory.
    """
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if not intersection:
        return 0.0
    union = len(a | b)
    if not union:
        return 0.0
    return max(0.0, min(1.0, intersection / union))


def abstract_state(
    *,
    task: str = "",
    observation: str = "",
    recent_actions: Sequence[str] = (),
    surface: str = "",
    limit: int = MAX_STATE_CHARS,
) -> StateView:
    """Compress one step's raw context into a retrieval-ready state.

    Args:
        task: User request or task instruction framing the step.
        observation: Raw observation produced by the environment.
        recent_actions: Local action history. Only the newest three entries
            participate, mirroring the paper's bounded context window.
        surface: Coarse partition key (platform, task family, ...) used by the
            optional retrieval gate.
        limit: Maximum length of the signature and text fields.

    Returns:
        A StateView. The result is deterministic: identical inputs always
        produce an identical view, without relying on hash ordering.
    """
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = MAX_STATE_CHARS

    task_source = "" if task is None else str(task)
    observation_source = "" if observation is None else str(observation)
    actions = [
        item if isinstance(item, str) else str(item) for item in (recent_actions or ())
    ]
    context = actions[-_MAX_ACTION_CONTEXT:]

    task_text = regularize(redact(task_source))
    observation_text = regularize(redact(observation_source))
    context_texts = [regularize(redact(item)) for item in context]

    tokens = tokenize(" ".join([task_text, observation_text, *context_texts]))
    signature = (
        f"surface={surface}|task={task_text}|obs={observation_text}"
        f"|ctx={' > '.join(context_texts)}"
    )[:limit]

    sections: list[str] = []
    if task_text:
        sections.append(f"task: {task_text[:_MAX_TASK_CHARS]}")
    if context_texts:
        rendered_context = " > ".join(
            item[:_MAX_ACTION_CHARS] for item in context_texts
        )
        sections.append(f"recent: {rendered_context}")
    if observation_text:
        sections.append(f"observed: {observation_text[:_MAX_OBSERVATION_CHARS]}")
    text = " | ".join(sections)[:limit]

    return StateView(
        signature=signature,
        tokens=tokens,
        text=text,
        surface=surface if surface else "",
        raw_chars=len(task_source) + len(observation_source),
    )


def normalize_action(
    *,
    kind: str,
    name: str,
    args: Mapping[str, Any] | None = None,
    text: str = "",
) -> ActionView:
    """Normalise an action so it can be aggregated across episodes.

    Instance-specific payloads are collapsed onto a signature: free-form
    strings become a string placeholder, numbers a numeric placeholder and so
    on, while short opaque enumerations such as "open", "zh" or "GET" are
    preserved so that action variants stay distinguishable.

    Args:
        kind: Raw action category; anything unknown becomes "custom".
        name: Raw tool or action name.
        args: Raw action arguments; deep-redacted before normalisation.
        text: Optional human-readable rendering supplied by the caller. When
            empty, a compact rendering is built from the arguments.

    Returns:
        The normalised ActionView.
    """
    normalized_kind: ActionKind = "custom"
    raw_kind = "" if kind is None else str(kind).strip().lower()
    if raw_kind in _ACTION_KINDS:
        normalized_kind = raw_kind  # type: ignore[assignment]

    raw_name = "" if name is None else str(name).strip().lower()
    normalized_name = _ACTION_NAME_RE.sub("_", raw_name) or normalized_kind

    clean_args: dict[str, Any] = {}
    if isinstance(args, Mapping):
        scrubbed = redact_payload(dict(args))
        if isinstance(scrubbed, dict):
            clean_args = scrubbed

    signature_parts: list[str] = []
    for key in sorted(clean_args):
        value = clean_args[key]
        if isinstance(value, bool):
            shown = "true" if value else "false"
        elif isinstance(value, str):
            candidate = value.strip()
            if (
                0 < len(candidate) <= _MAX_ENUM_ARG_CHARS
                and " " not in candidate
                and _ENUM_ARG_RE.fullmatch(candidate)
            ):
                shown = candidate
            else:
                shown = "<str>"
        elif isinstance(value, int | float):
            shown = "<num>"
        elif isinstance(value, list | tuple | set | frozenset):
            shown = "<list>"
        elif isinstance(value, Mapping):
            shown = "<obj>"
        elif value is None:
            shown = "<null>"
        else:
            shown = "<str>"
        signature_parts.append(f"{key}={shown}")
    action_key = f"{normalized_kind}:{normalized_name}({','.join(signature_parts)})"

    caller_text = "" if text is None else str(text).strip()
    if caller_text:
        action_text = redact(caller_text)
    elif clean_args:
        rendered: list[str] = []
        for key in sorted(clean_args):
            shown_value = str(clean_args[key]).replace("\n", " ")[:_MAX_ARG_TEXT_CHARS]
            rendered.append(f"{key}={shown_value}")
        action_text = f"{normalized_name}({', '.join(rendered)})"
    else:
        action_text = f"{normalized_name}()"

    return ActionView(
        kind=normalized_kind,
        name=normalized_name,
        key=action_key,
        text=action_text,
        args=clean_args,
    )


def describe_action(action: ActionView) -> str:
    """Render one action as a single prompt-friendly line.

    Args:
        action: Normalised action to render.

    Returns:
        The action kind in brackets followed by its text, capped at 160
        characters.
    """
    text = "" if action is None else str(action.text or "")
    kind = "" if action is None else str(action.kind or "")
    return f"[{kind}] {text[:_MAX_DESCRIBE_CHARS]}"


def redact_payload(value: Any) -> Any:
    """Recursively redact a JSON-like payload.

    Args:
        value: Arbitrary payload. String values are scrubbed with redact(),
            dict keys are scrubbed as well, and containers are rebuilt with
            the same type.

    Returns:
        A structurally identical payload with secrets removed. Nesting deeper
        than six levels is replaced by a "deep" placeholder so that a hostile
        payload cannot exhaust the stack.
    """

    def walk(item: Any, depth: int) -> Any:
        """Redact one node of the payload tree.

        Args:
            item: Current node.
            depth: Current nesting depth.

        Returns:
            The redacted node.
        """
        if depth > _MAX_PAYLOAD_DEPTH:
            return "<deep>"
        if isinstance(item, str):
            return redact(item)
        if isinstance(item, Mapping):
            return {
                redact(str(key)): walk(child, depth + 1) for key, child in item.items()
            }
        if isinstance(item, list):
            return [walk(child, depth + 1) for child in item]
        if isinstance(item, tuple):
            return tuple(walk(child, depth + 1) for child in item)
        if isinstance(item, set | frozenset):
            return type(item)(walk(child, depth + 1) for child in item)
        return item

    return walk(value, 0)
