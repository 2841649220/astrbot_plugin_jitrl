"""Typed view over the plugin configuration.

AstrBot hands the plugin a plain `dict` produced from `_conf_schema.json`.
Reading it ad hoc spreads string keys and `dict.get` chains across the whole
codebase, so the raw dict is normalised once into `JitRLConfig` and passed
around instead. Missing or malformed values fall back to the paper's
hyperparameters rather than raising: a bad config must never break a chat.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Paper hyperparameters (Appendix H) used whenever a value is absent/invalid.
_DEFAULT_GAMMA = 0.5
_DEFAULT_K = 10
_DEFAULT_SIMILARITY_THRESHOLD = 0.8
_DEFAULT_EXPLORATION_RATE = 0.3
_DEFAULT_UCB_ALPHA = 5.0
_DEFAULT_CANDIDATE_LIMIT = 3


def _as_bool(raw: Any, default: bool) -> bool:
    """Coerce a configuration value to `bool`.

    Args:
        raw: Raw value read from the plugin configuration.
        default: Value returned when `raw` is missing or unusable.

    Returns:
        The coerced boolean.
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on", "y"}
    if isinstance(raw, int | float):
        return bool(raw)
    return default


def _as_int(
    raw: Any, default: int, *, low: int | None = None, high: int | None = None
) -> int:
    """Coerce a configuration value to `int` and clamp it.

    Args:
        raw: Raw value read from the plugin configuration.
        default: Value returned when `raw` is missing or unusable.
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        The clamped integer.
    """
    try:
        value = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _as_float(
    raw: Any, default: float, *, low: float | None = None, high: float | None = None
) -> float:
    """Coerce a configuration value to `float` and clamp it.

    Args:
        raw: Raw value read from the plugin configuration.
        default: Value returned when `raw` is missing or unusable.
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        The clamped float.
    """
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        value = default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _as_str(raw: Any, default: str, *, allowed: tuple[str, ...] | None = None) -> str:
    """Coerce a configuration value to `str` constrained to a choice set.

    Args:
        raw: Raw value read from the plugin configuration.
        default: Value returned when `raw` is missing or not allowed.
        allowed: Permitted values; `None` accepts any string.

    Returns:
        The validated string.
    """
    if not isinstance(raw, str) or not raw:
        return default
    value = raw.strip().lower()
    if allowed is not None and value not in allowed:
        return default
    return value


def _as_str_list(raw: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    """Coerce a configuration value to a tuple of non-empty strings.

    Args:
        raw: Raw value read from the plugin configuration.
        default: Value returned when `raw` is missing or unusable.

    Returns:
        A tuple of strings with blanks removed.
    """
    if not isinstance(raw, list | tuple):
        return default
    items = [str(item).strip() for item in raw if str(item).strip()]
    return tuple(items) if items else default


@dataclass(slots=True)
class LearningConfig:
    """Experience-capture settings.

    Attributes:
        enabled: Whether completed runs are turned into experiences.
        evaluator_mode: `auto`, `llm`, `heuristic` or `off`.
        evaluator_provider_id: Provider used by the LLM evaluator; empty means
            "reuse the session provider".
        evaluator_timeout: Hard timeout for a single evaluator call.
        gamma: Discount factor `gamma` of Eq. 3.
        heuristic_weight: Weight of rule-based signals when fused with LLM
            scores, in `[0, 1]`.
        min_steps: Skip episodes shorter than this.
        async_learn: Run evaluation in a background task.
        success_detection: How task success is inferred.
    """

    enabled: bool = True
    evaluator_mode: str = "auto"
    evaluator_provider_id: str = ""
    evaluator_timeout: float = 30.0
    gamma: float = _DEFAULT_GAMMA
    heuristic_weight: float = 0.5
    min_steps: int = 1
    async_learn: bool = True
    success_detection: str = "heuristic"


@dataclass(slots=True)
class RetrievalConfig:
    """Memory retrieval and neighbourhood settings.

    Attributes:
        k: Neighbourhood size `k`.
        similarity_threshold: Minimum Jaccard similarity `r`.
        surface_gate: Restrict retrieval to the same `surface`.
        scope_isolation: `session`, `global` or `hybrid`.
        prefilter_limit: Upper bound on rows scored per query.
        recency_half_life_days: Half-life of the recency weight; `0` disables.
        min_count: Minimum supporting neighbours for an action to be scored.
    """

    k: int = _DEFAULT_K
    similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
    surface_gate: bool = False
    scope_isolation: str = "hybrid"
    prefilter_limit: int = 4000
    recency_half_life_days: float = 30.0
    min_count: int = 1


@dataclass(slots=True)
class PolicyConfig:
    """Closed-form policy-update settings.

    Attributes:
        beta: KL temperature `beta` of Eq. 9/10.
        exploration_rate: Probability `lambda` of the optimism bonus.
        ucb_alpha: Exploration bonus scale `alpha`.
        normalize_advantage: Apply Eq. 25 before the update.
        sample_temperature: Temperature applied to the post-update softmax.
        candidate_limit: Maximum size of the candidate set `C`.
        tool_gate: Mask tools with confidently negative advantage.
        tool_gate_min_count: Minimum samples before a tool may be masked.
        tool_gate_threshold: Advantage below which a tool is masked.
        max_hint_chars: Cap on injected prompt text.
    """

    beta: float = 1.5
    exploration_rate: float = _DEFAULT_EXPLORATION_RATE
    ucb_alpha: float = _DEFAULT_UCB_ALPHA
    normalize_advantage: bool = True
    sample_temperature: float = 1.0
    candidate_limit: int = _DEFAULT_CANDIDATE_LIMIT
    tool_gate: bool = False
    tool_gate_min_count: int = 3
    tool_gate_threshold: float = -0.3
    max_hint_chars: int = 900


@dataclass(slots=True)
class MemoryConfig:
    """Persistence settings for the experience bank.

    Attributes:
        max_entries: Hard cap on stored experiences.
        redact_secrets: Scrub credentials and personal data before storage.
        prune_strategy: `low_value`, `oldest` or `none`.
        keep_episodes: Number of episode headers retained.
        checkpoint_interval: Episodes between automatic checkpoints; 0 disables.
        keep_checkpoints: Automatic/safety snapshots retained per history.
    """

    max_entries: int = 20000
    redact_secrets: bool = True
    prune_strategy: str = "low_value"
    keep_episodes: int = 500
    checkpoint_interval: int = 50
    keep_checkpoints: int = 20


@dataclass(slots=True)
class FeedbackConfig:
    """Natural-language user-feedback settings.

    Attributes:
        enabled: Whether feedback keywords are recognised.
        positive_keywords: Terms signalling approval.
        negative_keywords: Terms signalling disapproval.
        window_seconds: How long a feedback reply stays associated with the
            previous episode.
    """

    enabled: bool = True
    positive_keywords: tuple[str, ...] = (
        "对",
        "正确",
        "没错",
        "谢谢",
        "好的",
        "太棒了",
        "有用",
        "helpful",
        "thanks",
        "correct",
        "great",
    )
    negative_keywords: tuple[str, ...] = (
        "不对",
        "错了",
        "不是",
        "答非所问",
        "没用",
        "垃圾",
        "wrong",
        "incorrect",
        "useless",
        "no",
    )
    window_seconds: int = 180


@dataclass(slots=True)
class CommandsConfig:
    """Surface-area settings for commands and agent tools.

    Attributes:
        enabled: Whether the `/jitrl` command group is registered.
        admin_only: Restrict commands to AstrBot admins.
        enable_agent_tools: Expose `jitrl_recall` to the agent.
    """

    enabled: bool = True
    admin_only: bool = True
    enable_agent_tools: bool = True


@dataclass(slots=True)
class DebugConfig:
    """Diagnostics settings.

    Attributes:
        log_decisions: Log every policy decision at INFO level.
    """

    log_decisions: bool = False


@dataclass(slots=True)
class JitRLConfig:
    """Normalised plugin configuration.

    Attributes:
        enabled: Master switch.
        apply_mode: `off`, `hint` or `logit`.
        learning: Experience-capture settings.
        retrieval: Retrieval settings.
        policy: Policy-update settings.
        memory: Persistence settings.
        feedback: Feedback settings.
        commands: Command settings.
        debug: Diagnostics settings.
    """

    enabled: bool = True
    apply_mode: str = "hint"
    learning: LearningConfig = field(default_factory=LearningConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    commands: CommandsConfig = field(default_factory=CommandsConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)

    @classmethod
    def from_raw(cls, raw: dict[str, Any] | None) -> JitRLConfig:
        """Build a normalised configuration from the plugin's raw dict.

        Args:
            raw: Dictionary produced by AstrBot from `_conf_schema.json`.

        Returns:
            A fully populated, range-clamped configuration. Never raises.
        """
        raw = raw if isinstance(raw, dict) else {}
        learning = raw.get("learning") if isinstance(raw.get("learning"), dict) else {}
        retrieval = (
            raw.get("retrieval") if isinstance(raw.get("retrieval"), dict) else {}
        )
        policy = raw.get("policy") if isinstance(raw.get("policy"), dict) else {}
        memory = raw.get("memory") if isinstance(raw.get("memory"), dict) else {}
        feedback = raw.get("feedback") if isinstance(raw.get("feedback"), dict) else {}
        commands = raw.get("commands") if isinstance(raw.get("commands"), dict) else {}
        debug = raw.get("debug") if isinstance(raw.get("debug"), dict) else {}

        return cls(
            enabled=_as_bool(raw.get("enabled"), True),
            apply_mode=_as_str(
                raw.get("apply_mode"), "hint", allowed=("off", "hint", "logit")
            ),
            learning=LearningConfig(
                enabled=_as_bool(learning.get("enabled"), True),
                evaluator_mode=_as_str(
                    learning.get("evaluator_mode"),
                    "auto",
                    allowed=("auto", "llm", "heuristic", "off"),
                ),
                evaluator_provider_id=_as_str(
                    learning.get("evaluator_provider_id"), ""
                ),
                evaluator_timeout=_as_float(
                    learning.get("evaluator_timeout"), 30.0, low=3.0, high=180.0
                ),
                gamma=_as_float(
                    learning.get("gamma"), _DEFAULT_GAMMA, low=0.01, high=0.99
                ),
                heuristic_weight=_as_float(
                    learning.get("heuristic_weight"), 0.5, low=0.0, high=1.0
                ),
                min_steps=_as_int(learning.get("min_steps"), 1, low=1, high=50),
                async_learn=_as_bool(learning.get("async_learn"), True),
                success_detection=_as_str(
                    learning.get("success_detection"),
                    "heuristic",
                    allowed=("off", "heuristic"),
                ),
            ),
            retrieval=RetrievalConfig(
                k=_as_int(retrieval.get("k"), _DEFAULT_K, low=1, high=64),
                similarity_threshold=_as_float(
                    retrieval.get("similarity_threshold"),
                    _DEFAULT_SIMILARITY_THRESHOLD,
                    low=0.0,
                    high=1.0,
                ),
                surface_gate=_as_bool(retrieval.get("surface_gate"), False),
                scope_isolation=_as_str(
                    retrieval.get("scope_isolation"),
                    "hybrid",
                    allowed=("session", "global", "hybrid"),
                ),
                prefilter_limit=_as_int(
                    retrieval.get("prefilter_limit"), 4000, low=50, high=200000
                ),
                recency_half_life_days=_as_float(
                    retrieval.get("recency_half_life_days"), 30.0, low=0.0, high=3650.0
                ),
                min_count=_as_int(retrieval.get("min_count"), 1, low=1, high=20),
            ),
            policy=PolicyConfig(
                beta=_as_float(policy.get("beta"), 1.5, low=0.0, high=20.0),
                exploration_rate=_as_float(
                    policy.get("exploration_rate"),
                    _DEFAULT_EXPLORATION_RATE,
                    low=0.0,
                    high=1.0,
                ),
                ucb_alpha=_as_float(
                    policy.get("ucb_alpha"), _DEFAULT_UCB_ALPHA, low=0.0, high=50.0
                ),
                normalize_advantage=_as_bool(policy.get("normalize_advantage"), True),
                sample_temperature=_as_float(
                    policy.get("sample_temperature"), 1.0, low=0.05, high=5.0
                ),
                candidate_limit=_as_int(
                    policy.get("candidate_limit"),
                    _DEFAULT_CANDIDATE_LIMIT,
                    low=2,
                    high=16,
                ),
                tool_gate=_as_bool(policy.get("tool_gate"), False),
                tool_gate_min_count=_as_int(
                    policy.get("tool_gate_min_count"), 3, low=1, high=100
                ),
                tool_gate_threshold=_as_float(
                    policy.get("tool_gate_threshold"), -0.3, low=-1.0, high=0.0
                ),
                max_hint_chars=_as_int(
                    policy.get("max_hint_chars"), 900, low=0, high=8000
                ),
            ),
            memory=MemoryConfig(
                max_entries=_as_int(
                    memory.get("max_entries"), 20000, low=100, high=1000000
                ),
                redact_secrets=_as_bool(memory.get("redact_secrets"), True),
                prune_strategy=_as_str(
                    memory.get("prune_strategy"),
                    "low_value",
                    allowed=("low_value", "oldest", "none"),
                ),
                keep_episodes=_as_int(
                    memory.get("keep_episodes"), 500, low=10, high=100000
                ),
                checkpoint_interval=_as_int(
                    memory.get("checkpoint_interval"), 50, low=0, high=10000
                ),
                keep_checkpoints=_as_int(
                    memory.get("keep_checkpoints"), 20, low=1, high=200
                ),
            ),
            feedback=FeedbackConfig(
                enabled=_as_bool(feedback.get("enabled"), True),
                positive_keywords=_as_str_list(
                    feedback.get("positive_keywords"),
                    FeedbackConfig().positive_keywords,
                ),
                negative_keywords=_as_str_list(
                    feedback.get("negative_keywords"),
                    FeedbackConfig().negative_keywords,
                ),
                window_seconds=_as_int(
                    feedback.get("window_seconds"), 180, low=10, high=86400
                ),
            ),
            commands=CommandsConfig(
                enabled=_as_bool(commands.get("enabled"), True),
                admin_only=_as_bool(commands.get("admin_only"), True),
                enable_agent_tools=_as_bool(commands.get("enable_agent_tools"), True),
            ),
            debug=DebugConfig(
                log_decisions=_as_bool(debug.get("log_decisions"), False),
            ),
        )
