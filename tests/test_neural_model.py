"""Neural path tests use actual forward/backward computation on CPU."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from jitrl.neural.config import LensConfig  # noqa: E402
from jitrl.neural.contracts import (  # noqa: E402
    Intervention,
    LensDecision,
    MemoryRecord,
    ReplaySample,
    RetrievalSnapshot,
)
from jitrl.neural.encoding import canonicalize, engram_addresses  # noqa: E402
from jitrl.neural.learner import train_model  # noqa: E402
from jitrl.neural.model import (  # noqa: E402
    EngramLookup,
    GroupedMemoryAttention,
    create_model,
    make_batch,
)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_pool():
    """Avoid excessive thread startup cost for tiny test matrix products."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg():
    return LensConfig(
        hidden_dim=24,
        embedding_dim=32,
        summary_dim=16,
        blocks=2,
        engram_buckets=31,
        engram_dim=4,
        bases=2,
        rank=2,
        min_samples=4,
        train_steps=25,
        learning_rate=0.003,
        batch_size=8,
    )


def records():
    old = canonicalize(
        scope="scope-a",
        goal="find the report",
        events=("tool:search", "result:ok"),
        episode_id="old",
        decision_id="old-decision",
        observed_at=10,
    )
    state = canonicalize(
        scope="scope-a",
        goal="find the report",
        events=("tool:search", "result:ok"),
        episode_id="current",
        decision_id="current-decision",
        observed_at=30,
    )
    memory = MemoryRecord(
        "experience-old",
        old,
        "tool:search",
        "search",
        "NO_OP",
        3,
        "environment",
        1,
        20,
        0.9,
    )
    snapshot = RetrievalSnapshot("snapshot", state.scope, 30, (memory,))
    return state, snapshot


def test_deterministic_initialization_preserves_host_rng(cfg):
    torch.manual_seed(123)
    original = torch.get_rng_state().clone()
    first = create_model(cfg)
    assert torch.equal(original, torch.get_rng_state())
    second = create_model(cfg)
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())
    )
    counts = first.parameter_counts()
    assert (
        counts["engram_tables"]
        == (cfg.engram_buckets + 1) * cfg.engram_dim * cfg.engram_heads * 2
    )
    assert counts["total"] == counts["active_neural"] + counts["engram_tables"]
    assert counts["frozen_encoder"] == 0


def test_initialization_does_not_reseed_host_cuda_generator(cfg, monkeypatch):
    def forbidden_gpu_seed(*args, **kwargs):
        raise AssertionError("local CPU controller reseeded the host GPU")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", forbidden_gpu_seed)
    assert create_model(cfg) is not None


def test_candidate_conditioning_and_all_major_modules_receive_gradient(cfg):
    state, snapshot = records()
    batch = make_batch(
        [state], [snapshot], [["tool:search", "tool:delete"]], ["NO_OP"], cfg
    )
    model = create_model(cfg).train()
    output = model(batch)
    assert output["candidate_values"].shape == (1, 2)
    assert output["uncertainty"].min() > 0
    assert not torch.allclose(
        output["candidate_values"][:, 0], output["candidate_values"][:, 1]
    )
    loss = (
        output["candidate_values"].sum()
        + output["pre_value"].sum()
        + output["relevance"].sum()
    )
    loss.backward()
    for component in (
        model.state_encoder,
        model.engram,
        model.memory_encoder,
        model.compiler,
        model.adapter,
        model.attention,
        model.blocks,
        model.candidate_encoder,
        model.candidate_head,
        model.pre_head,
    ):
        assert any(
            p.grad is not None
            and torch.isfinite(p.grad).all()
            and p.grad.abs().sum() > 0
            for p in component.parameters()
        )


def test_empty_memory_is_finite_and_fast_modulation_is_exact_zero(cfg):
    state, snapshot = records()
    empty = replace(snapshot, memories=())
    batch = make_batch([state], [empty], [[]], ["NO_OP"], cfg)
    output = create_model(cfg)(batch)
    assert batch["memory_mask"].shape == (1, 1)
    assert not batch["memory_mask"].any()
    assert not batch["candidate_mask"].any()
    assert torch.equal(output["alpha"], torch.zeros_like(output["alpha"]))
    assert torch.equal(
        output["adapter_residual"], torch.zeros_like(output["adapter_residual"])
    )
    assert all(torch.isfinite(t).all() for t in output.values())


def test_padding_cannot_change_memory_attention_or_compiler(cfg):
    state, snapshot = records()
    batch = make_batch([state], [snapshot], [["tool:search"]], ["NO_OP"], cfg)
    padded = {key: value.clone() for key, value in batch.items()}
    for name in ("memory_states", "memory_actions", "memory_meta"):
        value = padded[name]
        padded[name] = torch.cat((value, torch.ones_like(value) * 999), dim=1)
    padded["memory_mask"] = torch.tensor([[True, False]])
    model = create_model(cfg)
    with torch.no_grad():
        before, after = model(batch), model(padded)
    assert torch.allclose(before["hidden"], after["hidden"], atol=1e-6)
    assert torch.allclose(
        before["candidate_values"], after["candidate_values"], atol=1e-6
    )


def test_grouped_kv_broadcast_matches_explicit_reference(cfg):
    layer = GroupedMemoryAttention(cfg)
    state = torch.randn(2, cfg.hidden_dim)
    memory = torch.randn(2, 3, cfg.hidden_dim)
    valid = torch.tensor([[True, False, True], [True, True, True]])
    query = layer.query(state).reshape(2, cfg.query_heads, 1, layer.head_dim)
    key = layer.key(memory).reshape(2, 3, cfg.kv_heads, layer.head_dim).transpose(1, 2)
    value = (
        layer.value(memory).reshape(2, 3, cfg.kv_heads, layer.head_dim).transpose(1, 2)
    )
    key = key.repeat_interleave(3, dim=1)
    value = value.repeat_interleave(3, dim=1)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=valid[:, None, None, :]
    )
    expected = layer.output(expected.reshape(2, cfg.hidden_dim))
    assert torch.allclose(layer(state, memory, valid), expected, atol=1e-6)
    assert torch.equal(
        layer(state, memory, torch.zeros_like(valid)), torch.zeros_like(state)
    )


def test_event_lookup_convolution_is_causal_and_ordered(cfg):
    layer = EngramLookup(cfg)
    state = torch.randn(1, cfg.hidden_dim)
    events = ("tool:search", "outcome:error", "tool:retry", "outcome:ok", "reply")
    addresses = torch.tensor(
        [engram_addresses(events, buckets=cfg.engram_buckets, heads=cfg.engram_heads)]
    )
    changed = addresses.clone()
    changed[:, 3:] = (changed[:, 3:] % cfg.engram_buckets) + 1
    mask = torch.ones(1, len(events), dtype=torch.bool)
    before = layer(state, addresses, mask)
    after = layer(state, changed, mask)
    assert torch.equal(before[:, :3], after[:, :3])
    assert not torch.equal(before[:, 3:], after[:, 3:])
    assert engram_addresses(events, buckets=31) != engram_addresses(
        tuple(reversed(events)), buckets=31
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "scope",
        "environment",
        "version",
        "future",
        "same_episode",
        "self",
        "duplicate",
        "snapshot_future",
        "before_decision",
    ],
)
def test_replay_rejects_causal_or_scope_violations(cfg, mutation):
    state, snapshot = records()
    memory = snapshot.memories[0]
    if mutation == "scope":
        memory = replace(memory, state=replace(memory.state, scope="other"))
    elif mutation == "environment":
        memory = replace(memory, state=replace(memory.state, environment="new"))
    elif mutation == "version":
        memory = replace(memory, state=replace(memory.state, versions=("old",)))
    elif mutation == "future":
        memory = replace(memory, settled_at=30)
    elif mutation == "before_decision":
        memory = replace(memory, settled_at=9)
    elif mutation == "same_episode":
        memory = replace(
            memory, state=replace(memory.state, episode_id=state.episode_id)
        )
    elif mutation == "self":
        memory = replace(
            memory, state=replace(memory.state, decision_id=state.decision_id)
        )
    if mutation == "duplicate":
        snapshot = replace(snapshot, memories=(memory, memory))
    elif mutation == "snapshot_future":
        snapshot = replace(snapshot, cutoff=31)
    else:
        snapshot = replace(snapshot, memories=(memory,))
    with pytest.raises(ValueError):
        make_batch([state], [snapshot], [["tool:search"]], ["NO_OP"], cfg)


def test_coarse_memory_order_is_preserved_without_silent_truncation(cfg):
    state, snapshot = records()
    memories = tuple(
        replace(snapshot.memories[0], experience_id=f"old-{i}", return_g=float(i))
        for i in range(cfg.max_memory + 1)
    )
    snapshot = replace(snapshot, memories=memories)
    batch = make_batch([state], [snapshot], [["tool:search"]], ["NO_OP"], cfg)
    assert batch["memory_mask"].sum() == cfg.max_memory + 1
    assert batch["memory_meta"][0, :3, 0].tolist() == pytest.approx([0, 1 / 6, 2 / 6])


def samples(count=8):
    state, snapshot = records()
    result = []
    for i in range(count):
        current = replace(
            state,
            episode_id=f"episode-{i}",
            decision_id=f"decision-{i}",
            observed_at=30 + i,
        )
        snap = replace(snapshot, cutoff=current.observed_at)
        decision = LensDecision(current, snap, Intervention(), "", "untrained")
        result.append(
            ReplaySample(f"observed-{i}", decision, "tool:search", 3.0, 1.0, 50 + i)
        )
    return result


def test_real_optimizer_reduces_loss_and_does_not_mutate_serving_or_host_rng(cfg):
    model = create_model(cfg)
    serving = {key: value.clone() for key, value in model.state_dict().items()}
    random_state = torch.get_rng_state().clone()
    updated, metrics = train_model(model, cfg, samples())
    assert metrics["after_loss"] < metrics["before_loss"] * 0.6
    assert metrics["samples"] == 8 and metrics["steps"] == cfg.train_steps
    assert "uncalibrated" in metrics["uncertainty"]
    assert torch.equal(random_state, torch.get_rng_state())
    assert all(
        torch.equal(serving[key], value) for key, value in model.state_dict().items()
    )
    assert any(
        not torch.equal(serving[key], value)
        for key, value in updated.state_dict().items()
    )
    assert not updated.training


@pytest.mark.parametrize("mutation", ["few", "foreign", "pending", "nan", "duplicate"])
def test_learner_rejects_invalid_labels_and_scope(cfg, mutation):
    replay = samples()
    if mutation == "few":
        replay = replay[:1]
    elif mutation == "foreign":
        decision = replace(
            replay[-1].decision,
            state=replace(replay[-1].decision.state, scope="foreign"),
        )
        replay[-1] = replace(replay[-1], decision=decision)
    elif mutation == "pending":
        replay[-1] = replace(replay[-1], settled_at=0)
    elif mutation == "nan":
        replay[-1] = replace(replay[-1], return_g=float("nan"))
    elif mutation == "duplicate":
        replay[-1] = replay[0]
    with pytest.raises(ValueError):
        train_model(create_model(cfg), cfg, replay)
