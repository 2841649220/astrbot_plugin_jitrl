"""MiTA is exercised inside the actual Lens, replay and checkpoint lifecycle."""

from __future__ import annotations

import hashlib
import time
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from jitrl.neural.attention import GroupedSelfAttention, MiTAAttention  # noqa: E402
from jitrl.neural.config import LensConfig  # noqa: E402
from jitrl.neural.contracts import (  # noqa: E402
    Intervention,
    LensDecision,
    MemoryRecord,
    RetrievalSnapshot,
)
from jitrl.neural.encoding import canonicalize, frozen_embedding  # noqa: E402
from jitrl.neural.model import (  # noqa: E402
    GroupedMemoryAttention,
    LensTokenBlock,
    SwiGLUBlock,
    create_model,
    make_batch,
)
from jitrl.neural.runtime import (  # noqa: E402
    _LEGACY_ARCHITECTURE,
    IncompatibleArchitecture,
    LensRuntime,
)


@pytest.fixture(scope="module", autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def cfg():
    return LensConfig(
        mode="active",
        hidden_dim=24,
        embedding_dim=32,
        summary_dim=16,
        blocks=4,
        engram_buckets=31,
        engram_dim=4,
        bases=2,
        rank=2,
        min_samples=4,
        batch_size=4,
        train_steps=3,
        mita_landmarks=3,
        mita_topk=2,
        mita_query_chunk=3,
        min_support=1,
        relevance_threshold=0.0,
        max_uncertainty=2.0,
    )


def records(*, identity="current", events=4, memories=2):
    current = canonicalize(
        scope="private-a",
        goal="Locate and verify the report",
        observation="Report listing",
        events=tuple(f"event:{i}:search" for i in range(events)),
        allowed_actions=("search", "reply"),
        episode_id=f"episode-{identity}",
        decision_id=f"decision-{identity}",
        observed_at=100.0,
    )
    history = []
    for i in range(memories):
        old = replace(
            current,
            episode_id=f"old-episode-{i}",
            decision_id=f"old-decision-{i}",
            observed_at=10.0 + i,
        )
        history.append(
            MemoryRecord(
                f"experience-{i}",
                old,
                "tool:search",
                "search",
                "NO_OP",
                2.0,
                "environment",
                1.0,
                20.0 + i,
                0.8,
            )
        )
    return current, RetrievalSnapshot(
        f"snapshot-{identity}", current.scope, 100.0, tuple(history)
    )


def batch_for(cfg, *, events=4, memories=2, candidates=None):
    state, snapshot = records(events=events, memories=memories)
    return make_batch(
        [state], [snapshot], [candidates or ["tool:search", "reply"]], ["NO_OP"], cfg
    )


def test_default_backbone_executes_all_four_mita_layers_and_backpropagates(cfg):
    assert LensConfig().attention_backend == "mita"
    assert cfg.blocks == 4
    model = create_model(cfg)
    assert not hasattr(model, "attention")
    assert all(
        isinstance(block, LensTokenBlock) and isinstance(block.attention, MiTAAttention)
        for block in model.blocks
    )
    observed = []
    handles = []
    for index, block in enumerate(model.blocks):

        def record_call(module, args, output, index=index):
            observed.append((index, tuple(args[0].shape), tuple(output.shape)))

        handles.append(block.attention.register_forward_hook(record_call))
    batch = batch_for(cfg)
    output = model(batch)
    loss = (
        output["pre_value"].square().sum()
        + output["candidate_values"].square().sum()
        + output["relevance"].square().sum()
    )
    loss.backward()
    for handle in handles:
        handle.remove()
    tokens = 1 + batch["events"].shape[1] + batch["memory_mask"].shape[1]
    assert observed == [(i, (1, tokens, 24), (1, tokens, 24)) for i in range(4)]
    for block in model.blocks:
        for parameter in block.attention.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    assert model.attention_inventory() == {
        "backend": "mita",
        "mita_layers": 4,
        "gqa_layers": 0,
        "total_attention_layers": 4,
    }


@pytest.mark.parametrize(
    "backend,mi,full", [("mita", 4, 0), ("hybrid", 3, 1), ("gqa", 0, 4)]
)
def test_backend_inventory_counts_real_executed_classes(cfg, backend, mi, full):
    model = create_model(replace(cfg, attention_backend=backend))
    counts = {"mita": 0, "gqa": 0}
    handles = []
    for block in model.blocks:
        kind = "mita" if isinstance(block.attention, MiTAAttention) else "gqa"
        assert isinstance(block.attention, MiTAAttention | GroupedSelfAttention)

        def count_call(module, args, output, kind=kind):
            counts[kind] += 1

        handles.append(block.attention.register_forward_hook(count_call))
    with torch.no_grad():
        model(batch_for(cfg))
    for handle in handles:
        handle.remove()
    assert counts == {"mita": mi, "gqa": full}
    assert model.attention_inventory()["mita_layers"] == mi
    assert model.attention_inventory()["gqa_layers"] == full


def test_dense_and_sparse_backbones_share_identical_initial_parameters(cfg):
    sparse = create_model(cfg)
    for backend in ("hybrid", "gqa"):
        control = create_model(replace(cfg, attention_backend=backend))
        assert sparse.state_dict().keys() == control.state_dict().keys()
        for name, tensor in sparse.state_dict().items():
            assert torch.equal(tensor, control.state_dict()[name]), name


def test_legacy_backbone_retains_original_parameter_path_and_output_contract(cfg):
    cfg = replace(cfg, attention_backend="legacy")
    model = create_model(cfg)
    assert isinstance(model.attention, GroupedMemoryAttention)
    assert all(isinstance(block, SwiGLUBlock) for block in model.blocks)
    assert not hasattr(model, "event_encoder") and not hasattr(model, "token_types")
    tensors = model.state_dict()
    assert tensors["attention.query.weight"].shape == (24, 24)
    assert tensors["attention.key.weight"].shape == (8, 24)
    assert tensors["blocks.0.up.weight"].shape == (96, 24)
    assert tensors["blocks.3.down.weight"].shape == (24, 48)
    assert not any(name.startswith("blocks.0.attention.") for name in tensors)
    assert model.attention_inventory() == {
        "backend": "legacy",
        "mita_layers": 0,
        "gqa_layers": 1,
        "total_attention_layers": 1,
    }
    output = model(batch_for(cfg))
    assert set(output) == {
        "pre_value",
        "pre_uncertainty",
        "candidate_values",
        "uncertainty",
        "relevance",
        "alpha",
        "adapter_residual",
        "hidden",
    }
    assert output["candidate_values"].shape == (1, 2)


@pytest.mark.parametrize("backend", ["mita", "hybrid", "gqa", "legacy"])
def test_output_tensor_api_and_padding_are_compatible(cfg, backend):
    cfg = replace(cfg, attention_backend=backend)
    model = create_model(cfg)
    short, short_snapshot = records(events=2, memories=1)
    long, long_snapshot = records(identity="long", events=11, memories=9)
    alone = make_batch(
        [short], [short_snapshot], [["search", "reply"]], ["SELECT_EVIDENCE"], cfg
    )
    mixed = make_batch(
        [short, long],
        [short_snapshot, long_snapshot],
        [["search", "reply"], ["search"]],
        ["SELECT_EVIDENCE", "NO_OP"],
        cfg,
    )
    with torch.no_grad():
        before, after = model(alone), model(mixed)
    expected = {
        "pre_value": (2,),
        "pre_uncertainty": (2,),
        "candidate_values": (2, 2),
        "uncertainty": (2, 2),
        "relevance": (2, 9),
        "alpha": (2, 2),
        "adapter_residual": (2, 24),
        "hidden": (2, 24),
    }
    assert set(after) == set(expected)
    for name, shape in expected.items():
        assert after[name].shape == shape and torch.isfinite(after[name]).all()
    for name in (
        "pre_value",
        "pre_uncertainty",
        "candidate_values",
        "uncertainty",
        "alpha",
        "adapter_residual",
        "hidden",
    ):
        torch.testing.assert_close(
            before[name][0], after[name][0], atol=2e-6, rtol=2e-5
        )
    assert after["candidate_values"][1, 1] == 0


def test_event_features_remain_ordered_and_drive_the_backbone(cfg):
    state, snapshot = records(events=6)
    batch = make_batch([state], [snapshot], [["tool:search"]], ["NO_OP"], cfg)
    for i, event in enumerate(state.events):
        torch.testing.assert_close(
            batch["event_features"][0, i],
            torch.tensor(frozen_embedding(event, cfg.embedding_dim)),
        )
    changed = {name: tensor.clone() for name, tensor in batch.items()}
    changed["event_features"] = changed["event_features"].flip(1)
    model = create_model(cfg)
    with torch.no_grad():
        first, second = model(batch), model(changed)
    assert not torch.allclose(first["hidden"], second["hidden"], atol=1e-5)


def test_invalid_padding_cannot_inject_nan_into_backbone_or_parameter_gradients(cfg):
    short, short_snapshot = records(events=2, memories=1)
    long, long_snapshot = records(identity="long", events=9, memories=7)
    batch = make_batch(
        [short, long],
        [short_snapshot, long_snapshot],
        [["search"], ["search"]],
        ["NO_OP", "NO_OP"],
        cfg,
    )
    for name in ("memory_states", "memory_actions", "memory_meta"):
        batch[name] = batch[name].masked_fill(
            ~batch["memory_mask"].unsqueeze(-1), float("nan")
        )
    batch["event_features"] = batch["event_features"].masked_fill(
        ~batch["event_mask"].unsqueeze(-1), float("nan")
    )
    model = create_model(cfg)
    output = model(batch)
    assert all(torch.isfinite(tensor).all() for tensor in output.values())
    output["candidate_values"].square().sum().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def seed_causal_training(runtime):
    for i in range(4):
        current = canonicalize(
            scope="private-a",
            goal="Locate report",
            observation="Report listing",
            events=("search", "observed"),
            episode_id=f"training-{i}",
            decision_id=f"training-decision-{i}",
            observed_at=10 + i * 10,
        )
        snapshot = runtime.store.retrieve(current, runtime.cfg)
        trace = LensDecision(current, snapshot, Intervention())
        epoch = runtime.store.epoch(current.scope)
        assert runtime.store.record_decision(trace, expected_epoch=epoch)
        experience = runtime.store.append_execution(
            current.decision_id,
            index=0,
            action_key="tool:search",
            action_name="search",
            expected_epoch=epoch,
            executed_at=current.observed_at + 1,
        )
        assert runtime.store.settle(
            experience,
            reward=2.0,
            return_g=2.0,
            source="environment",
            expected_epoch=epoch,
            settled_at=current.observed_at + 2,
        )


def snapshots_in_store(store):
    with store._lock:
        return [
            tuple(row)
            for row in store._conn.execute(
                "SELECT decision_id,snapshot_json FROM decisions ORDER BY decision_id"
            )
        ]


@pytest.mark.asyncio
async def test_old_checkpoint_loads_only_legacy_and_mita_retrains_same_causal_data(
    tmp_path, cfg
):
    legacy = LensRuntime(tmp_path, replace(cfg, attention_backend="legacy"))
    try:
        seed_causal_training(legacy)
        snapshots = snapshots_in_store(legacy.store)
        assert any("experience" in snapshot for _, snapshot in snapshots)
        trained = await legacy.train("private-a")
        assert trained["status"] == "trained"
        record = legacy.store.latest_checkpoint("private-a")
        path = legacy._path(record["filename"])
        payload = torch.load(path, map_location="cpu", weights_only=True)
        # Emulate the exact v0.2.0 header, before attention backend fields existed.
        payload["architecture"] = {
            name: payload["architecture"][name] for name in _LEGACY_ARCHITECTURE
        }
        torch.save(payload, path)
        with legacy.store._lock, legacy.store._conn:
            legacy.store._conn.execute(
                "UPDATE checkpoints SET sha256=? WHERE checkpoint_id=?",
                (
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    record["checkpoint_id"],
                ),
            )
        loaded, identifier = legacy._load_model("private-a")
        assert identifier == record["checkpoint_id"]
        assert loaded.attention_inventory()["backend"] == "legacy"
        assert all(
            torch.equal(tensor, loaded.state_dict()[name])
            for name, tensor in payload["state_dict"].items()
        )
    finally:
        await legacy.close()
    modern = LensRuntime(tmp_path, cfg)
    try:
        with pytest.raises(IncompatibleArchitecture):
            modern._load_model("private-a")
        assert modern.store.stats("private-a")["executions"] == 4
        result = await modern.train("private-a")
        assert (
            result["status"] == "trained"
            and result["initialization"] == "fresh_architecture"
        )
        assert result["attention"]["mita_layers"] == 4
        assert snapshots_in_store(modern.store) == snapshots
        assert len(modern.store.replay("private-a")) == 4
        model, new_identifier = modern._load_model("private-a")
        assert (
            new_identifier != identifier
            and model.attention_inventory()["backend"] == "mita"
        )
        future, epoch = await modern.prepare(
            canonicalize(
                scope="private-a",
                goal="Locate report",
                observation="Report listing",
                observed_at=time.time() + 1,
            )
        )
        assert future.checkpoint_id == new_identifier
        assert await modern.commit(future, epoch)
    finally:
        await modern.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "attention_backend",
        "mita_landmarks",
        "mita_topk",
        "mita_query_chunk",
        "memory_tokens",
    ],
)
async def test_cached_checkpoint_identity_includes_backend_and_sparse_budgets(
    tmp_path, cfg, change
):
    runtime = LensRuntime(tmp_path, cfg)
    try:
        seed_causal_training(runtime)
        assert (await runtime.train("private-a"))["status"] == "trained"
        first, identifier = runtime._load_model("private-a")
        assert runtime._models
        value = "gqa" if change == "attention_backend" else getattr(cfg, change) - 1
        runtime.cfg = replace(cfg, **{change: value})
        with pytest.raises(IncompatibleArchitecture):
            runtime._load_model("private-a", identifier)
        # Changing an operator gate is not an architecture change.
        runtime.cfg = replace(cfg, relevance_threshold=0.05)
        cached, again = runtime._load_model("private-a", identifier)
        assert cached is first and again == identifier
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_auto_device_falls_back_to_real_cpu_training_when_cuda_unavailable(
    tmp_path, cfg, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = LensRuntime(tmp_path, replace(cfg, device="auto"))
    try:
        assert runtime._device().type == "cpu"
        seed_causal_training(runtime)
        trained = await runtime.train("private-a")
        assert trained["status"] == "trained" and trained["device"] == "cpu"
        model, _ = runtime._load_model("private-a")
        assert next(model.parameters()).device.type == "cpu"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_explicit_unavailable_cuda_is_reported_without_publishing_weights(
    tmp_path, cfg, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    runtime = LensRuntime(tmp_path, replace(cfg, device="cuda"))
    try:
        with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
            runtime._device()
        seed_causal_training(runtime)
        trained = await runtime.train("private-a")
        assert trained["status"] == "failed" and trained["reason"] == "RuntimeError"
        assert runtime.store.latest_checkpoint("private-a") is None
        assert not list(tmp_path.rglob("*.pt"))
        assert len(runtime.store.replay("private-a")) == 4
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cuda_training_checkpoint_is_portable_to_cpu_and_preserves_host_rng(
    tmp_path, cfg
):
    if not torch.cuda.is_available():
        pytest.skip("optional CUDA hardware is unavailable")
    # Initialize the device before recording its host-owned RNG state. Model
    # initialization and minibatch choices must not consume this generator.
    before_cuda = torch.cuda.get_rng_state().clone()
    before_cpu = torch.get_rng_state().clone()
    runtime = LensRuntime(tmp_path, replace(cfg, device="cuda"))
    try:
        seed_causal_training(runtime)
        snapshots = snapshots_in_store(runtime.store)
        trained = await runtime.train("private-a")
        assert trained["status"] == "trained" and trained["device"].startswith("cuda")
        assert trained["attention"]["mita_layers"] == 4
        gpu_model, identifier = runtime._load_model("private-a")
        assert next(gpu_model.parameters()).device.type == "cuda"
        current, _ = records(identity="cuda-evaluation")
        current = replace(current, observed_at=time.time() + 1)
        snapshot = runtime.store.retrieve(current, runtime.cfg)
        gpu_output = runtime._forward(
            gpu_model, current, snapshot, ["tool:search", "reply"]
        )
        assert all(tensor.device.type == "cuda" for tensor in gpu_output.values())
        weights = {
            name: tensor.detach().cpu().clone()
            for name, tensor in gpu_model.state_dict().items()
        }
        record = runtime.store.latest_checkpoint("private-a")
        payload = torch.load(
            runtime._path(record["filename"]), map_location="cpu", weights_only=True
        )
        assert "device" not in payload["architecture"]
        # Switching device changes cache identity but is not an incompatible
        # architecture, and it must not silently retrain or change model ID.
        runtime.cfg = replace(cfg, device="cpu")
        cpu_model, again = runtime._load_model("private-a", identifier)
        assert again == identifier and cpu_model is not gpu_model
        assert next(cpu_model.parameters()).device.type == "cpu"
        for name, tensor in cpu_model.state_dict().items():
            assert torch.equal(weights[name], tensor), name
        cpu_output = runtime._forward(
            cpu_model, current, snapshot, ["tool:search", "reply"]
        )
        for name, tensor in gpu_output.items():
            torch.testing.assert_close(
                cpu_output[name], tensor.cpu(), atol=2e-5, rtol=2e-4
            )
        assert snapshots_in_store(runtime.store) == snapshots
        assert torch.equal(before_cpu, torch.get_rng_state())
        assert torch.equal(before_cuda, torch.cuda.get_rng_state())
    finally:
        await runtime.close()
