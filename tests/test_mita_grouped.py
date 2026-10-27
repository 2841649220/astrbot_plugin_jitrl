"""Equation and gradient checks for grouped expert execution, not a new model."""

from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from test_mita_attention import fixture_qkv, scalar_reference  # noqa: E402
from test_mita_backbone import (  # noqa: E402
    records,
    seed_causal_training,
    snapshots_in_store,
)

from jitrl.neural.attention import MiTAAttention, mita_attention  # noqa: E402
from jitrl.neural.config import LensConfig  # noqa: E402
from jitrl.neural.mita_grouped import grouped_mita_attention  # noqa: E402
from jitrl.neural.model import create_model, make_batch  # noqa: E402
from jitrl.neural.runtime import LensRuntime  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("landmarks,topk,chunk", [(1, 1, 1), (3, 2, 4), (8, 20, 2)])
def test_grouped_matches_eager_and_independent_joint_softmax_reference(
    landmarks, topk, chunk
):
    q, k, v = fixture_qkv()
    valid = torch.tensor([[True, False, True, True, False, True, True], [False] * 7])
    output, diagnostics = grouped_mita_attention(q, k, v, valid, landmarks, topk, chunk)
    eager, original = mita_attention(q, k, v, valid, landmarks, topk, chunk)
    reference = scalar_reference(q, k, v, valid, landmarks, topk)
    torch.testing.assert_close(output, eager, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(output, reference, atol=1e-12, rtol=1e-12)
    for name in (
        "routes",
        "expert_indices",
        "expert_valid",
        "landmark_valid",
        "landmark_queries",
        "landmark_values",
    ):
        torch.testing.assert_close(diagnostics[name], original[name])
    assert diagnostics["group_counts"].sum() == valid.sum() * q.shape[1]
    assert diagnostics["host_size_syncs"] == 1


@pytest.mark.parametrize("chunk", [1, 2, 4, 64])
def test_group_packing_and_chunk_sizes_preserve_all_qkv_gradients(chunk):
    q, k, v = (tensor.requires_grad_() for tensor in fixture_qkv(batch=1, length=6))
    valid = torch.tensor([[True, True, False, True, True, True]])
    eager, _ = mita_attention(q, k, v, valid, 3, 2, 3)
    grouped, _ = grouped_mita_attention(q, k, v, valid, 3, 2, chunk)
    weights = torch.arange(eager.numel(), dtype=eager.dtype).reshape_as(eager) / 10
    expected = torch.autograd.grad(
        (eager * weights).sum(), (q, k, v), retain_graph=True
    )
    actual = torch.autograd.grad((grouped * weights).sum(), (q, k, v))
    torch.testing.assert_close(grouped, eager, atol=1e-12, rtol=1e-12)
    for before, after in zip(expected, actual, strict=True):
        torch.testing.assert_close(before, after, atol=1e-11, rtol=1e-11)


def test_grouped_attention_restores_interleaved_query_order():
    q = torch.tensor(
        [[[[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0], [3.0, 0.0], [-3.0, 0.0]]]],
        dtype=torch.double,
    )
    k = torch.tensor(
        [[[[3.0, 1.0], [-4.0, 2.0], [1.0, 2.0], [-2.0, 1.0], [2.0, 4.0], [-1.0, 3.0]]]],
        dtype=torch.double,
    )
    v = torch.arange(12, dtype=torch.double).reshape(1, 1, 6, 2)
    valid = torch.ones(1, 6, dtype=torch.bool)
    output, diagnostics = grouped_mita_attention(q, k, v, valid, 2, 2, 2)
    assert diagnostics["routes"].tolist() == [[[0, 1, 0, 1, 0, 1]]]
    torch.testing.assert_close(
        output, scalar_reference(q, k, v, valid, 2, 2), atol=1e-12, rtol=1e-12
    )
    assert diagnostics["packed_blocks"] == 4


def test_grouped_padding_nan_poison_is_ignored_in_forward_and_backward():
    tensors = tuple(
        tensor.requires_grad_() for tensor in fixture_qkv(batch=1, length=5)
    )
    valid = torch.ones(1, 5, dtype=torch.bool)
    expected, _ = grouped_mita_attention(*tensors, valid, 3, 8, 2)
    locations = [1, 2, 4, 6, 8]
    padded = []
    for tensor in tensors:
        expanded = torch.full((1, 2, 10, 3), float("nan"), dtype=torch.double)
        expanded[:, :, locations] = tensor.detach()
        padded.append(expanded.requires_grad_())
    mask = torch.zeros(1, 10, dtype=torch.bool)
    mask[:, locations] = True
    actual, _ = grouped_mita_attention(*padded, mask, 3, 8, 4)
    torch.testing.assert_close(
        actual[:, :, locations], expected, atol=1e-12, rtol=1e-12
    )
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[:, :, ~mask[0]], torch.zeros_like(actual[:, :, ~mask[0]]))
    actual.square().sum().backward()
    for tensor in padded:
        assert torch.isfinite(tensor.grad).all()
        assert torch.equal(
            tensor.grad[:, :, ~mask[0]], torch.zeros_like(tensor.grad[:, :, ~mask[0]])
        )


def test_empty_groups_and_empty_sequence_have_zero_outputs_and_gradients():
    for length in (0, 5):
        for static in (False, True):
            q, k, v = (
                tensor.requires_grad_()
                for tensor in fixture_qkv(batch=1, length=length)
            )
            valid = torch.zeros(1, length, dtype=torch.bool)
            output, diagnostics = grouped_mita_attention(
                q, k, v, valid, 3, 2, 3, static_pack=static
            )
            assert output.shape == q.shape and torch.equal(output, torch.zeros_like(q))
            assert diagnostics["active_packed_blocks"] == 0
            output.sum().backward()
            for tensor in (q, k, v):
                assert torch.equal(tensor.grad, torch.zeros_like(tensor))


def test_grouped_gradcheck_with_fixed_discrete_routes():
    q, k, v = (
        tensor.requires_grad_()
        for tensor in fixture_qkv(batch=1, heads=1, length=4, dimension=2)
    )
    valid = torch.ones(1, 4, dtype=torch.bool)
    assert torch.autograd.gradcheck(
        lambda a, b, c: grouped_mita_attention(a, b, c, valid, 2, 2, 2)[0], (q, k, v)
    )


def test_diagnostics_can_be_disabled_without_changing_attention_or_gradients():
    q, k, v = (tensor.requires_grad_() for tensor in fixture_qkv(batch=1))
    valid = torch.ones(1, 7, dtype=torch.bool)
    before, full = grouped_mita_attention(q, k, v, valid, 3, 2, 4)
    after, empty = grouped_mita_attention(
        q, k, v, valid, 3, 2, 4, return_diagnostics=False
    )
    assert full and empty == {}
    torch.testing.assert_close(before, after)
    first = torch.autograd.grad(before.square().sum(), (q, k, v), retain_graph=True)
    second = torch.autograd.grad(after.square().sum(), (q, k, v))
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("chunk", [1, 3, 16, 64])
def test_static_block_packing_matches_dynamic_outputs_and_gradients(chunk):
    q, k, v = (tensor.requires_grad_() for tensor in fixture_qkv(batch=2, length=11))
    valid = torch.tensor(
        [
            [True, False, True, True, False, True, True, False, True, False, True],
            [False] * 11,
        ]
    )
    before, dynamic = grouped_mita_attention(q, k, v, valid, 4, 3, chunk)
    after, static = grouped_mita_attention(
        q, k, v, valid, 4, 3, chunk, static_pack=True
    )
    torch.testing.assert_close(before, after, atol=1e-12, rtol=1e-12)
    assert static["host_size_syncs"] == 0
    assert static["packed_blocks"] >= dynamic["packed_blocks"]
    assert static["active_packed_blocks"] == dynamic["packed_blocks"]
    first = torch.autograd.grad(before.square().sum(), (q, k, v), retain_graph=True)
    second = torch.autograd.grad(after.square().sum(), (q, k, v))
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


def test_static_pack_executes_without_host_scalar_extraction():
    from torch.utils._python_dispatch import TorchDispatchMode

    class NoHostScalar(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if "_local_scalar_dense" in str(func):
                raise AssertionError("static grouped MiTA extracted a host scalar")
            return func(*args, **(kwargs or {}))

    q, k, v = fixture_qkv(batch=1, length=17)
    valid = torch.ones(1, 17, dtype=torch.bool)
    with NoHostScalar():
        output, diagnostics = grouped_mita_attention(
            q, k, v, valid, 4, 3, 8, static_pack=True, return_diagnostics=False
        )
    assert diagnostics == {} and torch.isfinite(output).all()


def test_large_cuda_half_routes_are_repeatable_and_match_grouped_execution():
    if not torch.cuda.is_available():
        pytest.skip("optional CUDA hardware unavailable")
    for length in (161, 1024):
        generator = torch.Generator().manual_seed(1729 + length)
        q, k, v = (
            torch.randn(1, 6, length, 64, generator=generator).cuda().half()
            for _ in range(3)
        )
        valid = torch.ones(1, length, device="cuda", dtype=torch.bool)
        baseline, reference = mita_attention(q, k, v, valid, 8, 16, 64)
        for _ in range(3):
            repeat, again = mita_attention(q, k, v, valid, 8, 16, 64)
            grouped, packed = grouped_mita_attention(q, k, v, valid, 8, 16, 32)
            assert torch.equal(reference["routes"], again["routes"])
            assert torch.equal(reference["expert_indices"], again["expert_indices"])
            assert torch.equal(reference["routes"], packed["routes"])
            assert torch.equal(reference["expert_indices"], packed["expert_indices"])
            torch.testing.assert_close(repeat, baseline, atol=2e-3, rtol=3e-3)
            torch.testing.assert_close(grouped, baseline, atol=2e-3, rtol=3e-3)


@pytest.mark.parametrize("fault", ["shape", "mask", "budget"])
def test_grouped_rejects_invalid_input_contracts(fault):
    q, k, v = fixture_qkv(batch=1)
    valid = torch.ones(1, 7, dtype=torch.bool)
    if fault == "shape":
        k = k[:, :, :-1]
    if fault == "mask":
        valid = valid.float()
    with pytest.raises(ValueError):
        grouped_mita_attention(q, k, v, valid, 0 if fault == "budget" else 3, 2, 4)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("static", [False, True])
def test_actual_cuda_grouped_forward_and_gradients_match_eager(dtype, static):
    if not torch.cuda.is_available():
        pytest.skip("optional CUDA hardware unavailable")
    q, k, v = (
        tensor.to(device="cuda", dtype=dtype).requires_grad_()
        for tensor in fixture_qkv(batch=1, length=17, dimension=8)
    )
    valid = torch.ones(1, 17, device="cuda", dtype=torch.bool)
    valid[:, 3::4] = False
    first, _ = mita_attention(q, k, v, valid, 4, 5, 8)
    second, _ = grouped_mita_attention(q, k, v, valid, 4, 5, 8, static_pack=static)
    tolerance = 3e-5 if dtype == torch.float32 else 5e-3
    torch.testing.assert_close(first, second, atol=tolerance, rtol=tolerance)
    before = torch.autograd.grad(
        first.float().square().sum(), (q, k, v), retain_graph=True
    )
    after = torch.autograd.grad(second.float().square().sum(), (q, k, v))
    for expected, actual in zip(before, after, strict=True):
        torch.testing.assert_close(expected, actual, atol=tolerance, rtol=tolerance)


def integration_config(**changes):
    return replace(
        LensConfig(
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
            train_steps=3,
            batch_size=4,
            mita_landmarks=3,
            mita_topk=2,
            mita_query_chunk=64,
        ),
        **changes,
    )


def test_four_layer_lens_reference_and_grouped_outputs_and_gradients_match():
    reference_cfg = integration_config(mita_execution="reference")
    grouped_cfg = replace(reference_cfg, mita_execution="grouped")
    first = create_model(reference_cfg).double()
    second = create_model(grouped_cfg).double()
    second.load_state_dict(first.state_dict(), strict=True)
    state, snapshot = records(events=6, memories=7)
    batch = make_batch(
        [state],
        [snapshot],
        [["tool:search", "reply"]],
        ["SELECT_EVIDENCE"],
        grouped_cfg,
    )
    batch = {
        name: tensor.double() if tensor.is_floating_point() else tensor
        for name, tensor in batch.items()
    }
    before, after = first(batch), second(batch)
    assert len(first.blocks) == len(second.blocks) == 4
    for name, tensor in before.items():
        torch.testing.assert_close(tensor, after[name], atol=1e-10, rtol=1e-9)
    before_loss = (
        before["candidate_values"].square().sum()
        + before["pre_value"].square().sum()
        + before["relevance"].square().sum()
    )
    after_loss = (
        after["candidate_values"].square().sum()
        + after["pre_value"].square().sum()
        + after["relevance"].square().sum()
    )
    before_loss.backward()
    after_loss.backward()
    reference_parameters = dict(first.named_parameters())
    grouped_parameters = dict(second.named_parameters())
    assert reference_parameters.keys() == grouped_parameters.keys()
    for name, parameter in reference_parameters.items():
        actual = grouped_parameters[name]
        if parameter.grad is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(
                parameter.grad, actual.grad, atol=1e-8, rtol=1e-8, msg=name
            )


def test_production_forward_executes_grouped_kernel_for_each_layer_without_diagnostics(
    monkeypatch,
):
    import jitrl.neural.attention as attention_module
    import jitrl.neural.mita_grouped as grouped_module

    assert LensConfig().mita_execution == "grouped"
    calls = []
    original = grouped_module.grouped_mita_attention

    def record_grouped(*args, **kwargs):
        output, diagnostics = original(*args, **kwargs)
        calls.append((args[6], kwargs["return_diagnostics"], bool(diagnostics)))
        return output, diagnostics

    def reference_forbidden(*args, **kwargs):
        raise AssertionError(
            "grouped production execution fell back to reference kernel"
        )

    monkeypatch.setattr(grouped_module, "grouped_mita_attention", record_grouped)
    monkeypatch.setattr(attention_module, "mita_attention", reference_forbidden)
    cfg = integration_config()
    state, snapshot = records(events=6, memories=7)
    batch = make_batch([state], [snapshot], [["tool:search"]], ["NO_OP"], cfg)
    model = create_model(cfg)
    output = model(batch)
    assert all(torch.isfinite(tensor).all() for tensor in output.values())
    assert calls == [(32, False, False)] * 4
    tokens = torch.randn(
        1, 11, cfg.hidden_dim, generator=torch.Generator().manual_seed(81)
    )
    valid = torch.ones(1, 11, dtype=torch.bool)
    diagnostics = model.blocks[0].attention.inspect_routing(tokens, valid)
    assert diagnostics["group_counts"].sum() == cfg.query_heads * 11
    assert diagnostics["packed_blocks"] > 0
    assert calls[-1] == (32, True, True)


@pytest.mark.parametrize("execution", ["reference", "grouped"])
def test_context_diagnostic_switch_preserves_production_values(execution):
    layer = MiTAAttention(integration_config(mita_execution=execution))
    tokens = torch.randn(1, 9, 24, generator=torch.Generator().manual_seed(19))
    valid = torch.ones(1, 9, dtype=torch.bool)
    q, k, v = layer.project(tokens, valid)
    first, full = layer._context(q, k, v, valid, diagnostics=True)
    second, empty = layer._context(q, k, v, valid, diagnostics=False)
    assert full and empty == {}
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(layer(tokens, valid), layer.merge(second, valid))


@pytest.mark.asyncio
async def test_existing_mita_checkpoint_switches_execution_without_retraining_or_cache_mixup(
    tmp_path,
):
    reference_cfg = integration_config(mita_execution="reference")
    runtime = LensRuntime(tmp_path, reference_cfg)
    try:
        seed_causal_training(runtime)
        snapshots = snapshots_in_store(runtime.store)
        trained = await runtime.train("private-a")
        assert trained["status"] == "trained"
        first, identifier = runtime._load_model("private-a")
        record = runtime.store.latest_checkpoint("private-a")
        payload = torch.load(
            runtime._path(record["filename"]), map_location="cpu", weights_only=True
        )
        assert "mita_execution" not in payload["architecture"]
        state, snapshot = records(events=6, memories=7)
        reference_output = runtime._forward(
            first, state, snapshot, ["tool:search", "reply"]
        )
        runtime.cfg = replace(reference_cfg, mita_execution="grouped")
        second, again = runtime._load_model("private-a", identifier)
        assert second is not first and again == identifier
        assert all(block.attention.execution == "reference" for block in first.blocks)
        assert all(block.attention.execution == "grouped" for block in second.blocks)
        for name, tensor in first.state_dict().items():
            assert torch.equal(tensor, second.state_dict()[name]), name
        grouped_output = runtime._forward(
            second, state, snapshot, ["tool:search", "reply"]
        )
        for name, tensor in reference_output.items():
            torch.testing.assert_close(
                tensor, grouped_output[name], atol=2e-6, rtol=2e-5
            )
        assert (
            runtime.store.latest_checkpoint("private-a")["checkpoint_id"] == identifier
        )
        with runtime.store._lock:
            assert (
                runtime.store._conn.execute(
                    "SELECT COUNT(*) FROM checkpoints"
                ).fetchone()[0]
                == 1
            )
        assert snapshots_in_store(runtime.store) == snapshots
        runtime.cfg = reference_cfg
        cached, again = runtime._load_model("private-a", identifier)
        assert cached is first and again == identifier
    finally:
        await runtime.close()
