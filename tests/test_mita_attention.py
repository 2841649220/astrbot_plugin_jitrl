"""Independent equation-level tests for actual MiTA compression and routing."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from jitrl.neural.attention import (  # noqa: E402
    GroupedSelfAttention,
    MiTAAttention,
    masked_landmarks,
    mita_attention,
)
from jitrl.neural.config import LensConfig  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixture_qkv(batch=2, heads=2, length=7, dimension=3):
    generator = torch.Generator().manual_seed(451)
    return tuple(
        torch.randn(
            batch, heads, length, dimension, generator=generator, dtype=torch.double
        )
        for _ in range(3)
    )


def scalar_reference(query, key, value, valid, landmarks, topk):
    """Literal Eq. 5-12, without using production pooling/routing helpers."""
    batches = []
    dimension = query.shape[-1]
    for b in range(query.shape[0]):
        positions = [i for i in range(query.shape[2]) if bool(valid[b, i])]
        heads = []
        for h in range(query.shape[1]):
            if not positions:
                heads.append(torch.zeros_like(query[b, h]))
                continue
            windows = [[] for _ in range(landmarks)]
            for rank, position in enumerate(positions):
                windows[rank * landmarks // len(positions)].append(position)
            active = [i for i, group in enumerate(windows) if group]
            landmark = torch.stack([query[b, h, windows[i]].mean(0) for i in active])
            keys = key[b, h, positions]
            values = value[b, h, positions]
            compressed = []
            experts = []
            for i in range(len(active)):
                scores = keys @ landmark[i] / math.sqrt(dimension)
                compressed.append(scores.softmax(-1) @ values)
                experts.append(
                    torch.argsort(scores, descending=True)[: min(topk, len(positions))]
                )
            compressed = torch.stack(compressed)
            outputs = []
            for position in range(query.shape[2]):
                if position not in positions:
                    outputs.append(torch.zeros_like(query[b, h, position]))
                    continue
                q = query[b, h, position]
                chosen = int((landmark @ q).argmax())
                merged_key = torch.cat((landmark, keys[experts[chosen]]))
                merged_value = torch.cat((compressed, values[experts[chosen]]))
                weights = (merged_key @ q / math.sqrt(dimension)).softmax(-1)
                outputs.append(weights @ merged_value)
            heads.append(torch.stack(outputs))
        batches.append(torch.stack(heads))
    return torch.stack(batches)


@pytest.mark.parametrize("landmarks,topk,chunk", [(1, 1, 1), (3, 2, 4), (8, 20, 2)])
def test_matches_independent_equation_reference(landmarks, topk, chunk):
    query, key, value = fixture_qkv()
    valid = torch.tensor([[True, False, True, True, False, True, True], [False] * 7])
    result, diagnostics = mita_attention(
        query, key, value, valid, landmarks, topk, chunk
    )
    reference = scalar_reference(query, key, value, valid, landmarks, topk)
    torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
    assert torch.equal(result[1], torch.zeros_like(result[1]))
    assert torch.isfinite(result).all()
    assert not diagnostics["landmark_valid"][1].any()
    assert torch.equal(
        diagnostics["routes"][1], torch.full_like(diagnostics["routes"][1], -1)
    )


def test_single_joint_softmax_is_not_an_average_of_experts():
    query = torch.tensor([[[[4.0, 0.0], [1.0, 1.0], [0.0, 1.0]]]], dtype=torch.double)
    key = torch.tensor([[[[3.0, 0.0], [0.0, 2.0], [-1.0, 0.0]]]], dtype=torch.double)
    value = torch.tensor(
        [[[[10.0, -2.0], [0.0, 7.0], [-6.0, 5.0]]]], dtype=torch.double
    )
    valid = torch.ones(1, 3, dtype=torch.bool)
    actual, _ = mita_attention(query, key, value, valid, 1, 1, 2)
    landmark = query[0, 0].mean(0)
    compression_weights = (key[0, 0] @ landmark / math.sqrt(2)).softmax(-1)
    compressed_value = compression_weights @ value[0, 0]
    selected = (key[0, 0] @ landmark).argmax()
    wrong_average = 0.5 * (compressed_value + value[0, 0, selected])
    torch.testing.assert_close(actual, scalar_reference(query, key, value, valid, 1, 1))
    assert (actual[0, 0, 0] - wrong_average).abs().max() > 0.1


def test_padding_insertion_and_poison_do_not_change_valid_outputs():
    query, key, value = fixture_qkv(batch=1, length=5)
    valid = torch.ones(1, 5, dtype=torch.bool)
    before, _ = mita_attention(query, key, value, valid, 4, 8, 2)
    locations = [1, 2, 4, 6, 8]
    padded = []
    for original in (query, key, value):
        tensor = torch.full((1, 2, 10, 3), float("nan"), dtype=torch.double)
        tensor[:, :, locations] = original
        padded.append(tensor)
    padded_valid = torch.zeros(1, 10, dtype=torch.bool)
    padded_valid[:, locations] = True
    after, _ = mita_attention(*padded, padded_valid, 4, 8, 3)
    torch.testing.assert_close(after[:, :, locations], before, rtol=1e-12, atol=1e-12)
    assert torch.equal(
        after[:, :, ~padded_valid[0]], torch.zeros_like(after[:, :, ~padded_valid[0]])
    )
    assert torch.isfinite(after).all()


@pytest.mark.parametrize("chunk", [1, 2, 3, 64])
def test_query_chunk_size_does_not_change_outputs_or_gradients(chunk):
    q, k, v = [x.requires_grad_() for x in fixture_qkv(batch=1)]
    valid = torch.tensor([[True, True, False, True, True, False, True]])
    reference, _ = mita_attention(q, k, v, valid, 3, 2, 64)
    reference_grad = torch.autograd.grad(
        reference.square().sum(), (q, k, v), retain_graph=True
    )
    result, _ = mita_attention(q, k, v, valid, 3, 2, chunk)
    gradients = torch.autograd.grad(result.square().sum(), (q, k, v))
    torch.testing.assert_close(result, reference, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(gradients, reference_grad, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)


def test_topk_experts_are_reused_and_depend_on_key_content():
    query = torch.tensor(
        [[[[3.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0]]]]
    )
    key = query.clone()
    value = torch.arange(12, dtype=torch.float32).reshape(1, 1, 6, 2)
    valid = torch.ones(1, 6, dtype=torch.bool)
    _, before = mita_attention(query, key, value, valid, 2, 2, 2)
    assert before["expert_indices"].shape == (1, 1, 2, 2)
    assert before["routes"][0, 0].tolist() == [0, 0, 0, 1, 1, 1]
    assert set(before["expert_indices"][0, 0, 0].tolist()) == {0, 1}
    changed_key = key.clone()
    changed_key[0, 0, 5] = torch.tensor([9.0, 0.0])
    _, after = mita_attention(query, changed_key, value, valid, 2, 2, 2)
    assert 5 in after["expert_indices"][0, 0, 0].tolist()
    assert torch.equal(before["routes"], after["routes"])


def test_masked_pooling_matches_equal_valid_order_windows():
    query = torch.arange(18, dtype=torch.double).reshape(1, 1, 6, 3)
    valid = torch.tensor([[False, True, True, False, True, True]])
    landmarks, support = masked_landmarks(query, valid, 2)
    expected = torch.stack((query[0, 0, [1, 2]].mean(0), query[0, 0, [4, 5]].mean(0)))
    torch.testing.assert_close(landmarks[0, 0], expected)
    assert support.tolist() == [[True, True]]


def test_gradient_matches_finite_differences_away_from_routing_ties():
    args = tuple(
        x.requires_grad_() for x in fixture_qkv(batch=1, heads=1, length=4, dimension=2)
    )
    valid = torch.ones(1, 4, dtype=torch.bool)
    assert torch.autograd.gradcheck(
        lambda q, k, v: mita_attention(q, k, v, valid, 2, 2, 2)[0], args
    )
    output, diagnostics = mita_attention(*args, valid, 2, 2, 2)
    output.square().sum().backward()
    for tensor in args:
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0
    assert all(not tensor.requires_grad for tensor in diagnostics.values())


@pytest.mark.parametrize("layer_type", [MiTAAttention, GroupedSelfAttention])
def test_grouped_projection_shape_masking_and_parameter_gradients(layer_type):
    cfg = LensConfig(
        hidden_dim=24, query_heads=6, kv_heads=2, mita_landmarks=3, mita_topk=2
    )
    layer = layer_type(cfg).double()
    tokens = torch.randn(2, 7, 24, dtype=torch.double, requires_grad=True)
    valid = torch.tensor([[True, False, True, True, False, True, True], [False] * 7])
    poisoned = tokens.masked_fill(~valid.unsqueeze(-1), float("nan"))
    output = layer(poisoned, valid)
    assert output.shape == tokens.shape and torch.isfinite(output).all()
    assert torch.equal(output[~valid], torch.zeros_like(output[~valid]))
    output.square().sum().backward()
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert torch.equal(tokens.grad[~valid], torch.zeros_like(tokens.grad[~valid]))


def test_gqa_control_uses_identical_projection_layout_and_matches_sdpa():
    cfg = LensConfig(hidden_dim=24, query_heads=6, kv_heads=2)
    sparse, dense = MiTAAttention(cfg).double(), GroupedSelfAttention(cfg).double()
    dense.load_state_dict(sparse.state_dict(), strict=True)
    assert set(sparse.state_dict()) == {
        "query.weight",
        "key.weight",
        "value.weight",
        "output.weight",
    }
    tokens = torch.randn(2, 5, 24, dtype=torch.double)
    valid = torch.tensor([[True, False, True, True, True], [True] * 5])
    q, k, v = dense.project(tokens, valid)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=valid[:, None, None, :]
    )
    torch.testing.assert_close(dense(tokens, valid), dense.merge(expected, valid))


def test_mita_never_forms_a_full_query_key_score_matrix():
    from torch.utils._python_dispatch import TorchDispatchMode

    class MatrixShapes(TorchDispatchMode):
        def __init__(self):
            self.shapes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            if "bmm" in str(func):
                self.shapes.append(tuple(result.shape))
            return result

    q, k, v = fixture_qkv(batch=1, heads=2, length=37, dimension=4)
    valid = torch.ones(1, 37, dtype=torch.bool)
    matrices = MatrixShapes()
    with matrices:
        result, diagnostics = mita_attention(q, k, v, valid, 3, 5, 7)
    assert result.shape == q.shape
    assert matrices.shapes
    assert not any(shape[-2:] == (37, 37) for shape in matrices.shapes)
    assert diagnostics["score_elements"] == 2 * 37 * (2 * 3 + 5)
    assert diagnostics["peak_score_elements"] < 2 * 37 * 37


@pytest.mark.parametrize("layer_type", [MiTAAttention, GroupedSelfAttention])
def test_zero_length_sequence_is_well_defined(layer_type):
    cfg = LensConfig(hidden_dim=24)
    layer = layer_type(cfg)
    tokens = torch.empty(2, 0, 24)
    valid = torch.empty(2, 0, dtype=torch.bool)
    assert layer(tokens, valid).shape == tokens.shape


@pytest.mark.parametrize("fault", ["shape", "mask", "budget"])
def test_malformed_tensor_contracts_are_rejected(fault):
    q, k, v = fixture_qkv(batch=1)
    valid = torch.ones(1, 7, dtype=torch.bool)
    if fault == "shape":
        k = k[:, :, :-1]
    if fault == "mask":
        valid = valid.float()
    with pytest.raises(ValueError):
        mita_attention(q, k, v, valid, 0 if fault == "budget" else 3, 2, 4)
