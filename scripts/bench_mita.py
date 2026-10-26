#!/usr/bin/env python
"""Reproduce CPU/CUDA attention timings without AstrBot or a remote API.

Compare the genuine local MiTA kernel with dense grouped-query self-attention
using exactly matched Q/K/V/output projections, inputs, dtype and masks. The
logical tensor capacities are reported separately from measured CUDA allocator
peaks. Autograd can retain query gathers from multiple chunks during training.

Usage::

    python scripts/bench_mita.py --json benchmarks/mita-attention.json
    python scripts/bench_mita.py --lengths 32,128,512,1024 --repeats 7
    python scripts/bench_mita.py --device cuda --model --json benchmarks/mita-attention-cuda.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HARNESS_VERSION = 2
PROJECTIONS = ("query.", "key.", "value.", "output.")


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _lengths(value: str) -> list[int]:
    try:
        result = [_positive(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "comma-separated positive lengths required"
        ) from exc
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("sequence lengths must be distinct")
    return result


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lengths", type=_lengths, default=[32, 128, 512])
    parser.add_argument("--repeats", type=_positive, default=7)
    parser.add_argument("--warmup", type=_positive, default=2)
    parser.add_argument("--threads", type=_positive, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--batch-size", type=_positive, default=1)
    parser.add_argument("--hidden-dim", type=_positive, default=384)
    parser.add_argument("--query-heads", type=_positive, default=6)
    parser.add_argument("--kv-heads", type=_positive, default=2)
    parser.add_argument("--landmarks", type=_positive, default=8)
    parser.add_argument("--topk", type=_positive, default=16)
    parser.add_argument("--query-chunk", type=_positive, default=64)
    parser.add_argument("--seed", type=_positive, default=1729)
    parser.add_argument("--padding-fraction", type=float, default=0.0)
    parser.add_argument(
        "--model",
        action="store_true",
        help="also time all four local Lens backends on identical precomputed feature tensors",
    )
    parser.add_argument("--json", type=Path, dest="output")
    args = parser.parse_args()
    if not math.isfinite(args.padding_fraction) or not 0 <= args.padding_fraction < 1:
        parser.error("--padding-fraction must be finite and in [0,1)")
    if args.device != "cuda" and args.dtype == "float16":
        parser.error("--dtype float16 is supported only with --device cuda")
    return args


def _revision() -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {"head": revision, "working_tree_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "working_tree_dirty": None}


def _distribution(milliseconds: list[float]) -> dict[str, Any]:
    ordered = sorted(milliseconds)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[index],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples_ms": milliseconds,
    }


def _measure(
    function: Any, warmup: int, repeats: int, *, device: Any, prepare: Any = None
) -> dict[str, Any]:
    import torch

    cuda = device.type == "cuda"

    def synchronize() -> None:
        if cuda:
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        if prepare is not None:
            prepare()
        synchronize()
        function()
        synchronize()
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        samples = []
        memory_samples = []
        for _ in range(repeats):
            if prepare is not None:
                prepare()
            synchronize()
            if cuda:
                baseline = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter_ns()
            function()
            synchronize()
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
            if cuda:
                peak = torch.cuda.max_memory_allocated(device)
                memory_samples.append(
                    {
                        "baseline_allocated_bytes": baseline,
                        "peak_allocated_bytes": peak,
                        "peak_minus_baseline_bytes": max(0, peak - baseline),
                    }
                )
    finally:
        if was_enabled:
            gc.enable()
    result = _distribution(samples)
    if cuda:
        result["cuda_allocator"] = {
            "kind": "measured_pytorch_cuda_allocated_tensor_memory_not_reserved_or_process_memory",
            "max_peak_allocated_bytes": max(
                row["peak_allocated_bytes"] for row in memory_samples
            ),
            "max_peak_minus_baseline_bytes": max(
                row["peak_minus_baseline_bytes"] for row in memory_samples
            ),
            "samples": memory_samples,
            "baseline": "all currently resident benchmark weights/inputs; gradients cleared before each training sample; excludes reserved cache, driver and non-PyTorch allocations",
        }
    return result


def interaction_counts(cfg: Any, length: int, batch_size: int) -> dict[str, Any]:
    """Account for compression, expert routing and final shared+sparse access.

    These formulas count physical sequence slots. They omit projection FLOPs,
    pooling, top-k selection, gathering, softmax and framework overhead. Routing
    computes scores without an additional weighted value sum.
    """
    # The physical budget is retained even when some windows are masked.
    landmarks = cfg.mita_landmarks
    topk = min(cfg.mita_topk, length)
    multiplier = batch_size * cfg.query_heads
    head_dim = cfg.hidden_dim // cfg.query_heads
    compression = landmarks * length
    routing = length * landmarks
    final = length * (landmarks + topk)
    dense = length * length
    # The final shared expert reuses the router's Nm logits, rather than
    # recomputing a second query-to-landmark score matrix.
    reused = length * landmarks
    sparse_scores = compression + routing + final - reused
    sparse_values = compression + final
    return {
        "effective_landmarks": landmarks,
        "effective_topk": topk,
        "routed_experts_per_query": 1,
        "dense": {
            "score_dot_product_pairs": multiplier * dense,
            "weighted_value_pairs": multiplier * dense,
            "attention_only_scalar_flop_estimate": multiplier * dense * 4 * head_dim,
        },
        "mita": {
            "compression_score_pairs": multiplier * compression,
            "routing_score_pairs": multiplier * routing,
            "shared_plus_routed_score_pairs": multiplier * final,
            "reused_routing_score_pairs_in_final_support": multiplier * reused,
            "score_dot_product_pairs": multiplier * sparse_scores,
            "weighted_value_pairs": multiplier * sparse_values,
            "attention_only_scalar_flop_estimate": multiplier
            * (sparse_scores + sparse_values)
            * 2
            * head_dim,
        },
        "mita_to_dense_score_pair_ratio": sparse_scores / dense,
        "mita_to_dense_attention_flop_ratio": (sparse_scores + sparse_values)
        / (2 * dense),
    }


def tensor_capacities(
    cfg: Any, length: int, batch_size: int, element_bytes: int
) -> dict[str, Any]:
    """Logical tensor sizes, explicitly distinct from process/allocator memory.

    Shared landmark vectors are reused; only routed original KV vectors are
    gathered per query. Training may retain gathers across every query chunk.
    """
    h = batch_size * cfg.query_heads
    d = cfg.hidden_dim // cfg.query_heads
    m = cfg.mita_landmarks
    k = min(cfg.mita_topk, length)
    q = min(cfg.mita_query_chunk, length)
    return {
        "unit": "bytes",
        "element_bytes": element_bytes,
        "kind": "individual_logical_tensor_capacity_estimates_not_measured_allocator_peak",
        "dense_explicit_score_matrix": h * length * length * element_bytes,
        "mita_compression_scores": h * m * length * element_bytes,
        "mita_full_routing_scores_upper_bound": h * length * m * element_bytes,
        "mita_one_chunk_final_scores": h * q * (m + k) * element_bytes,
        "mita_one_chunk_routed_key_and_value_gather": 2 * h * q * k * d * element_bytes,
        "mita_reusable_expert_key_and_value_buffer": 2 * h * m * k * d * element_bytes,
        "mita_landmark_query_and_compressed_value": 2 * h * m * d * element_bytes,
        "mita_all_queries_routed_key_and_value_gather": 2
        * h
        * length
        * k
        * d
        * element_bytes,
        "caveat": "Several intermediates may coexist. Backward may retain tensors from every chunk. Dense optimized SDPA may avoid materializing its full logical score matrix. These values do not measure either allocator or process peak memory.",
    }


def _matched_modules(cfg: Any, torch: Any) -> tuple[Any, Any, int]:
    from jitrl.neural.attention import GroupedSelfAttention, MiTAAttention

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(cfg.seed)
        dense = GroupedSelfAttention(cfg).cpu()
        sparse = MiTAAttention(cfg).cpu()
    reference = {
        name: value
        for name, value in dense.state_dict().items()
        if name.startswith(PROJECTIONS)
    }
    actual = sparse.state_dict()
    if not reference or any(
        name not in actual or actual[name].shape != value.shape
        for name, value in reference.items()
    ):
        raise RuntimeError(
            "attention kernels do not expose identical projection parameter shapes"
        )
    actual.update(reference)
    sparse.load_state_dict(actual, strict=True)
    if any(
        not torch.equal(sparse.state_dict()[name], value)
        for name, value in reference.items()
    ):
        raise RuntimeError("failed to match attention projection weights")
    return dense, sparse, sum(value.numel() for value in reference.values())


def _model_forward(cfg: Any, args: argparse.Namespace, torch: Any) -> dict[str, Any]:
    """Time the complete network on fixed feature tensors; no API or store I/O."""
    from jitrl.neural.model import create_model

    b, events, memory, actions = args.batch_size, cfg.max_events, cfg.memory_tokens, 3
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    def features(*shape: int) -> Any:
        value = torch.randn(*shape, generator=generator)
        return torch.nn.functional.normalize(value, dim=-1)

    batch = {
        "state": features(b, cfg.embedding_dim),
        "events": torch.randint(
            1,
            cfg.engram_buckets + 1,
            (b, events, 2 * cfg.engram_heads),
            generator=generator,
        ),
        "event_features": features(b, events, cfg.embedding_dim),
        "event_mask": torch.ones(b, events, dtype=torch.bool),
        "memory_states": features(b, memory, cfg.embedding_dim),
        "memory_actions": features(b, memory, cfg.embedding_dim),
        "memory_meta": torch.rand(b, memory, 6, generator=generator),
        "memory_mask": torch.ones(b, memory, dtype=torch.bool),
        "candidates": features(b, actions, cfg.embedding_dim),
        "candidate_mask": torch.ones(b, actions, dtype=torch.bool),
        "pre_index": torch.zeros(b, dtype=torch.long),
    }
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    batch = {
        name: value.to(
            device=device, dtype=dtype if value.is_floating_point() else value.dtype
        )
        for name, value in batch.items()
    }
    reference = create_model(replace(cfg, attention_backend="gqa")).to(
        device=device, dtype=dtype
    )
    result = {
        "workload": {
            "batch_size": b,
            "events": events,
            "memories": memory,
            "candidates": actions,
            "backbone_tokens": 1 + events + memory,
        },
        "timing_scope": "network forward on precomputed seeded synthetic features; excludes retrieval, canonicalization, frozen hashing, training and remote API",
        "weight_matching": "gqa/mita/hybrid share every state_dict tensor; legacy has a different scalar-query architecture and only shares the seed and input feature workload",
        "backends": {},
    }
    for backend in ("legacy", "gqa", "hybrid", "mita"):
        model = (
            reference
            if backend == "gqa"
            else create_model(replace(cfg, attention_backend=backend))
        )
        model = model.to(device=device, dtype=dtype)
        if backend in {"mita", "hybrid"}:
            model.load_state_dict(reference.state_dict(), strict=True)
        with torch.inference_mode():
            output = model(batch)
            if any(not torch.isfinite(value).all() for value in output.values()):
                raise RuntimeError("nonfinite full-model benchmark output")
            del output
            timing = _measure(
                lambda layer=model: layer(batch),
                args.warmup,
                args.repeats,
                device=device,
            )
        result["backends"][backend] = {
            "forward": timing,
            "attention_inventory": model.attention_inventory(),
            "parameters": model.parameter_counts(),
        }
    return result


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from jitrl.neural.config import LensConfig

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    device, dtype = torch.device(args.device), getattr(torch, args.dtype)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = LensConfig(
        hidden_dim=args.hidden_dim,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        attention_backend="mita",
        mita_landmarks=args.landmarks,
        mita_topk=args.topk,
        mita_query_chunk=args.query_chunk,
        seed=args.seed,
        device=args.device,
    )
    dense, sparse, matched = _matched_modules(cfg, torch)
    dense = dense.to(device=device, dtype=dtype)
    sparse = sparse.to(device=device, dtype=dtype)
    rows = []
    for length in args.lengths:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + length)
        tokens = torch.randn(
            args.batch_size,
            length,
            cfg.hidden_dim,
            generator=generator,
            dtype=torch.float32,
        )
        valid = torch.ones(args.batch_size, length, dtype=torch.bool)
        active = max(1, math.floor(length * (1 - args.padding_fraction)))
        valid[:, active:] = False
        tokens = tokens.to(device=device, dtype=dtype)
        valid = valid.to(device=device)
        row = {
            "length": length,
            "valid_tokens_per_batch_row": active,
            "interactions": interaction_counts(cfg, length, args.batch_size),
            "tensor_capacities": tensor_capacities(
                cfg, length, args.batch_size, tokens.element_size()
            ),
        }
        with torch.no_grad():
            outputs = [module.eval()(tokens, valid) for module in (dense, sparse)]
            if any(
                output.shape != tokens.shape or not torch.isfinite(output).all()
                for output in outputs
            ):
                raise RuntimeError("invalid attention output")
            difference = (outputs[1].float() - outputs[0].float()).norm() / outputs[
                0
            ].float().norm().clamp_min(1e-12)
            row["relative_output_l2_difference"] = float(difference)
            del outputs
        for name, module in (("gqa", dense), ("mita", sparse)):
            print(
                f"{args.device.upper()} {args.dtype} N={length} {name}: forward and backward",
                file=sys.stderr,
                flush=True,
            )
            module.eval()
            with torch.inference_mode():
                forward = _measure(
                    lambda layer=module: layer(tokens, valid),
                    args.warmup,
                    args.repeats,
                    device=device,
                )
            module.train()
            training_tokens = tokens.detach().clone().requires_grad_(True)

            def train_step(layer: Any = module, data: Any = training_tokens) -> None:
                layer.zero_grad(set_to_none=True)
                data.grad = None
                layer(data, valid).float().square().mean().backward()

            def clear_gradients(
                layer: Any = module, data: Any = training_tokens
            ) -> None:
                layer.zero_grad(set_to_none=True)
                data.grad = None

            backward = _measure(
                train_step,
                args.warmup,
                args.repeats,
                device=device,
                prepare=clear_gradients,
            )
            if (
                training_tokens.grad is None
                or not torch.isfinite(training_tokens.grad).all()
            ):
                raise RuntimeError("invalid attention input gradient")
            module.zero_grad(set_to_none=True)
            training_tokens.grad = None
            del train_step, clear_gradients, training_tokens
            row[name] = {"forward": forward, "forward_and_backward": backward}
        row["mita_to_gqa_median_latency_ratio"] = {
            phase: row["mita"][phase]["median_ms"] / row["gqa"][phase]["median_ms"]
            for phase in ("forward", "forward_and_backward")
        }
        rows.append(row)
    result = {
        "harness_version": HARNESS_VERSION,
        "meta": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch_version": str(torch.__version__),
            "device": args.device,
            "dtype": args.dtype,
            "cuda": {
                "gpu_name": torch.cuda.get_device_name(device),
                "total_memory_bytes": torch.cuda.get_device_properties(
                    device
                ).total_memory,
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "runtime_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
            if device.type == "cuda"
            else None,
            "cuda_synchronized_before_and_after_every_sample_and_warmup": device.type
            == "cuda",
            "tf32_matmul_and_cudnn_enabled": False,
            "seed": args.seed,
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "batch_size": args.batch_size,
            "warmup_per_phase": args.warmup,
            "repeats_per_phase": args.repeats,
            "padding_fraction": args.padding_fraction,
            "matched_projection_parameters": matched,
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "jitrl/neural/attention.py",
                    "jitrl/neural/config.py",
                    "scripts/bench_mita.py",
                    *(["jitrl/neural/model.py"] if args.model else []),
                )
            },
            "config": asdict(cfg),
            "revision": _revision(),
            "forward_and_backward_includes": "zero_grad(set_to_none), input grad reset, attention forward, mean-square synthetic loss and backward; excludes optimizer and data generation",
        },
        "limitations": [
            "Kernel microbenchmark on synthetic Gaussian inputs; no end-to-end Agent, remote API, training quality or task-success claim.",
            "MiTA approximates dense attention; matched projections do not imply identical output functions.",
            "Interaction/FLOP formulas omit projections, pooling, top-k, gathers and framework overhead; lower interaction counts do not guarantee lower latency.",
            "Tensor capacity estimates are not measured allocator peaks; forward query chunking does not guarantee one-chunk training activation retention.",
            "CUDA allocator peaks, when present, measure PyTorch allocated tensors in this benchmark process. They exclude reserved cache, driver, other processes and non-PyTorch allocations.",
            "Timings depend on machine load, thread settings, synchronization overhead and PyTorch implementation; raw samples are retained.",
        ],
        "results": rows,
    }
    if args.model:
        result["model_forward"] = _model_forward(cfg, args, torch)
    return result


def main() -> int:
    args = _arguments()
    result = benchmark(args)
    serialized = (
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print("N    GQA forward / fwd+bwd ms    MiTA forward / fwd+bwd ms")
        for row in result["results"]:
            print(
                f"{row['length']:<5}{row['gqa']['forward']['median_ms']:>9.3f} / {row['gqa']['forward_and_backward']['median_ms']:<9.3f}   {row['mita']['forward']['median_ms']:>9.3f} / {row['mita']['forward_and_backward']['median_ms']:<9.3f}"
            )
        for backend, row in result.get("model_forward", {}).get("backends", {}).items():
            print(
                f"Lens {backend}: forward {row['forward']['median_ms']:.3f} ms; inventory={row['attention_inventory']}"
            )
        print(f"JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
