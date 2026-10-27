#!/usr/bin/env python
"""Diagnose the actual production MiTA/GQA operators on CPU and CUDA.

Unprofiled CPU wall, CUDA host-dispatch wall, synchronized wall and CUDA-event
elapsed time are measured separately. Kineto instrumentation is a separate
diagnostic run: its overhead is never substituted for production latency.

Examples::

    python scripts/profile_mita.py --json benchmarks/mita-profile.json
    python scripts/profile_mita.py --chunks 64,256,1024 --repeats 9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAMILIES = {
    "gather": {"aten::gather"},
    "topk": {"aten::topk"},
    "argmax": {"aten::argmax"},
    "softmax": {"aten::softmax", "aten::_softmax"},
    "bmm": {"aten::bmm"},
    "projections_mm": {"aten::mm", "aten::linear"},
    "mul": {"aten::mul"},
    "sum": {"aten::sum"},
    "copy": {"aten::copy_", "aten::_to_copy", "aten::clone", "aten::contiguous"},
    "repeat_interleave": {"aten::repeat_interleave"},
    "mask": {
        "aten::masked_fill",
        "aten::masked_fill_",
        "aten::where",
        "aten::bitwise_not",
    },
    "shape_views": {
        "aten::reshape",
        "aten::view",
        "aten::slice",
        "aten::select",
        "aten::as_strided",
        "aten::unsqueeze",
        "aten::expand",
    },
    "pooling": {"aten::scatter_add", "aten::scatter_add_", "aten::cumsum"},
}


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("a positive integer is required")
    return number


def _numbers(value: str) -> list[int]:
    return [_positive(item.strip()) for item in value.split(",")]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--lengths", type=_numbers, default=[161, 1024])
    parser.add_argument("--chunks", type=_numbers, default=[64, 256, 1024])
    parser.add_argument("--repeats", type=_positive, default=9)
    parser.add_argument("--profile-repeats", type=_positive, default=2)
    parser.add_argument("--warmup", type=_positive, default=3)
    parser.add_argument("--threads", type=_positive, default=1)
    parser.add_argument("--seed", type=_positive, default=1729)
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument(
        "--mita-execution", choices=("reference", "grouped"), default="grouped"
    )
    parser.add_argument(
        "--compare-reference",
        action="store_true",
        help="capture reference and grouped production execution against identical GQA projections",
    )
    parser.add_argument("--json", type=Path, dest="output")
    return parser.parse_args()


def _distribution(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "min_ms": ordered[0],
        "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "samples_ms": values,
    }


def _timings(
    layer: Any, tokens: Any, valid: Any, args: argparse.Namespace, torch: Any
) -> dict[str, Any]:
    cuda = tokens.device.type == "cuda"
    wall, dispatch, elapsed = [], [], []
    with torch.inference_mode():
        for _ in range(args.warmup):
            if cuda:
                torch.cuda.synchronize(tokens.device)
            layer(tokens, valid)
            if cuda:
                torch.cuda.synchronize(tokens.device)
        for _ in range(args.repeats):
            if cuda:
                torch.cuda.synchronize(tokens.device)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            started = time.perf_counter_ns()
            output = layer(tokens, valid)
            submitted = time.perf_counter_ns()
            if cuda:
                end_event.record()
                torch.cuda.synchronize(tokens.device)
            finished = time.perf_counter_ns()
            wall.append((finished - started) / 1_000_000)
            dispatch.append((submitted - started) / 1_000_000)
            if cuda:
                elapsed.append(start_event.elapsed_time(end_event))
            del output
    result = {"synchronized_wall" if cuda else "cpu_wall": _distribution(wall)}
    if cuda:
        result.update(
            {
                "host_dispatch_wall": _distribution(dispatch),
                "cuda_event_elapsed": _distribution(elapsed),
                "cuda_event_note": "stream elapsed interval includes idle gaps between CPU kernel submissions; it is not the sum of active GPU kernel execution time",
            }
        )
    return result


@contextmanager
def _projection_scopes(layer: Any, torch: Any):
    """Temporary diagnostic scopes; hooks and open scopes never escape."""
    handles = []
    active: dict[str, list[Any]] = defaultdict(list)
    for name in ("query", "key", "value", "output"):

        def begin(module, args, label=name):
            scope = torch.profiler.record_function("projection::" + label)
            scope.__enter__()
            active[label].append(scope)

        def end(module, args, result, label=name):
            if active[label]:
                active[label].pop().__exit__(None, None, None)

        handles.append(getattr(layer, name).register_forward_pre_hook(begin))
        handles.append(
            getattr(layer, name).register_forward_hook(end, always_call=True)
        )
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for stack in active.values():
            while stack:
                stack.pop().__exit__(None, None, None)


def _metric(event: Any, name: str) -> float:
    return float(getattr(event, name, 0.0))


def _row(event: Any, cuda_traced: bool) -> dict[str, Any]:
    return {
        "name": event.key,
        "device_type": str(event.device_type),
        "count": int(event.count),
        "input_shapes": event.input_shapes,
        "self_cpu_us": _metric(event, "self_cpu_time_total"),
        "total_cpu_us": _metric(event, "cpu_time_total"),
        "self_device_us": _metric(event, "self_device_time_total")
        if cuda_traced
        else None,
        "total_device_us": _metric(event, "device_time_total") if cuda_traced else None,
        "net_self_cpu_bytes": int(_metric(event, "self_cpu_memory_usage")),
        "net_self_device_bytes": int(_metric(event, "self_device_memory_usage"))
        if cuda_traced
        else None,
    }


def _profile(
    layer: Any, tokens: Any, valid: Any, args: argparse.Namespace, torch: Any
) -> dict[str, Any]:
    supported = torch.profiler.supported_activities()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.profiler.ProfilerActivity.CUDA in supported:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.inference_mode(), _projection_scopes(layer, torch):
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                for _ in range(args.profile_repeats):
                    with torch.profiler.record_function("production_attention_forward"):
                        layer(tokens, valid)
                        torch.cuda.synchronize(tokens.device)
        raw = prof.events()
        raw_cuda = [e for e in raw if str(e.device_type).endswith("CUDA")]
        # Kineto also emits CUDA-side user-annotation ranges for record_function
        # scopes. Their spans contain kernels/idle gaps and must not be counted
        # a second time as active kernel execution.
        cuda_events = [
            e
            for e in raw_cuda
            if not getattr(e, "is_user_annotation", False)
            and e.name != "production_attention_forward"
            and not e.name.startswith("projection::")
        ]
        traced = bool(cuda_events)
        averages = list(prof.key_averages(group_by_input_shape=True))
        aten = [e for e in averages if e.key.startswith("aten::")]
        aggregate = list(prof.key_averages())
        operators = [_row(event, traced) for event in aten]
        raw_aten = [e for e in raw if e.name.startswith("aten::")]
        families = {}
        for name, members in FAMILIES.items():
            rows = [row for row in operators if row["name"] in members]
            events = [event for event in raw_aten if event.name in members]
            families[name] = {
                "count": sum(row["count"] for row in rows),
                "self_cpu_us": sum(row["self_cpu_us"] for row in rows),
                "self_device_us": sum(row["self_device_us"] for row in rows)
                if traced
                else None,
                "positive_self_device_allocated_bytes": sum(
                    max(0, int(_metric(e, "self_device_memory_usage"))) for e in events
                )
                if traced
                else None,
                "net_self_device_bytes": sum(
                    int(_metric(e, "self_device_memory_usage")) for e in events
                )
                if traced
                else None,
                "rows": rows,
            }
        kernels: dict[str, dict[str, Any]] = {}
        for event in cuda_events:
            row = kernels.setdefault(
                event.name, {"name": event.name, "count": 0, "device_us": 0.0}
            )
            row["count"] += 1
            row["device_us"] += _metric(event, "device_time_total")
        backend_ops = sorted(e.key for e in aggregate if "scaled_dot_product" in e.key)
        return {
            "status": "profiled",
            "profile_samples": args.profile_repeats,
            "cuda_trace_status": "available"
            if traced
            else "unavailable_no_cuda_events_recorded",
            "profile_overhead_not_production_latency": True,
            "sdpa_backend_operators": backend_ops,
            "cuda_event_count": len(cuda_events) if traced else None,
            "cuda_annotation_ranges_excluded": len(raw_cuda) - len(cuda_events)
            if traced
            else None,
            "cuda_launch_kernel_calls": sum(
                e.count
                for e in aggregate
                if e.key.startswith(("cudaLaunchKernel", "cuLaunchKernel"))
            ),
            "cuda_memcpy_async_calls": sum(
                e.count for e in aggregate if e.key == "cudaMemcpyAsync"
            ),
            "sum_raw_cuda_event_device_us": sum(
                _metric(e, "device_time_total") for e in cuda_events
            )
            if traced
            else None,
            "top_self_cpu": [
                _row(e, traced)
                for e in sorted(
                    aggregate,
                    key=lambda e: _metric(e, "self_cpu_time_total"),
                    reverse=True,
                )[:15]
            ],
            "top_aten_self_cpu": sorted(
                operators, key=lambda row: row["self_cpu_us"], reverse=True
            )[:15],
            "top_aten_self_device": sorted(
                operators, key=lambda row: row["self_device_us"], reverse=True
            )[:15]
            if traced
            else None,
            "top_cuda_kernels": sorted(
                kernels.values(), key=lambda row: row["device_us"], reverse=True
            )[:15]
            if traced
            else None,
            "projection_scopes": [
                _row(e, traced) for e in aggregate if e.key.startswith("projection::")
            ],
            "operation_families": families,
            "memory": {
                "positive_aten_self_device_allocation_bytes": sum(
                    max(0, int(_metric(e, "self_device_memory_usage")))
                    for e in raw_aten
                )
                if traced
                else None,
                "positive_aten_self_cpu_allocation_bytes": sum(
                    max(0, int(_metric(e, "self_cpu_memory_usage"))) for e in raw_aten
                ),
                "note": "allocation-event volume and signed operator net memory are not peak allocated/reserved/process memory",
            },
        }
    except (RuntimeError, AssertionError) as exc:
        return {
            "status": "profiling_unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "cuda_trace_status": "unavailable",
            "profile_overhead_not_production_latency": True,
        }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from jitrl.neural.attention import GroupedSelfAttention, MiTAAttention
    from jitrl.neural.config import LensConfig

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        return {
            "status": "cuda_unavailable",
            "torch_version": str(torch.__version__),
            "no_cuda_work_claimed": True,
        }
    dtype = getattr(torch, args.dtype)
    cfg = LensConfig(
        device="cuda",
        mita_landmarks=8,
        mita_topk=16,
        mita_query_chunk=64,
        mita_execution=args.mita_execution,
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(args.seed)
        reference = GroupedSelfAttention(cfg).eval()
    matched_weights = {
        name: tensor.clone() for name, tensor in reference.state_dict().items()
    }
    del reference
    rows = []
    for length in args.lengths:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + length)
        cpu_tokens = torch.randn(1, length, cfg.hidden_dim, generator=generator)
        cpu_valid = torch.ones(1, length, dtype=torch.bool)
        gpu_tokens = cpu_tokens.to(device="cuda", dtype=dtype)
        gpu_valid = cpu_valid.cuda()
        reference_output = None
        executions = (
            ["reference", "grouped"]
            if args.compare_reference
            else [args.mita_execution]
        )
        cases = [("gqa", 64, args.mita_execution)]
        cases.extend(
            ("mita_" + execution, chunk, execution)
            for execution in executions
            for chunk in args.chunks
        )
        for name, chunk, execution in cases:
            selected_cfg = replace(
                cfg, mita_query_chunk=chunk, mita_execution=execution
            )
            cls = GroupedSelfAttention if name == "gqa" else MiTAAttention
            cpu_layer = cls(replace(selected_cfg, device="cpu")).eval()
            cpu_layer.load_state_dict(matched_weights, strict=True)
            cpu_timing = _timings(cpu_layer, cpu_tokens, cpu_valid, args, torch)
            gpu_layer = cls(selected_cfg).to(device="cuda", dtype=dtype).eval()
            gpu_layer.load_state_dict(matched_weights, strict=True)
            with torch.inference_mode():
                output = gpu_layer(gpu_tokens, gpu_valid)
                if not torch.isfinite(output).all():
                    raise RuntimeError("nonfinite production attention output")
                equivalence = None
                if name != "gqa":
                    if reference_output is None:
                        reference_output = output.detach().clone()
                    equivalence = float((output - reference_output).abs().max())
                    if not torch.allclose(
                        output,
                        reference_output,
                        atol=0.002 if dtype == torch.float16 else 0.00001,
                        rtol=0.002 if dtype == torch.float16 else 0.00001,
                    ):
                        raise RuntimeError(
                            "reference/grouped execution or query chunk sweep changed attention semantics"
                        )
                del output
            routing = {}
            if name != "gqa":
                # Diagnostic tensor construction and scalar transfers remain
                # outside unprofiled timings and profiler captures.
                with torch.inference_mode():
                    data = gpu_layer.inspect_routing(gpu_tokens, gpu_valid)
                for key in (
                    "packed_blocks",
                    "packed_query_slots",
                    "host_size_syncs",
                    "active_packed_blocks",
                    "score_elements",
                    "peak_score_elements",
                ):
                    if key in data:
                        routing[key] = int(data[key].item())
                if "packed_query_slots" in routing:
                    routing["valid_query_slots"] = (
                        int(gpu_valid.sum().item()) * cfg.query_heads
                    )
                    routing["packed_padding_fraction"] = 1.0 - routing[
                        "valid_query_slots"
                    ] / max(1, routing["packed_query_slots"])
            gpu_timing = _timings(gpu_layer, gpu_tokens, gpu_valid, args, torch)
            print(
                f"N={length} {name} chunk={chunk}: CPU={cpu_timing['cpu_wall']['median_ms']:.3f}ms CUDAwall={gpu_timing['synchronized_wall']['median_ms']:.3f}ms host={gpu_timing['host_dispatch_wall']['median_ms']:.3f}ms event={gpu_timing['cuda_event_elapsed']['median_ms']:.3f}ms",
                file=sys.stderr,
                flush=True,
            )
            diagnostic = _profile(gpu_layer, gpu_tokens, gpu_valid, args, torch)
            assert not gpu_layer._forward_hooks and not gpu_layer._forward_pre_hooks
            assert all(
                not getattr(gpu_layer, part)._forward_hooks
                and not getattr(gpu_layer, part)._forward_pre_hooks
                for part in ("query", "key", "value", "output")
            )
            rows.append(
                {
                    "backend": name,
                    "length": length,
                    "query_chunk": chunk,
                    "mita_execution": execution if name != "gqa" else None,
                    "routing_diagnostics_outside_timing": routing,
                    "cpu_dtype": "float32",
                    "cuda_dtype": args.dtype,
                    "chunk_equivalence_max_abs": equivalence,
                    "unprofiled_cpu": cpu_timing,
                    "unprofiled_cuda": gpu_timing,
                    "diagnostic_profile": diagnostic,
                }
            )
            del cpu_layer, gpu_layer
    return {
        "status": "complete",
        "harness_version": 2,
        "meta": {
            "torch_version": str(torch.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "kineto_available": torch.profiler.kineto_available(),
            "supported_profiler_activities": sorted(
                str(activity) for activity in torch.profiler.supported_activities()
            ),
            "threads": args.threads,
            "seed": args.seed,
            "config": asdict(cfg),
            "repeats": args.repeats,
            "warmup": args.warmup,
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "scripts/profile_mita.py",
                    "jitrl/neural/attention.py",
                    "jitrl/neural/mita_grouped.py",
                    "jitrl/neural/config.py",
                )
            },
        },
        "limitations": [
            "Only actual production modules are executed. Diagnostic projection hooks are present only inside profiler captures and removed afterwards.",
            "Profiler CPU/CUDA/memory instrumentation adds substantial overhead. Use separate unprofiled samples for latency conclusions.",
            "CUDA events include stream idle time while the CPU dispatches operations; they do not equal summed active kernel durations.",
            "Profile counts and allocation volume cover the stated profile_samples, not one forward. Signed net memory and allocation-event volume are not allocator peaks.",
            "Fixed synthetic token workload and single machine; no Agent task-quality or universal speed claim.",
        ],
        "results": rows,
    }


def main() -> int:
    args = _arguments()
    result = run(args)
    serialized = (
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(f"{result['status']}: {args.output}")
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
