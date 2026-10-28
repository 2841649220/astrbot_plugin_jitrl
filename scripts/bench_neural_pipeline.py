"""Synthetic Neural Lens retrieval and CPU feature-encoding benchmarks.

No remote calls, learned-weight changes, or user databases are involved.
Retrieval includes SQLite and snapshot construction. Encoding uses fixed
historical inputs independent of any retrieval-policy changes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_source_parser = argparse.ArgumentParser(add_help=False)
_source_parser.add_argument("--source-root", type=Path, default=ROOT)
SOURCE_ROOT = _source_parser.parse_known_args()[0].source_root.resolve()
sys.path.insert(0, str(SOURCE_ROOT))

from jitrl.neural.config import LensConfig  # noqa: E402
from jitrl.neural.contracts import (  # noqa: E402
    Intervention,
    LensDecision,
    MemoryRecord,
    RetrievalSnapshot,
)
from jitrl.neural.encoding import canonicalize  # noqa: E402
from jitrl.neural.store import LensStore  # noqa: E402

NOW = 1_800_000_000.0


def state(index, at, *, unique=False):
    return canonicalize(
        scope="bench-private",
        decision_id=f"decision-{index}",
        episode_id=f"episode-{index}",
        goal=f"Find and verify report {index if unique else index % 32}: compare the documented result with the source, inspect the details and summarize the evidence for this request.",
        observation=f"report section {index if unique else index % 16} ready for review; check the current search results before choosing a tool",
        events=("tool:search", "result:ok", "tool:verify", "result:ok"),
        allowed_actions=("search", "verify"),
        observed_at=at,
    )


def memories(count):
    return tuple(
        MemoryRecord(
            f"memory-{i}",
            state(1000000 + i, NOW - 200000 - i),
            "tool:search" if i % 2 else "tool:verify",
            "search" if i % 2 else "verify",
            "NO_OP",
            float(i % 7 - 3),
            "environment",
            1.0,
            NOW - 100000 - i,
            0.7,
        )
        for i in range(count)
    )


def timed(function, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - start) * 1000)
    return result, {
        "p50_ms": round(statistics.median(samples), 4),
        "p95_ms": round(sorted(samples)[int(0.95 * (len(samples) - 1))], 4),
    }


def retrieval(root, size, repeats, *, unique=False):
    cfg = LensConfig()
    store = LensStore(root / f"bank-{size}-{unique}.sqlite3")
    history = memories(16)
    try:
        for i in range(size):
            current = state(i, NOW - 10000 + i, unique=unique)
            snapshot = RetrievalSnapshot(
                f"snapshot-{i}", current.scope, current.observed_at, history
            )
            assert store.record_decision(
                LensDecision(current, snapshot, Intervention()), expected_epoch=0
            )
            experience = store.append_execution(
                current.decision_id,
                index=0,
                action_key="tool:search",
                action_name="search",
                expected_epoch=0,
                executed_at=current.observed_at + 0.1,
            )
            assert store.settle(
                experience,
                reward=1,
                return_g=1,
                source="environment",
                expected_epoch=0,
                settled_at=current.observed_at + 0.2,
            )
        query = state(999999, NOW, unique=unique)
        _, cold = timed(lambda: store.retrieve(query, cfg), 1)
        result, warm = timed(lambda: store.retrieve(query, cfg), repeats)
        tracemalloc.start()
        store.retrieve(query, cfg)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "name": "retrieve_unique" if unique else "retrieve",
            "rows": size,
            "cold_ms": cold["p50_ms"],
            **warm,
            "returned": len(result.memories),
            "query_peak_bytes": peak,
        }
    finally:
        store.close()


def encoding(repeats):
    import torch

    from jitrl.neural.model import create_model, make_batch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    cfg = LensConfig()
    current = state(999999, NOW)
    snapshot = RetrievalSnapshot(
        "fixed-128", current.scope, current.observed_at, memories(128)
    )
    rows = []
    try:
        for batch_size in (1, 16):

            def run():
                return make_batch(
                    [current] * batch_size,
                    [snapshot] * batch_size,
                    [["tool:search", "tool:verify"]] * batch_size,
                    ["NO_OP"] * batch_size,
                    cfg,
                )

            run()
            _, result = timed(run, repeats)
            rows.append(
                {"name": "make_batch", "batch": batch_size, "memories": 128, **result}
            )
        # Distinct state text, frozen inputs, actual four-layer MiTA on CPU.
        batch = make_batch(
            [current], [snapshot], [["tool:search", "tool:verify"]], ["NO_OP"], cfg
        )
        model = create_model(cfg)
        with torch.inference_mode():
            _, result = timed(lambda: model(batch), repeats)
        rows.append(
            {
                "name": "model_preencoded_cpu",
                "attention": model.attention_inventory(),
                **result,
            }
        )
        return rows
    finally:
        torch.set_num_threads(previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--source-label", default="working-tree")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="jitrl-pipeline-bench-") as folder:
        rows = [retrieval(Path(folder), n, args.repeats) for n in (128, 2048)]
        rows.append(retrieval(Path(folder), 2048, args.repeats, unique=True))
        rows.extend(encoding(args.repeats))
    report = {
        "meta": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "source_label": args.source_label,
            "source_sha256": hashlib.sha256(
                b"".join(
                    (SOURCE_ROOT / "jitrl" / "neural" / name).read_bytes()
                    for name in ("store.py", "model.py", "runtime.py", "encoding.py")
                )
            ).hexdigest(),
            "repeats": args.repeats,
            "synthetic_only": True,
            "threads_for_encoding_and_model": 1,
            "retrieval_workload": "16 memories per historical snapshot; repeated text has 32 templates; unique text uses distinct report/section identifiers; default replay/coarse budgets",
        },
        "scenarios": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
