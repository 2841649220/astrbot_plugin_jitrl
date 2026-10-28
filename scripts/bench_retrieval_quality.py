"""Deterministic exact-recall and dense-index benchmarks; no remote model calls.

Run the same harness before and after a change. Timings cover ExperienceIndex,
not the host, SQLite, or a model; recall is against an exhaustive Jaccard oracle.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jitrl.index import ExperienceIndex  # noqa: E402

NOW = 1_800_000_000.0


def scenario(name, size, *, mixed=False, scope="s1", mode="global", repeats=25):
    query = frozenset(f"q{i}" for i in range(100))
    index = ExperienceIndex()
    rows = []
    for entry_id in range(1, size + 1):
        if mixed and entry_id <= 10:
            tokens = query - {f"q{entry_id}"}
        elif mixed:
            tokens = query | frozenset(f"extra{i}" for i in range(90))
        else:
            tokens = query
        created_at = NOW + entry_id
        index.append(
            entry_id=entry_id,
            tokens=tokens,
            scope="s1",
            surface="web",
            action_key="tool:search()",
            discounted_return=1.0,
            reward=1.0,
            created_at=created_at,
        )
        if mode == "session" and scope and scope != "s1":
            continue
        similarity = len(query & tokens) / len(query | tokens)
        if similarity < 0.5:
            continue
        if mode == "hybrid" and scope and scope != "s1":
            similarity *= 0.85
        rows.append((similarity, created_at, entry_id))
    expected = sorted(rows, reverse=True)[:10]
    kwargs = {
        "k": 10,
        "threshold": 0.5,
        "scope": scope,
        "scope_mode": mode,
        "prefilter_limit": 32 if mixed else 4000,
        "recency_half_life_days": 30.0,
        "now": NOW,
    }
    for _ in range(3):
        index.search(query, **kwargs)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        hits = index.search(query, **kwargs)
        samples.append((time.perf_counter() - start) * 1000)
    actual = [(hit.similarity, hit.created_at, hit.entry_id) for hit in hits]
    expected_ids = {item[2] for item in expected}
    actual_ids = {item[2] for item in actual}
    tracemalloc.start()
    index.search(query, **kwargs)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "name": name,
        "n": size,
        "p50_ms": round(statistics.median(samples), 4),
        "p95_ms": round(sorted(samples)[int(0.95 * (len(samples) - 1))], 4),
        "query_peak_bytes": peak,
        "recall_at_10": len(expected_ids & actual_ids) / len(expected_ids)
        if expected_ids
        else float(not actual_ids),
        "exact_ranking": actual == expected,
        "expected_ids": [item[2] for item in expected],
        "actual_ids": [item[2] for item in actual],
        "parameters": kwargs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=25)
    args = parser.parse_args()
    rows = [
        scenario("dense_ties", n, repeats=args.repeats) for n in (1000, 5000, 20000)
    ]
    rows.extend(
        [
            scenario(
                "padded_decoys_small_budget", 5000, mixed=True, repeats=args.repeats
            ),
            scenario(
                "unseen_session",
                1000,
                scope="new",
                mode="session",
                repeats=args.repeats,
            ),
            scenario(
                "unseen_hybrid", 1000, scope="new", mode="hybrid", repeats=args.repeats
            ),
        ]
    )
    report = {
        "meta": {
            "harness": "scripts/bench_retrieval_quality.py",
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "repeats": args.repeats,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "boundary": "synthetic index-only workload; no task success or remote-logit measurement",
        },
        "scenarios": rows,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for row in rows:
        print(
            f"{row['name']:32} N={row['n']:5} p50={row['p50_ms']:9.4f} ms recall={row['recall_at_10']:.2f} exact={row['exact_ranking']}"
        )


if __name__ == "__main__":
    main()
