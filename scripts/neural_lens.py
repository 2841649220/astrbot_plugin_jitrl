"""Offline local Neural Lens training and a disposable, synthetic smoke run.

No provider is called and no real plugin memory is modified by ``smoke``.
``train`` requires an explicit existing neural data directory and private scope.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def bootstrap() -> None:
    if importlib.util.find_spec("astrbot") is not None:
        return
    host, api = ModuleType("astrbot"), ModuleType("astrbot.api")
    host.__path__ = []
    api.logger = SimpleNamespace(
        warning=lambda message, *args, **kwargs: print(message % args, file=sys.stderr)
    )
    host.api = api
    sys.modules["astrbot"] = host
    sys.modules["astrbot.api"] = api


async def smoke(steps: int, backend: str = "mita", device: str = "cpu") -> dict:
    import torch

    from jitrl.neural.config import LensConfig
    from jitrl.neural.contracts import CandidateSpec
    from jitrl.neural.encoding import canonicalize
    from jitrl.neural.runtime import LensRuntime

    # The demonstration uses the shipped 384d/4-block architecture. Gates are
    # deliberately permissive so the mechanical PRE/POST paths are exercised;
    # they are NOT recommended deployment thresholds.
    cfg = replace(
        LensConfig(),
        mode="active",
        post_enabled=True,
        min_samples=32,
        train_steps=steps,
        min_support=2,
        max_uncertainty=2.0,
        relevance_threshold=0.0,
        evidence_tokens=256,
        attention_backend=backend,
        device=device,
    )
    scope = "synthetic-private-scope"
    with tempfile.TemporaryDirectory(prefix="jitrl-neural-smoke-") as folder:
        runtime = LensRuntime(Path(folder), cfg)
        try:
            for index in range(40):
                state = canonicalize(
                    scope=scope,
                    goal="Find and verify the report",
                    observation="Report listing available",
                    events=("tool_call:search", "tool_result:observed"),
                    allowed_actions=("verify", "guess", "reply"),
                    episode_id=f"synthetic-episode-{index}",
                )
                decision, epoch = await runtime.prepare(state)
                assert await runtime.commit(decision, epoch)
                action = "verify" if index % 2 else "guess"
                experience_id = runtime.store.append_execution(
                    state.decision_id,
                    index=0,
                    action_key=f"tool:{action}",
                    action_name=action,
                    expected_epoch=epoch,
                )
                reward = 3.0 if action == "verify" else -3.0
                assert runtime.store.settle(
                    experience_id,
                    reward=reward,
                    return_g=reward,
                    source="environment",
                    expected_epoch=epoch,
                )
            trained = await runtime.train(scope)
            if trained["status"] != "trained":
                raise RuntimeError(f"training did not publish: {trained}")
            # Evaluation decisions are created after checkpoint publication.
            # Their labels do not enter the store or the model inputs.
            squared_errors = []
            selections = []
            rendered_bytes = []
            for index in range(4):
                state = canonicalize(
                    scope=scope,
                    goal="Find and verify the report",
                    observation=f"Report listing available, page {index + 1}",
                    events=("tool_call:search", "tool_result:observed"),
                    allowed_actions=("verify", "guess", "reply"),
                    episode_id=f"synthetic-evaluation-{index}",
                )
                decision, epoch = await runtime.prepare(state)
                assert await runtime.commit(decision, epoch)
                candidates = [
                    CandidateSpec("guess", "tool:guess", "guess", 0.0, "RANK_ONLY"),
                    CandidateSpec(
                        "verify", "tool:verify", "verify", -0.05, "RANK_ONLY"
                    ),
                ]
                scores = await runtime.score_candidates(decision, candidates)
                for score in scores:
                    if score.local_value is None:
                        raise RuntimeError("candidate neural path was not exercised")
                    target = 0.5 if score.candidate.action_name == "verify" else -0.5
                    squared_errors.append((score.local_value - target) ** 2)
                    assert (
                        math.isfinite(score.score)
                        and abs(score.adjustment) <= cfg.max_adjustment
                    )
                selections.append(
                    max(scores, key=lambda score: score.score).candidate.candidate_id
                )
                rendered_bytes.append(len(decision.rendered.encode("utf-8")))
            return {
                "evidence": "synthetic-mechanical-smoke-only-no-agent-performance-claim",
                "architecture": {
                    "hidden_dim": cfg.hidden_dim,
                    "blocks": cfg.blocks,
                    "query_heads": cfg.query_heads,
                    "kv_heads": cfg.kv_heads,
                    "bases": cfg.bases,
                    "rank": cfg.rank,
                    "attention_backend": cfg.attention_backend,
                    "mita_landmarks": cfg.mita_landmarks,
                    "mita_topk": cfg.mita_topk,
                    "memory_tokens": cfg.memory_tokens,
                },
                "torch_version": str(torch.__version__),
                "device": trained["device"],
                "training": {k: v for k, v in trained.items() if k != "checkpoint_id"},
                "future_synthetic_candidate_mse": sum(squared_errors)
                / len(squared_errors),
                "synthetic_candidate_selections": selections,
                "rendered_utf8_bytes": rendered_bytes,
                "real_llm_calls": 0,
                "note": "permissive smoke gates; no real task improvement measured",
            }
        finally:
            await runtime.close()


async def run(args: argparse.Namespace) -> dict:
    if args.command == "smoke":
        return await smoke(args.steps, args.backend, args.device)
    from jitrl.neural.config import LensConfig
    from jitrl.neural.runtime import LensRuntime

    root = await asyncio.to_thread(Path(args.data_dir).resolve)
    if not await asyncio.to_thread((root / "neural.sqlite3").is_file):
        raise ValueError(
            "--data-dir must contain an existing neural.sqlite3; legacy databases are not imported"
        )
    cfg = replace(
        LensConfig(),
        train_steps=args.steps,
        attention_backend=args.backend,
        device=args.device,
    )
    runtime = LensRuntime(root, cfg)
    try:
        return (
            await runtime.train(args.scope)
            if args.command == "train"
            else await runtime.status(args.scope)
        )
    finally:
        await runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser(
        "smoke", help="disposable synthetic end-to-end CPU demonstration"
    )
    demo.add_argument("--steps", type=int, default=20)
    demo.add_argument("--json", type=Path)
    demo.add_argument(
        "--backend", choices=("mita", "hybrid", "gqa", "legacy"), default="mita"
    )
    demo.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    for name in ("train", "status"):
        command = sub.add_parser(name)
        command.add_argument("--data-dir", required=True)
        command.add_argument("--scope", required=True)
        command.add_argument("--steps", type=int, default=20)
        command.add_argument(
            "--backend", choices=("mita", "hybrid", "gqa", "legacy"), default="mita"
        )
        command.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    args = parser.parse_args()
    if not 1 <= args.steps <= 1000:
        parser.error("--steps must be in [1,1000]")
    bootstrap()
    try:
        if args.command != "status":
            import torch

            # This is a dedicated CLI process, not the AstrBot host.
            torch.set_num_threads(2)
        result = asyncio.run(run(args))
    except ImportError:
        print(
            "Install optional requirements-neural.txt with the PyTorch CPU wheel index first.",
            file=sys.stderr,
        )
        return 2
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if getattr(args, "json", None):
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n", encoding="utf-8", newline="\n")
    print(output)
    return (
        0
        if result.get("status", "trained") in {"trained", "insufficient_samples"}
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
