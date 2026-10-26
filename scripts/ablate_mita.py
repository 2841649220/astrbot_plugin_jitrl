#!/usr/bin/env python
"""Train matched local backbones on a synthetic memory-conditioned rule task.

Rule bindings change every episode. Paired examples have identical observable
query states and executed actions but complementary historical rulebooks and
opposite observed returns. Held-out episodes use unseen complete binding maps.
Only executed-action returns enter train_model; full candidate ground truth is
used exclusively for evaluation. This is not an Agent performance benchmark.

Usage::

    python scripts/ablate_mita.py --json benchmarks/mita-memory-ablation.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KEYS = (
    "amber",
    "birch",
    "cobalt",
    "delta",
    "ember",
    "frost",
    "garnet",
    "hazel",
    "indigo",
    "jade",
)
ACTIONS = ("tool:alpha", "tool:beta")
HARNESS_VERSION = 1


@dataclass(frozen=True)
class Example:
    sample: Any
    correct_index: int
    observed_index: int
    binding: tuple[int, ...]
    query_key: str


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _seeds(value: str) -> list[int]:
    result = [_positive(part.strip()) for part in value.split(",")]
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("seeds must be distinct")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=_seeds, default=[1729, 2718])
    parser.add_argument("--steps", type=_positive, default=100)
    parser.add_argument("--train-examples", type=_positive, default=128)
    parser.add_argument("--eval-examples", type=_positive, default=64)
    parser.add_argument("--batch-size", type=_positive, default=16)
    parser.add_argument("--threads", type=_positive, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--backends", default="mita,gqa,hybrid")
    parser.add_argument(
        "--json", type=Path, default=ROOT / "benchmarks/mita-memory-ablation.json"
    )
    args = parser.parse_args()
    args.backends = args.backends.split(",")
    if (
        not args.backends
        or not set(args.backends) <= {"mita", "gqa", "hybrid"}
        or len(args.backends) != len(set(args.backends))
    ):
        parser.error("--backends requires distinct choices from mita,gqa,hybrid")
    if args.train_examples % 2 or args.eval_examples % 2:
        parser.error("paired train/evaluation example counts must be even")
    if args.train_examples + args.eval_examples > math.comb(len(KEYS), len(KEYS) // 2):
        parser.error("not enough distinct balanced rulebooks for requested split")
    if not math.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 0.01:
        parser.error("--learning-rate must be finite and in (0,0.01]")
    return args


def dataset(seed: int, train_count: int, eval_count: int) -> tuple[list, list, dict]:
    from jitrl.neural.contracts import (
        Intervention,
        LensDecision,
        MemoryRecord,
        ReplaySample,
        RetrievalSnapshot,
    )
    from jitrl.neural.encoding import canonicalize, digest

    rng = random.Random(seed)
    maps = []
    for selected in itertools.combinations(range(len(KEYS)), len(KEYS) // 2):
        binding = tuple(int(index in selected) for index in range(len(KEYS)))
        inverse = tuple(1 - item for item in binding)
        if binding < inverse:
            maps.append(binding)
    rng.shuffle(maps)
    scope = "synthetic-rulebook-private-scope"
    environment = "synthetic-rulebook-v1"

    def build(split: str, bindings: list, offset: int) -> list[Example]:
        examples = []
        for pair, assignment in enumerate(bindings):
            query_index = rng.randrange(len(KEYS))
            observed_index = rng.randrange(len(ACTIONS))
            # Pair members share time and all current-state features. Only
            # their independently observed historical rulebooks differ.
            timestamp = float(offset + pair * 1000)
            for variant, binding in enumerate(
                (assignment, tuple(1 - item for item in assignment))
            ):
                identifier = f"{split}-pair-{pair}-variant-{variant}"

                def state(code: str, name: str, *, at: float, episode: str):
                    return canonicalize(
                        scope=scope,
                        goal="Choose the successful action for the named code",
                        observation=f"query code {code}",
                        events=("rulebook:available", f"code:{code}", "choice:pending"),
                        allowed_actions=("alpha", "beta"),
                        environment=environment,
                        episode_id=episode,
                        decision_id=name,
                        observed_at=at,
                    )

                current = state(
                    KEYS[query_index],
                    f"decision-{identifier}",
                    at=timestamp,
                    episode=f"episode-{identifier}",
                )
                memories = []
                for code_index, code in enumerate(KEYS):
                    for action_index, action_key in enumerate(ACTIONS):
                        historical_id = f"history-{identifier}-{code}-{action_index}"
                        historical = state(
                            code,
                            f"decision-{historical_id}",
                            at=timestamp - 200,
                            episode=f"episode-{historical_id}",
                        )
                        memories.append(
                            MemoryRecord(
                                experience_id=historical_id,
                                state=historical,
                                action_key=action_key,
                                action_name=("alpha", "beta")[action_index],
                                intervention="NO_OP",
                                return_g=3.0
                                if action_index == binding[code_index]
                                else -3.0,
                                reward_source="environment",
                                reliability=1.0,
                                settled_at=timestamp - 100,
                                similarity=0.5,
                            )
                        )
                rng.shuffle(memories)
                snapshot = RetrievalSnapshot(
                    f"snapshot-{identifier}", scope, timestamp, tuple(memories)
                )
                correct_index = binding[query_index]
                # This is the one actually selected synthetic environment
                # action. Neither of the other candidate returns is added to
                # ReplaySample or to train_model's features/labels.
                observed_return = 3.0 if observed_index == correct_index else -3.0
                sample = ReplaySample(
                    experience_id=f"experience-{identifier}",
                    decision=LensDecision(current, snapshot, Intervention()),
                    action_key=ACTIONS[observed_index],
                    return_g=observed_return,
                    reliability=1.0,
                    settled_at=timestamp + 1,
                )
                examples.append(
                    Example(
                        sample,
                        correct_index,
                        observed_index,
                        binding,
                        KEYS[query_index],
                    )
                )
        return examples

    train_pairs, eval_pairs = train_count // 2, eval_count // 2
    train = build("train", maps[:train_pairs], 1000)
    evaluation = build(
        "evaluation", maps[train_pairs : train_pairs + eval_pairs], 1_000_000
    )
    if {item.binding for item in train} & {item.binding for item in evaluation}:
        raise AssertionError("rulebook binding leaked across train/evaluation split")
    for examples in (train, evaluation):
        for index in range(0, len(examples), 2):
            first, second = examples[index : index + 2]
            if (
                first.sample.decision.state.state_hash
                != second.sample.decision.state.state_hash
                or first.sample.action_key != second.sample.action_key
                or first.sample.return_g != -second.sample.return_g
                or first.correct_index == second.correct_index
            ):
                raise AssertionError("paired construction leaked a state-only shortcut")
    report = {
        "seed": seed,
        "train_examples": len(train),
        "evaluation_examples": len(evaluation),
        "train_fingerprint": digest([asdict(example) for example in train]),
        "evaluation_fingerprint": digest([asdict(example) for example in evaluation]),
        "heldout_unit": "complete binding assignment and episode; code vocabulary is shared",
        "binding_assignment_overlap": 0,
        "keys": list(KEYS),
        "memory_records_per_snapshot": 2 * len(KEYS),
        "rulebook_positive_binding_counts": {"alpha": 5, "beta": 5},
        "train_positive_observed_returns": sum(
            item.sample.return_g > 0 for item in train
        ),
        "evaluation_positive_observed_returns": sum(
            item.sample.return_g > 0 for item in evaluation
        ),
        "zero_return_predictor_evaluation_mse": 9.0,
        "state_only_best_pair_balanced_ranking_accuracy": 0.5,
        "label_policy": "train_model receives one executed action and its +/-3 observed return; both-candidate ground truth exists only in evaluation metadata",
    }
    return train, evaluation, report


def evaluate(
    model: Any, cfg: Any, examples: list[Example], mode: str = "memory"
) -> dict:
    import torch

    from jitrl.neural.model import make_batch

    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(examples), cfg.batch_size):
            chunk = examples[start : start + cfg.batch_size]
            snapshots = []
            for local, example in enumerate(chunk):
                snapshot = example.sample.decision.snapshot
                if mode == "empty_memory":
                    snapshot = replace(snapshot, memories=())
                elif mode == "complemented_rulebook":
                    paired = examples[(start + local) ^ 1].sample.decision.snapshot
                    snapshot = replace(snapshot, memories=paired.memories)
                snapshots.append(snapshot)
            batch = make_batch(
                [item.sample.decision.state for item in chunk],
                snapshots,
                [list(ACTIONS) for _ in chunk],
                ["NO_OP"] * len(chunk),
                cfg,
            )
            values = model(batch)["candidate_values"] * cfg.return_scale
            if not torch.isfinite(values).all():
                raise ValueError("nonfinite candidate prediction")
            for example, scores in zip(chunk, values.tolist(), strict=True):
                predicted_index = int(scores[1] > scores[0])
                tie = abs(scores[0] - scores[1]) <= 1e-8
                predictions.append(
                    {
                        "experience_id": example.sample.experience_id,
                        "query_key": example.query_key,
                        "observed_action_index": example.observed_index,
                        "correct_action_index_evaluation_only": example.correct_index,
                        "observed_return": example.sample.return_g,
                        "candidate_return_predictions": scores,
                        "ranking_credit": 0.5
                        if tie
                        else float(predicted_index == example.correct_index),
                    }
                )
    return {
        "condition": mode,
        "observed_return_mse": statistics.mean(
            (
                row["candidate_return_predictions"][row["observed_action_index"]]
                - row["observed_return"]
            )
            ** 2
            for row in predictions
        ),
        "candidate_ranking_accuracy": statistics.mean(
            row["ranking_credit"] for row in predictions
        ),
        "tie_rule": "0.5 credit when candidate return scores differ by at most 1e-8",
        "predictions": predictions,
    }


def weight_fingerprint(model: Any) -> str:
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode("utf-8"))
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def experiment(args: argparse.Namespace) -> dict:
    import torch

    from jitrl.neural.config import LensConfig
    from jitrl.neural.learner import train_model
    from jitrl.neural.model import create_model

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rows, reports = [], []
    for seed in args.seeds:
        train, evaluation, report = dataset(
            seed, args.train_examples, args.eval_examples
        )
        reports.append(report)
        base = replace(
            LensConfig(),
            hidden_dim=48,
            embedding_dim=64,
            summary_dim=32,
            blocks=4,
            engram_buckets=127,
            engram_dim=4,
            bases=4,
            rank=4,
            max_events=8,
            min_samples=2,
            max_memory=20,
            coarse_limit=20,
            memory_tokens=20,
            mita_landmarks=4,
            mita_topk=4,
            mita_query_chunk=32,
            batch_size=args.batch_size,
            train_steps=args.steps,
            learning_rate=args.learning_rate,
            seed=seed,
        )
        initial_fingerprint = None
        for backend in args.backends:
            cfg = replace(base, attention_backend=backend)
            model = create_model(cfg)
            fingerprint = weight_fingerprint(model)
            if initial_fingerprint is None:
                initial_fingerprint = fingerprint
            elif fingerprint != initial_fingerprint:
                raise AssertionError(
                    "matched backbones did not share exact initial weights"
                )
            initial = evaluate(model, cfg, evaluation)
            print(
                f"seed={seed} backend={backend}: training {args.steps} updates",
                flush=True,
            )
            started = time.perf_counter()
            trained, metrics = train_model(model, cfg, [item.sample for item in train])
            seconds = time.perf_counter() - started
            delta = math.sqrt(
                sum(
                    float((updated.detach() - original.detach()).square().sum())
                    for original, updated in zip(
                        model.parameters(), trained.parameters(), strict=True
                    )
                )
            )
            conditions = {
                mode: evaluate(trained, cfg, evaluation, mode)
                for mode in ("memory", "empty_memory", "complemented_rulebook")
            }
            row = {
                "seed": seed,
                "backend": backend,
                "config": asdict(cfg),
                "initial_weights_sha256": fingerprint,
                "trained_weights_sha256": weight_fingerprint(trained),
                "parameter_l2_change": delta,
                "parameters": model.parameter_counts(),
                "attention_inventory": model.attention_inventory(),
                "training_seconds_including_encoding": seconds,
                "train_model_metrics": metrics,
                "initial_heldout": initial,
                "heldout_conditions": conditions,
            }
            rows.append(row)
            memory = conditions["memory"]
            empty = conditions["empty_memory"]
            print(
                f"seed={seed} {backend}: train {metrics['before_loss']:.4f}->{metrics['after_loss']:.4f}; heldout MSE={memory['observed_return_mse']:.4f}, ranking={memory['candidate_ranking_accuracy']:.3f}, empty-memory ranking={empty['candidate_ranking_accuracy']:.3f}; {seconds:.2f}s",
                flush=True,
            )
    aggregate = {}
    for backend in args.backends:
        selected = [row for row in rows if row["backend"] == backend]
        aggregate[backend] = {}
        for condition in ("memory", "empty_memory", "complemented_rulebook"):
            aggregate[backend][condition] = {}
            for metric in ("observed_return_mse", "candidate_ranking_accuracy"):
                values = [
                    row["heldout_conditions"][condition][metric] for row in selected
                ]
                aggregate[backend][condition][metric] = {
                    "mean": statistics.mean(values),
                    "sample_std": statistics.stdev(values) if len(values) > 1 else None,
                    "per_seed": values,
                }
    return {
        "harness_version": HARNESS_VERSION,
        "task": "paired episodically randomized code-to-action rulebooks",
        "metadata": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "seeds": args.seeds,
            "all_backends_have_exact_equal_initial_weights_per_seed": True,
            "selection_layer": "raw candidate head; runtime confidence/support gates are not evaluated",
            "source_sha256": {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in (
                    "scripts/ablate_mita.py",
                    "jitrl/neural/attention.py",
                    "jitrl/neural/model.py",
                    "jitrl/neural/learner.py",
                    "jitrl/neural/config.py",
                    "jitrl/neural/contracts.py",
                    "jitrl/neural/encoding.py",
                )
            },
        },
        "protocol": {
            "train_examples": args.train_examples,
            "evaluation_examples": args.eval_examples,
            "updates": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "memory": "20 shuffled historical environment-labelled records; both observed actions for every code; only 5 alpha-positive and 5 beta-positive bindings in every rulebook",
            "pair_balance": "same query state and executed action, complementary binding map and opposite observed return; no deterministic state-only candidate ranking can exceed 0.5 over pairs",
            "training_labels": "one executed action's observed +/-3 reward per current decision; no unexecuted candidate label is passed to train_model",
            "temporal_boundary": "all memory observations and settlements precede the decision, current settlement follows it; evaluation decision timestamps follow training examples",
            "split": "disjoint complete mapping assignments and episode/experience IDs; vocabulary is shared; evaluation targets never select training hyperparameters",
            "evaluation_controls": "empty_memory removes all evidence; complemented_rulebook replaces the evidence by the paired inverse mapping while retaining original targets to probe dependence on bindings",
        },
        "datasets": reports,
        "aggregate": aggregate,
        "results": rows,
        "limitations": [
            "Synthetic memory binding task, not a remote LLM, real Agent or real-user performance result.",
            "Small CPU models, limited updates and two default seeds do not establish a statistically reliable superiority claim.",
            "Ranking ground truth for both candidates is generated by the synthetic environment and used only for evaluation, never training.",
            "Learner also predicts PRE return without knowing the randomly executed action; its auxiliary target has irreducible conditional noise on this task.",
            "Empty/inverted-memory evaluation is a diagnostic distribution shift, not a causal intervention estimate for a deployed Agent.",
            "Training timings include encoding, private model copy, optimizer, objective evaluations and updates; machine load and backend execution order affect latency.",
        ],
    }


def main() -> int:
    args = arguments()
    result = experiment(args)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"JSON: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
