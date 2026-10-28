"""Compare the current encoder/backbone with a trusted local Git release.

Requires optional PyTorch and the reference tag in the local repository.
Historical source is executed locally; use only a trusted reference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch

    from jitrl.neural import model as current
    from jitrl.neural.config import LensConfig
    from jitrl.neural.contracts import RetrievalSnapshot
    from scripts.bench_neural_pipeline import NOW, memories, state

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="v0.2.2")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    source = subprocess.check_output(
        ["git", "show", f"{args.reference}:jitrl/neural/model.py"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    )
    old = {"__name__": "jitrl.neural._trusted_reference", "__package__": "jitrl.neural"}
    exec(compile(source, "reference/model.py", "exec"), old)
    torch.set_num_threads(1)
    cfg = LensConfig()
    states, snapshots = [], []
    for i, count in enumerate((0, 1, 7, 128)):
        query = replace(
            state(900000 + i, NOW + i),
            scope=f"scope-{i % 2}",
            events=tuple(f"event:{j}" for j in range((0, 1, 9, 32)[i])),
        )
        history = tuple(
            replace(m, state=replace(m.state, scope=query.scope))
            for m in memories(count)
        )
        states.append(query)
        snapshots.append(
            RetrievalSnapshot(f"gold-{i}", query.scope, query.observed_at, history)
        )
    candidates = [[], ["tool:search"], ["tool:search", "tool:verify"], ["tool:verify"]]
    pre = ["NO_OP", "SELECT_EVIDENCE", "NO_OP", "SELECT_EVIDENCE"]
    cases = []
    for dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        for indices in ([0], [3], [0, 1, 2, 3]):
            arguments = (
                [states[i] for i in indices],
                [snapshots[i] for i in indices],
                [candidates[i] for i in indices],
                [pre[i] for i in indices],
                cfg,
            )
            before, after = (
                old["make_batch"](*arguments),
                current.make_batch(*arguments),
            )
            assert before.keys() == after.keys()
            for key in before:
                assert (
                    before[key].dtype == after[key].dtype
                    and before[key].shape == after[key].shape
                ), key
                assert torch.equal(
                    before[key].contiguous().view(torch.uint8),
                    after[key].contiguous().view(torch.uint8),
                ), key
            cases.append(
                {
                    "dtype": str(dtype),
                    "rows": indices,
                    "all_fields_byte_identical": True,
                }
            )
    torch.set_default_dtype(torch.float32)
    arguments = (states, snapshots, candidates, pre, cfg)
    before, after = old["make_batch"](*arguments), current.make_batch(*arguments)
    outputs = []
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        prior, updated = (
            old["create_model"](cfg).to(device),
            current.create_model(cfg).to(device),
        )
        updated.load_state_dict(prior.state_dict(), strict=True)
        assert prior.attention_inventory() == updated.attention_inventory()
        with torch.inference_mode():
            a = prior({k: v.to(device) for k, v in before.items()})
            b = updated({k: v.to(device) for k, v in after.items()})
        error = max(float((a[k] - b[k]).abs().max()) for k in a)
        for key in a:
            torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=2e-5)
        outputs.append(
            {
                "device": device,
                "max_abs_output_error": error,
                "reference_state_dict_loads_strictly": True,
                "attention": updated.attention_inventory(),
            }
        )
    assert (
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                args.reference,
                "--",
                "jitrl/neural/attention.py",
                "jitrl/neural/mita_grouped.py",
            ],
            cwd=ROOT,
        ).returncode
        == 0
    )
    report = {
        "reference": args.reference,
        "torch": torch.__version__,
        "encoding_cases": cases,
        "outputs": outputs,
        "mita_kernel_sources_unchanged": True,
        "boundary": "identical historical inputs and checkpoint weights; retrieval/gating policies intentionally changed",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
