"""
Runs all 3 leave-one-domain-out rotations x all 4 methods (12 runs in total)
and aggregates the mean and the sample standard deviation (n-1) of the
held-out macro-F1 per method.

    held out = plantvillage  (train on plantdoc + plantwild)
    held out = plantdoc      (train on plantvillage + plantwild)
    held out = plantwild     (train on plantvillage + plantdoc)

Usage (run from repo root, after data/plantvillage, data/plantdoc,
data/plantwild are all populated):

    python scripts/run_all_rotations.py --config configs/base.yaml

Results are saved to results/rotation_results.json and a summary table is
printed at the end. Each run also appends a row to results/summary.csv (written
by src/train.py), and its checkpoint is stored under
checkpoints/rotation_<held_out_domain>/. The script can be re-run to resume:
completed runs are skipped on the next invocation (checked via the results file).
"""

import argparse
import copy
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

from src.train import train as train_fn

METHODS = ["linear_probe", "lora_plain", "full_finetune", "doga"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--results_path", type=str, default="results/rotation_results.json")
    parser.add_argument("--methods", type=str, nargs="+", default=METHODS,
                         help=f"Subset of methods to run, default all: {METHODS}")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    domain_names = [d["name"] for d in base_cfg["data"]["domains"]]
    if len(domain_names) < 3:
        raise ValueError(
            f"Only {len(domain_names)} domains configured ({domain_names}). "
            f"True leave-one-domain-out rotation needs 3+ domains - add plantwild "
            f"via scripts/integrate_plantwild.py first."
        )

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)

    # Load any already-completed results so this script is resumable
    if os.path.exists(args.results_path):
        with open(args.results_path, "r") as f:
            results = json.load(f)
        print(f"Resuming - {sum(len(v) for v in results.values())} runs already completed.")
    else:
        results = {method: {} for method in args.methods}

    total_runs = len(args.methods) * len(domain_names)
    done_runs = sum(len(v) for v in results.values())
    print(f"Plan: {len(args.methods)} methods x {len(domain_names)} rotations = {total_runs} runs "
          f"({done_runs} already done, {total_runs - done_runs} remaining)\n")

    for held_out in domain_names:
        for method in args.methods:
            results.setdefault(method, {})
            if held_out in results[method]:
                print(f"[skip - already done] method={method} held_out={held_out}: "
                      f"{results[method][held_out]:.4f}")
                continue

            print(f"\n{'='*70}")
            print(f"RUN: method={method} | held_out_domain={held_out}")
            print(f"{'='*70}")

            run_cfg = copy.deepcopy(base_cfg)
            run_cfg["data"]["held_out_domain"] = held_out
            run_cfg["experiment"]["checkpoint_dir"] = os.path.join(
                base_cfg["experiment"]["checkpoint_dir"], f"rotation_{held_out}"
            )

            try:
                held_out_f1 = train_fn(run_cfg, mode_override=method)
                results[method][held_out] = held_out_f1
            except Exception as e:
                print(f"FAILED: method={method} held_out={held_out}: {e}")
                results[method][held_out] = None

            # save after every run so progress is never lost if something crashes later
            with open(args.results_path, "w") as f:
                json.dump(results, f, indent=2)

    print(f"\n\n{'='*70}")
    print("FINAL SUMMARY - mean +/- std held-out macro-F1 across all rotations")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Mean':>8} {'Std':>8}   Per-rotation scores")
    for method in args.methods:
        scores = [v for v in results.get(method, {}).values() if v is not None]
        if not scores:
            print(f"{method:<20} {'N/A':>8} {'N/A':>8}   (no successful runs)")
            continue
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if len(scores) > 1 else 0.0
        detail = ", ".join(f"{d}={results[method].get(d, 'N/A')}" for d in domain_names)
        print(f"{method:<20} {mean:>8.4f} {std:>8.4f}   {detail}")

    print(f"\nFull results saved to {args.results_path}")
    print("(std is the sample standard deviation over the rotations, n-1.)")


if __name__ == "__main__":
    main()
