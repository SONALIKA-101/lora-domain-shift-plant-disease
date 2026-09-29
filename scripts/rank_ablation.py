"""
LoRA rank ablation for DoGA (Table 7 in the paper): one rotation (PlantWild
held out, the same rotation as the lambda sweep), varying lora.r while every
other setting from the config stays fixed, including lora.alpha. Because alpha
is fixed, the LoRA scaling alpha/r changes with the rank.

Usage (run from the repository root):
    python scripts/rank_ablation.py --config configs/base.yaml --ranks 4 8 16 32

Results are written to results/rank_ablation.csv; each run's checkpoint is
stored under checkpoints/rank_ablation_r<rank>/.
"""

import argparse
import copy
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

from src.train import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32])
    parser.add_argument("--held_out_domain", type=str, default="plantwild",
                         help="Matches the lambda sweep's rotation for direct comparability")
    parser.add_argument("--mode", type=str, default="doga",
                         choices=["lora_plain", "doga"],
                         help="Method to ablate the rank for (default: doga)")
    parser.add_argument("--results_path", type=str, default="results/rank_ablation.csv")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.results_path), exist_ok=True)
    rows = []

    for r in args.ranks:
        print(f"\n{'='*70}")
        print(f"RANK ABLATION: r={r} | mode={args.mode} | held_out={args.held_out_domain}")
        print(f"{'='*70}")

        run_cfg = copy.deepcopy(base_cfg)
        run_cfg["data"]["held_out_domain"] = args.held_out_domain
        run_cfg["lora"]["r"] = r
        # lora.alpha is kept at its config value (16), so the scaling alpha/r
        # changes with r. To scale alpha with the rank instead, uncomment:
        # run_cfg["lora"]["alpha"] = 2 * r
        run_cfg["experiment"]["checkpoint_dir"] = os.path.join(
            base_cfg["experiment"]["checkpoint_dir"], f"rank_ablation_r{r}"
        )

        try:
            held_out_f1 = train(run_cfg, mode_override=args.mode)
            rows.append({"rank": r, "mode": args.mode, "held_out_domain": args.held_out_domain,
                         "held_out_macro_f1": round(held_out_f1, 4), "status": "ok"})
        except Exception as e:
            print(f"FAILED: rank={r}: {e}")
            rows.append({"rank": r, "mode": args.mode, "held_out_domain": args.held_out_domain,
                         "held_out_macro_f1": None, "status": f"failed: {e}"})

        with open(args.results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "mode", "held_out_domain", "held_out_macro_f1", "status"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n{'='*70}")
    print("RANK ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Rank':<8} {'Held-out macro-F1':<20} {'Status'}")
    for row in rows:
        f1_str = f"{row['held_out_macro_f1']:.4f}" if row["held_out_macro_f1"] is not None else "N/A"
        print(f"{row['rank']:<8} {f1_str:<20} {row['status']}")
    print(f"\nSaved to {args.results_path}")
    print("Only lora.r is varied between runs; lora.alpha stays at its config value.")


if __name__ == "__main__":
    main()
