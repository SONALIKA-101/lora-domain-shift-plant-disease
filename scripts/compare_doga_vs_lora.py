"""
Runs McNemar's test and bootstrap CIs comparing doga vs lora_plain, per
held-out rotation, using the saved predictions in results/preds/.

Run from the repository root:
    python -m scripts.compare_doga_vs_lora

In the McNemar detail, b01 counts held-out images that only lora_plain
classifies correctly and b10 those that only doga classifies correctly
(doga is the first argument of mcnemar_test).
"""

import numpy as np
from sklearn.metrics import f1_score

from src.stats_utils import mcnemar_test, bootstrap_ci

ROTATIONS = ["plantvillage", "plantdoc", "plantwild"]
PRED_DIR = "results/preds"


def macro_f1(labels, preds):
    return f1_score(labels, preds, average="macro")


def main():
    print(f"{'Rotation':<14} {'lora_plain F1':>14} {'doga F1':>10} "
          f"{'diff':>8} {'McNemar p':>11} {'sig?':>6}")
    print("-" * 70)

    for domain in ROTATIONS:
        labels = np.load(f"{PRED_DIR}/holdout-{domain}_labels.npy")
        lora_preds = np.load(f"{PRED_DIR}/lora_plain_holdout-{domain}_preds.npy")
        doga_preds = np.load(f"{PRED_DIR}/doga_holdout-{domain}_preds.npy")

        lora_f1 = macro_f1(labels, lora_preds)
        doga_f1 = macro_f1(labels, doga_preds)

        mcnemar_result = mcnemar_test(doga_preds, lora_preds, labels)

        sig = "yes" if mcnemar_result["p_value"] < 0.05 else "no"
        print(f"{domain:<14} {lora_f1:>14.4f} {doga_f1:>10.4f} "
              f"{doga_f1 - lora_f1:>+8.4f} {mcnemar_result['p_value']:>11.4f} {sig:>6}")
        print(f"  McNemar detail: b01(lora-only-right)={mcnemar_result['b01']}, "
              f"b10(doga-only-right)={mcnemar_result['b10']}, "
              f"n_discordant={mcnemar_result['n_discordant']}, "
              f"test={mcnemar_result['test_used']}")

        lora_point, lora_lo, lora_hi = bootstrap_ci(lora_preds, labels, macro_f1, n_bootstrap=1000)
        doga_point, doga_lo, doga_hi = bootstrap_ci(doga_preds, labels, macro_f1, n_bootstrap=1000)
        print(f"  95% CI  lora_plain: [{lora_lo:.4f}, {lora_hi:.4f}]   "
              f"doga: [{doga_lo:.4f}, {doga_hi:.4f}]")
        print()


if __name__ == "__main__":
    main()
