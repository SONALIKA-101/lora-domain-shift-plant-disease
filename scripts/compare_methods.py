"""
Compares two trained checkpoints (e.g. doga vs lora_plain) on the same
held-out domain with proper statistical analysis, not just a bare accuracy
difference.

Usage (run from the repository root):
    python scripts/compare_methods.py --config configs/base.yaml \\
        --checkpoint_a checkpoints/rotation_plantwild/doga_holdout-plantwild_best.pt --mode_a doga \\
        --checkpoint_b checkpoints/rotation_plantwild/lora_plain_holdout-plantwild_best.pt --mode_b lora_plain

The held-out domain comes from data.held_out_domain in the config and must match
the domain the checkpoints were trained with. Checkpoints written by
scripts/run_all_rotations.py are stored in checkpoints/rotation_<domain>/;
runs started directly with src.train write to checkpoints/.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score

from src.data import build_domain_split
from src.model import DoGAModel
from src.stats_utils import mcnemar_test, bootstrap_ci, generalization_gap


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels, _domain_ids in loader:
        images = images.to(device)
        out = model(images)
        preds = out["class_logits"].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


def macro_f1_fn(labels, preds):
    return f1_score(labels, preds, average="macro")


def load_model(cfg, mode, checkpoint_path, num_classes, num_domains, device):
    model = DoGAModel(
        backbone_name=cfg["model"]["backbone"], num_classes=num_classes,
        num_domains=num_domains, lora_cfg=cfg["lora"], mode=mode,
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--checkpoint_a", type=str, required=True)
    parser.add_argument("--mode_a", type=str, required=True,
                         choices=["linear_probe", "lora_plain", "full_finetune", "doga"])
    parser.add_argument("--checkpoint_b", type=str, required=True)
    parser.add_argument("--mode_b", type=str, required=True,
                         choices=["linear_probe", "lora_plain", "full_finetune", "doga"])
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = build_domain_split(cfg)

    model_a = load_model(cfg, args.mode_a, args.checkpoint_a, split.num_classes, split.num_train_domains, device)
    model_b = load_model(cfg, args.mode_b, args.checkpoint_b, split.num_classes, split.num_train_domains, device)

    preds_a, labels = get_predictions(model_a, split.held_out_loader, device)
    preds_b, _ = get_predictions(model_b, split.held_out_loader, device)

    val_preds_a, val_labels = get_predictions(model_a, split.val_loader, device)
    val_preds_b, _ = get_predictions(model_b, split.val_loader, device)

    print(f"\n{'='*70}")
    print(f"Comparing {args.mode_a} vs {args.mode_b} on held-out domain: {cfg['data']['held_out_domain']}")
    print(f"{'='*70}\n")

    for name, preds in [(args.mode_a, preds_a), (args.mode_b, preds_b)]:
        acc, acc_lo, acc_hi = bootstrap_ci(preds, labels, accuracy_score, args.n_bootstrap)
        f1, f1_lo, f1_hi = bootstrap_ci(preds, labels, macro_f1_fn, args.n_bootstrap)
        print(f"{name}:")
        print(f"  Accuracy: {acc:.4f}  (95% CI: [{acc_lo:.4f}, {acc_hi:.4f}])")
        print(f"  Macro-F1: {f1:.4f}  (95% CI: [{f1_lo:.4f}, {f1_hi:.4f}])")

    print(f"\n--- McNemar's test: {args.mode_a} vs {args.mode_b} (paired, same held-out samples) ---")
    result = mcnemar_test(preds_a, preds_b, labels)
    print(f"  {args.mode_a} uniquely correct on {result['b10']} samples that {args.mode_b} got wrong")
    print(f"  {args.mode_b} uniquely correct on {result['b01']} samples that {args.mode_a} got wrong")
    print(f"  Test used: {result['test_used']}")
    print(f"  p-value: {result['p_value']:.4f}", end="")
    if result['p_value'] < 0.05:
        print(f"  -> statistically significant difference (p < 0.05)")
    else:
        print(f"  -> NOT statistically significant at p < 0.05")

    # This gap is accuracy-based; Eq. (6) in the paper uses macro-F1 (see README).
    print(f"\n--- Generalization gap (val accuracy - held-out accuracy) ---")
    for name, preds, val_p in [(args.mode_a, preds_a, val_preds_a), (args.mode_b, preds_b, val_preds_b)]:
        val_acc = accuracy_score(val_labels, val_p)
        test_acc = accuracy_score(labels, preds)
        gap = generalization_gap(val_acc, test_acc)
        print(f"  {name}: val_acc={val_acc:.4f}, held_out_acc={test_acc:.4f}, gap={gap:.4f} "
              f"(smaller gap = better generalization)")

    print(f"\nReminder: n=1 rotation here. For the across-rotation mean+/-std,")
    print(f"use scripts/run_all_rotations.py - that number is descriptive (n=3),")
    print(f"not a substitute for this within-rotation significance test.")


if __name__ == "__main__":
    main()
