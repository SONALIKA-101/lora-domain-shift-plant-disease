"""
Standalone evaluation of one checkpoint on the held-out domain, with a per-class
breakdown.

    python -m src.eval --config configs/base.yaml --mode doga \\
        --checkpoint checkpoints/rotation_plantwild/doga_holdout-plantwild_best.pt \\
        --save_dir results/preds

The held-out domain is taken from data.held_out_domain in the config and must
match the domain the checkpoint was trained with. With --save_dir, the per-image
predictions and labels are saved as {mode}_holdout-{domain}_preds.npy and
holdout-{domain}_labels.npy; scripts/compare_doga_vs_lora.py reads these files
for the McNemar tests and bootstrap intervals.
"""

import argparse
import os

import numpy as np

import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from src.data import build_domain_split
from src.model import DoGAModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--mode", type=str, required=True,
                         choices=["linear_probe", "lora_plain", "full_finetune", "doga"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None,
                         help="If set, saves preds/labels as .npy files here for later stats tests.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = build_domain_split(cfg)
    num_domains = split.num_train_domains

    model = DoGAModel(
        backbone_name=cfg["model"]["backbone"],
        num_classes=split.num_classes,
        num_domains=num_domains,
        lora_cfg=cfg["lora"],
        mode=args.mode,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels, _domain_ids in split.held_out_loader:
            images = images.to(device)
            out = model(images)
            preds = out["class_logits"].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        held_out = cfg["data"]["held_out_domain"]
        preds_path = os.path.join(args.save_dir, f"{args.mode}_holdout-{held_out}_preds.npy")
        labels_path = os.path.join(args.save_dir, f"holdout-{held_out}_labels.npy")
        np.save(preds_path, np.array(all_preds))
        np.save(labels_path, np.array(all_labels))
        print(f"Saved predictions to {preds_path}")
        print(f"Saved labels to {labels_path}")

    print(f"=== Held-out domain: {cfg['data']['held_out_domain']} | mode: {args.mode} ===")
    print(classification_report(all_labels, all_preds, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(all_labels, all_preds))


if __name__ == "__main__":
    main()
