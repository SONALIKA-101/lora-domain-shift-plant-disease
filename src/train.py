"""
Training entrypoint. Run from repo root:

    python -m src.train --config configs/base.yaml

To run a baseline instead of DoGA, override the mode:

    python -m src.train --config configs/base.yaml --mode lora_plain
    python -m src.train --config configs/base.yaml --mode linear_probe
"""

import argparse
import csv
import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from src.data import build_domain_split
from src.model import DoGAModel


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def grl_lambda_schedule(step: int, warmup_steps: int, max_lambda: float) -> float:
    """Sigmoid ramp of the gradient-reversal strength from 0 to max_lambda over
    warmup_steps mini-batches (the schedule of Ganin and Lempitsky, 2015)."""
    if warmup_steps <= 0:
        return max_lambda
    p = min(1.0, step / warmup_steps)
    return max_lambda * (2.0 / (1.0 + np.exp(-10 * p)) - 1.0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels, _domain_ids in loader:
        images = images.to(device)
        out = model(images)
        preds = out["class_logits"].argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

def train(cfg: dict, mode_override: str = None):
    set_seed(cfg["experiment"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split = build_domain_split(cfg)
    num_domains = split.num_train_domains  # number of domains present in the training set

    mode = mode_override or cfg["baselines"]["mode"]
    use_domain_loss = (mode == "doga") and cfg["domain_invariance"]["enabled"]

    if mode == "doga" and num_domains < 2:
        raise ValueError(
            f"doga mode needs 2+ domains present during training for the domain-invariance "
            f"term to be meaningful, but only {num_domains} found. Check data.split_strategy "
            f"in the config: 'few_shot_target' works with 2 total domains, while "
            f"'leave_one_out' needs 3 or more."
        )

    model = DoGAModel(
        backbone_name=cfg["model"]["backbone"],
        num_classes=split.num_classes,
        num_domains=num_domains,
        lora_cfg=cfg["lora"],
        mode=mode,
    ).to(device)

    print(f"[mode={mode}] trainable params: {model.trainable_parameter_count():,} "
          f"/ total: {model.total_parameter_count():,} "
          f"({100 * model.trainable_parameter_count() / model.total_parameter_count():.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg["train"]["lr"]), weight_decay=float(cfg["train"]["weight_decay"]),
    )
    scaler = GradScaler(device, enabled=cfg["train"]["mixed_precision"])
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    lambda_domain_max = cfg["domain_invariance"]["lambda_domain"]
    warmup_steps = cfg["domain_invariance"]["grl_warmup_steps"]

    global_step = 0
    best_val_f1 = -1.0
    best_held_out_at_best_val = -1.0
    os.makedirs(cfg["experiment"]["checkpoint_dir"], exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        pbar = tqdm(split.train_loader, desc=f"epoch {epoch+1}/{cfg['train']['epochs']} [{mode}]")
        optimizer.zero_grad()
        for step, (images, labels, domain_ids) in enumerate(pbar):
            images, labels, domain_ids = images.to(device), labels.to(device), domain_ids.to(device)

            domain_lambda = grl_lambda_schedule(global_step, warmup_steps, lambda_domain_max) \
                if use_domain_loss else 0.0

            with autocast(device, enabled=cfg["train"]["mixed_precision"]):
                out = model(images, domain_lambda=domain_lambda)
                loss = class_criterion(out["class_logits"], labels)
                if use_domain_loss:
                    # lambda enters only through the gradient reversal layer (model.py),
                    # not as a weight on this loss.
                    domain_loss = domain_criterion(out["domain_logits"], domain_ids)
                    loss = loss + domain_loss

                loss = loss / cfg["train"]["grad_accum_steps"]

            scaler.scale(loss).backward()

            if (step + 1) % cfg["train"]["grad_accum_steps"] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * cfg["train"]["grad_accum_steps"]
            global_step += 1

            if step % cfg["train"]["log_every"] == 0:
                pbar.set_postfix(loss=running_loss / (step + 1))

        if cfg["train"]["eval_every_epoch"]:
            val_metrics = evaluate(model, split.val_loader, device)
            print(f"  [val, from training domains] "
                  f"acc={val_metrics['accuracy']:.4f} macro_f1={val_metrics['macro_f1']:.4f}")

            # Monitoring only: held-out (test) domain performance is printed for
            # diagnostics and is never used to pick the checkpoint. Selecting on
            # it would be tuning on the test set.
            held_out_metrics = evaluate(model, split.held_out_loader, device)
            print(f"  [held-out={cfg['data']['held_out_domain']}, MONITORING ONLY] "
                  f"acc={held_out_metrics['accuracy']:.4f} macro_f1={held_out_metrics['macro_f1']:.4f} "
                  f"(epoch time {time.time()-epoch_start:.0f}s)")

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_held_out_at_best_val = held_out_metrics["macro_f1"]
                ckpt_name = f"{mode}_holdout-{cfg['data']['held_out_domain']}_best.pt"
                ckpt_path = os.path.join(cfg["experiment"]["checkpoint_dir"], ckpt_name)
                torch.save(model.state_dict(), ckpt_path)

    print(f"\nBest val macro_f1 ({mode}): {best_val_f1:.4f}")
    print(f"Held-out ({cfg['data']['held_out_domain']}) macro_f1 AT that checkpoint: {best_held_out_at_best_val:.4f}")
    print("^ This held-out score is from the checkpoint selected by validation macro_f1")
    print("  (the held-out domain is never used for checkpoint selection).")

    os.makedirs(cfg["experiment"]["output_dir"], exist_ok=True)
    summary_path = os.path.join(cfg["experiment"]["output_dir"], "summary.csv")
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "held_out_domain": cfg["data"]["held_out_domain"],
        "best_val_macro_f1": round(best_val_f1, 4),
        "held_out_macro_f1_at_best_val": round(best_held_out_at_best_val, 4),
        "epochs": cfg["train"]["epochs"],
        "lora_r": cfg["lora"]["r"] if cfg["lora"]["enabled"] else None,
        "lambda_domain": cfg["domain_invariance"]["lambda_domain"] if use_domain_loss else None,
    }
    write_header = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"Logged this run to {summary_path}")

    return best_held_out_at_best_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--mode", type=str, default=None,
                         choices=["linear_probe", "lora_plain", "full_finetune", "doga"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg, mode_override=args.mode)
