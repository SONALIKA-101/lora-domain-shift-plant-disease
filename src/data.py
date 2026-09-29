"""
Multi-domain dataset loading for leave-one-domain-out disease classification.

Expected folder layout per domain (standard ImageFolder style):
    data/<domain_name>/<class_name>/<image>.jpg

Class names are harmonized across the datasets in two ways: the download
scripts (scripts/download_real_data.py, scripts/integrate_plantwild.py) place
each dataset's images in folders named with the shared class name, and
harmonize() normalizes folder names to lowercase with underscores at load
time. CLASS_NAME_MAP below is an optional explicit override and is empty by
default.
"""

import os
import random
from dataclasses import dataclass
from typing import List, Optional

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Optional explicit mapping from a raw folder name to the shared class name.
# Empty by default: the folders are already named consistently (see the module
# docstring).
CLASS_NAME_MAP = {
    # "Tomato___Early_blight": "tomato_early_blight",
    # "Tomato Early blight leaf": "tomato_early_blight",
}


def harmonize(name: str) -> str:
    return CLASS_NAME_MAP.get(name, name.strip().lower().replace(" ", "_"))


class DomainTaggedDataset(Dataset):
    """Wraps an ImageFolder-style dataset and attaches a domain id to every sample."""

    def __init__(self, root: str, domain_id: int, transform=None, class_to_idx: Optional[dict] = None):
        self.base = ImageFolder(root, transform=transform)
        self.domain_id = domain_id

        if class_to_idx is not None:
            # Remap this dataset's local class indices onto the shared global class space.
            local_to_global = {}
            for local_name, local_idx in self.base.class_to_idx.items():
                global_name = harmonize(local_name)
                if global_name not in class_to_idx:
                    raise ValueError(
                        f"Class '{global_name}' (from '{local_name}') not found in shared "
                        f"class_to_idx. Update CLASS_NAME_MAP in src/data.py."
                    )
                local_to_global[local_idx] = class_to_idx[global_name]
            self.remap = local_to_global
        else:
            self.remap = None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, local_label = self.base[idx]
        label = self.remap[local_label] if self.remap is not None else local_label
        return img, label, self.domain_id


def build_shared_class_index(domain_roots: List[str]) -> dict:
    """Scan all domains, harmonize class names, return a single global class_to_idx."""
    all_classes = set()
    for root in domain_roots:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            if os.path.isdir(os.path.join(root, entry)):
                all_classes.add(harmonize(entry))
    classes = sorted(all_classes)
    return {c: i for i, c in enumerate(classes)}


def get_transforms(image_size: int, train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@dataclass
class LeaveOneDomainOutSplit:
    train_loader: DataLoader
    val_loader: DataLoader          # carved from the training domains only; used for checkpoint selection
    held_out_loader: DataLoader     # unseen test domain; never used for training or checkpoint selection
    num_classes: int
    domain_names: List[str]
    num_train_domains: int  # how many distinct domain_ids appear in train_loader


def _stratified_index_split(base: DomainTaggedDataset, holdout_fraction: float, seed: int):
    """Splits a dataset's indices into (kept, holdout), stratified per class.
    Used both for validation carve-out and for the few-shot target-domain slice."""
    rng = random.Random(seed)
    by_class = {}
    for idx, (_path, local_label) in enumerate(base.base.samples):
        global_label = base.remap[local_label] if base.remap else local_label
        by_class.setdefault(global_label, []).append(idx)

    kept_indices, holdout_indices = [], []
    for _label, idxs in by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_holdout = max(1, int(round(len(idxs) * holdout_fraction))) if len(idxs) > 1 else 0
        holdout_indices.extend(idxs[:n_holdout])
        kept_indices.extend(idxs[n_holdout:])
    return kept_indices, holdout_indices


def build_leave_one_domain_out(cfg) -> LeaveOneDomainOutSplit:
    """Build train / val / held-out loaders for strict leave-one-domain-out evaluation.

    The held-out domain is never used for training, validation or checkpoint
    selection. A stratified validation slice (val_fraction) is carved from each
    training domain and used only for checkpoint selection. DoGA needs at least
    two training domains, i.e. three configured domains in total.
    """
    domains = cfg["data"]["domains"]
    held_out_name = cfg["data"]["held_out_domain"]
    image_size = cfg["data"]["image_size"]
    val_fraction = cfg["data"].get("val_fraction", 0.1)
    seed = cfg["experiment"].get("seed", 42)

    roots = [d["root"] for d in domains]
    class_to_idx = build_shared_class_index(roots)
    num_classes = len(class_to_idx)

    train_sets, val_sets, held_out_set = [], [], None
    train_domain_ids = set()
    train_domain_counter = 0  # contiguous 0..N-1 among TRAINING domains only -
    # using raw list position here would break whenever the held-out domain
    # isn't last in the list, since the domain classifier's output size only
    # covers the training domains, not the full configured list.
    for i, d in enumerate(domains):
        is_held_out = d["name"] == held_out_name
        if is_held_out:
            tfm = get_transforms(image_size, train=False)
            ds = DomainTaggedDataset(d["root"], domain_id=i, transform=tfm, class_to_idx=class_to_idx)
            held_out_set = ds
            continue

        this_domain_id = train_domain_counter
        train_domain_counter += 1

        # Training domain: carve a stratified validation slice from it (never
        # from the held-out domain). Checkpoint selection uses this slice, so the
        # held-out domain plays no role in model selection.
        train_tfm = get_transforms(image_size, train=True)
        val_tfm = get_transforms(image_size, train=False)
        train_view = DomainTaggedDataset(d["root"], domain_id=this_domain_id, transform=train_tfm, class_to_idx=class_to_idx)
        val_view = DomainTaggedDataset(d["root"], domain_id=this_domain_id, transform=val_tfm, class_to_idx=class_to_idx)

        train_idx, val_idx = _stratified_index_split(train_view, val_fraction, seed + i)
        train_sets.append(_IndexSubsetWithDomain(train_view, train_idx, domain_id_override=this_domain_id))
        val_sets.append(_IndexSubsetWithDomain(val_view, val_idx, domain_id_override=this_domain_id))
        train_domain_ids.add(this_domain_id)

    if held_out_set is None:
        raise ValueError(f"held_out_domain '{held_out_name}' not found in configured domains.")

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    held_out_loader = DataLoader(
        held_out_set, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    return LeaveOneDomainOutSplit(
        train_loader=train_loader,
        val_loader=val_loader,
        held_out_loader=held_out_loader,
        num_classes=num_classes,
        domain_names=[d["name"] for d in domains],
        num_train_domains=len(train_domain_ids),
    )


class _IndexSubsetWithDomain(Dataset):
    """Wraps a DomainTaggedDataset restricted to a fixed list of indices,
    optionally overriding the domain_id (used to relabel a target-domain
    slice as its own domain id distinct from its held-out counterpart)."""

    def __init__(self, base: DomainTaggedDataset, indices: List[int], domain_id_override: int):
        self.base = base
        self.indices = indices
        self.domain_id_override = domain_id_override

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label, _orig_domain = self.base[self.indices[idx]]
        return img, label, self.domain_id_override


def build_few_shot_target_split(cfg) -> LeaveOneDomainOutSplit:
    """
    Alternative to strict leave-one-domain-out for when you only have 2
    domains. Source domain(s) train fully as normal. The target (held-out)
    domain is split class-stratified into a small labeled slice that joins
    training (with its OWN domain id, so the domain classifier has 2 real
    domains to distinguish - making doga's invariance term meaningful) and a
    larger slice held out purely for evaluation, never seen during training.

    A validation set is further carved from the TRAINING portions only
    (source domain + the small target-domain training slice) - never from
    the held-out evaluation slice - so checkpoint selection never touches
    the genuinely-unseen data.

    This is still a genuine unseen-SAMPLE generalization test (the held-out
    slice is never touched during training), just not a genuine unseen-DOMAIN
    test (the target domain is partially represented). Upgrade to
    build_leave_one_domain_out() once you have 3+ domains for the stronger claim.

    Config keys used (with sensible defaults if absent):
        data.target_train_fraction: float, default 0.2
            fraction of the held-out domain's images (per class, stratified)
            that join training as a labeled "target" domain slice.
        data.val_fraction: float, default 0.1
            fraction of each TRAINING portion (source + target-train-slice)
            further carved out for validation/checkpoint selection.
    """
    domains = cfg["data"]["domains"]
    held_out_name = cfg["data"]["held_out_domain"]
    image_size = cfg["data"]["image_size"]
    target_train_fraction = cfg["data"].get("target_train_fraction", 0.2)
    val_fraction = cfg["data"].get("val_fraction", 0.1)
    seed = cfg["experiment"].get("seed", 42)

    roots = [d["root"] for d in domains]
    class_to_idx = build_shared_class_index(roots)
    num_classes = len(class_to_idx)

    train_sets, val_sets = [], []
    held_out_set = None
    train_domain_ids = set()
    source_domain_id = 0
    target_domain_id = 1  # the small labeled slice of the target domain gets its own id

    for i, d in enumerate(domains):
        is_held_out = d["name"] == held_out_name
        if not is_held_out:
            train_tfm = get_transforms(image_size, train=True)
            val_tfm = get_transforms(image_size, train=False)
            train_view = DomainTaggedDataset(d["root"], domain_id=source_domain_id, transform=train_tfm, class_to_idx=class_to_idx)
            val_view = DomainTaggedDataset(d["root"], domain_id=source_domain_id, transform=val_tfm, class_to_idx=class_to_idx)
            tr_idx, va_idx = _stratified_index_split(train_view, val_fraction, seed + i)
            train_sets.append(_IndexSubsetWithDomain(train_view, tr_idx, domain_id_override=source_domain_id))
            val_sets.append(_IndexSubsetWithDomain(val_view, va_idx, domain_id_override=source_domain_id))
            train_domain_ids.add(source_domain_id)
            continue

        # held-out (target) domain: stratified split into a small train slice + eval slice
        train_tfm = get_transforms(image_size, train=True)
        eval_tfm = get_transforms(image_size, train=False)
        base_train_view = DomainTaggedDataset(d["root"], domain_id=target_domain_id, transform=train_tfm, class_to_idx=class_to_idx)
        base_eval_view = DomainTaggedDataset(d["root"], domain_id=target_domain_id, transform=eval_tfm, class_to_idx=class_to_idx)

        kept_idx, eval_idx = _stratified_index_split(base_train_view, 1.0 - target_train_fraction, seed + i)
        # kept_idx = the target_train_fraction slice that joins training; eval_idx = held out for testing

        # further carve a small validation slice from the target-train slice
        # (operate on a wrapped view so _stratified_index_split's per-class
        # grouping - which reads base.base.samples - still works correctly)
        target_train_view = _IndexSubsetWithDomain(base_train_view, kept_idx, domain_id_override=target_domain_id)
        # simple validation carve: shuffle kept_idx directly since it's already a flat index list
        rng = random.Random(seed + i + 1000)
        kept_idx_shuffled = kept_idx[:]
        rng.shuffle(kept_idx_shuffled)
        n_val = max(1, int(round(len(kept_idx_shuffled) * val_fraction))) if len(kept_idx_shuffled) > 1 else 0
        target_val_idx = kept_idx_shuffled[:n_val]
        target_final_train_idx = kept_idx_shuffled[n_val:]

        train_sets.append(_IndexSubsetWithDomain(base_train_view, target_final_train_idx, domain_id_override=target_domain_id))
        val_sets.append(_IndexSubsetWithDomain(base_eval_view, target_val_idx, domain_id_override=target_domain_id))
        held_out_set = _IndexSubsetWithDomain(base_eval_view, eval_idx, domain_id_override=target_domain_id)

        train_domain_ids.add(target_domain_id)

    if held_out_set is None:
        raise ValueError(f"held_out_domain '{held_out_name}' not found in configured domains.")

    train_ds = ConcatDataset(train_sets)
    val_ds = ConcatDataset(val_sets)

    # Domain-balanced sampling: without this, since the target slice is much
    # smaller than the source domain, most batches would contain zero target
    # examples, starving the domain classifier of signal. Weight each sample
    # inversely to its domain's size so batches see both domains regularly.
    domain_id_per_sample = [ds.domain_id_override for ds in train_sets for _ in range(len(ds))]
    domain_counts = {}
    for did in domain_id_per_sample:
        domain_counts[did] = domain_counts.get(did, 0) + 1
    sample_weights = [1.0 / domain_counts[did] for did in domain_id_per_sample]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["data"]["batch_size"], sampler=sampler,
        num_workers=cfg["data"]["num_workers"], pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    held_out_loader = DataLoader(
        held_out_set, batch_size=cfg["data"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    return LeaveOneDomainOutSplit(
        train_loader=train_loader,
        val_loader=val_loader,
        held_out_loader=held_out_loader,
        num_classes=num_classes,
        domain_names=[d["name"] for d in domains],
        num_train_domains=len(train_domain_ids),
    )


def build_domain_split(cfg) -> LeaveOneDomainOutSplit:
    """Dispatches to the right split builder based on cfg['data']['split_strategy'].
    'few_shot_target' (default, works with 2 domains) or 'leave_one_out'
    (needs 3+ domains for doga to be meaningful). The study reported in the paper
    uses 'leave_one_out' (set in configs/base.yaml)."""
    strategy = cfg["data"].get("split_strategy", "few_shot_target")
    if strategy == "leave_one_out":
        return build_leave_one_domain_out(cfg)
    elif strategy == "few_shot_target":
        return build_few_shot_target_split(cfg)
    else:
        raise ValueError(f"Unknown split_strategy '{strategy}'. Use 'few_shot_target' or 'leave_one_out'.")
