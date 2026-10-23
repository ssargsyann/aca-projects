"""Explicit test-set evaluation, separate from training and checkpoint selection."""
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from .checkpoint import load_model
from .data import CLASSES, image_transform, stratified_limit
from .engine import run_epoch


def evaluate(checkpoint_path, data_root, output_path, limit=None, download=False):
    torch.set_num_threads(2)
    model, checkpoint = load_model(checkpoint_path)
    dataset = CIFAR10(data_root, train=False, download=download, transform=image_transform(False))
    indices = stratified_limit(np.arange(len(dataset)), dataset.targets, limit, checkpoint["config"]["seed"] + 2)
    subset = Subset(dataset, indices.tolist())
    metrics = run_epoch(model, DataLoader(subset, batch_size=128, shuffle=False, num_workers=0), "cpu")
    metrics.update({"split": "official_test_subset" if limit else "official_test_full", "checkpoint_data_kind": checkpoint["data_kind"], "classes": CLASSES, "checkpoint_epoch": checkpoint["epoch"]})
    metrics["checkpoint_sha256"] = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    metrics["selection_seed"] = checkpoint["config"]["seed"] + 2
    if limit is not None:
        metrics["selected_test_indices"] = indices.tolist()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
