"""Train on CIFAR-10 training data; select checkpoints using validation only."""
import json
import platform
import random
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import CLASSES, training_sets
from .engine import run_epoch
from .model import AdvancedCNN


def setup(config):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    requested = config["device"]
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available; choose cpu or auto.")
    return torch.device(device)


def train(data_root, config_path, output_dir, download=False, datasets_override=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config["epochs"] < 1 or config["batch_size"] < 1 or config["cpu_threads"] < 1 or config["num_workers"] < 0:
        raise ValueError("Epochs, batch size, and CPU threads must be positive; workers cannot be negative.")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new or empty run directory; existing runs are preserved.")
    device = setup(config)
    if datasets_override is None:
        training, validation, train_indices, val_indices = training_sets(data_root, config, download)
        kind = "CIFAR10_SUBSET_SMOKE" if config["train_limit"] or config["validation_limit"] else "CIFAR10_FULL_TRAIN_VALIDATION"
    else:
        training, validation = datasets_override
        train_indices = val_indices = None
        kind = "SYNTHETIC_DEMO_ONLY"
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(training, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"], generator=generator)
    validation_loader = DataLoader(validation, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    model = AdvancedCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    history, best_accuracy, best_epoch = [], -1.0, None
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config["epochs"] + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        fitted = run_epoch(model, train_loader, device, optimizer)
        validated = run_epoch(model, validation_loader, device)
        row = {"epoch": epoch, "learning_rate": learning_rate, "train_loss": fitted["loss"], "train_accuracy": fitted["accuracy"], "validation_loss": validated["loss"], "validation_accuracy": validated["accuracy"]}
        history.append(row)
        scheduler.step()
        if validated["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validated["accuracy"], epoch
            torch.save({"architecture": "AdvancedCNN-v1", "state_dict": model.state_dict(), "classes": CLASSES, "normalization": {"mean": [0.5] * 3, "std": [0.5] * 3}, "config": config, "epoch": epoch, "validation_accuracy": best_accuracy, "data_kind": kind}, output_dir / "best.pt")
        print(f"Epoch {epoch}/{config['epochs']} | train loss {fitted['loss']:.4f} | validation accuracy {validated['accuracy']:.4f}", flush=True)
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    metadata = {
        "config": config, "data_kind": kind, "train_samples": len(training), "validation_samples": len(validation),
        "best_epoch": best_epoch, "best_validation_accuracy": best_accuracy,
        "device": str(device), "python": platform.python_version(),
        "versions": {name: version(name) for name in ["torch", "torchvision", "numpy", "pillow"]},
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "test_used_during_training": False,
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if train_indices is not None:
        (output_dir / "split-indices.json").write_text(json.dumps({"train": train_indices.tolist(), "validation": val_indices.tolist()}), encoding="utf-8")
    return metadata
