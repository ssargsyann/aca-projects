"""Configurable synthetic training, best-validation checkpointing and inference."""

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .common import new_directory, sha256, write_json
from .data import Shapes
from .metrics import bce_dice, overlaps
from .model import UNet


def configure(config):
    for name in [
        "train_count",
        "validation_count",
        "test_count",
        "epochs",
        "batch_size",
        "base_channels",
        "cpu_threads",
    ]:
        if (
            isinstance(config[name], bool)
            or not isinstance(config[name], int)
            or config[name] < 1
        ):
            raise ValueError(f"{name} must be a positive integer")
    if (
        config["batch_size"] < 2
        or config["train_count"] < 2
        or config["learning_rate"] <= 0
    ):
        raise ValueError(
            "Training needs batch size/count >=2 and a positive learning rate"
        )
    if len(config["image_size"]) != 2 or min(config["image_size"]) < 32:
        raise ValueError("Image height and width must be at least 32")
    if not 0 <= config["threshold"] <= 1 or config["seed"] < 0:
        raise ValueError("Invalid threshold or seed")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if (
        device not in {"cpu", "cuda"}
        or device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ValueError("Choose an available cpu/cuda device")
    return device


@torch.no_grad()
def evaluate(model, loader, device, threshold):
    model.eval()
    totals = {"loss": 0.0, "iou": 0.0, "dice": 0.0}
    count = 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        count += len(images)
        totals["loss"] += float(bce_dice(logits, masks)) * len(images)
        for name, values in overlaps(logits, masks, threshold).items():
            totals[name] += float(values.sum())
    return {name: value / count for name, value in totals.items()}


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "aca-synthetic-unet-v1" or saved.get("threshold") is None:
        raise ValueError("Unsupported checkpoint format")
    model = UNet(num_classes=1, base_channels=saved["base_channels"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval(), saved


def train(config_path, output):
    config = json.loads(Path(config_path).read_text())
    device = configure(config)
    output = new_directory(output)
    datasets = {
        split: Shapes(config[key], config["image_size"], config["seed"], split)
        for split, key in [
            ("train", "train_count"),
            ("validation", "validation_count"),
            ("test", "test_count"),
        ]
    }
    loaders = {
        split: DataLoader(
            data,
            batch_size=config["batch_size"],
            shuffle=split == "train",
            num_workers=0,
            generator=torch.Generator().manual_seed(config["seed"]),
        )
        for split, data in datasets.items()
    }
    model = UNet(num_classes=1, base_channels=config["base_channels"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best, history = -1.0, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        total, count = 0.0, 0
        for images, masks in loaders["train"]:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = bce_dice(model(images), masks)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(images)
            count += len(images)
        validation = evaluate(model, loaders["validation"], device, config["threshold"])
        row = {
            "epoch": epoch,
            "train_loss": total / count,
            **{"validation_" + k: v for k, v in validation.items()},
        }
        history.append(row)
        if validation["iou"] > best:
            best = validation["iou"]
            torch.save(
                {
                    "format": "aca-synthetic-unet-v1",
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "base_channels": config["base_channels"],
                    "threshold": config["threshold"],
                    "best_epoch": epoch,
                    "config": config,
                },
                output / "best.pt",
            )
        print(json.dumps(row), flush=True)
    model, saved = load_checkpoint(output / "best.pt", device)
    # The test split is visited only after validation-based model selection.
    validation = evaluate(model, loaders["validation"], device, config["threshold"])
    test = evaluate(model, loaders["test"], device, config["threshold"])
    result = {
        "scope": "fixed_synthetic_shapes_only_not_real_world_benchmark",
        "config": config,
        "generator": "pillow_shapes_v1_seed_sequence_seed_split_index",
        "split_ids": {"train": 0, "validation": 1, "test": 2},
        "best_epoch": saved["best_epoch"],
        "validation": validation,
        "test": test,
        "history": history,
        "parameters": sum(p.numel() for p in model.parameters()),
        "checkpoint_sha256": sha256(output / "best.pt"),
        "versions": {
            name: version(name) for name in ["torch", "numpy", "pillow", "matplotlib"]
        },
    }
    write_json(output / "results.json", result)
    render_report(output, model, datasets["test"], device)
    return result


def render_report(output, model, dataset, device):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    results = json.loads((output / "results.json").read_text())
    history = results["history"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = [row["epoch"] for row in history]
    for name in ["train_loss", "validation_loss"]:
        axes[0].plot(epochs, [row[name] for row in history], label=name)
    for name in ["validation_iou", "validation_dice"]:
        axes[1].plot(epochs, [row[name] for row in history], label=name)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "training.png", dpi=140)
    plt.close(figure)
    figure, axes = plt.subplots(3, 3, figsize=(8, 8), squeeze=False)
    model.eval()
    for i in range(3):
        image, mask = dataset[i % len(dataset)]
        with torch.inference_mode():
            pred = (
                model(image.unsqueeze(0).to(device)).sigmoid().cpu()[0, 0]
                > results["config"]["threshold"]
            )
        for axis, data, title in zip(
            axes[i],
            [image.permute(1, 2, 0), mask[0], pred],
            ["Synthetic test image", "Ground truth", "Selected checkpoint"],
        ):
            axis.imshow(data.numpy(), cmap="gray", vmin=0, vmax=1)
            axis.axis("off")
            axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output / "predictions.png", dpi=140)
    plt.close(figure)


def predict(checkpoint, image_path, output, device="cpu"):
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    model, saved = load_checkpoint(checkpoint, device)
    with Image.open(image_path) as opened:
        image = np.array(opened.convert("RGB"))
    if min(image.shape[:2]) < 16:
        raise ValueError("Image dimensions must be at least 16")
    output = new_directory(output)
    tensor = (
        torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255
    )
    with torch.inference_mode():
        probabilities = model(tensor).sigmoid().cpu()[0, 0].numpy()
    np.save(output / "probabilities.npy", probabilities)
    Image.fromarray((probabilities > saved["threshold"]).astype(np.uint8) * 255).save(
        output / "mask.png"
    )
    result = {
        "image_size": [image.shape[1], image.shape[0]],
        "image_sha256": sha256(image_path),
        "checkpoint_sha256": sha256(checkpoint),
        "threshold": saved["threshold"],
        "scope": "synthetic_shapes_model_no_real_world_validity",
    }
    write_json(output / "prediction.json", result)
    return result
