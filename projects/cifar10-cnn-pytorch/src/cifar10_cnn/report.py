"""Aggregate training diagnostics; historical and smoke runs stay labeled."""
import json
import hashlib
from pathlib import Path

import numpy as np


def report(run_dir, output_dir, evaluation_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    metadata = json.loads((run_dir / "run.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())
    evaluated = None
    if evaluation_path:
        evaluated = json.loads(Path(evaluation_path).read_text())
        if evaluated.get("checkpoint_sha256") != hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest():
            raise ValueError("Evaluation checkpoint does not match this run's best.pt.")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    epochs = [row["epoch"] for row in history]
    for key, title, axis in [("loss", "Cross-entropy loss", axes[0]), ("accuracy", "Accuracy", axes[1])]:
        axis.plot(epochs, [row[f"train_{key}"] for row in history], marker="o", label="Train")
        axis.plot(epochs, [row[f"validation_{key}"] for row in history], marker="o", label="Validation")
        axis.set(xlabel="Epoch", ylabel=title)
        axis.set_xticks(epochs if len(epochs) <= 15 else np.unique(np.linspace(1, max(epochs), 8, dtype=int)))
        axis.set_xlim(0.5, max(epochs) + 0.5)
        if key == "accuracy":
            axis.set_ylim(0, 1)
        else:
            axis.set_ylim(bottom=0)
        axis.legend()
    labels = {"SYNTHETIC_DEMO_ONLY": "Synthetic images: execution check", "CIFAR10_SUBSET_SMOKE": "CIFAR-10 subset: short execution check", "CIFAR10_FULL_TRAIN_VALIDATION": "CIFAR-10 training and validation"}
    figure.suptitle(labels.get(metadata["data_kind"], metadata["data_kind"]), fontsize=12)
    figure.tight_layout()
    figure.savefig(output_dir / "training-history.png", dpi=160)
    plt.close(figure)
    if evaluated is not None:
        matrix = np.array(evaluated["confusion_matrix"])
        figure, axis = plt.subplots(figsize=(8, 7))
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(10), evaluated["classes"], rotation=45, ha="right")
        axis.set_yticks(range(10), evaluated["classes"])
        split_label = "Official test subset" if evaluated["split"] == "official_test_subset" else "Official test set"
        checkpoint_label = labels.get(evaluated["checkpoint_data_kind"], evaluated["checkpoint_data_kind"])
        axis.set(xlabel="Predicted class", ylabel="True class", title=f"{split_label}: {evaluated['samples']} images, accuracy={evaluated['accuracy']:.1%}\n{checkpoint_label}")
        for row in range(10):
            for column in range(10):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7, color="white" if matrix[row, column] > matrix.max() / 2 else "black")
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(output_dir / "confusion-matrix.png", dpi=160)
        plt.close(figure)
    return metadata
