"""Standalone figures from measured history and correctly aligned attention rows."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def training_figure(result, path):
    history = result["history"]
    epochs = [row["epoch"] for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(epochs, [row["train_nll"] for row in history], label="Train NLL")
    axes[0].plot(
        epochs,
        [row["validation_teacher_forced_nll"] for row in history],
        label="Validation NLL",
    )
    axes[0].set_ylabel("Mean loss per non-padding target token")
    axes[1].plot(
        epochs,
        [row["validation_greedy_exact_match"] for row in history],
        label="Greedy exact date + EOS",
    )
    axes[1].plot(
        epochs,
        [row["validation_teacher_forced_token_accuracy"] for row in history],
        label="Teacher-forced token accuracy",
    )
    axes[1].set_ylim(0, 1.03)
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def attention_figure(result, path):
    values = np.asarray(result["attention"])
    labels = result["attention_rows"]
    columns = result["attention_columns"]
    if values.shape != (len(labels), len(columns)) or not np.isfinite(values).all():
        raise ValueError("Attention rows/columns must match token labels exactly")
    figure, axis = plt.subplots(figsize=(10, 5))
    plot = axis.imshow(values, cmap="viridis", aspect="auto", vmin=0)
    axis.set_xticks(
        range(len(columns)),
        ["space" if char == " " else char for char in columns],
        rotation=90,
    )
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Source characters (padding omitted)")
    axis.set_ylabel("Generated token, including EOS when emitted")
    axis.set_title(
        f"Last-layer mean cross-attention: {result['source']} → {result['prediction']}",
        fontsize=10,
    )
    figure.colorbar(plot, ax=axis, label="Attention weight")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
