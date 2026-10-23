"""Aggregate diagnostic charts from a completed segmentation run."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def report(run_dir, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text())
    profiles = pd.read_csv(run_dir / "cluster_profiles.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    cumulative = np.cumsum(metrics["pca_variance_spectrum"])
    axes[0].plot(range(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    axes[0].axvline(metrics["pca_components"], linestyle="--", color="gray", label="Selected components")
    axes[0].set(xlabel="Principal components", ylabel="Cumulative explained variance", title="PCA retained variance", ylim=(0, 1.05))
    axes[0].legend()
    axes[1].plot([row["k"] for row in metrics["candidates"]], [row["silhouette"] for row in metrics["candidates"]], marker="o")
    axes[1].axvline(metrics["selected_k"], linestyle="--", color="gray")
    axes[1].set(xlabel="K-Means clusters (K)", ylabel="Silhouette score", title="In-sample model selection")
    figure.tight_layout()
    figure.savefig(output_dir / "model-selection.png", dpi=160)
    plt.close(figure)
    columns = ["Income", "total_spend", "NumDealsPurchases", "Recency"]
    values = profiles[columns].to_numpy()
    relative = (values - values.mean(axis=0)) / np.where(values.std(axis=0) > 0, values.std(axis=0), 1)
    figure, axis = plt.subplots(figsize=(8.5, max(3, len(profiles) * 0.6)))
    image = axis.imshow(relative, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    axis.set_xticks(range(len(columns)), ["Income", "Total spend", "Discount purchases", "Recency"])
    axis.set_yticks(range(len(profiles)), [f"Segment {int(row.segment)} (n={int(row.customers)})" for row in profiles.itertuples()])
    for row in range(len(profiles)):
        for column in range(len(columns)):
            axis.text(column, row, f"{values[row, column]:,.1f}", ha="center", va="center", fontsize=9)
    axis.set_title("Observed segment means\nColor: standardized across segment means; text: original source units")
    figure.colorbar(image, ax=axis, label="Relative profile level")
    figure.tight_layout()
    figure.savefig(output_dir / "segment-profiles.png", dpi=160)
    plt.close(figure)
    return metrics
