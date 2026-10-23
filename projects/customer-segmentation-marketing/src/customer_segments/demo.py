"""An invented-data workflow that does not require a Kaggle account."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .predict import predict
from .train import train

FEATURES = ["Year_Birth", "Income", "Kidhome", "Teenhome", "Recency", "MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds", "NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases", "NumWebVisitsMonth"]


def make_demo_data(rows=180, seed=42):
    if rows < 40:
        raise ValueError("Demo requires at least 40 rows.")
    rng = np.random.default_rng(seed)
    groups = np.arange(rows) % 3
    frame = pd.DataFrame({
        "ID": np.arange(rows), "Year_Birth": rng.integers(1950, 1995, rows),
        "Income": 25000 + groups * 22000 + rng.normal(0, 1500, rows),
        "Kidhome": rng.integers(0, 3, rows), "Teenhome": rng.integers(0, 3, rows),
        "Recency": rng.integers(0, 100, rows),
    })
    for column in FEATURES[5:11]:
        frame[column] = np.maximum(0, 20 + groups * 130 + rng.normal(0, 10, rows))
    for column in FEATURES[11:]:
        frame[column] = rng.integers(0, 10, rows)
    frame.loc[0:4, "Income"] = np.nan
    return frame


def run_demo(output_dir, seed=42):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Demo output directory must be new or empty.")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "synthetic.csv"
    make_demo_data(seed=seed).to_csv(data_path, index=False)
    config = {
        "data_kind": "SYNTHETIC_DEMO_ONLY", "features": FEATURES,
        "variance_threshold": 0.8, "clip_quantiles": [0.01, 0.99],
        "k_values": [2, 3, 4], "random_state": seed, "n_init": 10,
        "silhouette_sample_size": 3000, "stability_seeds": [seed + 1],
        "boundary_margin": 0.1, "gmm_reg_covar": 0.000001,
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics = train(data_path, config_path, output_dir / "run")
    predict(output_dir / "run/segmentation.joblib", data_path, output_dir / "reloaded-assignments.csv")
    (output_dir / "README.txt").write_text("SYNTHETIC DEMO ONLY\nInvented records; not a marketing benchmark or evidence of campaign effectiveness.\n", encoding="utf-8")
    return metrics
