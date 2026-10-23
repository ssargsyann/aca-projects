"""A self-contained synthetic demonstration, not an ICU benchmark."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .predict import predict
from .train import train


def make_demo_data(rows=240, seed=42):
    """Generate invented features and arbitrary labels for exercising the code."""
    if rows < 40:
        raise ValueError("The synthetic demo requires at least 40 rows.")
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "encounter_id": np.arange(1, rows + 1),
        "age": rng.integers(18, 90, rows).astype(float),
        "heart_rate": rng.normal(85, 18, rows),
        "unit_type": rng.choice(["medical", "surgical"], rows),
    })
    # Labels are intentionally arbitrary and balanced, not a medical model.
    frame["hospital_death"] = rng.permutation(np.arange(rows) % 2)
    frame.loc[rng.choice(rows, rows // 10, replace=False), "age"] = np.nan
    return frame


def run_demo(output_dir, rows=240, seed=42, with_explanations=False):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Demo output directory must be empty or new; choose a fresh --output path.")
    frame = make_demo_data(rows, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "synthetic_train.csv"
    inference_path = output_dir / "synthetic_inference.csv"
    frame.to_csv(training_path, index=False)
    frame.drop(columns="hospital_death").head(12).to_csv(inference_path, index=False)
    config = {
        "data_kind": "SYNTHETIC_DEMO_ONLY",
        "target": "hospital_death", "test_size": 0.2, "random_state": seed,
        "missing_threshold": 0.65, "drop_columns": ["encounter_id"],
        "model": {"n_estimators": 20, "learning_rate": 0.05, "max_depth": 3, "n_jobs": 1},
    }
    config_path = output_dir / "demo-config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics = train(training_path, config_path, output_dir / "model")
    model_path = output_dir / "model/pipeline.joblib"
    predict(model_path, inference_path, output_dir / "predictions.csv")
    if with_explanations:
        from .explain import explain
        explain(model_path, inference_path, output_dir / "explanations", max_rows=12)
    summary = {
        "data_kind": "SYNTHETIC_DEMO_ONLY",
        "notice": "Invented data and arbitrary labels. Metrics do not measure ICU mortality prediction quality.",
        "rows": rows, "seed": seed, "prediction_rows": 12,
        "explanations_generated": with_explanations,
        "metrics": metrics,
    }
    (output_dir / "demo-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "SYNTHETIC DEMO ONLY\nThese are invented records with arbitrary labels, not patient data.\n"
        "The inference example reuses training-file rows to exercise prediction; it is not an additional evaluation.\n"
        "Do not report these metrics as ICU performance. See demo-summary.json and model/run.json.\n",
        encoding="utf-8",
    )
    return summary
