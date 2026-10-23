"""Reconstruct the recorded validation split and save aggregate diagnostics."""
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, average_precision_score,
                             classification_report, roc_auc_score)
from sklearn.model_selection import train_test_split

from .data import validate_target


def create_report(run_dir, data_path, output_dir):
    run_dir, data_path, output_dir = Path(run_dir), Path(data_path), Path(output_dir)
    metadata = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if hashlib.sha256(data_path.read_bytes()).hexdigest() != metadata["data_sha256"]:
        raise ValueError("Dataset checksum differs from the recorded training input.")
    config = metadata["config"]
    frame = pd.read_csv(data_path)
    labels = validate_target(frame, config["target"])
    _, X_val, _, y_val = train_test_split(
        frame.drop(columns=config["target"]), labels,
        test_size=config["test_size"], random_state=config["random_state"], stratify=labels,
    )
    pipeline = joblib.load(run_dir / "pipeline.joblib")
    probabilities = pipeline.predict_proba(X_val)[:, 1]
    predictions = pipeline.predict(X_val)
    summary = {
        "evaluation": "recorded_internal_validation_split",
        "data_kind": config.get("data_kind", "USER_SUPPLIED_DATA"),
        "validation_rows": len(y_val),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "average_precision": float(average_precision_score(y_val, probabilities)),
        "positive_rate": float(y_val.mean()),
        "classification_report": classification_report(y_val, predictions, output_dict=True, zero_division=0),
    }
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    kind = "Synthetic demo" if config.get("data_kind") == "SYNTHETIC_DEMO_ONLY" else "Recovered local data"
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    RocCurveDisplay.from_predictions(y_val, probabilities, ax=axes[0], name="XGBoost")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random ranking")
    axes[0].legend(loc="lower right")
    PrecisionRecallDisplay.from_predictions(y_val, probabilities, ax=axes[1], name="XGBoost")
    axes[1].axhline(float(y_val.mean()), linestyle="--", color="gray", label="Positive prevalence")
    axes[1].legend(loc="upper right")
    figure.suptitle(f"{kind}: internal validation (not external test)")
    figure.tight_layout()
    figure.savefig(output_dir / "validation-curves.png", dpi=160)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    ConfusionMatrixDisplay.from_predictions(y_val, predictions, ax=axis, display_labels=["0", "1"], colorbar=False)
    axis.set_title(f"{kind}\nValidation confusion matrix, default threshold")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion-matrix.png", dpi=160)
    plt.close(figure)
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
