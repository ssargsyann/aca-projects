"""Training and validation entry point."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import validate_target
from .model import build_pipeline


def train(data_path, config_path, output_dir):
    data_path, output_dir = Path(data_path), Path(output_dir)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    frame = pd.read_csv(data_path)
    labels = validate_target(frame, config["target"])
    features = frame.drop(columns=[config["target"]])
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=config["test_size"],
        random_state=config["random_state"], stratify=labels,
    )
    pipeline = build_pipeline(config, y_train)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_val)[:, 1]
    predictions = pipeline.predict(X_val)
    metrics = {
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "average_precision": float(average_precision_score(y_val, probabilities)),
        "accuracy": float(accuracy_score(y_val, predictions)),
        "classification_report": classification_report(y_val, predictions, output_dict=True, zero_division=0),
        "train_rows": len(y_train), "validation_rows": len(y_val),
        "validation_positive_rate": float(y_val.mean()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_dir / "pipeline.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    metadata = {
        "config": config, "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "features": pipeline.named_steps["select"].columns_,
        "versions": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "xgboost", "joblib"]},
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics
