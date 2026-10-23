"""Assign rows using previously fitted preprocessing and segment models."""
from pathlib import Path

import joblib
from threadpoolctl import threadpool_limits

from .data import load_data
from .profiles import assignments


def predict(model_path, data_path, output_path):
    bundle = joblib.load(model_path)
    frame = load_data(data_path)
    with threadpool_limits(limits=1):
        result = assignments(bundle, frame)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
