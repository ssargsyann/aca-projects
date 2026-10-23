"""Score input rows with a trusted, previously trained pipeline."""
from pathlib import Path

import joblib
import pandas as pd


def predict(model_path, data_path, output_path):
    pipeline = joblib.load(model_path)
    frame = pd.read_csv(data_path)
    result = pd.DataFrame({
        "row_number": range(len(frame)),
        "mortality_score": pipeline.predict_proba(frame)[:, 1],
        "predicted_class": pipeline.predict(frame),
    })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result
