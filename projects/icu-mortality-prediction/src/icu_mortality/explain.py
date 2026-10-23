"""Optional SHAP plots for a trusted saved pipeline."""
from pathlib import Path

import joblib
import pandas as pd


def explain(model_path, data_path, output_dir, max_rows=100):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    if max_rows < 1:
        raise ValueError("max_rows must be positive.")
    pipeline = joblib.load(model_path)
    frame = pd.read_csv(data_path).head(max_rows)
    selected = pipeline.named_steps["select"].transform(frame)
    preprocess = pipeline.named_steps["preprocess"]
    transformed = pd.DataFrame(preprocess.transform(selected), columns=preprocess.get_feature_names_out())
    values = shap.TreeExplainer(pipeline.named_steps["model"])(transformed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shap.plots.beeswarm(values, show=False)
    plt.savefig(output_dir / "global-shap.png", bbox_inches="tight")
    plt.close("all")
    shap.plots.waterfall(values[0], show=False)
    plt.savefig(output_dir / "first-row-shap.png", bbox_inches="tight")
    plt.close("all")
