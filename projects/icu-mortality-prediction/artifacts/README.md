# Generated artifacts

Training writes `pipeline.joblib`, `metrics.json`, and `run.json` into the chosen output directory. Prediction and optional SHAP plots also belong here. Generated artifacts are ignored by Git.

`run.json` records the configuration, data checksum, retained features, and package versions. Model scores are not calibrated probabilities or clinical recommendations. Load only trusted joblib files; deserialization can execute code. Compatible package versions are required when reloading a model.
