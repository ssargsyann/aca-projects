"""Fit a descriptive batch segmentation and save its complete inference state."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import NumericFeatures, load_data
from .profiles import assignments, summarize


def train(data_path, config_path, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty; existing runs are preserved.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not 0 < config["variance_threshold"] < 1:
        raise ValueError("variance_threshold must be between 0 and 1.")
    if not 0 <= config["boundary_margin"] <= 1:
        raise ValueError("boundary_margin must be in [0, 1].")
    frame = load_data(data_path)
    preprocessor = Pipeline([
        ("features", NumericFeatures(config["features"], config["clip_quantiles"])),
        ("scale", StandardScaler()),
    ])
    with threadpool_limits(limits=1):
        scaled = preprocessor.fit_transform(frame)
        if not np.any(np.var(scaled, axis=0) > 0):
            raise ValueError("At least one configured feature must vary.")
        pca = PCA(n_components=config["variance_threshold"], svd_solver="full")
        projected = pca.fit_transform(scaled)
        spectrum = PCA(svd_solver="full").fit(scaled).explained_variance_ratio_.tolist()
        distinct = len(np.unique(projected, axis=0))
        candidates, models = [], {}
        for clusters in sorted(set(config["k_values"])):
            if not isinstance(clusters, int) or not 2 <= clusters < len(frame) or clusters > distinct:
                raise ValueError("Each candidate K must be an integer >= 2, below row count, and <= distinct feature rows.")
            model = KMeans(n_clusters=clusters, random_state=config["random_state"], n_init=config["n_init"])
            labels = model.fit_predict(projected)
            sample_size = min(config["silhouette_sample_size"], len(frame))
            score = silhouette_score(projected, labels, sample_size=sample_size, random_state=config["random_state"])
            candidates.append({"k": clusters, "silhouette": float(score), "inertia": float(model.inertia_)})
            models[clusters] = model
        if not candidates:
            raise ValueError("At least one candidate K is required.")
        best = max(candidates, key=lambda result: result["silhouette"])
        final = models[best["k"]]
        stability = []
        for seed in config["stability_seeds"]:
            alternate = KMeans(n_clusters=best["k"], random_state=seed, n_init=config["n_init"]).fit_predict(projected)
            stability.append({"seed": seed, "adjusted_rand_index": float(adjusted_rand_score(final.labels_, alternate))})
        gmm = GaussianMixture(n_components=best["k"], random_state=config["random_state"], reg_covar=config["gmm_reg_covar"], n_init=3, max_iter=300).fit(projected)
    pipeline = Pipeline([("preprocess", preprocessor), ("pca", pca), ("kmeans", final)])
    profiles = summarize(frame, final.labels_)
    bundle = {"pipeline": pipeline, "gmm": gmm, "config": config, "profiles": profiles.to_dict(orient="records")}
    assigned = assignments(bundle, frame)
    metrics = {
        "evaluation": "in_sample_descriptive_clustering", "rows": len(frame),
        "input_columns": len(frame.columns), "features": len(config["features"]),
        "missing_feature_values": int(frame[config["features"]].isna().sum().sum()),
        "pca_components": int(pca.n_components_), "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "selected_k": best["k"], "silhouette": best["silhouette"], "candidates": candidates,
        "silhouette_sample_rows": min(config["silhouette_sample_size"], len(frame)),
        "initialization_stability": stability, "gmm_converged": bool(gmm.converged_),
        "gmm_boundary_rows": int(assigned["gmm_boundary"].sum()),
        "pca_variance_spectrum": spectrum,
    }
    metadata = {
        "config": config, "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ["pandas", "numpy", "scikit-learn", "joblib", "threadpoolctl"]},
        "feature_medians": preprocessor.named_steps["features"].medians_.to_dict(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "segmentation.joblib")
    profiles.to_csv(output_dir / "cluster_profiles.csv", index=False)
    assigned.to_csv(output_dir / "assignments.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics
