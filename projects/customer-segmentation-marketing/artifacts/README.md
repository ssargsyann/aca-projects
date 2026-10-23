# Local run artifacts

Training creates a complete `segmentation.joblib` bundle, `metrics.json`, `run.json`, `cluster_profiles.csv`, and row-level `assignments.csv`. Reporting adds aggregate figures. These generated outputs are ignored by Git. Existing nonempty training/demo run directories are preserved; choose a fresh output path for another experiment.

`run.json` records the input checksum, configuration, fitted feature medians, and package versions. The saved bundle includes preprocessing, PCA, K-Means, GMM, and descriptive segment names. Load only trusted joblib files; deserialization can execute code.

K-Means segment numbers and GMM component numbers are independent. GMM probabilities describe a fitted mixture model, not the probability that a customer will respond to a campaign. Segment numbers and names may change when refitting; retain the saved model for consistent future assignments.
