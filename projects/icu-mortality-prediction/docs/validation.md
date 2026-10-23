# Validation record

Checked on 2026-10-23 with Python 3.12 on Windows:

- All eight unittest tests passed, including reproducible demo generation and refusal to overwrite existing demo outputs.
- The installed `icu-mortality --help` command exposed training, prediction, and explanation commands.
- A separate synthetic-data smoke run trained a small XGBoost model, reloaded the saved pipeline, and wrote predictions for all 120 input rows.
- The optional explanation command generated nonempty global beeswarm and first-row waterfall PNG files for 20 synthetic rows.

Core tests cover train-only missingness selection, invalid labels, missing features, exclusion of identifiers and configured leakage columns, pipeline serialization, and unseen categories. The smoke run verifies command wiring and optional SHAP integration. Synthetic metrics and generated synthetic plots are deliberately not presented as ICU performance results.

Real-data training remains unverified because the source dataset is absent. Dataset provenance, licensing, feature timing, patient overlap, appropriate holdout design, and model quality still need to be assessed against the actual dataset. The historical notebook is preserved unchanged and is not the maintained training entry point.

The maintained `demo` command and Jupyter quickstart now provide a dataset-free entry point. GitHub Actions additionally runs the command-line demo. Every generated demo run includes synthetic-data notices and a configuration marker; no synthetic scores are claimed as ICU benchmarks.
