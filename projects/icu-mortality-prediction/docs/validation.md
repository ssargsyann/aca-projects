# Validation record

Checked on 2026-10-23 with Python 3.12 on Windows:

- All eight unittest tests passed, including reproducible demo generation and refusal to overwrite existing demo outputs.
- The installed `icu-mortality --help` command exposed training, prediction, and explanation commands.
- A separate synthetic-data smoke run trained a small XGBoost model, reloaded the saved pipeline, and wrote predictions for all 120 input rows.
- The optional explanation command generated nonempty global beeswarm and first-row waterfall PNG files for 20 synthetic rows.

Core tests cover train-only missingness selection, invalid labels, missing features, exclusion of identifiers and configured leakage columns, pipeline serialization, and unseen categories. The smoke run verifies command wiring and optional SHAP integration. Synthetic metrics and generated synthetic plots are deliberately not presented as ICU performance results.

Real-data training and saved-model inference have now been verified on the recovered local CSVs. See [the real-data run report](real-data-results.md) for the exact split and metrics. Dataset provenance, licensing, feature timing, site effects, and external validation still need review. The historical notebook is preserved unchanged and is not the maintained training entry point.

The maintained `demo` command and Jupyter quickstart now provide a dataset-free entry point. GitHub Actions additionally runs the command-line demo. Every generated demo run includes synthetic-data notices and a configuration marker; no synthetic scores are claimed as ICU benchmarks.

The checksum-verified `report` command reconstructed the real-data validation split and reproduced ROC-AUC 0.892606 and average precision 0.539299. Its aggregate ROC/precision-recall and confusion-matrix figures are included in the run report. The regression suite checks rejection of a mismatched report dataset; GitHub Actions also runs reporting on the synthetic demo.

Tested runtime versions are recorded in `constraints-py312-tested.txt`. The model card and data-provenance note document intended use, remaining evidence gaps, and relevant official upstream references without asserting an unverified local download source.

Both maintained Jupyter notebooks (`quickstart.ipynb` and `real-data-workflow.ipynb`) were executed end to end with a fresh workspace test kernel in the installed notebook environment. All cells completed successfully, including synthetic SHAP plots, actual local-data training, validation charts, and optional inference on the supplied hidden-test file. Published notebook sources have cleared outputs; executed copies and generated artifacts remain local. Documentation links were checked and aggregate figures were visually inspected.
