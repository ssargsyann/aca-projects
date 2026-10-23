# Model card: educational ICU mortality classifier

## Intended use

Portfolio demonstration and reproducible machine-learning experimentation on an authorized tabular dataset. The model estimates the supplied binary `hospital_death` label. It is not validated for diagnosis, triage, resource allocation, or patient-care decisions.

## Implementation

Training-only missingness filtering, median/mode imputation, one-hot encoding, and a weighted XGBoost classifier. Identifiers and configured APACHE mortality-probability fields are excluded. All fitted transformations are saved with the model in one pipeline. The default classifier threshold is used; no calibrated risk interpretation is established.

## Evaluation

The recovered local CSV was divided into a stratified random 80/20 split with seed 42. Internal validation ROC-AUC was 0.8926 and average precision was 0.5393. Positive-class precision was 0.3125 and recall was 0.7697. See [the full run report](docs/real-data-results.md) for counts, version information, comparison with a majority baseline, and aggregate plots.

Prediction completed on the separate unlabeled local file. No hidden-test evaluation is claimed because the correspondence between its rows and the separate labels file is not verified.

## Limits

Exact upstream data provenance and permitted use remain unresolved; see [data provenance](docs/data-provenance.md). Feature timing has not been audited. The random split may share site-specific patterns between training and validation. No temporal or external validation, calibration study, subgroup performance audit, threshold optimization, or prospective evaluation has been performed. The internal split should not be interpreted as evidence of performance at another hospital or on a future population.

## Artifacts and reproducibility

Models and predictions stay local. Training records the CSV checksum, configuration, retained columns, and installed dependency versions in `run.json`. Tested Python 3.12 dependency constraints are provided; the lightweight synthetic demo works without private data. Load only trusted joblib artifacts. The original notebook is retained as historical reference; use the maintained package for new experiments.
