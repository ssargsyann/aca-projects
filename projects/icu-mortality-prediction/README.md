# ICU Mortality Prediction

An educational classification project using tabular ICU features, XGBoost, and optional SHAP explanations. The original notebook is preserved in [notebooks](notebooks/README.md); maintained training and inference code lives in a Python package.

[Automated checks](https://github.com/ssargsyann/aca-projects/actions/workflows/icu-checks.yml) · [Model card](MODEL_CARD.md) · [Real-data results](docs/real-data-results.md) · [Data provenance](docs/data-provenance.md)

## Scope and results

The target is `hospital_death` (0 or 1). The recovered local dataset has now been used for an actual training and validation run: **ROC-AUC 0.8926**, **average precision 0.5393**, and **positive-class recall 0.7697** on a stratified 20% validation partition. See the [real-data run report](docs/real-data-results.md) for precision, sample counts, reproducibility details, and limits. Prediction also completed on 13,757 unlabeled test rows; no hidden-test accuracy is claimed.

Raw datasets, trained models, and row-level predictions remain local and are excluded from Git. Dataset provenance, redistribution rights, and feature timing still need confirmation. This project is not clinically validated or ready for patient-care use.

![Internal validation ROC and precision-recall curves](docs/figures/validation-curves.png)

## Structure

```text
configs/default.json        Training and feature-selection settings
src/icu_mortality/
  data.py                   Label validation and train-fitted feature selection
  model.py                  Preprocessing and XGBoost pipeline
  train.py                  Training, validation metrics, and run metadata
  predict.py                Saved-pipeline inference
  explain.py                Optional SHAP plots
  report.py                 Checksum-verified validation charts
  cli.py                    Command-line interface
notebooks/                  Original experiment, preserved for reference
tests/                      Synthetic-data regression tests
data/                       Local dataset instructions
artifacts/                  Generated outputs (ignored by Git)
```

## Install

Use Python 3.10 or newer. From the repository root:

```bash
cd projects/icu-mortality-prediction
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell, or `source .venv/bin/activate` on macOS/Linux, then:

```bash
python -m pip install -e .
```

For notebooks and SHAP, install `python -m pip install -e ".[notebook]"`. Dependency ranges are in `pyproject.toml`; each training run records the actual versions.

For the tested Python 3.12 runtime, report, and SHAP versions, install with:

```bash
python -m pip install -c constraints-py312-tested.txt -e ".[reports,explain]"
```

These constraints record the tested Windows environment, not a complete cross-platform Jupyter lock. Other supported Python versions resolve compatible dependencies from the declared ranges.

## Data

No dataset download is needed to try the complete synthetic demo below. For real-data training, provide your original CSV separately.

Read [data/README.md](data/README.md), then supply `data/icu_train.csv` with a binary `hospital_death` column. The original repository does not specify a verified download source or license. Inference CSVs need the retained predictor columns; the target is optional and unused during inference. Extra columns are ignored, missing required features are rejected, and unseen categories are accepted.

## Try it immediately

After installation, run:

```bash
icu-mortality demo --output artifacts/demo
```

One command generates invented data, trains a small model, saves metrics and run metadata, reloads the model, and writes example predictions. Labels are arbitrary and balanced; no real patient records are used. Output is explicitly marked `SYNTHETIC_DEMO_ONLY`. These metrics are not ICU benchmarks. The prediction example reuses rows from the generated file to demonstrate inference, not to create a second evaluation.

For SHAP plots too:

```bash
python -m pip install -e ".[explain]"
icu-mortality demo --output artifacts/demo-with-shap --explain
```

Use a fresh output directory for each run; the demo refuses to overwrite an existing nonempty directory. You can choose `--rows` (at least 40) and `--seed`.

Prefer Jupyter? Install `python -m pip install -e ".[notebook]"`, start `jupyter lab`, and open [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb). Select the environment where this package is installed. This notebook runs the maintained synthetic demo and displays its outputs.

### What the demo creates

```text
artifacts/demo/
  README.txt                 Synthetic-data notice
  synthetic_train.csv        Invented labeled rows
  synthetic_inference.csv    Example predictor-only rows
  demo-config.json           Exact demonstration settings
  demo-summary.json          Labeled summary and synthetic metrics
  predictions.csv            Example scores and classes
  model/
    pipeline.joblib
    metrics.json
    run.json
```

See [the input schema](docs/input-schema.md) for the demo column definitions and the real-data contract.

## Train and evaluate

Run from this project directory:

```bash
icu-mortality train --data data/icu_train.csv --config configs/default.json --output artifacts/run
```

The pipeline splits rows with stratification (80/20 by default), then learns missingness-based column selection, median/mode imputation, and one-hot encoding only on training rows. Class weighting also uses training labels only. One saved pipeline contains all preprocessing and the classifier.

Outputs:

- `pipeline.joblib`: fitted preprocessing and classifier.
- `metrics.json`: ROC-AUC, average precision, accuracy, per-class precision/recall/F1, and sample counts.
- `run.json`: configuration, dataset checksum, retained features, and dependency versions.

The validation split is not a separate final test set. Repeated tuning against it can overfit evaluation. Patient overlap, site effects, feature timing, calibration, and external validation require a dataset-specific review.

## Predict

```bash
icu-mortality predict --model artifacts/run/pipeline.joblib --data data/new_encounters.csv --output artifacts/predictions.csv
```

Rows preserve input order with zero-based `row_number`, `mortality_score`, and `predicted_class`. Scores are uncalibrated model outputs. The default threshold is not clinically selected. Load only trusted model files: joblib deserialization can execute code.

## Explain (optional)

```bash
python -m pip install -e ".[explain]"
icu-mortality explain --model artifacts/run/pipeline.joblib --data data/new_encounters.csv --output artifacts/explanations --max-rows 100
```

This writes a global SHAP beeswarm for the first selected rows and a waterfall for the first row. SHAP describes model behavior, not causal effects. This optional path requires compatible SHAP/XGBoost versions and is separate from core training tests.

## Verify

```bash
python -m unittest discover -s tests -v
icu-mortality --help
```

Tests use synthetic data to cover training-only missingness selection, invalid targets, schema errors, excluded columns, serialization, and unseen categories. They do not establish ICU predictive quality. GitHub Actions runs these checks when this project changes.

See the [validation record](docs/validation.md) for completed checks and remaining limits.

## Recreate validation charts

```bash
python -m pip install -e ".[reports]"
icu-mortality report --run artifacts/real-data-run --data data/icu_train.csv --output artifacts/real-data-report
```

The report verifies the training-file checksum against `run.json`, reconstructs the recorded validation split, and writes ROC/precision-recall curves, a confusion matrix, and aggregate metrics. It does not train a new model or evaluate the hidden test set. Only load trusted run directories.

For the full real-data Jupyter workflow, open [notebooks/real-data-workflow.ipynb](notebooks/real-data-workflow.ipynb). It trains on your authorized local CSV and generates the same diagnostics. Both maintained notebooks leave raw records and model artifacts local.
