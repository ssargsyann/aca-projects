# Input schema

## Synthetic demonstration

The `demo` command generates these invented columns. These are illustrative names and distributions, not a clinical data specification:

| Column | Type | Purpose |
| --- | --- | --- |
| encounter_id | Integer | Row identifier, excluded from training |
| age | Numeric, some missing | Example numeric predictor |
| heart_rate | Numeric | Example numeric predictor |
| unit_type | String | Example categorical predictor |
| hospital_death | 0 or 1 | Arbitrary balanced label, not medically derived |

## Original dataset

The original local dataset was recovered and its column presence was verified by a successful training run: 77,956 rows, 186 columns including `hospital_death`, and 122 predictors retained before encoding. A separate unlabeled file with 13,757 rows passed inference schema checks. The raw files are not included in Git; their upstream provenance and full feature definitions remain unverified. The configured identifier and APACHE probability columns are excluded. Do not assume the synthetic demo reproduces this dataset.

- Training CSV: must include the configured target, both classes 0 and 1, no missing labels, and sufficient rows per class for the configured stratified split.
- Predictors: numeric and categorical columns. Missing values are imputed using training statistics. Columns exceeding the configured training missingness threshold are dropped.
- Excluded columns: identifiers and configured APACHE mortality probabilities. Verify all remaining columns are available at the intended prediction time.
- Inference CSV: must contain every retained feature recorded in `run.json`; extra columns are ignored. Target labels are not required. Unseen categorical values are accepted.
- Units, coding conventions, and meaning of missing values must match training data. The pipeline checks column presence; it does not establish semantic compatibility.

## Recovering the real data

Look for the original `icu_train.csv` beside the notebook on your computer, in the Jupyter working directory, or in the original course/download location. A notebook references a CSV but does not bundle it. Before adding a source link or download command, verify the actual dataset identity, version, and permitted use. Store local records in `data/`, which is ignored by Git; publish retrieval instructions only when their provenance and permissions are known.
