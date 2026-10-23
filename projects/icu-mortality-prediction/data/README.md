# Data

To try the code without locating the original dataset, run `icu-mortality demo --output artifacts/demo`. It generates clearly marked synthetic inputs under the output directory. Jupyter users can open `notebooks/quickstart.ipynb`. See `docs/input-schema.md` for the demo fields and real-data requirements.

The source notebook expects a local CSV named `icu_train.csv` with a binary `hospital_death` target. The dataset itself, its download URL, license, and exact release are not recorded in the original repository and have not been verified.

The original local CSV was recovered from the user's Jupyter working files on 2026-10-23. It contains 77,956 rows and 186 columns, including the target (6,728 positives). The maintained pipeline completed training and validation on that file. A separate local `icu_hidden_test.csv` contains 13,757 rows and 185 predictor columns; saved-model inference completed on all rows. See `docs/real-data-results.md`. These files are not redistributed in this repository.

Place an appropriately authorized dataset at `data/icu_train.csv`. Do not commit patient records or credentials. Before reporting results, document the source, version, permitted use, feature definitions, and availability of each feature at prediction time.

Training requires both target classes, numeric and/or categorical predictors, and enough examples per class for a stratified split. The configured identifiers and APACHE mortality-probability columns are removed. This exclusion list alone does not establish that the remaining features are free of leakage.

The current split is stratified by row. If a dataset contains repeated patients, correlated encounters, multiple hospitals, or a deployment time horizon, use an appropriate grouped, site-based, or temporal validation design before interpreting performance.
