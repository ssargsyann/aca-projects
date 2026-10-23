# Validation record

Checked on 2026-10-23 with Python 3.12 in the workspace environment:

- Five regression tests passed. Cases cover fitted-statistic reuse, missing columns, invalid/infinite/all-missing inputs, CSV/TSV equivalence, reproducible demo generation, saved-model round trips, preserved row order, normalized GMM probabilities, independent component IDs, exact boundary flags, ignored ID changes, and preservation of existing runs.
- Training completed on all 2,240 source rows with 24 missing income values imputed. K=2, silhouette 0.360505, seven PCs, retained variance 0.804103. GMM converged.
- The bundle reloaded and wrote 2,240 local assignments.
- Aggregate report figures were generated and visually inspected.

The synthetic demo provides code verification without a dataset download. Real-data run results are documented separately in `results.md`; model quality is descriptive and in-sample. Initialization checks are not a substitute for sampling/temporal stability. Raw records, models, and assignments are not committed.

Both maintained notebooks were executed end to end in a fresh workspace-only kernel: the synthetic quickstart and the real-data workflow completed every cell, including fitting, aggregate figures, and saved-model assignment. Published notebook sources have cleared outputs. Relative documentation links and notebook code were checked.

The versioned KaggleHub retrieval helper was tested with KaggleHub 0.4.3 against the existing source cache. It copied version 1 to a fresh destination and produced SHA-256 `cd0affa36b1b981e80ba0e27767e9b3ab723f7a3dba948f722af81abc6b990ea`, matching the trained input. This tests the helper/cache path; a forced uncached download and a fresh-account authentication flow were not exercised.

Git ignore checks confirmed that the local source CSV, saved model, and row assignments are excluded. Only aggregate result metadata and figures are published. GitHub Actions runs the regression suite and the synthetic demo/report commands without downloading the dataset.
