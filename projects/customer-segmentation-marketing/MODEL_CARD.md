# Model card: descriptive customer segments

## Intended use

Educational analysis of the supplied retail dataset, reproducible segment profiling, and consistent assignment of new rows using a saved model. Segment descriptions summarize observed spending and discounted-purchase behavior relative to other segment means. They are not validated customer identities, predictions of future sales, or evidence that a particular campaign will work.

## Method

Sixteen explicitly selected numeric demographic/purchasing fields are median-imputed, clipped at fitted 1st/99th percentiles, standardized, and projected with PCA retaining at least 80% variance. K-Means candidates 2 through 8 are compared by in-sample silhouette. A separate Gaussian mixture with the selected number of components supplies model-specific membership probabilities and boundary flags.

The full fitted batch is used for descriptive clustering. Silhouette is not supervised accuracy or a held-out generalization score. Repeat-initialization adjusted Rand indices assess sensitivity to optimization seeds on the same rows; they do not measure sampling stability or temporal robustness.

## Interpretation

Observed profiles are reported in source units, before clipping/imputation, with explicit counts. A descriptive segment name includes its segment number, relative spending level, and relative discounted-purchase count. Repeated fitting can change cluster numbering. GMM labels have no one-to-one semantic mapping to K-Means labels. Boundary flags use an unrounded gap between the largest two GMM probabilities.

## Limits

The cohort is small and historical. Geographic/temporal representativeness, campaign lift, subgroup effects, and out-of-time stability have not been established. Numeric birth year, income, and household fields affect distances; text categories and dates are excluded in this baseline. Clipping reduces extreme-value influence but is a modeling choice, not a verified correction of data errors. New behavior outside the training ranges is clipped and may be assigned poorly. Marketing actions require separate business review and experiments.

## Data and artifacts

The notebook identifies the Kaggle Customer Personality Analysis dataset; see [data provenance and schema](data/README.md). Raw records, fitted models, and row assignments remain local. Only aggregate run results and figures are published. Saved joblib artifacts must be trusted. The original notebook remains available as a distinct historical experiment.
