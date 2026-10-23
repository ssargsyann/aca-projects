# Dataset and input contract

The original notebook explicitly downloads [Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis), published by Akash Patel, and reads `marketing_campaign.csv`. The provider credits Dr. Omar Romero-Hernandez and lists the dataset license as **CC0: Public Domain**. Source checked on 2026-10-23. The dataset license is distinct from a software license for this project's code.

The tested input was recovered from the same KaggleHub dataset's local version-1 cache. It has 2,240 rows, 29 columns, and 24 missing income entries. Raw records are kept local as a repository convention, not because the source prohibits redistribution.

## Retrieve version 1

```bash
python -m pip install -e ".[download]"
customer-segments download --output data/marketing_campaign.csv
```

The helper uses the official [KaggleHub library](https://github.com/Kaggle/kagglehub) and the versioned handle `imakash3011/customer-personality-analysis/versions/1`. It does not overwrite an existing destination. Kaggle may require you to complete its normal authentication or consent flow; never commit tokens. You may also download from the source page and place the file at the same path.

## Schema

The default configuration explicitly selects 16 numeric fields: `Year_Birth`, `Income`, `Kidhome`, `Teenhome`, `Recency`, the six `Mnt*` spending columns, and `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`.

- Tab-delimited source files and comma-delimited CSVs are detected from their header.
- Identifiers, campaign acceptance/response fields, constants, text categories, and enrollment dates are not used by the default model.
- Missing numeric entries are imputed with fitted medians; no rows are dropped.
- Values are clipped to fitted 1st/99th feature percentiles before scaling. These limits are reused for new customers.
- Required fields must be numeric or missing. Missing columns, infinite values, and entirely missing training features are rejected.
- Extra columns are ignored at inference. Output `row_number` preserves zero-based input row order.
- Aggregate profiles use original observed values, not clipped values, and skip missing income values. `income_observed` records the corresponding denominator.

Use the provider's data dictionary for definitions. Monetary amounts are reported in source units; this repository does not assume a particular currency. `Recency` refers to the source observation snapshot, not today's date.
