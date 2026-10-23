# Dataset provenance and regeneration

No external files, downloads, credentials or pretrained models are required. `data.py` deterministically creates canonical English/ISO pairs from all valid Gregorian dates between 1950-01-01 and 2050-12-31. Leap years and month lengths follow Python's calendar implementation. A fixed local seed shuffles the full calendar once; train/validation/test are nonoverlapping slices, without duplicate dates within a split. They share the same range/format distribution, so this is not an out-of-distribution test.

`date-transformer dataset --config configs/dates-cpu.json --output artifacts/exported-dates` writes exact CSVs. The same exporter runs automatically during training. CSV encoding is UTF-8, header `human_date,iso_date`, LF newlines and standard CSV quoting. [The manifest](dataset-manifest.json) records full file SHA-256 values, counts, date range, seed, vocabulary and generator version. The measured test membership can also be audited through [all test outputs](test-predictions.jsonl).

The fixed vocabulary uses only the declared format's month-name letters, digits, spaces, commas and hyphens, plus PAD/BOS/EOS. Nothing is fitted on test data. Input supports canonical spelling/capitalization, no leading-zero day, valid calendar dates and the declared year range. Encoding raises on unknown characters/overflow instead of truncation.

Raw generated CSVs and checkpoints remain ignored local artifacts. Only small reports, illustrative generated examples and prediction records are committed. The original notebook used random draws with possible duplicate dates and only days 1–28, without held-out splits; its results are not directly comparable to the maintained generator.
