# Original exploration

`original_experiment.ipynb` preserves the original notebook unchanged for reference. It uses working-directory-relative paths and filters high-missingness columns before splitting the data. Its output and claims have not been independently reproduced.

Use the maintained command-line pipeline in `src/icu_mortality/` for new runs. That pipeline learns feature missingness, imputers, encoders, and class weighting exclusively from the training partition. Install the `notebook` optional dependencies if you want to inspect or run the historical notebook.
