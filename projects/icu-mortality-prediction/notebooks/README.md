# Notebooks

- `quickstart.ipynb`: maintained synthetic demonstration requiring no dataset download.
- `real-data-workflow.ipynb`: maintained training, aggregate validation plots, and optional hidden-file inference using your authorized local CSVs.
- `original_experiment.ipynb`: unchanged historical experiment, with the limitations below.

For a maintained, immediately runnable Jupyter example, open `quickstart.ipynb`. It generates explicitly synthetic data and uses the packaged pipeline. The notes below apply to the historical notebook only.

`original_experiment.ipynb` preserves the original notebook unchanged for reference. It uses working-directory-relative paths and filters high-missingness columns before splitting the data. Its output and claims have not been independently reproduced.

Use the maintained command-line pipeline in `src/icu_mortality/` for new runs. That pipeline learns feature missingness, imputers, encoders, and class weighting exclusively from the training partition. Install the `notebook` optional dependencies if you want to inspect or run the historical notebook.
