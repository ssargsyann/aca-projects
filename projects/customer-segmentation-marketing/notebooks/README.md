# Notebooks

- `quickstart.ipynb`: maintained, dataset-free synthetic demonstration using the package.
- `real-data-workflow.ipynb`: maintained Kaggle-data fitting, aggregate profiles, report plots, and saved-model inference.
- `original_experiment.ipynb`: unchanged original exploration, including its KaggleHub download and optional UMAP sandbox.

Install the `notebook` extras and select the environment containing this package. Start Jupyter from this project directory or its `notebooks/` directory. The real-data notebook expects the documented local CSV. Both maintained notebooks generate fresh run folders and keep records/model files local.

The historical notebook uses a different baseline: it drops missing rows, selects all remaining numeric fields, and does not apply the maintained model's quantile clipping. Its GMM identifiers must not be interpreted as K-Means persona identifiers. UMAP plots are exploratory visualizations, not evidence of campaign effectiveness. Install the separate `legacy` extras only to inspect that historical workflow; it is not the maintained model entry point.
