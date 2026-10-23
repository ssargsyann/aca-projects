# Notebooks

- `quickstart.ipynb`: maintained synthetic demo without a download.
- `real-data-workflow.ipynb`: maintained one-epoch real CIFAR smoke workflow, checkpoint reload, explicitly limited test evaluation, and report generation.
- `original_experiment.ipynb`: unchanged historical CUDA experiment and saved training log.

Install the `notebook` extras and select that environment. Start Jupyter from this project or its `notebooks/` directory. The real-data notebook reads `data/` by default and also accepts a `CIFAR10_DATA_ROOT` environment variable for an existing local cache. Set that variable before launching Jupyter. Generated run directories are unique and outputs stay local.

The original notebook checks official test accuracy after every epoch and reports 81.56% at epoch 15. The maintained trainer instead uses a separate validation partition and leaves official test evaluation to an explicit command. Original result claims and new smoke evidence are documented separately. Published maintained notebooks have cleared outputs.
