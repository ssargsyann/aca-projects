# Maintained notebooks

- `quickstart.ipynb`: tiny two-epoch smoke, saved-model evaluation/prediction and offline mechanics. It verifies the execution path and is not expected to reach the reported run's accuracy.
- `verified-date-workflow.ipynb`: complete measured ten-epoch configuration, validation-selected checkpoint, held-out greedy metrics, reloaded example and labeled attention visualization. No downloads needed.
- `original_experiment.ipynb`: unchanged imported notebook with historical code/outputs. Its external tokenizer demo, unsplit date sampling, timing/position claims and attention labeling issues are documented separately. Use maintained notebooks for reproducible execution.

Launch Jupyter from the project or notebooks directory and select the environment containing this package. Every maintained execution chooses a unique output directory under `artifacts/`. Committed maintained sources have cleared outputs; validated outputs live separately from source notebooks. Domain is canonical English dates in 1950–2050, with no product or general translation claim.
