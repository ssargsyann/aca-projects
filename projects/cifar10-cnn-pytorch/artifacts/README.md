# Local artifacts

Training writes `best.pt`, `history.json`, `run.json`, and (for CIFAR data) `split-indices.json`. Checkpoints are selected by validation accuracy only, with the earliest epoch kept for ties. The state dictionary, architecture/class order, normalization, configuration, selected epoch, and data-kind marker are stored together. Optimizer/resume state is not saved; this is an inference checkpoint.

Run metadata records actual sample counts, device, Python/package versions, parameter count, and whether the run is synthetic, subset smoke, or full training. Full training does not automatically evaluate the test set. `evaluate` writes aggregate loss, accuracy, and confusion counts; `predict` writes class names and all ten scores.

Dataset images, checkpoints, and generated outputs remain local. Only aggregate evidence and figures are committed. Load trusted checkpoints only; loading uses `weights_only=True`, strict state-dict matching, and checks class order/normalization. Restricted loading reduces deserialization exposure but is not a blanket guarantee for arbitrary files.
