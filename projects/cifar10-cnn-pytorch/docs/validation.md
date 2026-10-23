# Validation record

Verified in the workspace on 2026-10-23:

- Seven network-free regression tests passed, covering disjoint stratified/reproducible splits and limits, training-pool-only dataset requests, deterministic normalized evaluation transforms, finite forward/backward gradients, deterministic eval mode, sample-weighted epoch loss, checkpoint round-trip, grayscale/RGB inference, probability sums, class-order/normalization rejection, existing-run preservation, and rejection of reports from another checkpoint.
- The installed command interface exposes `download`, `demo`, `train`, `evaluate`, `predict`, and `report`.
- Real CIFAR training completed with the explicit one-epoch/1,000-train/200-validation smoke configuration.
- A separate 500-image stratified test-subset evaluation completed and wrote aggregate confusion counts. Report generation verified the evaluation's checkpoint hash against the selected `best.pt`.
- The checkpoint reloaded for single-image inference. Synthetic demo/checkpoint inference was also exercised in the regression suite.

This verifies executable workflows, not a fresh full benchmark. The historical 81.56% is preserved as archived notebook evidence and remains distinct from smoke results. Raw data, sample images, and checkpoints are not committed. See `results.md` and the model card for scope and limitations.

Both maintained notebooks were executed end to end using a workspace-only kernel with all cells completing successfully. This includes the synthetic demo and the actual one-epoch CIFAR subset training, independent limited test evaluation, checkpoint reload, and report generation. Published maintained notebooks have cleared outputs; executed copies stay local. The official archive MD5 matched the source value, and aggregate figures were visually inspected.
