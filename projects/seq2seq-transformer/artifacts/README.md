# Local run artifacts

Each train command uses a new/empty output directory and writes exact split CSVs/manifests, a validation-selected `best.pt` state-dictionary checkpoint, JSON metrics/history, test predictions, training/attention PNGs and attention JSON. Prediction and evaluation use separate new/empty directories. The checkpoint records architecture, vocabulary, selected epoch and split manifest, and is loaded with restricted tensor-state deserialization and strict keys.

Weights and generated CSVs remain ignored. Small measured reports/figures and all held-out test prediction records are committed under `docs/`. Failed/partial runs are retained for inspection and never silently overwritten.
