# Local outputs

Choose a new/empty directory for each command. U-Net training saves the validation-selected state-dictionary checkpoint, history, final validation/test scores, and figures. Prediction writes original-size probability/mask outputs. YOLO training saves local dataset YAML, input hashes, `run/results.csv`, `run/weights/best.pt`, `last.pt`, and final metrics JSON. YOLO prediction writes JSONL incrementally, up to four previews, and MP4 for video. Settings/runtime files also stay local. Interrupted outputs are retained for inspection and are not reused automatically.

Artifacts and weights are ignored. Committed `docs/` copies contain small reports/figures and source attribution, not full datasets/videos or model binaries.
