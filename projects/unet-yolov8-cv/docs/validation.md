# Validation record

Checked on 2026-10-23 with Windows, Python 3.12 and CPU PyTorch. See recorded dependency versions and exact configurations in the result JSON files.

- Nine network-free regression tests passed, covering deterministic sample/split identity, even/odd/rectangular/minimum shapes, crop/pad alignment, known/empty-mask overlap arithmetic, finite loss gradients, a tiny train/select/checkpoint/predict round trip, invalid configuration, dataset row/pair/duplicate validation, and early missing-checkpoint failure before importing YOLO.
- The six-epoch width-8 U-Net smoke completed with separately seeded fixed train/validation/test sets; the selected checkpoint was reloaded before final evaluation. The larger width-64 preset was not retrained.
- Official asset retrieval completed; all three downloads matched the recovered local base/video and COCO8 file contents. Their full hashes are pinned in the fetch helper and committed manifest.
- Actual YOLOv8n training completed one epoch on four COCO8 train images with four validation images. The final best-checkpoint validation and epoch CSV are recorded separately. Reloaded best.pt image inference returned one detection.
- The pretrained base processed 48 real traffic frames without accumulating results in a list; empty outputs were retained in JSONL. No traffic ground-truth quality evaluation was performed.
- Full-video inference processed all 377 frames and wrote 165 frame-wise boxes in 113 frames. The output MP4 was decoded again and contained all 377 frames. Its JSONL counts and class totals were checked against the report; the earlier 48-frame output was also decoded and verified.
- Every cell in both maintained notebooks executed successfully in fresh kernels, including another synthetic training/prediction run and actual COCO8 training/reloaded-checkpoint image/eight-frame video inference. Published notebook sources have cleared outputs.
- The training curve, synthetic predictions and selected traffic-frame figure were visually inspected. Conflicting labels and the image's cat-versus-dog disagreement are disclosed in the results report.

GitHub checks run the package regression suite without assets or YOLO dependencies. Full model binaries, raw datasets and source/annotated videos remain local. See the model/use card and provenance for scope.
