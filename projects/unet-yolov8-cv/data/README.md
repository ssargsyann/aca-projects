# Data

Synthetic U-Net samples are generated in memory and require no downloads. Each split/index is deterministic. The exact generator/configuration is recorded in run JSON.

YOLO uses a local COCO80-format directory containing `images/train`, `images/val`, `labels/train`, `labels/val`. Each image needs a same-stem text label, whose rows are `class_id center_x center_y width height` with normalized coordinates. The supported training command uses the official COCO80 class names, not arbitrary custom classes. It checks readable files, paired labels, valid rows and duplicate image contents before training. Empty label files are allowed for background images.

Use the explicit asset fetcher or supply an existing dataset directory. A new manifest is saved for every training run. Raw images, caches and labels stay ignored. See [provenance](../docs/provenance.md).
