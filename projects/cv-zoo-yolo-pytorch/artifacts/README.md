# Local generated outputs

Geometry creates `geometry.json` and `grid-anchors.png`. The synthetic demo also creates an invented image and annotation file, with an explicit notice that no trained detector is being used.

Pretrained inference writes `results.json`, task preview PNGs, `instance-masks.npz`, and `semantic-labels.npy`. Detection/instance boxes are in original-image pixel coordinates. Semantic logits are resized to original image dimensions before argmax, so the label map and overlays align with the original RGB image. The results record thresholds, detector resize limits, weight identities/hashes, image hash, device, and library versions.

Run commands require new/empty output directories. Images, full-resolution masks, and model weights stay local. Only illustrative previews and auditable result metadata are published. A single image does not establish classification accuracy, detection mAP, mask IoU, or relative model speed.
