# Model/use card: CV Zoo and educational geometry

## Intended use

Educational comparison of four pretrained image-understanding tasks and numerical exploration of grid/anchor concepts. There is no custom detector training, deployed service, or complete YOLO implementation. The synthetic geometry demo uses invented rectangles and requires no models/downloads.

## Pretrained tasks

- ResNet-50, ImageNet-1K V2: top-K image labels using the official weight transform.
- Faster R-CNN ResNet50-FPN V2, COCO V1: score-filtered boxes and categories.
- Mask R-CNN ResNet50-FPN V2, COCO V1: filtered object masks and boxes.
- DeepLabV3 ResNet50, COCO with VOC labels V1: 21-way semantic prediction, including background, restored to original-image coordinates.

The label spaces differ across tasks and are not interchangeable. COCO metadata contains label-index placeholders; its metadata list length should not be mistaken for a count of distinct usable object classes. DeepLab's weights use COCO images restricted to VOC labels, not arbitrary open-world semantic classes.

## Interpretation and limitations

The recorded run is qualitative inference on one image, not an evaluation set. Model scores are uncalibrated and threshold-dependent. No accuracy, mAP, mask-quality benchmark, latency comparison, or deployment safety claim is made. Resizing, source domain, occlusion, and unsupported objects affect outputs. Semantic regions do not distinguish individual objects; instance masks do.

Teaching annotations are approximate, and the geometry module has no collision policy or complete detector supervision/loss. See [geometry notes](docs/geometry-notes.md) for the version-specific tensor corrections and separate shape/position comparisons.

## Reproducibility and assets

Configuration, explicit enum identities, checksum verification, input hash, output coordinate frame, device, and dependency versions are recorded. Network access is explicit, weights are loaded sequentially from a chosen local directory, and inference rejects missing/corrupt caches. Only trusted official model files should be used. Source image/weight attribution is in [assets/README.md](assets/README.md). The earlier blanket MIT claim has been removed because the repository did not contain the referenced LICENSE file; code, images, weights, and model training datasets must not be assumed to share a license.
