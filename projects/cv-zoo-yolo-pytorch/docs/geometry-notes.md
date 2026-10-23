# Grid and anchor geometry: scope and corrections

This package computes educational geometry, not a YOLO training loss or a complete detector assignment procedure. Multiple objects can occupy the same cell; collision resolution, multi-scale assignment, objectness targets, loss construction, and NMS are outside this toy workflow.

## Coordinate contract

Boxes are finite `[xmin, ymin, xmax, ymax]` values. Reversed boxes are rejected; zero-area boxes have defined IoU zero in the general overlap function. Teaching annotation boxes must have positive width/height and fit in the annotated image. Empty `[0,4]` inputs are supported by pairwise IoU.

Image size is `(width, height)`. Cells are returned as `(column, row)`. With grid size S, the center is divided by `(width/S, height/S)` and floored. An exact right/bottom outer-boundary coordinate maps to the final cell; genuinely out-of-image centers are rejected instead of silently reassigned. Non-square images use separate horizontal/vertical scales.

## Two different IoU questions

**Shape-only matching** compares each annotation's width/height with anchor width/height at a common center. The intersection is `min(w_box,w_anchor) * min(h_box,h_anchor)`; IoU divides this by the corresponding union area. This depends only on size and aspect ratio, not the object's location.

**Cell-centered positional IoU** places anchor templates at the responsible cell center and compares the resulting absolute boxes. It includes center displacement. The JSON reports this second value separately. The original notebook's description claimed a common center while its code used the cell center; these should not be conflated.

The maintained figure draws the shape-matched anchor at the object center, with its responsible cell shaded separately. The optional cell-centered comparison remains numerical, explicitly labeled. Example anchors and three manual boxes are chosen for teaching and are not learned from a dataset.

## YOLO versions and tensor notation

[YOLOv1](https://arxiv.org/abs/1506.02640) assigns objects according to their centers and shares class predictions within a cell. Its illustrative output layout is `S x S x (B*5+C)`, with B predicted boxes rather than anchor priors; the paper's S=7, B=2, C=20 example gives 7x7x30.

[YOLOv2/YOLO9000](https://arxiv.org/abs/1612.08242) introduces anchor-based prediction and per-anchor class/objectness predictions. A conventional single-scale representation for B anchors and C classes is `S x S x B x (5+C)` (equivalently flattened final channels). This differs from YOLOv1's shared-class representation. Later YOLO families vary further and can be anchor-free; this project does not claim one formula or assignment rule describes every version.

Faster R-CNN and Mask R-CNN use proposal-based detection; ResNet classification and DeepLab semantic prediction are not both two-stage detectors. The original notebook's broad grouping of these architectures was inaccurate. This repository does not measure a universal speed advantage for any family.
