# Validation record

Checked on 2026-10-23 in the Python 3.12 workspace environment:

- Eight network-free regression tests passed. They cover randomized custom IoU against Torchvision's reference, empty/zero-area/invalid boxes, non-square grids and exact boundaries, anchor shape invariance to object translation, offline-demo persistence guards, filtered/empty instance outputs, semantic original-size alignment, and missing/corrupt weight-cache rejection without silent downloads.
- All four actual pinned pretrained models loaded strict state dictionaries and completed CPU inference on the real sample.
- Original-image coordinate dimensions were preserved for boxes/masks, and the semantic output was explicitly realigned.
- The explicit sample retrieval fetched the public Darknet image to a fresh destination and verified its hash matched the used local image.
- Full weight hashes were recorded, the montage and geometry figures were generated, and machine-readable outputs were saved separately.
- Every cell in both maintained notebooks (`quickstart.ipynb` and `real-inference.ipynb`) executed successfully in a clean kernel; the real notebook ran all four models again. Published notebook sources have cleared outputs.
- The published montage and grid/anchor figures were visually inspected for readable layout and correct image alignment.

Inference uses existing trusted weights and does not silently access the network. The explicit downloader for the large weight files was not forced to redownload an already complete cache. Geometry is a teaching calculation, not a full YOLO training implementation. No model benchmark is implied. See the results report and model/use card for scope.
