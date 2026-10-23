# Data and asset provenance

## Synthetic shapes

No external dataset. `data.py` generates noisy gradient backgrounds with colored geometric shapes and exact binary masks using NumPy/Pillow. Generator version `pillow_shapes_v1_seed_sequence_seed_split_index`, seed/split IDs/index and image size determine each pair. This differs from the original stateful OpenCV random generator. The committed reports contain aggregate metrics and three generated examples; no external clinical or personal data is used.

## YOLOv8n and COCO8

The local base checkpoint was recovered from the user's existing files, then compared byte-for-byte by SHA-256 with a fresh download from [the official v8.4.0 release](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt). It is the base used for fresh training and video prediction, not one of the recovered historical fine-tuned weights.

COCO8's local 4/4 images and labels were compared with a fresh [official archive](https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip). File identities are in [the manifest](coco8-manifest.json). The installed Ultralytics official `coco8.yaml` supplies the 80 class names. Only path/train/val/names are copied into the generated local YAML; no download directive is propagated.

[Ultralytics COCO8 documentation](https://docs.ultralytics.com/datasets/detect/coco8/) describes its 4-train/4-validation debugging purpose and COCO train2017 origin. That origin can overlap pretrained COCO training data. It is not held-out evidence for generalization. See [COCO terms](https://cocodataset.org/#termsofuse) for image/annotation terms and individual image rights; dataset packaging and image licenses are not interchangeable. COCO photos and full weights are not committed here. The [Ultralytics license page](https://www.ultralytics.com/license) explains its upstream license choices; this project makes no unsupported MIT claim.

## Traffic sample

The notebook used Intel IoT DevKit's [car-detection.mp4](https://github.com/intel-iot-devkit/sample-videos/blob/master/car-detection.mp4). The recovered local file matches the independently downloaded upstream bytes. Intel's sample-videos repository declares [CC BY 4.0](https://github.com/intel-iot-devkit/sample-videos/blob/master/LICENSE). The documentation credits the source of its extracted preview; full source/annotated videos remain ignored local files.

[Download manifest](download-manifest.json) records exact source URLs, bytes and full SHA-256 values. `fetch-assets` pins these observed hashes; it fails if upstream bytes change. Existing destinations are preserved. Downloads are explicit and separate from training/inference. Fresh asset retrieval and local file comparisons were verified during this restructuring.
