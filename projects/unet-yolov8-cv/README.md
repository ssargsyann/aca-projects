# U-Net Segmentation and YOLOv8 Detection

Two independent computer-vision workflows: a U-Net implemented from convolution blocks for synthetic binary segmentation, and YOLOv8n fine-tuning/inference on local COCO8/image/video inputs. Includes configurable commands, validation-selected checkpoints, deterministic synthetic data, reproducible reports, and maintained notebooks. The original notebook is preserved unchanged.

[Verified results](docs/results.md) · [Model/use card](MODEL_CARD.md) · [Data and asset provenance](docs/provenance.md) · [Automated checks](https://github.com/ssargsyann/aca-projects/actions/workflows/unet-yolo-checks.yml)

## Verified behavior

- **U-Net:** six CPU epochs with width 8 and fixed 192/48/48 synthetic train/validation/test samples at 64x64. Best epoch 6: validation mean foreground IoU **0.8919**, test mean IoU **0.9022** and Dice **0.9478**. This measures the synthetic shape generator only.
- **YOLOv8n:** a fresh one-epoch CPU fine-tune on COCO8 (4 train / 4 validation images), plus reloaded-checkpoint image inference. Final best-checkpoint validation mAP50 **0.5792** is a tiny pipeline smoke result, not a generalization claim.
- **Video:** the official pretrained base processed all **377 frames** in bounded memory and produced a decodable annotated MP4 and per-frame JSONL. At 320px/confidence 0.25 it returned **165 frame-wise detections across 113 frames**, including conflicting `car`/`cell phone` outputs. These are not counts of distinct vehicles or a ground-truth accuracy score. The separate first-48-frame smoke had zero detections.

![U-Net predictions on three fixed synthetic test samples](docs/figures/unet-predictions.png)

## Structure

```text
configs/                         U-Net smoke/full and YOLO smoke settings
src/unet_yolo/
  model.py                       Original encoder/decoder, configurable width
  data.py                        Deterministic shape generation by split/index
  metrics.py                     BCE + Dice and per-image overlap metrics
  segmentation.py                Train/select/checkpoint/predict/report
  detection.py                   Validated local data, fine-tune, streamed frames
  assets.py                      Explicit checksum-pinned asset downloads
  cli.py                         Command-line entry points
tests/                           Numerical, split, checkpoint and I/O contracts
notebooks/                       Two maintained workflows and original notebook
docs/                            Verified JSON/CSV reports and figures
assets/ data/ artifacts/          Local assets/data/outputs; ignored except guides
```

## Install

Python 3.10+; tested with Python 3.12 on Windows CPU. From the collection root:

```bash
cd projects/unet-yolov8-cv
python -m venv .venv
```

Activate `.venv\Scripts\Activate.ps1` in Windows PowerShell or `source .venv/bin/activate` on macOS/Linux. Install CPU PyTorch first:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[yolo,notebook]"
```

For U-Net only, `python -m pip install -e .` is sufficient after PyTorch installation; YOLO/OpenCV/Jupyter are optional. For CUDA, choose a compatible build from [PyTorch](https://pytorch.org/get-started/locally/). See `constraints-py312-cpu-tested.txt` for recorded direct-dependency versions, not a complete cross-platform lock.

## Synthetic U-Net: no dataset download

```bash
unet-yolo unet-train --config configs/unet-smoke.json --output artifacts/unet-demo
unet-yolo unet-predict --checkpoint artifacts/unet-demo/best.pt --image YOUR_SHAPE_IMAGE.png --output artifacts/unet-prediction
```

Training writes `best.pt`, `results.json`, `training.png`, and `predictions.png`. The checkpoint contains a tensor state dictionary, width, threshold, configuration and selected epoch; loading uses `weights_only=True` and strict keys. Inference saves a same-size `mask.png`, `probabilities.npy` and provenance JSON. Arbitrary real photographs are outside this synthetic model's validated domain.

`configs/unet-full.json` preserves the original width 64 and 15-epoch scale as an **unverified longer-run preset**. The smoke uses width 8 (486,553 parameters); it is not equivalent to a full-width training result. Odd and rectangular inputs preserve their dimensions. Inference requires height/width >=16, while the maintained generator/training path requires >=32. U-Net does not accept literally every possible resolution.

## Explicit assets and YOLOv8

```bash
unet-yolo fetch-assets --output assets/downloaded
unet-yolo yolo-train --checkpoint assets/downloaded/yolov8n.pt --dataset assets/downloaded/coco8 --config configs/yolo-smoke.json --output artifacts/yolo-smoke
unet-yolo yolo-predict --checkpoint artifacts/yolo-smoke/run/weights/best.pt --source assets/downloaded/coco8/images/val/000000000042.jpg --output artifacts/yolo-image
unet-yolo yolo-predict --checkpoint assets/downloaded/yolov8n.pt --source assets/downloaded/traffic_video.mp4 --max-frames 48 --output artifacts/video
```

The fetch command downloads only the named official YOLOv8n weights, COCO8 archive and original public traffic sample, verifying their recorded SHA-256 values. It checks archive paths and size limits before extraction. Asset terms are recorded in [provenance](docs/provenance.md).

Training requires the verified official base checkpoint and a local COCO80-format `images/train`, `images/val`, `labels/train`, `labels/val` tree. It validates paired labels, normalized boxes, readable images, and duplicate image content, then writes its exact manifest and an absolute local dataset YAML with no download directive. CPU is the tested route; `device: "0"` selects a CUDA device explicitly. Workers stay at zero for this workflow. Missing local files fail before model initialization. Runtime disables optional Ultralytics auto-install/network checks; downloads belong to the explicit fetch step.

Image/video inference accepts local checkpoints, never a model name or remote stream URL. It retains one frame's result at a time, writes `detections.jsonl`, at most four preview JPEGs, and an annotated MP4 for video. `--max-frames` bounds a smoke run; omit it to process the full file. Frames use original-image xyxy coordinates. Empty detections are valid, not silently discarded. There is no automatic CPU retry or exception-swallowing success message. Use a new or empty output directory for each command.

YOLO `.pt` files are loaded through Ultralytics and can contain serialized model objects; use only trusted official files or checkpoints you produced. This differs from U-Net's restricted state-dictionary loader.

## Notebooks and checks

```bash
jupyter lab
python -m unittest discover -s tests -v
```

Open [synthetic-unet.ipynb](notebooks/synthetic-unet.ipynb) for a complete synthetic training/prediction workflow or [local-yolov8.ipynb](notebooks/local-yolov8.ipynb) for a real local COCO8 training/image/video smoke. Both use unique output directories. Retrieve assets first for the latter. `UNET_YOLO_ASSETS` and `UNET_YOLO_DATA` can select existing local files. See [notebook guide](notebooks/README.md).

Nine regression tests cover fixed split identity, odd/minimum shapes, crop/pad alignment, overlap arithmetic, loss gradients, checkpoint/inference round trips, invalid config, dataset validation and missing-checkpoint behavior. CI runs these offline without YOLO downloads. Real YOLO training/video validation is recorded separately in [validation](docs/validation.md).

The maintained project corrects the original notebook's changing validation samples, missing best-checkpoint selection, unbounded video result list and overly broad resolution/production claims. Its historical outputs remain historical evidence, not fresh measurements. U-Net and YOLO remain separate tasks; no combined detector/segmenter or deployment service is claimed. No missing-license MIT badge is retained; consult the upstream [Ultralytics license](https://www.ultralytics.com/license) and asset terms before reuse.
