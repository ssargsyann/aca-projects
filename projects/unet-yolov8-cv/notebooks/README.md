# Notebook workflows

- `synthetic-unet.ipynb`: full maintained six-epoch synthetic smoke, saved-checkpoint test metrics, figures and odd-sized image inference. No network/assets required.
- `local-yolov8.ipynb`: local COCO8 one-epoch smoke, reloaded fine-tuned image inference and eight-frame pretrained video smoke. Retrieve assets first; no silent downloads. Defaults to `assets/downloaded`; `UNET_YOLO_ASSETS` and `UNET_YOLO_DATA` can use existing local paths.
- `original_experiment.ipynb`: unchanged imported notebook. Historical code/results may contain changing validation samples, unbounded video result accumulation, implicit downloads and unsupported general claims corrected in maintained code/docs. It is preserved for traceability, not the recommended execution path.

Launch Jupyter from this project directory and choose the environment containing `aca-unet-yolo` plus the relevant extras. Maintained notebooks also recognize launch from their own folder. Each execution uses a unique directory under `artifacts/`. Published maintained sources have cleared outputs; run results are retained separately in the reports. Real YOLO loading uses trusted serialized checkpoints; U-Net loading is restricted state-dictionary deserialization.
