"""Local-only YOLOv8 training and bounded-memory image/video inference."""

import json
import os
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image

from .common import new_directory, sha256, write_json


def runtime(output):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(output).resolve() / "runtime"))
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    import torch

    torch.set_num_threads(2)
    return YOLO


def local_file(path, suffix=None):
    path = Path(path).resolve()
    if not path.is_file() or suffix and path.suffix.lower() != suffix:
        raise FileNotFoundError(
            f"Expected an existing local {suffix or 'input'} file: {path}"
        )
    return path


def dataset_manifest(dataset_root, names):
    root = Path(dataset_root).resolve()
    rows, seen = [], set()
    if not isinstance(names, (dict, list)) or not names:
        raise ValueError("A nonempty class-name mapping is required")
    if isinstance(names, dict) and set(names) != set(range(len(names))):
        raise ValueError("Class IDs must be contiguous from zero")
    for split in ["train", "val"]:
        images = sorted((root / "images" / split).glob("*"))
        images = [
            path for path in images if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if not images:
            raise ValueError(f"No images in images/{split}")
        for image in images:
            with Image.open(image) as opened:
                opened.verify()
            label = root / "labels" / split / (image.stem + ".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing paired YOLO label: {label}")
            digest = sha256(image)
            if digest in seen:
                raise ValueError(
                    "Duplicate image content across dataset entries/splits"
                )
            seen.add(digest)
            objects = 0
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                values = np.array([float(value) for value in line.split()])
                if values.shape != (5,) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid YOLO detection row: {label.name}")
                cls, x, y, w, h = values
                if (
                    not cls.is_integer()
                    or not 0 <= cls < len(names)
                    or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1)
                ):
                    raise ValueError(f"Invalid class or normalized box: {label.name}")
                # Ultralytics clips the uncommon boxes whose center is in-frame but extents cross its boundary.
                objects += 1
            rows.append(
                {
                    "split": split,
                    "image": image.relative_to(root).as_posix(),
                    "image_sha256": digest,
                    "label": label.relative_to(root).as_posix(),
                    "label_sha256": sha256(label),
                    "objects": objects,
                }
            )
    return rows


def train(checkpoint, dataset_root, config_path, output):
    checkpoint = local_file(checkpoint, ".pt")
    from .assets import EXPECTED

    if sha256(checkpoint) != EXPECTED["yolov8n.pt"]:
        raise ValueError(
            "Training expects the verified official YOLOv8n base checkpoint; see fetch-assets"
        )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    for key in ["epochs", "imgsz", "batch"]:
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], int)
            or config[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    if config["imgsz"] < 32 or config["imgsz"] % 32:
        raise ValueError("imgsz must be a positive multiple of 32")
    if config["device"] not in {"cpu", "0"} or config["workers"] != 0:
        raise ValueError("Use device cpu/0 and workers=0 for this supported workflow")
    output = new_directory(output).resolve()
    YOLO = runtime(output)
    import ultralytics
    import yaml

    official = yaml.safe_load(
        (Path(ultralytics.__file__).parent / "cfg/datasets/coco8.yaml").read_text(
            encoding="utf-8"
        )
    )
    names = official["names"]
    manifest = dataset_manifest(dataset_root, names)
    data = {
        "path": str(Path(dataset_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names,
    }
    (output / "dataset.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    write_json(output / "dataset-manifest.json", manifest)
    model = YOLO(str(checkpoint), task="detect")
    model.train(
        data=str(output / "dataset.yaml"),
        project=str(output),
        name="run",
        exist_ok=False,
        amp=False,
        plots=False,
        save=True,
        val=True,
        deterministic=True,
        **config,
    )
    best = output / "run/weights/best.pt"
    result = {
        "scope": "tiny_coco8_smoke_not_independent_coco_benchmark",
        "config": config,
        "initial_checkpoint_sha256": sha256(checkpoint),
        "best_checkpoint_sha256": sha256(best),
        "metrics": {
            key: float(value)
            for key, value in model.trainer.validator.metrics.results_dict.items()
        },
        "train_images": sum(row["split"] == "train" for row in manifest),
        "validation_images": sum(row["split"] == "val" for row in manifest),
        "versions": {
            name: version(name)
            for name in ["ultralytics", "torch", "torchvision", "opencv-python"]
        },
    }
    write_json(output / "results.json", result)
    return result


def predict(
    checkpoint,
    source,
    output,
    imgsz=320,
    confidence=0.25,
    max_frames=None,
    device="cpu",
):
    checkpoint, source = local_file(checkpoint, ".pt"), local_file(source)
    if (
        not 0 <= confidence <= 1
        or imgsz < 32
        or imgsz % 32
        or max_frames is not None
        and max_frames < 1
    ):
        raise ValueError("Invalid threshold, image size, or maximum frame count")
    if device not in {"cpu", "0"}:
        raise ValueError("Use device cpu or 0")
    output = new_directory(output)
    YOLO = runtime(output)
    model = YOLO(str(checkpoint), task="detect")
    import cv2

    video = source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    capture, writer = None, None
    count, objects = 0, 0
    preview_paths = []
    if video:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise ValueError("Cannot open source video")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            capture.release()
            raise ValueError("Video has invalid frame rate")
    try:
        with (output / "detections.jsonl").open("w", encoding="utf-8") as stream:
            while max_frames is None or count < max_frames:
                if video:
                    valid, frame = capture.read()
                    if not valid:
                        break
                elif count:
                    break
                else:
                    frame = cv2.imread(str(source))
                    if frame is None:
                        raise ValueError("Cannot decode source image")
                # Only one decoded frame and its one result are retained at a time.
                result = model.predict(
                    frame,
                    imgsz=imgsz,
                    conf=confidence,
                    device=device,
                    verbose=False,
                    save=False,
                )[0]
                rows = []
                for box in result.boxes:
                    cls = int(box.cls.item())
                    rows.append(
                        {
                            "class_id": cls,
                            "class_name": result.names[cls],
                            "confidence": float(box.conf.item()),
                            "xyxy": box.xyxy[0].cpu().tolist(),
                        }
                    )
                stream.write(
                    json.dumps(
                        {
                            "frame_index": count,
                            "size": [frame.shape[1], frame.shape[0]],
                            "objects": rows,
                        }
                    )
                    + "\n"
                )
                annotated = result.plot()
                if count < 4:
                    name = f"frame-{count:04d}.jpg"
                    if not cv2.imwrite(str(output / name), annotated):
                        raise OSError("Could not save preview frame")
                    preview_paths.append(name)
                if video:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(output / "annotated.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            fps,
                            (frame.shape[1], frame.shape[0]),
                        )
                        if not writer.isOpened():
                            raise OSError("Cannot create annotated video")
                    writer.write(annotated)
                count += 1
                objects += len(rows)
                del result, frame, annotated
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
    if count == 0:
        raise ValueError("Source contained no decodable frames")
    result = {
        "scope": "qualitative_predictions_no_ground_truth_scoring",
        "frames_processed": count,
        "detections": objects,
        "max_frames": max_frames,
        "previews": preview_paths,
        "checkpoint_sha256": sha256(checkpoint),
        "source_sha256": sha256(source),
        "imgsz": imgsz,
        "confidence_threshold": confidence,
        "device": device,
        "ultralytics_version": version("ultralytics"),
        "coordinate_frame": "original_frame_xyxy_pixels",
        "video": video,
        "video_fps": fps if video else None,
    }
    write_json(output / "results.json", result)
    return result
