"""Pinned Torchvision weight identities and explicit cache/download handling."""
import hashlib
from pathlib import Path
import re
from urllib.parse import urlparse

import torch
from torchvision import models


def specifications():
    return {
        "classification": (models.ResNet50_Weights.IMAGENET1K_V2, models.resnet50),
        "detection": (models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, models.detection.fasterrcnn_resnet50_fpn_v2),
        "instance": (models.detection.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1, models.detection.maskrcnn_resnet50_fpn_v2),
        "semantic": (models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1, models.segmentation.deeplabv3_resnet50),
    }


def verify_file(path, prefix):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing weights: {path.name}. Run the explicit weights command or supply an existing cache.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    if not value.startswith(prefix):
        raise ValueError(f"Weight checksum does not match the official filename prefix: {path.name}")
    return value


def prepare_weights(cache_dir, tasks, download=False):
    specs = specifications()
    result = {}
    for task in tasks:
        if task not in specs:
            raise ValueError(f"Unknown task: {task}")
        weight, _ = specs[task]
        filename = Path(urlparse(weight.url).path).name
        prefix = re.search(r"-([0-9a-f]+)\.pth$", filename).group(1)
        path = Path(cache_dir) / filename
        if download and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(weight.url, str(path), hash_prefix=prefix, progress=True)
        result[task] = {"enum": str(weight), "url": weight.url, "filename": filename, "sha256": verify_file(path, prefix)}
    return result


def load_task(task, cache_dir, config, device):
    weight, builder = specifications()[task]
    arguments = {"weights": None}
    if task in {"detection", "instance"}:
        arguments.update(weights_backbone=None, num_classes=len(weight.meta["categories"]), min_size=config["detection_min_size"], max_size=config["detection_max_size"])
    elif task == "semantic":
        arguments.update(weights_backbone=None, num_classes=len(weight.meta["categories"]), aux_loss=True)
    model = builder(**arguments)
    filename = Path(urlparse(weight.url).path).name
    state = torch.load(Path(cache_dir) / filename, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), weight
