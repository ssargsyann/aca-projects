"""Coordinate-preserving output filtering and semantic output alignment."""
import torch
from torch.nn.functional import interpolate


def filter_objects(prediction, categories, score_threshold, mask_threshold=0.5):
    if not 0 <= score_threshold <= 1 or not 0 <= mask_threshold <= 1:
        raise ValueError("Score/mask thresholds must lie in [0,1].")
    selected = prediction["scores"] > score_threshold
    boxes = prediction["boxes"][selected].detach().cpu()
    scores = prediction["scores"][selected].detach().cpu()
    labels = prediction["labels"][selected].detach().cpu()
    result = [{"box_xyxy": box.tolist(), "class_id": int(label), "class_name": categories[int(label)], "score": float(score)} for box, label, score in zip(boxes, labels, scores)]
    masks = None
    if "masks" in prediction:
        masks = (prediction["masks"][selected, 0] > mask_threshold).detach().cpu()
        for row, mask in zip(result, masks):
            row["mask_pixels"] = int(mask.sum())
    return boxes, masks, result


def align_semantic(logits, image_size):
    width, height = image_size
    return interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)[0].argmax(dim=0).detach().cpu()
