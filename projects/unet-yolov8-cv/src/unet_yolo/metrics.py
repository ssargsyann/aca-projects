"""Per-image foreground overlap, averaged across images, including empty masks."""

import torch
from torch.nn import functional as F


def bce_dice(logits, targets):
    probabilities = logits.sigmoid().flatten(1)
    truth = targets.flatten(1)
    dice = (2 * (probabilities * truth).sum(1) + 1) / (
        probabilities.sum(1) + truth.sum(1) + 1
    )
    return 0.5 * F.binary_cross_entropy_with_logits(logits, targets) + 0.5 * (
        1 - dice.mean()
    )


@torch.no_grad()
def overlaps(logits, targets, threshold=0.5):
    if logits.shape != targets.shape or not 0 <= threshold <= 1:
        raise ValueError("Matching shapes and threshold in [0,1] required")
    pred, truth = logits.sigmoid() > threshold, targets > 0.5
    axes = tuple(range(1, logits.ndim))
    intersection = (pred & truth).sum(axes).double()
    union = (pred | truth).sum(axes).double()
    total = pred.sum(axes) + truth.sum(axes)
    iou = torch.where(union > 0, intersection / union.clamp_min(1), 1.0)
    dice = torch.where(total > 0, 2 * intersection / total.clamp_min(1), 1.0)
    return {"iou": iou, "dice": dice}
