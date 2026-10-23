"""Vectorized box overlap, bounded grid assignment, and shape-only anchor matching."""
import numpy as np


def boxes_array(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("Boxes must be finite [N,4] xyxy values, including [0,4] for an empty set.")
    if np.any(values[:, 2:] < values[:, :2]):
        raise ValueError("Box maxima cannot precede minima.")
    return values


def pairwise_iou(first, second):
    first, second = boxes_array(first), boxes_array(second)
    low = np.maximum(first[:, None, :2], second[None, :, :2])
    high = np.minimum(first[:, None, 2:], second[None, :, 2:])
    overlap = np.prod(np.maximum(high - low, 0), axis=-1)
    area_first = np.prod(first[:, 2:] - first[:, :2], axis=-1)
    area_second = np.prod(second[:, 2:] - second[:, :2], axis=-1)
    union = area_first[:, None] + area_second[None, :] - overlap
    return np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)


def responsible_cells(centers, image_size, grid_size):
    centers = np.asarray(centers, dtype=np.float64)
    dimensions = np.asarray(image_size, dtype=np.float64)
    if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
        raise ValueError("image_size must contain positive finite width and height.")
    if not isinstance(grid_size, int) or isinstance(grid_size, bool) or grid_size < 1:
        raise ValueError("grid_size must be a positive integer.")
    if centers.ndim != 2 or centers.shape[1] != 2 or not np.isfinite(centers).all():
        raise ValueError("Centers must be finite [N,2] pixel coordinates.")
    if np.any(centers < 0) or np.any(centers > dimensions):
        raise ValueError("Centers must lie within the image, including its outer boundary.")
    return np.minimum(np.floor(centers / dimensions * grid_size).astype(int), grid_size - 1)


def shape_iou(box_sizes, anchor_sizes):
    box_sizes, anchor_sizes = np.asarray(box_sizes, dtype=float), np.asarray(anchor_sizes, dtype=float)
    for sizes in (box_sizes, anchor_sizes):
        if sizes.ndim != 2 or sizes.shape[1] != 2 or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("Sizes must be finite positive [N,2] widths and heights.")
    intersection = np.prod(np.minimum(box_sizes[:, None, :], anchor_sizes[None, :, :]), axis=-1)
    union = np.prod(box_sizes, axis=1)[:, None] + np.prod(anchor_sizes, axis=1)[None, :] - intersection
    return intersection / union


def centered_anchors(centers, anchor_sizes):
    centers, sizes = np.asarray(centers, dtype=float), np.asarray(anchor_sizes, dtype=float)
    return np.concatenate([centers[:, None, :] - sizes[None, :, :] / 2, centers[:, None, :] + sizes[None, :, :] / 2], axis=-1)


def analyze(spec):
    boxes = boxes_array([obj["box"] for obj in spec["objects"]])
    sizes = boxes[:, 2:] - boxes[:, :2]
    dimensions = np.asarray(spec["image_size"], dtype=float)
    if np.any(boxes[:, :2] < 0) or np.any(boxes[:, 2:] > dimensions):
        raise ValueError("Demonstration boxes must fit inside the annotated image.")
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2
    cells = responsible_cells(centers, spec["image_size"], spec["grid_size"])
    anchors = np.asarray([anchor["size"] for anchor in spec["anchors"]], dtype=float)
    scores = shape_iou(sizes, anchors)
    cell_centers = (cells + 0.5) * dimensions / spec["grid_size"]
    positioned = centered_anchors(cell_centers, anchors)
    position_scores = np.stack([pairwise_iou(boxes[index:index + 1], positioned[index])[0] for index in range(len(boxes))])
    results = []
    for index, obj in enumerate(spec["objects"]):
        winner, position_winner = int(scores[index].argmax()), int(position_scores[index].argmax())
        results.append({"name": obj["name"], "box_xyxy": boxes[index].tolist(), "center_xy": centers[index].tolist(), "cell_col_row": cells[index].tolist(), "shape_ious": scores[index].tolist(), "shape_winner": spec["anchors"][winner]["name"], "cell_centered_ious": position_scores[index].tolist(), "cell_centered_winner": spec["anchors"][position_winner]["name"]})
    return {"scope": "educational_geometry_not_detector_training", "image_size": spec["image_size"], "grid_size": spec["grid_size"], "anchors": spec["anchors"], "objects": results}
