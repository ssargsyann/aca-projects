"""Generate reproducible annotation/grid/anchor diagnostics without model weights."""
import json
from pathlib import Path

from PIL import Image, ImageDraw
import numpy as np

from .geometry import analyze, centered_anchors


def geometry(image_path, annotation_path, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    spec = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    if list(image.size) != spec["image_size"]:
        raise ValueError("Image dimensions do not match annotation image_size; rescale annotations explicitly.")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new or empty geometry output directory.")
    results = analyze(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "geometry.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    count = len(results["objects"])
    figure, axes = plt.subplots(1, count, figsize=(5 * count, 4.8), squeeze=False)
    width, height = image.size
    for axis, obj in zip(axes[0], results["objects"]):
        axis.imshow(image)
        for column in range(spec["grid_size"] + 1):
            axis.axvline(column * width / spec["grid_size"], color="white", alpha=0.4, linewidth=0.7)
            axis.axhline(column * height / spec["grid_size"], color="white", alpha=0.4, linewidth=0.7)
        x1, y1, x2, y2 = obj["box_xyxy"]
        axis.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, edgecolor="red", facecolor="none", linewidth=2))
        column, row = obj["cell_col_row"]
        axis.add_patch(Rectangle((column * width / spec["grid_size"], row * height / spec["grid_size"]), width / spec["grid_size"], height / spec["grid_size"], facecolor="yellow", alpha=0.35))
        scores = np.array(obj["shape_ious"])
        winner = scores.argmax()
        chosen = centered_anchors(np.array([obj["center_xy"]]), np.array([anchor["size"] for anchor in spec["anchors"]]))[0, winner]
        axis.add_patch(Rectangle(chosen[:2], chosen[2] - chosen[0], chosen[3] - chosen[1], edgecolor="lime", facecolor="none", linewidth=2, linestyle="--"))
        axis.plot(*obj["center_xy"], marker="o", color="red")
        axis.set_title(f"{obj['name']}: cell ({column}, {row})\nShape anchor: {obj['shape_winner']} (IoU {scores[winner]:.3f})")
        axis.set(xlim=(0, width), ylim=(height, 0))
        axis.axis("off")
    figure.suptitle("Demonstration boxes (red), responsible cell (yellow), shape-matched anchor at object center (green)", fontsize=11)
    figure.tight_layout()
    figure.savefig(output_dir / "grid-anchors.png", dpi=150)
    plt.close(figure)
    return results


def demo(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new or empty demo directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (640, 480), (35, 45, 65))
    spec = {"image_size": [640, 480], "grid_size": 7, "objects": [{"name": "rectangle A", "box": [40, 60, 180, 280]}, {"name": "rectangle B", "box": [270, 290, 560, 410]}], "anchors": [{"name": "wide", "size": [260, 120]}, {"name": "tall", "size": [140, 220]}, {"name": "square", "size": [180, 180]}]}
    draw = ImageDraw.Draw(image)
    for obj in spec["objects"]:
        draw.rectangle(obj["box"], fill=(90, 120, 160), outline="white", width=3)
    image.save(output_dir / "synthetic.png")
    (output_dir / "annotations.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    result = geometry(output_dir / "synthetic.png", output_dir / "annotations.json", output_dir / "geometry")
    (output_dir / "README.txt").write_text("SYNTHETIC GEOMETRY DEMO\nNo model weights, network access, or trained detector outputs are used.\n", encoding="utf-8")
    return result
