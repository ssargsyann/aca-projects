"""Sequential pretrained inference with explicit weights and auditable outputs."""
import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from .models import prepare_weights, load_task
from .postprocess import align_semantic, filter_objects


def inference(image_path, config_path, weights_dir, output_dir):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if not config["tasks"] or len(set(config["tasks"])) != len(config["tasks"]):
        raise ValueError("Supply a nonempty list of unique task names.")
    if not 0 <= config["score_threshold"] <= 1 or not 0 <= config["mask_threshold"] <= 1:
        raise ValueError("Thresholds must be in [0,1].")
    if not 1 <= config["top_k"] <= 1000 or config["cpu_threads"] < 1:
        raise ValueError("top_k must be 1..1000 and cpu_threads must be positive.")
    if not 1 <= config["detection_min_size"] <= config["detection_max_size"]:
        raise ValueError("Detection resize limits must be positive and ordered.")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new or empty inference run directory.")
    weights = prepare_weights(weights_dir, config["tasks"], download=False)
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"} or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("Choose an available cpu/cuda device, or auto.")
    torch.set_num_threads(config["cpu_threads"])
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    image_tensor = pil_to_tensor(image)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"scope": "single_image_pretrained_inference_not_benchmark", "image_size": list(image.size), "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(), "device": device, "config": config, "weights": weights, "versions": {name: version(name) for name in ["torch", "torchvision", "numpy", "pillow"]}, "tasks": {}}
    for task in config["tasks"]:
        print(f"Running {task}...", flush=True)
        model, weight = load_task(task, weights_dir, config, device)
        categories = weight.meta["categories"]
        with torch.inference_mode():
            transformed = weight.transforms()(image if task == "classification" else image_tensor).to(device)
            if task == "classification":
                scores = model(transformed.unsqueeze(0))[0].softmax(dim=0)
                values, indices = scores.topk(config["top_k"])
                rows = [{"class_id": int(index), "class_name": categories[int(index)], "score": float(score)} for score, index in zip(values.cpu(), indices.cpu())]
                results["tasks"][task] = {"top_k": rows}
                image.save(output_dir / "classification.png")
            elif task in {"detection", "instance"}:
                prediction = model([transformed])[0]
                boxes, masks, rows = filter_objects(prediction, categories, config["score_threshold"], config["mask_threshold"])
                visual = image_tensor.clone()
                if masks is not None:
                    if len(masks):
                        visual = draw_segmentation_masks(visual, masks, alpha=0.5)
                    np.savez_compressed(output_dir / "instance-masks.npz", masks=masks.numpy())
                if len(boxes):
                    captions = [f"{row['class_name']} {row['score']:.2f}" for row in rows]
                    visual = draw_bounding_boxes(visual, boxes, labels=captions, colors="lime", width=3)
                to_pil_image(visual).save(output_dir / f"{task}.png")
                results["tasks"][task] = {"objects": rows, "count": len(rows), "coordinate_frame": "original_RGB_image_pixels"}
                del prediction, boxes, masks, visual
            else:
                logits = model(transformed.unsqueeze(0))["out"]
                class_map = align_semantic(logits, image.size)
                np.save(output_dir / "semantic-labels.npy", class_map.numpy().astype(np.uint8))
                palette = torch.tensor([(index * 37 % 255, index * 17 % 255, index * 89 % 255) for index in range(len(categories))], dtype=torch.uint8)
                colored = Image.fromarray(palette[class_map].numpy())
                Image.blend(image, colored, alpha=0.55).save(output_dir / "semantic.png")
                counts = torch.bincount(class_map.flatten(), minlength=len(categories))
                results["tasks"][task] = {"class_map_size": [image.height, image.width], "classes_present": [{"class_id": index, "class_name": categories[index], "pixels": int(count)} for index, count in enumerate(counts) if count > 0], "alignment": "bilinear_resize_logits_to_original_then_argmax"}
                del logits, class_map, colored, counts
        del model, transformed
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def montage(run_dir, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    run_dir, output_path = Path(run_dir), Path(output_path)
    result = json.loads((run_dir / "results.json").read_text())
    tasks = list(result["tasks"])
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, task in zip(axes.flat, tasks):
        with Image.open(run_dir / f"{task}.png") as visual:
            axis.imshow(visual)
        title = {"classification": "Classification", "detection": "Object detection", "instance": "Instance segmentation", "semantic": "Semantic segmentation"}[task]
        if task == "classification":
            top = result["tasks"][task]["top_k"][0]
            title += f"\nTop ImageNet label: {top['class_name']} ({top['score']:.1%})"
        elif task in {"detection", "instance"}:
            title += f"\n{result['tasks'][task]['count']} outputs above threshold"
        axis.set_title(title, fontsize=11)
    figure.suptitle("One image, four pretrained vision tasks (qualitative demonstration)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    return output_path
