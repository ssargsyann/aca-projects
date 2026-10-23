"""RGB image inference with the training normalization and explicit class order."""
import json
from pathlib import Path

from PIL import Image
import torch
from torchvision.transforms import Resize

from .checkpoint import load_model
from .data import CLASSES, image_transform


def predict(checkpoint_path, image_path, output_path):
    torch.set_num_threads(2)
    model, checkpoint = load_model(checkpoint_path)
    with Image.open(image_path) as opened:
        image = Resize((32, 32))(opened.convert("RGB"))
    tensor = image_transform(False)(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].tolist()
    best = max(range(10), key=lambda index: probabilities[index])
    result = {"class_index": best, "class_name": CLASSES[best], "softmax_score": probabilities[best], "scores": dict(zip(CLASSES, probabilities)), "checkpoint_data_kind": checkpoint["data_kind"]}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
