"""State-dict checkpoints loaded with restricted deserialization."""
import torch

from .data import CLASSES
from .model import AdvancedCNN


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "AdvancedCNN-v1" or checkpoint.get("classes") != CLASSES:
        raise ValueError("Checkpoint architecture or class order is incompatible.")
    if checkpoint.get("normalization") != {"mean": [0.5] * 3, "std": [0.5] * 3}:
        raise ValueError("Checkpoint normalization is incompatible.")
    model = AdvancedCNN().to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint
