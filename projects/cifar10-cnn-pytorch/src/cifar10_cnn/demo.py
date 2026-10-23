"""Fake images verify optimization/checkpoint wiring without a dataset download."""
import json
from pathlib import Path

from torchvision.datasets import FakeData

from .data import image_transform
from .train import train


def demo(output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new or empty demo directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"epochs": 1, "batch_size": 8, "learning_rate": 0.001, "weight_decay": 0.0001, "validation_fraction": 0.1, "seed": 42, "device": "cpu", "num_workers": 0, "cpu_threads": 2, "train_limit": None, "validation_limit": None}
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    datasets = (FakeData(size=24, image_size=(3, 32, 32), num_classes=10, transform=image_transform(False), random_offset=0), FakeData(size=12, image_size=(3, 32, 32), num_classes=10, transform=image_transform(False), random_offset=1000))
    metadata = train(None, config_path, output_dir / "run", datasets_override=datasets)
    FakeData(size=1, image_size=(3, 32, 32), random_offset=2000)[0][0].save(output_dir / "synthetic.png")
    from .predict import predict
    predict(output_dir / "run/best.pt", output_dir / "synthetic.png", output_dir / "prediction.json")
    (output_dir / "README.txt").write_text("SYNTHETIC DEMO ONLY\nRandom invented images and labels. No CIFAR-10 benchmark performance is claimed.\n", encoding="utf-8")
    return metadata
