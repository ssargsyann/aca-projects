import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from unet_yolo.data import Shapes
from unet_yolo.detection import dataset_manifest
from unet_yolo.detection import predict as yolo_predict
from unet_yolo.metrics import bce_dice, overlaps
from unet_yolo.model import UNet
from unet_yolo.segmentation import configure, load_checkpoint, predict, train

torch.set_num_threads(2)


class Contracts(unittest.TestCase):
    def test_shapes_are_index_stable_and_splits_are_separate(self):
        train_data = Shapes(4, seed=9)
        valid = Shapes(4, seed=9, split="validation")
        image, mask = train_data[1]
        self.assertTrue(torch.equal(image, train_data[1][0]))
        self.assertTrue(torch.equal(mask, train_data[1][1]))
        self.assertFalse(torch.equal(image, valid[1][0]))
        self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})
        # Evaluation ordering and loader batch sizes do not change sample identity.
        self.assertTrue(
            torch.equal(next(iter(DataLoader(valid, batch_size=1)))[0][0], valid[0][0])
        )
        with self.assertRaises(IndexError):
            train_data[4]

    def test_even_odd_rectangular_and_minimum_shapes(self):
        model = UNet(num_classes=1, base_channels=2).eval()
        with torch.inference_mode():
            for h, w in [(16, 16), (31, 35), (32, 48)]:
                self.assertEqual(
                    tuple(model(torch.rand(1, 3, h, w)).shape), (1, 1, h, w)
                )
        with self.assertRaises(ValueError):
            model(torch.rand(1, 3, 15, 16))

    def test_alignment_supports_crop_and_pad(self):
        skip = torch.ones(1, 2, 7, 9)
        for shape in [(1, 2, 8, 10), (1, 2, 6, 8), (1, 2, 6, 10)]:
            self.assertEqual(
                tuple(UNet._align(torch.ones(shape), skip).shape), tuple(skip.shape)
            )

    def test_overlap_empty_partial_and_sample_aggregation(self):
        truth = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]], [[[0.0, 0.0], [0.0, 0.0]]]])
        logits = torch.tensor(
            [[[[10.0, -10.0], [10.0, -10.0]]], [[[-10.0, -10.0], [-10.0, -10.0]]]]
        )
        result = overlaps(logits, truth)
        np.testing.assert_allclose(result["iou"], [1 / 3, 1])
        np.testing.assert_allclose(result["dice"], [0.5, 1])
        self.assertAlmostEqual(float(result["iou"].mean()), 2 / 3)
        with self.assertRaises(ValueError):
            overlaps(logits, truth, 0.0 - 1)

    def test_loss_has_finite_gradients(self):
        logits = torch.zeros(2, 1, 4, 5, requires_grad=True)
        loss = bce_dice(logits, torch.zeros_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_train_restore_and_odd_size_predict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {
                "seed": 4,
                "image_size": [32, 32],
                "train_count": 4,
                "validation_count": 3,
                "test_count": 3,
                "epochs": 1,
                "batch_size": 2,
                "base_channels": 2,
                "learning_rate": 0.001,
                "threshold": 0.5,
                "cpu_threads": 2,
                "device": "cpu",
            }
            config = root / "config.json"
            config.write_text(json.dumps(cfg))
            result = train(config, root / "run")
            _model, saved = load_checkpoint(root / "run/best.pt")
            self.assertEqual(saved["best_epoch"], 1)
            self.assertEqual(len(result["history"]), 1)
            for metric in ["iou", "dice"]:
                self.assertTrue(0 <= result["test"][metric] <= 1)
            Image.new("RGB", (35, 33)).save(root / "odd.png")
            prediction = predict(
                root / "run/best.pt", root / "odd.png", root / "prediction"
            )
            self.assertEqual(prediction["image_size"], [35, 33])
            self.assertEqual(
                np.load(root / "prediction/probabilities.npy").shape, (33, 35)
            )
            with self.assertRaisesRegex(ValueError, "new or empty"):
                train(config, root / "run")

    def test_invalid_config_fails_early(self):
        cfg = {
            "seed": 4,
            "image_size": [32, 32],
            "train_count": 4,
            "validation_count": 3,
            "test_count": 3,
            "epochs": 1,
            "batch_size": 2,
            "base_channels": 2,
            "learning_rate": 0.001,
            "threshold": 0.5,
            "cpu_threads": 2,
            "device": "cpu",
        }
        for key, value in [
            ("epochs", 0),
            ("threshold", 1.1),
            ("image_size", [15, 64]),
            ("device", "invalid"),
        ]:
            with self.assertRaises(ValueError):
                configure({**cfg, key: value})

    def test_dataset_pairs_labels_and_split_duplicate_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split, color in [("train", "red"), ("val", "blue")]:
                (root / "images" / split).mkdir(parents=True)
                (root / "labels" / split).mkdir(parents=True)
                Image.new("RGB", (32, 32), color).save(
                    root / "images" / split / "sample.png"
                )
                (root / "labels" / split / "sample.txt").write_text("0 .5 .5 .4 .4")
            self.assertEqual(len(dataset_manifest(root, ["object"])), 2)
            (root / "labels/val/sample.txt").write_text("1 .5 .5 .4 .4")
            with self.assertRaises(ValueError):
                dataset_manifest(root, ["object"])
            (root / "labels/val/sample.txt").write_text("0 .5 .5 .4 .4")
            (root / "images/val/sample.png").write_bytes(
                (root / "images/train/sample.png").read_bytes()
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                dataset_manifest(root, ["object"])

    def test_missing_checkpoint_fails_before_optional_model_import(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch("unet_yolo.detection.runtime") as runtime,
        ):
            with self.assertRaises(FileNotFoundError):
                yolo_predict(
                    Path(tmp) / "missing.pt",
                    Path(tmp) / "missing.mp4",
                    Path(tmp) / "out",
                )
            runtime.assert_not_called()


if __name__ == "__main__":
    unittest.main()
