"""Regression checks without network access or benchmark accuracy assertions."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset

from cifar10_cnn.checkpoint import load_model
from cifar10_cnn.data import image_transform, split_indices, stratified_limit, training_sets
from cifar10_cnn.demo import demo
from cifar10_cnn.engine import run_epoch
from cifar10_cnn.model import AdvancedCNN
from cifar10_cnn.predict import predict
from cifar10_cnn.report import report

torch.set_num_threads(2)


class DataTests(unittest.TestCase):
    def test_split_is_disjoint_stratified_and_reproducible(self):
        labels = np.repeat(np.arange(10), 20)
        training, validation = split_indices(labels, 0.2, 42)
        self.assertFalse(set(training) & set(validation))
        self.assertEqual(set(training) | set(validation), set(range(200)))
        np.testing.assert_array_equal(np.bincount(labels[validation]), [4] * 10)
        np.testing.assert_array_equal(training, split_indices(labels, 0.2, 42)[0])
        limited = stratified_limit(training, labels, 30, 42)
        np.testing.assert_array_equal(np.bincount(labels[limited]), [3] * 10)
        with self.assertRaises(ValueError):
            stratified_limit(training, labels, 5, 42)
        with self.assertRaises(ValueError):
            split_indices(labels, 0)

    def test_training_does_not_load_official_test_set(self):
        class Stub:
            def __init__(self, *args, **kwargs):
                self.targets = np.repeat(np.arange(10), 20).tolist()
            def __len__(self):
                return len(self.targets)
        with patch("cifar10_cnn.data.datasets.CIFAR10", side_effect=Stub) as loader:
            config = {"validation_fraction": 0.2, "seed": 42, "train_limit": None, "validation_limit": None}
            training_sets("unused", config)
            self.assertTrue(all(call.kwargs["train"] for call in loader.call_args_list))

    def test_eval_transform_shape_and_normalization(self):
        image = Image.new("RGB", (32, 32), color=(255, 0, 0))
        tensor = image_transform(False)(image)
        self.assertEqual(tuple(tensor.shape), (3, 32, 32))
        np.testing.assert_allclose(tensor[:, 0, 0], [1, -1, -1])
        torch.testing.assert_close(tensor, image_transform(False)(image))


class ModelTests(unittest.TestCase):
    def test_forward_backward_and_eval_determinism(self):
        model = AdvancedCNN()
        inputs = torch.randn(2, 3, 32, 32)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2, 10))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertTrue(torch.isfinite(model.conv1.weight.grad).all())
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), model(inputs), rtol=0, atol=0)

    def test_loss_is_weighted_by_samples_not_batches(self):
        logits = torch.randn(7, 10)
        labels = torch.arange(7)
        loader = DataLoader(TensorDataset(logits, labels), batch_size=4)
        result = run_epoch(torch.nn.Identity(), loader, "cpu")
        expected = torch.nn.functional.cross_entropy(logits, labels).item()
        self.assertAlmostEqual(result["loss"], expected, places=6)
        self.assertEqual(np.array(result["confusion_matrix"]).sum(), 7)

    def test_demo_checkpoint_reload_and_rgb_image_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "demo"
            result = demo(directory)
            self.assertEqual(result["data_kind"], "SYNTHETIC_DEMO_ONLY")
            self.assertFalse(result["test_used_during_training"])
            model, checkpoint = load_model(directory / "run/best.pt")
            self.assertFalse(model.training)
            image = directory / "gray.png"
            Image.new("L", (48, 48), color=128).save(image)
            predicted = predict(directory / "run/best.pt", image, directory / "gray-prediction.json")
            self.assertAlmostEqual(sum(predicted["scores"].values()), 1, places=5)
            self.assertIn(predicted["class_name"], checkpoint["classes"])
            with self.assertRaises(ValueError):
                demo(directory)
            checkpoint["classes"] = list(reversed(checkpoint["classes"]))
            torch.save(checkpoint, directory / "incompatible.pt")
            with self.assertRaisesRegex(ValueError, "class order"):
                load_model(directory / "incompatible.pt")
            checkpoint["classes"] = list(reversed(checkpoint["classes"]))
            checkpoint["normalization"]["mean"] = [0, 0, 0]
            torch.save(checkpoint, directory / "wrong-normalization.pt")
            with self.assertRaisesRegex(ValueError, "normalization"):
                load_model(directory / "wrong-normalization.pt")

    def test_report_refuses_another_checkpoints_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "run.json").write_text(json.dumps({"data_kind": "SYNTHETIC_DEMO_ONLY"}))
            (directory / "history.json").write_text("[]")
            (directory / "best.pt").write_bytes(b"fingerprint fixture; not deserialized")
            (directory / "wrong.json").write_text(json.dumps({"checkpoint_sha256": "wrong"}))
            with self.assertRaisesRegex(ValueError, "does not match"):
                report(directory, directory / "report", directory / "wrong.json")
            self.assertFalse((directory / "report").exists())


if __name__ == "__main__":
    unittest.main()
