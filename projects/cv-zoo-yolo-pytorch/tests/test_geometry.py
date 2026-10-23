"""Numerical contracts and postprocessing without network/pretrained weights."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torchvision.ops import box_iou

from cv_zoo.geometry import analyze, pairwise_iou, responsible_cells, shape_iou
from cv_zoo.geometry_workflow import demo
from cv_zoo.models import prepare_weights, verify_file
from cv_zoo.postprocess import align_semantic, filter_objects


class GeometryTests(unittest.TestCase):
    def test_random_iou_matches_torchvision(self):
        rng = np.random.default_rng(42)
        low = rng.uniform(-20, 100, (40, 2))
        boxes = np.concatenate([low, low + rng.uniform(0.1, 50, (40, 2))], axis=1)
        actual = pairwise_iou(boxes[:20], boxes[20:])
        expected = box_iou(torch.tensor(boxes[:20]), torch.tensor(boxes[20:])).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_iou_empty_degenerate_and_invalid(self):
        self.assertEqual(pairwise_iou(np.empty((0, 4)), [[0, 0, 1, 1]]).shape, (0, 1))
        self.assertEqual(pairwise_iou([[0, 0, 0, 0]], [[0, 0, 0, 0]])[0, 0], 0)
        for values in ([[1, 1, 0, 0]], [[0, 0, np.nan, 1]], [[1, 2]]):
            with self.assertRaises(ValueError):
                pairwise_iou(values, [[0, 0, 1, 1]])

    def test_non_square_grid_and_edge_contract(self):
        result = responsible_cells([[0, 0], [100, 50], [50, 25]], [100, 50], 5)
        np.testing.assert_array_equal(result, [[0, 0], [4, 4], [2, 2]])
        for centers, size, grid in [([[-1, 0]], [100, 50], 5), ([[1, 1]], [0, 50], 5), ([[1, 1]], [100, 50], 0), ([[1, 1]], [100, 50], True)]:
            with self.assertRaises(ValueError):
                responsible_cells(centers, size, grid)

    def test_shape_matching_is_not_changed_by_object_location(self):
        anchors = [{"name": "wide", "size": [20, 10]}, {"name": "tall", "size": [10, 20]}]
        base = {"image_size": [100, 100], "grid_size": 5, "anchors": anchors, "objects": [{"name": "a", "box": [0, 0, 20, 10]}]}
        shifted = {**base, "objects": [{"name": "a", "box": [7, 8, 27, 18]}]}
        np.testing.assert_allclose(analyze(base)["objects"][0]["shape_ious"], analyze(shifted)["objects"][0]["shape_ious"])
        self.assertEqual(analyze(base)["objects"][0]["shape_winner"], "wide")
        np.testing.assert_allclose(shape_iou([[20, 10]], [[20, 10], [10, 20]]), [[1, 1 / 3]])

    def test_offline_demo_and_existing_output_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            result = demo(output)
            self.assertEqual(len(result["objects"]), 2)
            self.assertTrue((output / "geometry/grid-anchors.png").is_file())
            with self.assertRaisesRegex(ValueError, "new or empty"):
                demo(output)


class ModelContracts(unittest.TestCase):
    def test_empty_and_filtered_instance_results(self):
        prediction = {"scores": torch.tensor([0.9, 0.3]), "labels": torch.tensor([1, 2]), "boxes": torch.tensor([[0, 0, 2, 2], [1, 1, 3, 3]], dtype=torch.float32), "masks": torch.ones(2, 1, 4, 5) * 0.6}
        boxes, masks, rows = filter_objects(prediction, ["background", "one", "two"], 0.7, 0.5)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["class_name"], "one")
        self.assertEqual(rows[0]["mask_pixels"], 20)
        self.assertEqual(tuple(masks.shape), (1, 4, 5))
        boxes, masks, rows = filter_objects(prediction, ["background", "one", "two"], 1)
        self.assertEqual(tuple(boxes.shape), (0, 4))
        self.assertEqual(tuple(masks.shape), (0, 4, 5))
        self.assertEqual(rows, [])

    def test_semantic_alignment_restores_original_size(self):
        logits = torch.zeros(1, 3, 2, 4)
        logits[:, 2] = 5
        restored = align_semantic(logits, [13, 7])
        self.assertEqual(tuple(restored.shape), (7, 13))
        self.assertTrue((restored == 2).all())

    def test_missing_or_corrupt_cache_never_silently_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corrupt.pth"
            path.write_bytes(b"not model weights")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_file(path, "00000000")
            with self.assertRaises(FileNotFoundError):
                prepare_weights(temporary, ["classification"], download=False)
            self.assertEqual(list(Path(temporary).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
