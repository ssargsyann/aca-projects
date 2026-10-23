"""Behavior tests for preprocessing, persistence, and independent soft clusters."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from customer_segments.data import NumericFeatures, load_data
from customer_segments.demo import make_demo_data, run_demo
from customer_segments.predict import predict
from customer_segments.profiles import assignments
from customer_segments.train import train


class DataTests(unittest.TestCase):
    def test_inference_uses_fitted_statistics(self):
        transform = NumericFeatures(["Income"], (0, 1)).fit(pd.DataFrame({"Income": [10, 20, 30]}))
        result = transform.transform(pd.DataFrame({"Income": [np.nan, 999], "ID": [1, 2]}))
        np.testing.assert_allclose(result[:, 0], [20, 30])

    def test_invalid_inputs_are_actionable(self):
        for frame in (pd.DataFrame({"other": [1]}), pd.DataFrame({"Income": ["text"]}), pd.DataFrame({"Income": [np.inf]}), pd.DataFrame({"Income": [np.nan]})):
            with self.subTest(frame=frame.columns.tolist()), self.assertRaises(ValueError):
                NumericFeatures(["Income"]).fit(frame)

    def test_csv_and_tsv_read_equivalently(self):
        frame = make_demo_data(40)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame.to_csv(directory / "comma.csv", index=False)
            frame.to_csv(directory / "tab.csv", index=False, sep="\t")
            pd.testing.assert_frame_equal(load_data(directory / "comma.csv"), load_data(directory / "tab.csv"))

    def test_demo_is_reproducible(self):
        pd.testing.assert_frame_equal(make_demo_data(seed=5), make_demo_data(seed=5))
        with self.assertRaises(ValueError):
            make_demo_data(5)


class WorkflowTests(unittest.TestCase):
    def test_roundtrip_probabilities_and_row_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "demo"
            metrics = run_demo(directory)
            self.assertEqual(metrics["rows"], 180)
            self.assertIn(metrics["selected_k"], [2, 3, 4])
            self.assertGreaterEqual(metrics["explained_variance"], 0.8)
            self.assertTrue(metrics["gmm_converged"])
            bundle = joblib.load(directory / "run/segmentation.joblib")
            frame = make_demo_data()
            original = pd.read_csv(directory / "run/assignments.csv")
            reloaded = predict(directory / "run/segmentation.joblib", directory / "synthetic.csv", directory / "predictions.csv")
            np.testing.assert_array_equal(original.kmeans_segment, reloaded.kmeans_segment)
            np.testing.assert_array_equal(reloaded.row_number, np.arange(len(frame)))
            probability_columns = [name for name in reloaded if name.startswith("gmm_probability_") and name != "gmm_probability_gap"]
            np.testing.assert_allclose(reloaded[probability_columns].sum(axis=1), 1)
            expected = reloaded[probability_columns].to_numpy().argsort(axis=1)[:, -1]
            np.testing.assert_array_equal(expected, reloaded.gmm_component)
            self.assertEqual(len(reloaded), len(frame))
            np.testing.assert_array_equal(reloaded.gmm_boundary, reloaded.gmm_probability_gap <= 0.1)
            changed_ids = frame.copy()
            changed_ids["ID"] += 100000
            np.testing.assert_array_equal(assignments(bundle, frame).kmeans_segment, assignments(bundle, changed_ids).kmeans_segment)
            with self.assertRaisesRegex(ValueError, "new or empty"):
                run_demo(directory)
            missing = frame.drop(columns="Income")
            with self.assertRaisesRegex(ValueError, "Income"):
                assignments(bundle, missing)
            profiles = pd.read_csv(directory / "run/cluster_profiles.csv")
            self.assertEqual(profiles.customers.sum(), 180)
            self.assertEqual(profiles.description.nunique(), metrics["selected_k"])
            config = json.loads((directory / "config.json").read_text())
            config["k_values"] = [len(frame)]
            (directory / "invalid.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "candidate K"):
                train(directory / "synthetic.csv", directory / "invalid.json", directory / "invalid-run")


if __name__ == "__main__":
    unittest.main()
