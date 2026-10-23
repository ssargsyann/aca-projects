"""Behavioral regression tests using synthetic data, never clinical benchmarks."""
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from icu_mortality.data import FeatureSelector, validate_target
from icu_mortality.predict import predict
from icu_mortality.train import train
from icu_mortality.demo import make_demo_data, run_demo
from icu_mortality.report import create_report

ROOT = Path(__file__).resolve().parents[1]


class DataTests(unittest.TestCase):
    def test_missingness_is_learned_only_during_fit(self):
        selector = FeatureSelector(0.65, ["patient_id"])
        training = pd.DataFrame({"age": [40, 50, 60], "sparse": [np.nan, np.nan, 1], "patient_id": [1, 2, 3]})
        selector.fit(training)
        validation = pd.DataFrame({"age": [np.nan], "sparse": [9], "patient_id": [4]})
        self.assertEqual(selector.transform(validation).columns.tolist(), ["age"])
        self.assertEqual(selector.columns_, ["age"])

    def test_missing_feature_is_actionable(self):
        selector = FeatureSelector().fit(pd.DataFrame({"age": [30]}))
        with self.assertRaisesRegex(ValueError, "age"):
            selector.transform(pd.DataFrame({"other": [1]}))

    def test_invalid_labels_are_rejected(self):
        for labels in ([0, 0, 0], [0, 1, np.nan], [0, 1, 2], [0, 0, 1]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                validate_target(pd.DataFrame({"hospital_death": labels}), "hospital_death")

    def test_missing_target_is_actionable(self):
        with self.assertRaisesRegex(ValueError, "hospital_death"):
            validate_target(pd.DataFrame({"age": [40]}), "hospital_death")

    def test_empty_features_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "No usable features"):
            FeatureSelector().fit(pd.DataFrame({"empty": [np.nan]}))


class PipelineTests(unittest.TestCase):
    def test_demo_is_reproducible_and_honestly_labeled(self):
        pd.testing.assert_frame_equal(make_demo_data(60, 7), make_demo_data(60, 7))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "demo"
            summary = run_demo(output, rows=60, seed=7)
            self.assertEqual(summary["data_kind"], "SYNTHETIC_DEMO_ONLY")
            self.assertEqual(len(pd.read_csv(output / "predictions.csv")), 12)
            self.assertNotIn("hospital_death", pd.read_csv(output / "synthetic_inference.csv").columns)
            self.assertTrue((output / "model/pipeline.joblib").exists())
            changed = output / "changed.csv"
            changed.write_text("hospital_death,age\n0,50\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checksum"):
                create_report(output / "model", changed, output / "report")
            with self.assertRaisesRegex(ValueError, "empty or new"):
                run_demo(output)

    def test_tiny_demo_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 40"):
            make_demo_data(5)

    def test_training_roundtrip_and_unseen_category(self):
        rng = np.random.default_rng(7)
        frame = pd.DataFrame({
            "age": rng.integers(18, 90, 100).astype(float),
            "unit": np.where(np.arange(100) % 2, "medical", "surgical"),
            "patient_id": np.arange(100),
            "apache_4a_hospital_death_prob": np.arange(100) % 2,
            "hospital_death": np.arange(100) % 2,
        })
        frame.loc[0:10, "age"] = np.nan
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data = directory / "train.csv"
            frame.to_csv(data, index=False)
            config = json.loads((ROOT / "configs/default.json").read_text())
            config["model"].update(n_estimators=4, max_depth=2, n_jobs=1)
            configuration = directory / "config.json"
            configuration.write_text(json.dumps(config))
            metrics = train(data, configuration, directory / "run")
            self.assertEqual(metrics["train_rows"], 80)
            self.assertEqual(metrics["validation_rows"], 20)
            self.assertTrue(0 <= metrics["roc_auc"] <= 1)
            model_path = directory / "run/pipeline.joblib"
            pipeline = joblib.load(model_path)
            self.assertNotIn("patient_id", pipeline.named_steps["select"].columns_)
            self.assertNotIn("apache_4a_hospital_death_prob", pipeline.named_steps["select"].columns_)
            inference = frame.drop(columns="hospital_death").head(3).copy()
            inference["unit"] = "previously_unseen"
            inference.to_csv(directory / "inference.csv", index=False)
            result = predict(model_path, directory / "inference.csv", directory / "predictions.csv")
            self.assertEqual(len(result), 3)
            self.assertTrue(result.mortality_score.between(0, 1).all())
            np.testing.assert_allclose(result.mortality_score, pipeline.predict_proba(inference)[:, 1])
            metadata = json.loads((directory / "run/run.json").read_text())
            self.assertEqual(len(metadata["data_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
