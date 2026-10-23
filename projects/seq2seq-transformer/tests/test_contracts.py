import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from date_transformer.data import (
    BOS,
    EOS,
    FIRST,
    IDS,
    LAST,
    PAD,
    VOCAB,
    Dates,
    encode,
    export_splits,
    make_splits,
    parse_human,
)
from date_transformer.generation import generate, is_iso
from date_transformer.mechanics import sinusoidal_positions
from date_transformer.model import Seq2SeqTransformer, causal_mask
from date_transformer.workflow import (
    evaluate_checkpoint,
    load_checkpoint,
    predict,
    train,
)

torch.set_num_threads(2)


class ScheduledModel(torch.nn.Module):
    max_length = 32

    def __init__(self, emit_eos=True):
        super().__init__()
        self.emit_eos = emit_eos

    def encode(self, source):
        return torch.zeros(len(source), source.shape[1], 4), source.eq(PAD)

    def decode(self, prefix, memory, padding, capture=False):
        scores = torch.full((len(prefix), len(prefix[0]), len(VOCAB)), -100.0)
        for index in range(len(prefix)):
            next_id = (
                EOS if self.emit_eos and prefix.shape[1] >= index + 2 else IDS["2"]
            )
            scores[index, -1, next_id] = 100.0
        weights = (~padding).float()
        weights = weights / weights.sum(1, keepdim=True)
        return scores, weights[:, None, :].repeat(
            1, prefix.shape[1], 1
        ) if capture else None


class Contracts(unittest.TestCase):
    def test_disjoint_reproducible_unique_calendar_splits(self):
        cfg = {"seed": 42, "train_count": 100, "validation_count": 20, "test_count": 20}
        splits = make_splits(cfg)
        self.assertEqual(splits, make_splits(cfg))
        all_dates = [target for pairs in splits.values() for _, target in pairs]
        self.assertEqual(len(set(all_dates)), 140)
        self.assertNotEqual(splits, make_splits({**cfg, "seed": 43}))
        total = (LAST - FIRST).days + 1
        complete = make_splits(
            {**cfg, "train_count": total - 2, "validation_count": 1, "test_count": 1}
        )
        dates = {target for pairs in complete.values() for _, target in pairs}
        self.assertIn("2000-02-29", dates)
        self.assertIn("2050-12-31", dates)
        with self.assertRaises(ValueError):
            make_splits({**cfg, "train_count": total})
        with tempfile.TemporaryDirectory() as tmp:
            first = export_splits(splits, Path(tmp) / "a", cfg)
            second = export_splits(splits, Path(tmp) / "b", cfg)
            self.assertEqual(first, second)

    def test_calendar_validation_and_strict_encoding(self):
        self.assertEqual(parse_human("February 29, 2000"), dt.date(2000, 2, 29))
        for value in [
            "February 29, 2001",
            "April 31, 2000",
            "August 03, 2026",
            "August 3, 2051",
            "august 3, 2026",
        ]:
            with self.assertRaises(ValueError):
                parse_human(value)
        with self.assertRaises(ValueError):
            encode("January", 3)
        with self.assertRaises(ValueError):
            encode("?", 20)
        self.assertTrue(is_iso("2000-02-29"))
        self.assertFalse(is_iso("2001-02-29"))

    def test_shifted_targets_and_padding(self):
        source, decoder_in, truth = Dates([("May 1, 2000", "2000-05-01")])[0]
        self.assertEqual(tuple(source.shape), (20,))
        self.assertEqual(int(decoder_in[0]), BOS)
        self.assertEqual(int(truth[10]), EOS)
        self.assertEqual(int(truth[11]), PAD)
        self.assertTrue(torch.equal(decoder_in[1:11], truth[:10]))

    def test_causal_future_tokens_cannot_change_earlier_logits(self):
        model = Seq2SeqTransformer(
            len(VOCAB), width=16, heads=2, layers=1, ff_width=32, dropout=0
        ).eval()
        source = torch.tensor([[3, 4, 5, 0, 0]])
        target = torch.tensor([[BOS, 6, 7, 8]])
        changed = target.clone()
        changed[:, -1] = 9
        with torch.no_grad():
            before, after = model(source, target), model(source, changed)
        torch.testing.assert_close(before[:, :3], after[:, :3], atol=1e-6, rtol=1e-6)
        expected = torch.tensor(
            [[False, True, True], [False, False, True], [False, False, False]]
        )
        self.assertTrue(torch.equal(causal_mask(3), expected))

    def test_source_and_target_padding_invariance_and_attention(self):
        model = Seq2SeqTransformer(
            len(VOCAB), width=16, heads=2, layers=1, ff_width=32, dropout=0
        ).eval()
        source = torch.tensor([[3, 4, 5]])
        padded = torch.tensor([[3, 4, 5, 0, 0]])
        target = torch.tensor([[BOS, 6, 7]])
        with torch.no_grad():
            a = model(source, target)
            b = model(padded, torch.tensor([[BOS, 6, 7, 0, 0]]))
            memory, padding = model.encode(padded)
            _, weights = model.decode(target, memory, padding, capture=True)
        torch.testing.assert_close(a, b[:, :3], atol=2e-6, rtol=2e-6)
        self.assertTrue(
            torch.equal(weights[:, :, 3:], torch.zeros_like(weights[:, :, 3:]))
        )
        torch.testing.assert_close(weights.sum(-1), torch.ones(1, 3))

    def test_model_dimensions_and_all_padding_are_rejected(self):
        with self.assertRaises(ValueError):
            Seq2SeqTransformer(len(VOCAB), width=15, heads=2)
        model = Seq2SeqTransformer(len(VOCAB), width=16, heads=2, layers=1, ff_width=32)
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 20, dtype=torch.long), torch.tensor([[BOS]]))
        with self.assertRaises(ValueError):
            model(torch.ones(1, 33, dtype=torch.long), torch.tensor([[BOS]]))

    def test_greedy_eos_budget_mixed_completion_and_attention_alignment(self):
        rows = generate(ScheduledModel(), ["May 1, 2000", "May 2, 2000"])
        self.assertEqual([row["prediction"] for row in rows], ["2", "22"])
        self.assertTrue(all(row["terminated"] for row in rows))
        attention = generate(ScheduledModel(), ["May 1, 2000"], attention=True)[0]
        self.assertEqual(attention["attention_rows"], ["2", "<EOS>"])
        self.assertEqual(len(attention["attention"]), 2)
        self.assertEqual(len(attention["attention"][0]), len(attention["source"]))
        unfinished = generate(ScheduledModel(False), ["May 1, 2000"], max_tokens=2)[0]
        self.assertEqual(unfinished["prediction"], "22")
        self.assertFalse(unfinished["terminated"])
        with self.assertRaises(ValueError):
            generate(ScheduledModel(), ["May 1, 2000"], max_tokens=33)

    def test_sinusoidal_odd_dimensions_are_finite(self):
        positions = sinusoidal_positions(7, 5)
        self.assertEqual(tuple(positions.shape), (7, 5))
        self.assertTrue(torch.isfinite(positions).all())
        torch.testing.assert_close(
            positions[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        )

    def test_train_checkpoint_evaluate_predict_round_trip(self):
        config = {
            "seed": 4,
            "train_count": 16,
            "validation_count": 4,
            "test_count": 4,
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
            "cpu_threads": 2,
            "device": "cpu",
            "model": {
                "width": 16,
                "heads": 2,
                "layers": 1,
                "ff_width": 32,
                "dropout": 0,
                "max_length": 32,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "config.json"
            path.write_text(json.dumps(config))
            report = train(path, root / "train")
            _model, saved = load_checkpoint(root / "train/best.pt")
            self.assertEqual(saved["selected_epoch"], 1)
            metrics = evaluate_checkpoint(root / "train/best.pt", root / "evaluation")
            self.assertEqual(metrics, report["test"])
            self.assertEqual(metrics["rule_parser_exact_match"], 1)
            # Verify the domain parser never silently substitutes its correct rule output.
            with patch(
                "date_transformer.workflow.generate",
                return_value=[{"prediction": "wrong", "terminated": False}],
            ) as decoding:
                result = predict(
                    root / "train/best.pt", "August 3, 2026", root / "prediction"
                )
                self.assertEqual(result["prediction"], "wrong")
                decoding.assert_called_once()
            with self.assertRaisesRegex(ValueError, "new or empty"):
                train(path, root / "train")
            with self.assertRaises(ValueError):
                predict(root / "train/best.pt", "April 31, 2000", root / "invalid")
            bad = root / "bad.pt"
            torch.save({"format": "bad", "vocabulary": VOCAB}, bad)
            with self.assertRaises(ValueError):
                load_checkpoint(bad)


if __name__ == "__main__":
    unittest.main()
