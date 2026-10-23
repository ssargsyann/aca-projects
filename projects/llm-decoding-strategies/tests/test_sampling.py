import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch

from llm_decoding.engine import generate, load_local
from llm_decoding.reports import demo
from llm_decoding.sampling import (
    SamplingConfig,
    distribution,
    entropy,
    temperature_logits,
    top_k_logits,
    top_p_logits,
)

torch.set_num_threads(2)


class Tokenizer:
    eos_token_id = 4

    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(
            "abcde"[i]
            for i in ids
            if not (skip_special_tokens and i == self.eos_token_id)
        )


class Model(torch.nn.Module):
    config = SimpleNamespace(n_positions=16)

    def __init__(self, eos=False):
        super().__init__()
        self.calls = []
        self.eos = eos

    def forward(self, input_ids, attention_mask, use_cache, past_key_values=None):
        length = input_ids.shape[1] + (past_key_values or 0)
        self.calls.append(input_ids.shape[1])
        scores = torch.tensor([2.0, 1.0, 0.7, 0.3, -10.0])
        if self.eos and length >= 4:
            scores[4] = 10.0
        return SimpleNamespace(
            logits=scores[None, None, :].repeat(1, input_ids.shape[1], 1),
            past_key_values=length,
        )


class SamplingTests(unittest.TestCase):
    def test_temperature_stability_zero_and_entropy(self):
        logits = torch.tensor([2.0, 1.0, -4.0])
        low = temperature_logits(logits, 0.5).softmax(-1)
        high = temperature_logits(logits, 2.0).softmax(-1)
        self.assertLess(float(entropy(low)), float(entropy(high)))
        torch.testing.assert_close(
            temperature_logits(logits, 0).softmax(-1),
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64),
        )
        tiny = temperature_logits(torch.tensor([1e38, -1e38]), 1e-300).softmax(-1)
        torch.testing.assert_close(tiny, torch.tensor([1.0, 0.0], dtype=torch.float64))
        self.assertTrue(torch.equal(logits, torch.tensor([2.0, 1.0, -4.0])))

    def test_invalid_logits_and_parameters_fail(self):
        for logits in [
            torch.tensor([]),
            torch.tensor([1, 2]),
            torch.tensor([float("nan"), 0.0]),
            torch.tensor([float("inf"), 0.0]),
            torch.tensor([float("-inf"), float("-inf")]),
        ]:
            with self.assertRaises(ValueError):
                temperature_logits(logits, 1.0)
        for value in [-1.0, float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                temperature_logits(torch.tensor([0.0]), value)
        for value in [0.0, -1.0, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                top_p_logits(torch.tensor([0.0]), value)
        for value in [-1, 1.5, True]:
            with self.assertRaises(ValueError):
                top_k_logits(torch.tensor([0.0]), value)
        with self.assertRaises(ValueError):
            SamplingConfig(max_new_tokens=-1).validate()

    def test_top_k_limits_ties_and_nonmutation(self):
        logits = torch.tensor([[3.0, 2.0, 2.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
        copy = logits.clone()
        masked = top_k_logits(logits, 2)
        self.assertEqual(torch.isfinite(masked).sum(-1).tolist(), [3, 2])
        self.assertTrue(torch.equal(logits, copy))
        self.assertTrue(torch.equal(top_k_logits(logits, 0), logits))
        self.assertTrue(torch.equal(top_k_logits(logits, 100), logits))

    def test_top_p_boundary_ties_masked_rows_and_minimum_token(self):
        uniform = torch.zeros(4)
        masked = top_p_logits(uniform, 0.5)
        self.assertEqual(torch.isfinite(masked).tolist(), [True, True, False, False])
        self.assertEqual(
            torch.isfinite(top_p_logits(torch.tensor([3.0, 2.0, 1.0]), 0.01)).sum(), 1
        )
        batch = torch.tensor([[1.0, float("-inf"), 0.0], [2.0, 1.0, 0.0]])
        probabilities = top_p_logits(batch, 0.8).softmax(-1)
        torch.testing.assert_close(
            probabilities.sum(-1), torch.ones(2, dtype=torch.float64)
        )
        self.assertEqual(float(probabilities[0, 1]), 0.0)
        self.assertTrue(torch.equal(top_p_logits(uniform, 1), uniform))

    def test_temperature_precedes_nucleus_filter(self):
        logits = torch.tensor([2.0, 1.0, 0.0])
        warm = distribution(
            logits, SamplingConfig(temperature=1.0, top_k=0, top_p=0.8)
        )[1]
        cold = distribution(
            logits, SamplingConfig(temperature=0.25, top_k=0, top_p=0.8)
        )[1]
        self.assertEqual(int(torch.isfinite(warm).sum()), 2)
        self.assertEqual(int(torch.isfinite(cold).sum()), 1)

    def test_generation_is_seeded_without_changing_global_rng(self):
        config = SamplingConfig(
            top_k=0, top_p=1.0, temperature=1.0, max_new_tokens=6, seed=7
        )
        state = torch.random.get_rng_state().clone()
        first = generate(Model(), Tokenizer(), "prompt", config)
        second = generate(Model(), Tokenizer(), "prompt", config)
        self.assertEqual(first, second)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(first["generated_tokens"], 6)

    def test_eos_context_budget_and_cache_call_shapes(self):
        greedy = SamplingConfig(strategy="greedy", max_new_tokens=6)
        cached = Model(eos=True)
        result = generate(cached, Tokenizer(), "prompt", greedy)
        self.assertEqual(result["stop_reason"], "eos")
        self.assertEqual(result["generated_tokens"], 3)
        self.assertEqual(cached.calls, [2, 1, 1])
        uncached = Model(eos=True)
        other = generate(
            uncached, Tokenizer(), "prompt", replace(greedy, use_cache=False)
        )
        self.assertEqual(result["generated_token_ids"], other["generated_token_ids"])
        self.assertEqual(uncached.calls, [2, 3, 4])
        empty = Model()
        zero = generate(empty, Tokenizer(), "prompt", replace(greedy, max_new_tokens=0))
        self.assertEqual(zero["generated_tokens"], 0)
        self.assertEqual(empty.calls, [])
        with self.assertRaises(ValueError):
            generate(Model(), Tokenizer(), "prompt", replace(greedy, max_new_tokens=15))
        with self.assertRaises(ValueError):
            generate(Model(), Tokenizer(), " ", greedy)

    def test_missing_model_fails_before_optional_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_local(Path(tmp) / "missing")
            with self.assertRaises(FileNotFoundError):
                load_local(tmp)

    def test_offline_demo_and_output_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "demo"
            record = demo(output)
            self.assertEqual(len(record["distributions"]), 4)
            self.assertTrue((output / "distributions.png").is_file())
            with self.assertRaisesRegex(ValueError, "new or empty"):
                demo(output)


@unittest.skipUnless(
    importlib.util.find_spec("transformers"),
    "Install the gpt2 extra for framework integration tests",
)
class FrameworkTests(unittest.TestCase):
    def test_filters_match_reference_on_nonboundary_logits(self):
        from transformers.generation.logits_process import (
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        generator = torch.Generator().manual_seed(10)
        logits = torch.randn(3, 31, dtype=torch.float64, generator=generator)
        ids = torch.zeros(3, 1, dtype=torch.long)
        ours = top_p_logits(
            top_k_logits(temperature_logits(logits, 0.7), 7), 0.83
        ).softmax(-1)
        expected = TopPLogitsWarper(0.83)(
            ids, TopKLogitsWarper(7)(ids, TemperatureLogitsWarper(0.7)(ids, logits))
        ).softmax(-1)
        torch.testing.assert_close(ours, expected, atol=1e-12, rtol=1e-12)

    def test_real_random_initialized_gpt2_cached_matches_uncached(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        torch.manual_seed(123)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=5,
                n_positions=16,
                n_ctx=16,
                n_embd=16,
                n_layer=1,
                n_head=2,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
            )
        ).eval()
        tokenizer = Tokenizer()
        tokenizer.eos_token_id = None
        config = SamplingConfig(strategy="greedy", max_new_tokens=5)
        cached = generate(model, tokenizer, "prompt", config)
        uncached = generate(
            model, tokenizer, "prompt", replace(config, use_cache=False)
        )
        self.assertEqual(cached["generated_token_ids"], uncached["generated_token_ids"])
        torch.testing.assert_close(
            torch.tensor([row["base_model_logprob"] for row in cached["trace"]]),
            torch.tensor([row["base_model_logprob"] for row in uncached["trace"]]),
            atol=1e-5,
            rtol=1e-5,
        )


if __name__ == "__main__":
    unittest.main()
