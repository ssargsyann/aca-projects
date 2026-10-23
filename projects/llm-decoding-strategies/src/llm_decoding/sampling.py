"""Nonmutating last-axis filters with validated, stable probability conversion."""

import math
from dataclasses import dataclass

import torch


def validate_logits(logits):
    if (
        not isinstance(logits, torch.Tensor)
        or not logits.is_floating_point()
        or logits.ndim < 1
        or logits.shape[-1] < 1
    ):
        raise ValueError(
            "Expected a nonempty floating logits tensor with vocabulary on the last axis"
        )
    if (
        torch.isnan(logits).any()
        or torch.isposinf(logits).any()
        or not torch.isfinite(logits).any(-1).all()
    ):
        raise ValueError("Each row needs a finite logit; NaN/+inf are unsupported")


def temperature_logits(logits, temperature=1.0):
    validate_logits(logits)
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and nonnegative")
    values = logits.to(torch.float64)
    if temperature == 0:
        output = torch.full_like(values, float("-inf"))
        return output.scatter(-1, values.argmax(-1, keepdim=True), 0.0)
    # Subtract before division, preventing +inf for tiny positive temperatures.
    return (values - values.max(-1, keepdim=True).values) / temperature


def top_k_logits(logits, k=0):
    validate_logits(logits)
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    if k == 0 or k >= logits.shape[-1]:
        return logits.clone()
    cutoff = logits.topk(k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def top_p_logits(logits, p=1.0):
    validate_logits(logits)
    if (
        isinstance(p, bool)
        or not isinstance(p, (int, float))
        or not math.isfinite(p)
        or not 0 < p <= 1
    ):
        raise ValueError("top_p must be finite and in (0,1]")
    if p == 1:
        return logits.clone()
    values, indices = torch.sort(logits.to(torch.float64), descending=True, stable=True)
    cumulative = values.softmax(-1).cumsum(-1)
    # Retain the smallest sorted prefix whose CDF reaches p, including its boundary token.
    threshold = torch.full(
        (*cumulative.shape[:-1], 1), p, dtype=cumulative.dtype, device=cumulative.device
    )
    last = torch.searchsorted(
        cumulative.contiguous(), threshold, right=False
    ).clamp_max(values.shape[-1] - 1)
    removed = torch.arange(values.shape[-1], device=values.device) > last
    values = values.masked_fill(removed, float("-inf"))
    return torch.empty_like(values).scatter(-1, indices, values)


@dataclass(frozen=True)
class SamplingConfig:
    strategy: str = "sample"
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    max_new_tokens: int = 24
    seed: int = 42
    use_cache: bool = True

    def validate(self):
        if self.strategy not in {"greedy", "sample"}:
            raise ValueError("Choose greedy or sample")
        temperature_logits(torch.tensor([0.0]), self.temperature)
        top_k_logits(torch.tensor([0.0]), self.top_k)
        top_p_logits(torch.tensor([0.0]), self.top_p)
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or self.max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("seed must be an integer in [0,2**63)")
        if not isinstance(self.use_cache, bool):
            raise ValueError("use_cache must be boolean")
        return self


def distribution(logits, config):
    config.validate()
    validate_logits(logits)
    if config.strategy == "greedy":
        scores = temperature_logits(logits, 0)
    else:
        scores = temperature_logits(logits, config.temperature)
        scores = top_k_logits(scores, config.top_k)
        scores = top_p_logits(scores, config.top_p)
    return scores.softmax(-1), scores


def entropy(probabilities):
    return torch.special.entr(probabilities).sum(-1)
