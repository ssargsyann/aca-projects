"""Explicit single-prompt raw-forward generation, optionally reusing the KV cache."""

from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import torch

from .assets import verify_model
from .sampling import distribution, entropy


def load_local(model_dir, device="cpu", cpu_threads=2):
    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            "Supply an existing local pinned model directory; run fetch-model explicitly"
        )
    if (
        device not in {"cpu", "cuda"}
        or device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ValueError("Choose an available cpu/cuda device")
    if (
        isinstance(cpu_threads, bool)
        or not isinstance(cpu_threads, int)
        or cpu_threads < 1
    ):
        raise ValueError("cpu_threads must be positive")
    manifest = verify_model(model_dir)
    torch.set_num_threads(cpu_threads)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(device)
        .eval()
    )
    manifest["runtime"] = {
        "device": device,
        "cpu_threads": cpu_threads,
        "dtype": "float32",
        "attention_implementation": "eager",
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "versions": {
            name: version(name)
            for name in ["torch", "transformers", "tokenizers", "safetensors"]
        },
    }
    return model, tokenizer, manifest


@torch.inference_mode()
def next_logits(model, tokenizer, prompt, device="cpu"):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must contain non-whitespace text")
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    ids = encoded["input_ids"].to(device)
    if ids.shape[1] == 0 or ids.shape[1] > model.config.n_positions:
        raise ValueError("Prompt exceeds the model context or contains no tokens")
    return model(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
    ).logits[0, -1].detach(), ids[0].cpu().tolist()


@torch.inference_mode()
def generate(model, tokenizer, prompt, config, device="cpu"):
    config.validate()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must contain non-whitespace text")
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)[
        "input_ids"
    ].to(device)
    prompt_ids = ids[0].cpu().tolist()
    if (
        ids.ndim != 2
        or ids.shape[0] != 1
        or not prompt_ids
        or len(prompt_ids) + config.max_new_tokens > model.config.n_positions
    ):
        raise ValueError(
            "Prompt plus requested generation exceeds context capacity; no truncation is performed"
        )
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.seed)
    past = None
    generated = []
    trace = []
    reason = "token_budget"
    for step in range(config.max_new_tokens):
        step_ids = ids[:, -1:] if config.use_cache and past is not None else ids
        kwargs = {
            "input_ids": step_ids,
            "attention_mask": torch.ones_like(ids),
            "use_cache": config.use_cache,
        }
        if config.use_cache and past is not None:
            kwargs["past_key_values"] = past
        outputs = model(**kwargs)
        logits = outputs.logits[0, -1]
        if config.use_cache:
            past = outputs.past_key_values
        probabilities, scores = distribution(logits, config)
        if config.strategy == "greedy" or config.temperature == 0:
            next_id = int(logits.argmax())
        else:
            next_id = int(torch.multinomial(probabilities, 1, generator=generator))
        base_logprob = float(logits.double().log_softmax(-1)[next_id])
        trace.append(
            {
                "step": step,
                "token_id": next_id,
                "token_text": tokenizer.decode([next_id], skip_special_tokens=False),
                "eligible_tokens": int(torch.isfinite(scores).sum()),
                "positive_probability_tokens": int((probabilities > 0).sum()),
                "entropy_nats": float(entropy(probabilities)),
                "sampling_probability": float(probabilities[next_id]),
                "base_model_logprob": base_logprob,
            }
        )
        generated.append(next_id)
        ids = torch.cat(
            [ids, torch.tensor([[next_id]], device=device, dtype=ids.dtype)], dim=1
        )
        if next_id == tokenizer.eos_token_id:
            reason = "eos"
            break
    visible_ids = [token for token in generated if token != tokenizer.eos_token_id]
    bigrams = list(zip(visible_ids, visible_ids[1:]))
    continuation = tokenizer.decode(generated, skip_special_tokens=True)
    return {
        "prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "continuation": continuation,
        "text": prompt + continuation,
        "generated_token_ids": generated,
        "generated_tokens": len(generated),
        "stop_reason": reason,
        "config": asdict(config),
        "trace": trace,
        "summary": {
            "mean_base_model_logprob": sum(row["base_model_logprob"] for row in trace)
            / len(trace)
            if trace
            else None,
            "mean_sampling_entropy_nats": sum(row["entropy_nats"] for row in trace)
            / len(trace)
            if trace
            else None,
            "distinct_1": len(set(visible_ids)) / len(visible_ids)
            if visible_ids
            else None,
            "distinct_2": len(set(bigrams)) / len(bigrams) if bigrams else None,
        },
        "scope": "qualitative_continuation_not_factuality_or_quality_benchmark",
    }
