"""Reproducible local-model experiments; no high-level generate() calls."""

import json
from pathlib import Path

from .common import new_output, write_json
from .engine import generate, load_local, next_logits
from .sampling import SamplingConfig


def generate_run(model_dir, prompt, config_path, output, device="cpu"):
    config = SamplingConfig(
        **json.loads(Path(config_path).read_text(encoding="utf-8"))
    ).validate()
    output = new_output(output)
    model, tokenizer, manifest = load_local(model_dir, device)
    result = generate(model, tokenizer, prompt, config, device)
    write_json(output / "generation.json", result)
    write_json(output / "model-manifest.json", manifest)
    return result


def compare(model_dir, config_path, output, device="cpu"):
    experiment = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if (
        not experiment["prompts"]
        or not experiment["seeds"]
        or not experiment["strategies"]
    ):
        raise ValueError("At least one prompt, seed and strategy required")
    if len({row["name"] for row in experiment["strategies"]}) != len(
        experiment["strategies"]
    ):
        raise ValueError("Strategy names must be unique")
    if (
        isinstance(experiment["max_new_tokens"], bool)
        or not isinstance(experiment["max_new_tokens"], int)
        or experiment["max_new_tokens"] < 1
    ):
        raise ValueError("Comparison requires a positive generation budget")
    if any(
        not isinstance(prompt, str) or not prompt.strip()
        for prompt in experiment["prompts"]
    ):
        raise ValueError("Every comparison prompt must contain text")
    configs = []
    for row in experiment["strategies"]:
        for seed in experiment["seeds"]:
            config = SamplingConfig(
                **{key: value for key, value in row.items() if key != "name"},
                seed=seed,
                max_new_tokens=experiment["max_new_tokens"],
            ).validate()
            configs.append((row["name"], config))
    output = new_output(output)
    model, tokenizer, manifest = load_local(model_dir, device)
    write_json(output / "model-manifest.json", manifest)
    write_json(output / "config.json", experiment)
    rows = []
    with (output / "continuations.jsonl").open("w", encoding="utf-8") as stream:
        for prompt_index, prompt in enumerate(experiment["prompts"]):
            for name, config in configs:
                result = generate(model, tokenizer, prompt, config, device)
                result.update(strategy_name=name, prompt_index=prompt_index)
                rows.append(result)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(
                    f"Completed prompt {prompt_index + 1}, {name}, seed {config.seed}",
                    flush=True,
                )
    from .reports import continuation_figure, distribution_figure, distributions

    logits, prompt_ids = next_logits(model, tokenizer, experiment["prompts"][0], device)
    record = distributions(
        logits, tokenizer, experiment["strategies"], experiment["prompts"][0]
    )
    record["prompt_token_ids"] = prompt_ids
    write_json(output / "first-token.json", record)
    distribution_figure(record, output / "first-token.png")
    continuation_figure(rows, output / "continuations.png")
    summary = {
        "scope": "small_predeclared_qualitative_comparison_no_quality_benchmark",
        "runs": len(rows),
        "prompts": len(experiment["prompts"]),
        "seeds": experiment["seeds"],
        "strategies": [name["name"] for name in experiment["strategies"]],
        "generated_tokens": sum(row["generated_tokens"] for row in rows),
        "model_revision": manifest["revision"],
        "runtime": manifest["runtime"],
        "observations": [
            {
                "strategy": name,
                "runs": sum(row["strategy_name"] == name for row in rows),
                "mean_entropy_nats": sum(
                    row["summary"]["mean_sampling_entropy_nats"]
                    for row in rows
                    if row["strategy_name"] == name
                )
                / sum(row["strategy_name"] == name for row in rows),
            }
            for name in [variant["name"] for variant in experiment["strategies"]]
        ],
    }
    write_json(output / "summary.json", summary)
    return summary
