"""Small standalone distribution and observed-continuation figures."""

from dataclasses import asdict

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import new_output, write_json
from .sampling import SamplingConfig, distribution, entropy


def distributions(logits, tokenizer, variants, prompt=None):
    records = []
    for variant in variants:
        config = SamplingConfig(
            **{key: value for key, value in variant.items() if key != "name"}
        )
        probabilities, scores = distribution(logits, config)
        values, indices = probabilities.topk(min(8, len(logits)))
        records.append(
            {
                "name": variant["name"],
                "config": asdict(config),
                "eligible_tokens": int(torch.isfinite(scores).sum()),
                "entropy_nats": float(entropy(probabilities)),
                "top_tokens": [
                    {
                        "id": int(index),
                        "text": tokenizer.decode(
                            [int(index)], skip_special_tokens=False
                        ),
                        "probability": float(value),
                    }
                    for value, index in zip(values, indices)
                    if value > 0
                ],
            }
        )
    return {
        "prompt": prompt,
        "scope": "same_fixed_next_token_logits_under_each_filter",
        "distributions": records,
    }


def distribution_figure(record, path):
    rows = record["distributions"]
    figure, axes = plt.subplots(
        1, len(rows), figsize=(3.2 * len(rows), 4), squeeze=False
    )
    for axis, row in zip(axes.flat, rows):
        tokens = row["top_tokens"][:6]
        axis.bar(range(len(tokens)), [token["probability"] for token in tokens])
        axis.set_xticks(
            range(len(tokens)),
            [repr(token["text"]) for token in tokens],
            rotation=65,
            ha="right",
        )
        axis.set_title(
            f"{row['name']}\nH={row['entropy_nats']:.3f}, eligible={row['eligible_tokens']}",
            fontsize=10,
        )
        axis.set_ylim(0, 1)
    axes[0, 0].set_ylabel("Renormalized next-token probability")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def demo(output):
    output = new_output(output)

    class Tokens:
        def decode(self, ids, **kwargs):
            return "".join(f"token-{index}" for index in ids)

    logits = torch.tensor([2.0, 0.5, -1.0, 8.0, 5.0, 4.5, -3.0, 1.0, 0.0, 7.5])
    variants = [
        {
            "name": "greedy",
            "strategy": "greedy",
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
        },
        {"name": "temperature", "temperature": 0.7, "top_k": 0, "top_p": 1.0},
        {"name": "top-k", "temperature": 0.7, "top_k": 3, "top_p": 1.0},
        {"name": "top-p", "temperature": 0.7, "top_k": 0, "top_p": 0.9},
    ]
    result = distributions(logits, Tokens(), variants)
    result["scope"] = "synthetic_logits_only_no_language_model"
    result["logits"] = logits.tolist()
    write_json(output / "distributions.json", result)
    distribution_figure(result, output / "distributions.png")
    return result


def continuation_figure(rows, path):
    names = list(dict.fromkeys(row["strategy_name"] for row in rows))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for index, name in enumerate(names):
        selected = [row for row in rows if row["strategy_name"] == name]
        values = [row["summary"]["mean_sampling_entropy_nats"] for row in selected]
        distinct = [
            row["summary"]["distinct_2"]
            for row in selected
            if row["summary"]["distinct_2"] is not None
        ]
        axes[0].scatter([index] * len(values), values, alpha=0.65)
        axes[1].scatter([index] * len(distinct), distinct, alpha=0.65)
    axes[0].set_ylabel("Mean filtered entropy per continuation (nats)")
    axes[1].set_ylabel("Distinct token-bigram ratio (length-dependent)")
    for axis in axes:
        axis.set_xticks(range(len(names)), names, rotation=35, ha="right")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
