"""Training with held-out greedy validation, safe checkpoint restoration and reports."""

import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import PAD, VOCAB, Dates, export_splits, make_splits, parse_human
from .generation import generate
from .model import Seq2SeqTransformer


def new_output(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Choose a new or empty output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure(config):
    for key in [
        "train_count",
        "validation_count",
        "test_count",
        "epochs",
        "batch_size",
        "cpu_threads",
    ]:
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], int)
            or config[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    if (
        not isinstance(config["seed"], int)
        or isinstance(config["seed"], bool)
        or config["seed"] < 0
    ):
        raise ValueError("seed must be a nonnegative integer")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if config["model"]["max_length"] < 20:
        raise ValueError("Positional capacity must cover the 20 source positions")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if (
        device not in {"cpu", "cuda"}
        or device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise ValueError("Choose an available cpu/cuda device")
    return device


@torch.no_grad()
def teacher_metrics(model, loader, device):
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for source, decoder_in, truth in loader:
        source, decoder_in, truth = (
            source.to(device),
            decoder_in.to(device),
            truth.to(device),
        )
        logits = model(source, decoder_in)
        loss_sum += float(
            F.cross_entropy(
                logits.flatten(0, 1), truth.flatten(), ignore_index=PAD, reduction="sum"
            )
        )
        valid = truth.ne(PAD)
        count += int(valid.sum())
        correct += int((logits.argmax(-1).eq(truth) & valid).sum())
    return {
        "teacher_forced_nll": loss_sum / count,
        "teacher_forced_token_accuracy": correct / count,
        "nonpad_target_tokens": count,
    }


@torch.no_grad()
def evaluate(model, pairs, batch_size, device):
    loader = DataLoader(
        Dates(pairs), batch_size=batch_size, shuffle=False, num_workers=0
    )
    metrics = teacher_metrics(model, loader, device)
    predictions = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        for result, (_, expected) in zip(
            generate(model, [source for source, _ in batch], device), batch
        ):
            result["expected"] = expected
            result["exact_match"] = (
                result["prediction"] == expected and result["terminated"]
            )
            predictions.append(result)
    metrics.update(
        count=len(pairs),
        greedy_exact_match=sum(row["exact_match"] for row in predictions) / len(pairs),
        greedy_valid_iso=sum(row["valid_iso"] for row in predictions) / len(pairs),
        greedy_eos_rate=sum(row["terminated"] for row in predictions) / len(pairs),
        rule_parser_exact_match=sum(
            parse_human(source).isoformat() == target for source, target in pairs
        )
        / len(pairs),
    )
    return metrics, predictions


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (
        saved.get("format") != "aca-date-transformer-v1"
        or saved.get("vocabulary") != VOCAB
    ):
        raise ValueError("Unsupported checkpoint format or vocabulary")
    model = Seq2SeqTransformer(len(VOCAB), **saved["config"]["model"])
    model.load_state_dict(saved["state_dict"], strict=True)
    return model.to(device).eval(), saved


def train(config_path, output):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    device = configure(config)
    splits = make_splits(config)
    output = new_output(output)
    manifest = export_splits(splits, output / "data", config)
    loader = DataLoader(
        Dates(splits["train"]),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    model = Seq2SeqTransformer(len(VOCAB), **config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best = (-1.0, float("-inf"))
    history = []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for source, decoder_in, truth in loader:
            source, decoder_in, truth = (
                source.to(device),
                decoder_in.to(device),
                truth.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(source, decoder_in)
            loss = F.cross_entropy(
                logits.flatten(0, 1), truth.flatten(), ignore_index=PAD
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tokens = int(truth.ne(PAD).sum())
            loss_sum += float(loss.detach()) * tokens
            count += tokens
        validation, _ = evaluate(
            model, splits["validation"], config["batch_size"], device
        )
        row = {
            "epoch": epoch,
            "train_nll": loss_sum / count,
            **{
                "validation_" + key: value
                for key, value in validation.items()
                if key != "rule_parser_exact_match"
            },
        }
        history.append(row)
        selection = (
            validation["greedy_exact_match"],
            -validation["teacher_forced_nll"],
        )
        if selection > best:
            best = selection
            torch.save(
                {
                    "format": "aca-date-transformer-v1",
                    "config": config,
                    "vocabulary": VOCAB,
                    "selected_epoch": epoch,
                    "state_dict": {
                        name: value.cpu() for name, value in model.state_dict().items()
                    },
                    "dataset_manifest": manifest,
                },
                output / "best.pt",
            )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_nll": row["train_nll"],
                    "validation_greedy_exact": validation["greedy_exact_match"],
                    "validation_nll": validation["teacher_forced_nll"],
                }
            ),
            flush=True,
        )
    model, saved = load_checkpoint(output / "best.pt", device)
    validation, _ = evaluate(model, splits["validation"], config["batch_size"], device)
    # Test is accessed for scoring only after validation-based checkpoint selection.
    test, predictions = evaluate(model, splits["test"], config["batch_size"], device)
    result = {
        "scope": "synthetic_in_distribution_date_normalization_not_machine_translation_benchmark",
        "config": config,
        "selected_epoch": saved["selected_epoch"],
        "validation": validation,
        "test": test,
        "history": history,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "dataset_manifest": manifest,
        "checkpoint_sha256": file_hash(output / "best.pt"),
        "versions": {name: version(name) for name in ["torch", "numpy", "matplotlib"]},
    }
    write_json(output / "results.json", result)
    (output / "test-predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    from .report import attention_figure, training_figure

    training_figure(result, output / "training.png")
    attention = generate(model, [splits["test"][0][0]], device, attention=True)[0]
    attention["expected"] = splits["test"][0][1]
    write_json(output / "attention.json", attention)
    attention_figure(attention, output / "attention.png")
    return result


def predict(checkpoint, text, output, attention=False, device="cpu"):
    parse_human(
        text
    )  # Domain validation only; the parser result never replaces neural output.
    if device not in {"cpu", "cuda"}:
        raise ValueError("Choose cpu or cuda")
    model, saved = load_checkpoint(checkpoint, device)
    torch.set_num_threads(saved["config"]["cpu_threads"])
    output = new_output(output)
    result = generate(model, [text], device, attention=attention)[0]
    result.update(
        checkpoint_sha256=file_hash(checkpoint), selected_epoch=saved["selected_epoch"]
    )
    write_json(output / "prediction.json", result)
    if attention:
        from .report import attention_figure

        attention_figure(result, output / "attention.png")
    return result


def evaluate_checkpoint(checkpoint, output, split="test", device="cpu"):
    if split not in {"validation", "test"}:
        raise ValueError("Choose validation or test")
    model, saved = load_checkpoint(checkpoint, device)
    torch.set_num_threads(saved["config"]["cpu_threads"])
    pairs = make_splits(saved["config"])[split]
    metrics, rows = evaluate(model, pairs, saved["config"]["batch_size"], device)
    output = new_output(output)
    write_json(
        output / "metrics.json",
        {
            "split": split,
            "metrics": metrics,
            "checkpoint_sha256": file_hash(checkpoint),
        },
    )
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return metrics
