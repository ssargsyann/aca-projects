"""Commands for deterministic datasets, training, evaluation, prediction and concepts."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("dataset")
    data.add_argument("--config", required=True)
    data.add_argument("--output", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    pred = sub.add_parser("predict")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--text", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--attention", action="store_true")
    pred.add_argument("--device", default="cpu")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--split", choices=["validation", "test"], default="test")
    evaluate.add_argument("--device", default="cpu")
    concepts = sub.add_parser("mechanics")
    concepts.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "dataset":
        from .data import export_splits, make_splits
        from .workflow import new_output

        config = json.loads(Path(args["config"]).read_text(encoding="utf-8"))
        value = export_splits(make_splits(config), new_output(args["output"]), config)
    elif command == "mechanics":
        from .mechanics import plot_mechanics

        value = plot_mechanics(**args)
    else:
        from . import workflow

        if command == "train":
            args["config_path"] = args.pop("config")
        function = {
            "train": workflow.train,
            "predict": workflow.predict,
            "evaluate": workflow.evaluate_checkpoint,
        }[command]
        value = function(**args)
    print(json.dumps(value, indent=2))
