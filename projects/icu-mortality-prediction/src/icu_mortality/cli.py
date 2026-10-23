"""Command-line interface."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Educational ICU mortality classification")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a complete synthetic demonstration without a dataset download")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--rows", type=int, default=240)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--explain", action="store_true", help="Also generate SHAP plots (requires explain extras)")
    training = commands.add_parser("train", help="Train and evaluate a pipeline")
    training.add_argument("--data", required=True)
    training.add_argument("--config", default="configs/default.json")
    training.add_argument("--output", default="artifacts/run")
    scoring = commands.add_parser("predict", help="Score rows using a trusted pipeline")
    scoring.add_argument("--model", required=True)
    scoring.add_argument("--data", required=True)
    scoring.add_argument("--output", default="artifacts/predictions.csv")
    explaining = commands.add_parser("explain", help="Save optional SHAP plots for input rows")
    explaining.add_argument("--model", required=True)
    explaining.add_argument("--data", required=True)
    explaining.add_argument("--output", default="artifacts/explanations")
    explaining.add_argument("--max-rows", type=int, default=100)
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import run_demo
        print(json.dumps(run_demo(args.output, args.rows, args.seed, args.explain), indent=2))
    elif args.command == "train":
        from .train import train
        print(json.dumps(train(args.data, args.config, args.output), indent=2))
    elif args.command == "predict":
        from .predict import predict
        result = predict(args.model, args.data, args.output)
        print(f"Wrote {len(result)} predictions to {args.output}")
    else:
        from .explain import explain
        explain(args.model, args.data, args.output, args.max_rows)
        print(f"Wrote SHAP plots to {args.output}")


if __name__ == "__main__":
    main()
