"""Small command-line interface for complete reproducible workflows."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Descriptive customer segmentation")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--data", required=True)
    fit.add_argument("--config", default="configs/default.json")
    fit.add_argument("--output", default="artifacts/run")
    scoring = commands.add_parser("predict")
    scoring.add_argument("--model", required=True)
    scoring.add_argument("--data", required=True)
    scoring.add_argument("--output", default="artifacts/predictions.csv")
    reporting = commands.add_parser("report")
    reporting.add_argument("--run", required=True)
    reporting.add_argument("--output", default="artifacts/report")
    example = commands.add_parser("demo")
    example.add_argument("--output", default="artifacts/demo")
    example.add_argument("--seed", type=int, default=42)
    fetching = commands.add_parser("download")
    fetching.add_argument("--output", default="data/marketing_campaign.csv")
    args = parser.parse_args()
    try:
        if args.command == "train":
            from .train import train
            print(json.dumps(train(args.data, args.config, args.output), indent=2))
        elif args.command == "predict":
            from .predict import predict
            print(f"Wrote {len(predict(args.model, args.data, args.output))} row assignments.")
        elif args.command == "report":
            from .report import report
            result = report(args.run, args.output)
            print(f"Saved aggregate charts for {result['rows']} fitted rows.")
        elif args.command == "demo":
            from .demo import run_demo
            print("SYNTHETIC DEMO ONLY")
            print(json.dumps(run_demo(args.output, args.seed), indent=2))
        else:
            from .download import download
            print(f"Saved dataset to {download(args.output)}")
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
