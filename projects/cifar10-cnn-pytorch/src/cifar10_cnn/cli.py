"""Explicit commands keep test evaluation separate from training."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="Custom CIFAR-10 CNN workflows")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--data-root", default="data")
    fit.add_argument("--config", default="configs/default.json")
    fit.add_argument("--output", default="artifacts/run")
    fit.add_argument("--download", action="store_true")
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--data-root", default="data")
    evaluate_parser.add_argument("--output", default="artifacts/test-metrics.json")
    evaluate_parser.add_argument("--limit", type=int)
    evaluate_parser.add_argument("--download", action="store_true")
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--image", required=True)
    prediction.add_argument("--output", default="artifacts/prediction.json")
    example = commands.add_parser("demo")
    example.add_argument("--output", default="artifacts/demo")
    reporting = commands.add_parser("report")
    reporting.add_argument("--run", required=True)
    reporting.add_argument("--output", default="artifacts/report")
    reporting.add_argument("--evaluation")
    fetching = commands.add_parser("download")
    fetching.add_argument("--data-root", default="data")
    args = parser.parse_args()
    try:
        if args.command == "train":
            from .train import train
            result = train(args.data_root, args.config, args.output, args.download)
        elif args.command == "evaluate":
            from .evaluate import evaluate
            result = evaluate(args.checkpoint, args.data_root, args.output, args.limit, args.download)
        elif args.command == "predict":
            from .predict import predict
            result = predict(args.checkpoint, args.image, args.output)
        elif args.command == "demo":
            from .demo import demo
            result = demo(args.output)
        elif args.command == "report":
            from .report import report
            result = report(args.run, args.output, args.evaluation)
        else:
            from torchvision.datasets import CIFAR10
            training = CIFAR10(args.data_root, train=True, download=True)
            testing = CIFAR10(args.data_root, train=False, download=True)
            result = {"train_images": len(training), "test_images": len(testing)}
        summary = {key: value for key, value in result.items() if key not in {"selected_test_indices", "confusion_matrix"}}
        print(json.dumps(summary, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
