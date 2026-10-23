import argparse
import json


def main():
    parser = argparse.ArgumentParser(
        description="Raw-logit sampling and explicit offline GPT-2 experiments"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", required=True)
    fetch = sub.add_parser("fetch-model")
    fetch.add_argument("--output", required=True)
    verify = sub.add_parser("verify-model")
    verify.add_argument("--model-dir", required=True)
    for command in ["generate", "compare"]:
        task = sub.add_parser(command)
        task.add_argument("--model-dir", required=True)
        task.add_argument("--config", required=True)
        task.add_argument("--output", required=True)
        task.add_argument("--device", default="cpu")
        if command == "generate":
            task.add_argument("--prompt", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "demo":
        from .reports import demo

        result = demo(**args)
    elif command == "fetch-model":
        from .assets import fetch_model

        result = fetch_model(**args)
    elif command == "verify-model":
        from .assets import verify_model

        result = verify_model(args["model_dir"])
    else:
        from .workflow import compare, generate_run

        args["config_path"] = args.pop("config")
        result = (compare if command == "compare" else generate_run)(**args)
    print(json.dumps(result, indent=2))
