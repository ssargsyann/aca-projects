"""Explicit offline geometry and pretrained-inference commands."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description="CV model zoo and educational YOLO geometry")
    commands = parser.add_subparsers(dest="command", required=True)
    demo_parser = commands.add_parser("demo", help="Generate a synthetic geometry example without model downloads")
    demo_parser.add_argument("--output", default="artifacts/demo")
    sample_parser = commands.add_parser("sample", help="Explicitly retrieve the public example image with checksum validation")
    sample_parser.add_argument("--output", default="assets/dog.jpg")
    geometry_parser = commands.add_parser("geometry")
    geometry_parser.add_argument("--image", required=True)
    geometry_parser.add_argument("--annotations", required=True)
    geometry_parser.add_argument("--output", default="artifacts/geometry")
    weights_parser = commands.add_parser("weights", help="Explicitly download/verify official pretrained files")
    weights_parser.add_argument("--cache-dir", default="artifacts/weights")
    weights_parser.add_argument("--tasks", nargs="+", default=["classification", "detection", "instance", "semantic"])
    infer_parser = commands.add_parser("infer")
    infer_parser.add_argument("--image", required=True)
    infer_parser.add_argument("--config", default="configs/inference.json")
    infer_parser.add_argument("--weights-dir", default="artifacts/weights")
    infer_parser.add_argument("--output", default="artifacts/inference")
    montage_parser = commands.add_parser("montage")
    montage_parser.add_argument("--run", required=True)
    montage_parser.add_argument("--output", default="artifacts/montage.png")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            from .geometry_workflow import demo
            result = demo(args.output)
        elif args.command == "geometry":
            from .geometry_workflow import geometry
            result = geometry(args.image, args.annotations, args.output)
        elif args.command == "weights":
            from .models import prepare_weights
            result = prepare_weights(args.cache_dir, args.tasks, download=True)
        elif args.command == "infer":
            from .inference import inference
            result = inference(args.image, args.config, args.weights_dir, args.output)
        elif args.command == "sample":
            from .sample import fetch_sample
            result = fetch_sample(args.output)
        else:
            from .inference import montage
            result = {"saved": str(montage(args.run, args.output))}
        print(json.dumps(result, indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
