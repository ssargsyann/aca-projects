"""Explicit commands with optional YOLO imports kept outside U-Net workflows."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("unet-train")
    train.add_argument("--config", required=True)
    train.add_argument("--output", required=True)
    pred = sub.add_parser("unet-predict")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--image", required=True)
    pred.add_argument("--output", required=True)
    pred.add_argument("--device", default="cpu")
    ytrain = sub.add_parser("yolo-train")
    ytrain.add_argument("--checkpoint", required=True)
    ytrain.add_argument("--dataset", required=True)
    ytrain.add_argument("--config", required=True)
    ytrain.add_argument("--output", required=True)
    ypred = sub.add_parser("yolo-predict")
    ypred.add_argument("--checkpoint", required=True)
    ypred.add_argument("--source", required=True)
    ypred.add_argument("--output", required=True)
    ypred.add_argument("--imgsz", type=int, default=320)
    ypred.add_argument("--confidence", type=float, default=0.25)
    ypred.add_argument("--max-frames", type=int)
    ypred.add_argument("--device", default="cpu")
    fetch = sub.add_parser("fetch-assets")
    fetch.add_argument("--output", required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "fetch-assets":
        from .assets import fetch

        value = fetch(**args)
    elif command.startswith("unet"):
        from . import segmentation

        if command == "unet-predict":
            args["image_path"] = args.pop("image")
        if command == "unet-train":
            args["config_path"] = args.pop("config")
        value = (
            segmentation.train if command == "unet-train" else segmentation.predict
        )(**args)
    else:
        from . import detection

        if command == "yolo-train":
            args["dataset_root"] = args.pop("dataset")
            args["config_path"] = args.pop("config")
        value = (detection.train if command == "yolo-train" else detection.predict)(
            **args
        )
    print(json.dumps(value, indent=2))
