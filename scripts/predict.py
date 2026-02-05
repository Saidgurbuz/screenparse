# training/predict_folder.py
"""Run YOLO prediction on a folder of images.

Usage:
    python scripts/predict.py --weights path/to/weights.pt --source path/to/images --conf 0.05
"""
import random
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def auto_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="images/val")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.01)
    ap.add_argument("--project", type=str, default="viz/preds")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-det", type=int, default=700, help="Maximum number of detections per image")
    ap.add_argument("--line-width", type=int, default=2, help="Bounding box line thickness")
    ap.add_argument("--font-size", type=int, default=None, help="Label font size")
    args = ap.parse_args()

    # Generate dynamic name if not provided
    if args.name is None:
        weights_path = Path(args.weights)
        source_path = Path(args.source)
        parent_folder = weights_path.parent.parent.name  # Gets parent of 'weights' folder
        source_name = source_path.name # Gets the name of the input folder
        conf_str = str(args.conf).replace('.', '')  # Remove decimal point
        args.name = f"run_{parent_folder}_{source_name}_c{conf_str}_imgsz{args.imgsz}_md{args.max_det}"

    dev = args.device or auto_device()
    model = YOLO(args.weights)
    # Configure visualization parameters
    model.overrides['line_width'] = 3
    source = args.source

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be a positive integer")
        source_path = Path(args.source)
        if source_path.is_dir():
            files = sorted(
                p
                for p in source_path.iterdir()
                if p.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            )
            if not files:
                raise FileNotFoundError(f"No image files found in {source_path}")
            # randomize the list of files
            random.shuffle(files)
    model.predict(
        source=source,
        batch=1,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.project,
        name=args.name,
        device=dev,
        max_det=args.max_det,
        line_width=args.line_width,
    )
    print(f"Overlays saved under {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
