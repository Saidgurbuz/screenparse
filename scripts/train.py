"""
YOLO training script for webshot dataset.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --epochs 200 --batch 32 --imgsz 1280
"""

import argparse
import os
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Install with:")
    print("  pip install ultralytics")
    sys.exit(1)


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_args_with_config(config: dict, args: argparse.Namespace) -> dict:
    """Merge command-line arguments with config file."""
    # Command-line args override config file
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.batch is not None:
        config["batch"] = args.batch
    if args.imgsz is not None:
        config["imgsz"] = args.imgsz
    if args.device is not None:
        config["device"] = args.device
    if args.model is not None:
        config["model"] = args.model
    if args.data is not None:
        config["data"] = args.data
    if args.name is not None:
        config["name"] = args.name
    if args.resume:
        config["resume"] = True
    if args.workers is not None:
        config["workers"] = args.workers

    return config


def validate_data_yaml(data_path: str) -> bool:
    """Check if data.yaml exists and is valid."""
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found: {data_path}")
        return False

    try:
        with open(data_path, "r") as f:
            data = yaml.safe_load(f)

        required = ["path", "train", "val", "nc", "names"]
        missing = [k for k in required if k not in data]
        if missing:
            print(f"ERROR: Missing keys in {data_path}: {missing}")
            return False

        # Check if paths exist
        base_path = Path(data["path"])
        for split in ["train", "val", "test"]:
            if split in data:
                split_path = base_path / data[split]
                if not split_path.exists():
                    print(f"WARNING: {split} path does not exist: {split_path}")

        return True

    except Exception as e:
        print(f"ERROR: Failed to parse {data_path}: {e}")
        return False


def print_training_info(config: dict):
    """Print training configuration summary."""
    print("\n" + "=" * 70)
    print("YOLO TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Model:       {config['model']}")
    print(f"Dataset:     {config['data']}")
    print(f"Epochs:      {config['epochs']}")
    print(f"Batch size:  {config['batch']}")
    print(f"Image size:  {config['imgsz']}")
    print(f"Device:      {config['device']}")
    print(f"Workers:     {config['workers']}")
    print(f"Optimizer:   {config['optimizer']}")
    print(f"Project:     {config['project']}")
    print(f"Name:        {config['name']}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO model on webshot dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python scripts/train.py

  # Train with custom config
  python scripts/train.py --config configs/custom_config.yaml

  # Override specific parameters
  python scripts/train.py --epochs 200 --batch 32 --imgsz 1280

  # Train on CPU
  python scripts/train.py --device cpu

  # Resume training
  python scripts/train.py --resume

  # Multi-GPU training
  python scripts/train.py --device 0,1,2,3
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training config YAML file",
    )
    parser.add_argument("--model", type=str, help="Model to train (e.g., yolov8n.pt)")
    parser.add_argument("--data", type=str, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch", type=int, help="Batch size")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--device", type=str, help="Device (0, cpu, 0,1,2,3)")
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--workers", type=int, help="Number of dataloader workers")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"ERROR: Config file not found: {args.config}")
        print("Creating default config at configs/train_config.yaml...")
        os.makedirs("configs", exist_ok=True)
        # Would create default config here
        sys.exit(1)

    config = load_config(args.config)
    config = merge_args_with_config(config, args)

    # Validate data.yaml
    if not validate_data_yaml(config["data"]):
        print("\nPlease run the dataset pipeline first:")
        print("  wsd pipeline --urls urls.csv --yolo-dir data/yolo")
        sys.exit(1)

    # Print configuration
    print_training_info(config)

    if args.dry_run:
        print("Dry run - exiting without training")
        return

    # Load model
    print(f"Loading model: {config['model']}")
    model = YOLO(config["model"])

    # Extract training kwargs
    train_kwargs = {k: v for k, v in config.items() if k not in ["model", "task"]}

    # Start training
    print("\nStarting training...")
    try:
        results = model.train(**train_kwargs)

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"\nModel saved to: {config['project']}/{config['name']}/weights/")
        print(f"  - best.pt  (best checkpoint)")
        print(f"  - last.pt  (last checkpoint)")
        print("\nTo validate:")
        print(
            f"  yolo val model={config['project']}/{config['name']}/weights/best.pt data={config['data']}"
        )
        print("\nTo predict:")
        print(
            f"  yolo predict model={config['project']}/{config['name']}/weights/best.pt source=test_image.png"
        )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
