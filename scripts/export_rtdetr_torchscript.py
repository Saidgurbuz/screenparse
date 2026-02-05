#!/usr/bin/env python3
"""
Export RT-DETR model to TorchScript format.

This script must be run from the RTDETRv2 environment (where rtdetrv2 is installed).
The resulting TorchScript model can be used for inference without rtdetrv2.

IMPORTANT: The model must be traced on the same device type it will be used for inference.
- Use --device cuda (default) for GPU inference
- Use --device cpu for CPU-only inference

Usage:
    python scripts/export_rtdetr_torchscript.py \
        -c path/to/config.yml \
        -m path/to/model.pth \
        -o path/to/output.torchscript.pt \
        --device cuda

The exported model can then be used in the evaluation framework:
    PYTHONPATH=src python -m evaluation.cli \
        --rtdetr path/to/model.torchscript.pt \
        --rtdetr-classes path/to/classes.txt \
        ...
"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Export RT-DETR model to TorchScript format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to RT-DETR config YAML file",
    )
    parser.add_argument(
        "--model-checkpoint", "-m", required=True,
        help="Path to checkpoint .pth file",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output path for TorchScript model (.pt)",
    )
    parser.add_argument(
        "--input-height", type=int, default=736,
        help="Input image height for tracing",
    )
    parser.add_argument(
        "--input-width", type=int, default=1280,
        help="Input image width for tracing",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="Device to trace on. Use 'cuda' for GPU inference, 'cpu' for CPU-only.",
    )
    args = parser.parse_args()

    # Import rtdetrv2 (must be available in this environment)
    try:
        from rtdetrv2.core import YAMLConfig
    except ImportError:
        raise RuntimeError(
            "rtdetrv2 is not installed. Run this script from the RTDETRv2 environment."
        )

    print(f"Loading config: {args.config}")
    cfg = YAMLConfig(args.config, resume=args.model_checkpoint)

    print(f"Loading checkpoint: {args.model_checkpoint}")
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
        print("Using EMA model state")
    else:
        state = checkpoint["model"]
        print("Using standard model state")
    cfg.model.load_state_dict(state)

    # Create deploy model wrapper
    class DeployModel(nn.Module):
        def __init__(self, model, postprocessor):
            super().__init__()
            self.model = model
            self.postprocessor = postprocessor

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    print("Creating deployment model")
    model = DeployModel(cfg.model.deploy(), cfg.postprocessor.deploy())
    model.eval()

    # Move model to device for tracing
    device = torch.device(args.device)
    print(f"Moving model to device: {device}")
    model = model.to(device)

    # Trace the model on the specified device
    print(f"Tracing model with input size [{args.input_height}, {args.input_width}] on {device}")
    dummy_input = torch.rand(1, 3, args.input_height, args.input_width, device=device)
    dummy_sizes = torch.tensor([[args.input_width, args.input_height]], device=device)

    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy_input, dummy_sizes))

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {args.output}")
    traced.save(str(output_path))

    # Verify on same device
    print("Verifying exported model...")
    loaded = torch.jit.load(str(output_path), map_location=device)
    with torch.no_grad():
        labels, boxes, scores = loaded(dummy_input, dummy_sizes)
    print(f"Output shapes: labels={labels.shape}, boxes={boxes.shape}, scores={scores.shape}")
    print(f"Model exported for device: {device}")
    print("Done!")


if __name__ == "__main__":
    main()
