"""
RT-DETR model runner using TorchScript for inference.

This module provides a standalone RT-DETR inference implementation using
TorchScript, with full CUDA support and no external dependencies on rtdetrv2.

IMPORTANT: Before using this runner, export your trained RT-DETR model to TorchScript:

    cd /path/to/RTDETRv2
    .venv/bin/python -c "
import torch
import torch.nn as nn
from rtdetrv2.core import YAMLConfig

cfg = YAMLConfig('./src/rtdetrv2/configs/model/rtdetrv2_r50vd_6x_coco.yml',
                 resume='./outputs/best.pth')
checkpoint = torch.load('./outputs/best.pth', map_location='cpu')
state = checkpoint.get('ema', {}).get('module') or checkpoint['model']
cfg.model.load_state_dict(state)

class DeployModel(nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor
    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)

model = DeployModel(cfg.model.deploy(), cfg.postprocessor.deploy())
model.eval()
traced = torch.jit.trace(model, (torch.rand(1,3,736,1280), torch.tensor([[1280,736]])))
traced.save('./outputs/model.torchscript.pt')
"
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
import torchvision.transforms as T
from PIL import Image

from ..datasets import element_from_obj
from ..types import EvaluationSample, UIElement
from .base import ModelRunner


def _load_class_names(classes_file: str | None) -> List[str] | None:
    """Load class names from a txt/json/yaml file."""
    if not classes_file:
        return None

    path = Path(classes_file)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]

    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "categories" in data:
            categories = sorted(data["categories"], key=lambda x: x.get("id", 0))
            return [str(c.get("name", "")) for c in categories if "name" in c]
        if isinstance(data, dict) and "classes" in data:
            return [str(x) for x in data["classes"]]
        if isinstance(data, list):
            return [str(x) for x in data]

    if suffix in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML required for YAML classes file") from exc
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict) and "classes" in data:
            return [str(x) for x in data["classes"]]
        if isinstance(data, list):
            return [str(x) for x in data]

    return None


class RTDETRModelRunner(ModelRunner):
    """
    RT-DETR runner using TorchScript for inference.

    This is a standalone implementation using a pre-exported TorchScript model,
    with full CUDA support and no dependency on rtdetrv2.

    Args:
        model_path: Path to the TorchScript model (.pt file)
        name: Optional model name for reporting
        conf: Confidence threshold for detections (default 0.10)
        device: Inference device - "cuda" or "cpu" (default: auto-detect)
        classes_file: Path to classes.txt/.json/.yaml for label mapping
        input_size: Model input size as (height, width). Default (736, 1280)
    """

    def __init__(
        self,
        model_path: str,
        name: str | None = None,
        conf: float = 0.50,
        device: str | None = None,
        classes_file: str | None = None,
        input_size: Tuple[int, int] | None = None,
    ):
        model_name = name or Path(model_path).stem
        super().__init__(model_name)

        # Validate model file exists
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"TorchScript model not found: {model_path}")

        # Setup device
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load TorchScript model
        self.model = torch.jit.load(str(model_file), map_location=self.device)
        self.model.eval()

        # Default RT-DETR input size (height, width)
        self.input_size = input_size or (736, 1280)

        # Image transforms
        self.transforms = T.Compose([
            T.Resize(self.input_size),
            T.ToTensor(),
        ])

        # Load class names
        self.class_names = _load_class_names(classes_file)
        self.score_threshold = float(conf)

    def _format_detections(
        self,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> List[dict]:
        """Format raw detections into detection dicts."""
        w, h = image_size
        if labels.numel() == 0:
            return []

        # Filter by confidence
        keep = scores >= self.score_threshold
        labels = labels[keep].cpu()
        boxes = boxes[keep].cpu()
        scores = scores[keep].cpu()

        # Clamp boxes to image bounds
        if boxes.numel() > 0:
            boxes[:, 0::2].clamp_(0, max(w - 1, 0))  # x coords
            boxes[:, 1::2].clamp_(0, max(h - 1, 0))  # y coords

        results: List[dict] = []
        for label, box, score in zip(labels, boxes, scores):
            idx = int(label.item())
            label_str = (
                self.class_names[idx]
                if self.class_names and 0 <= idx < len(self.class_names)
                else str(idx)
            )
            results.append({
                "label": label_str,
                "score": float(score.item()),
                "bbox_ltrb": [float(x) for x in box.tolist()],
            })
        return results

    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        """Predict UI elements for a single sample."""
        return self.predict_batch([sample])[0]

    def predict_batch(self, samples: Iterable[EvaluationSample]) -> List[Sequence[UIElement]]:
        """Predict UI elements for a batch of samples."""
        samples_list = list(samples)

        images: List[torch.Tensor] = []
        sizes: List[Tuple[int, int]] = []
        valid_idx: List[int] = []

        for idx, sample in enumerate(samples_list):
            try:
                img = Image.open(sample.image_path).convert("RGB")
                w, h = img.size
                sizes.append((w, h))
                images.append(self.transforms(img))
                valid_idx.append(idx)
            except Exception:
                sizes.append((0, 0))
                continue

        if not images:
            return [[] for _ in samples_list]

        # Stack into batch and move to device
        batch = torch.stack(images, dim=0).to(self.device)
        # orig_target_sizes expects [W, H] format
        orig_sizes = torch.tensor(
            [sizes[i] for i in valid_idx],
            dtype=torch.int64,
            device=self.device,
        )

        # Run inference
        with torch.no_grad():
            labels, boxes, scores = self.model(batch, orig_sizes)

        # Process results
        out: List[Sequence[UIElement]] = [[] for _ in samples_list]
        for out_idx, sample_idx in enumerate(valid_idx):
            dets = self._format_detections(
                labels[out_idx],
                boxes[out_idx],
                scores[out_idx],
                sizes[sample_idx],
            )
            elements: List[UIElement] = []
            for obj in dets:
                el = element_from_obj(obj, include_raw=False)
                if el:
                    elements.append(el)
            out[sample_idx] = elements
        return out

    def close(self):
        """Release resources."""
        try:
            import gc
            del self.model
            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass
