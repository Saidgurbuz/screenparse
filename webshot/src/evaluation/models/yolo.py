from __future__ import annotations

from pathlib import Path
from typing import Sequence

from webshot.inference_pipeline import InferencePipeline

from ..datasets import element_from_obj
from ..types import EvaluationSample, UIElement
from .base import ModelRunner


class YoloModelRunner(ModelRunner):
    """Thin wrapper around the existing inference pipeline for evaluation."""

    def __init__(
        self,
        weights_path: str,
        name: str | None = None,
        conf: float = 0.10,
        iou: float = 0.10,
        imgsz: int = 1280,
        device: str | None = None,
    ):
        model_name = name or Path(weights_path).stem
        super().__init__(model_name)
        # Skip OCR/hierarchy; we only need boxes + labels here.
        self.pipeline = InferencePipeline(
            weights_path=weights_path,
            device=device,
            conf=conf,
            iou=iou,
            ocr_engine=None,
            imgsz=imgsz,
        )

    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        raw_elements = self.pipeline.predict_image(sample.image_path, plot_save_path=None)
        elements: list[UIElement] = []
        for obj in raw_elements:
            el = element_from_obj(obj, include_raw=False)
            if el:
                elements.append(el)
        return elements

    def close(self):
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
