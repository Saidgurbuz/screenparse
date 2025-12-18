from __future__ import annotations

from typing import Sequence

import numpy as np
from PIL import Image
from .base import Metric
from ..types import BoundingBox, EvaluationSample, MetricResult, UIElement


class PageIoU(Metric):
    """
    Page-level IoU from MinerU 2.5.

    Builds coverage maps for prediction and ground truth and computes
    sum(min) / sum(max) over pixels. Optional downscaling keeps memory small.
    """

    def __init__(self, max_resolution: int = 0, name: str = "page_iou"):
        """
        Args:
            max_resolution: If >0, the longer page side is scaled to this many
                pixels before computing coverage. Set 0 to use native size.
        """
        super().__init__(name=name)
        self.max_resolution = max_resolution

    def _maybe_read_image_size(self, sample: EvaluationSample):
        if sample.image_size:
            return sample.image_size
        try:
            with Image.open(sample.image_path) as im:
                return im.size
        except Exception:
            return None

    def _render_mask(
        self,
        boxes: Sequence[BoundingBox],
        width: int,
        height: int,
    ) -> np.ndarray:
        if width <= 0 or height <= 0:
            return np.zeros((1, 1), dtype=bool)

        scale = 1.0
        target_w, target_h = width, height
        longest = max(width, height)
        if self.max_resolution and longest > self.max_resolution:
            scale = float(self.max_resolution) / float(longest)
            target_w = max(1, int(round(width * scale)))
            target_h = max(1, int(round(height * scale)))

        mask = np.zeros((target_h, target_w), dtype=bool)

        for box in boxes:
            clamped = box.clamp(width, height).scaled(scale).rounded()
            if clamped.w <= 0 or clamped.h <= 0:
                continue

            x1 = max(0, min(int(clamped.x), target_w))
            y1 = max(0, min(int(clamped.y), target_h))
            x2 = max(0, min(int(clamped.x + clamped.w), target_w))
            y2 = max(0, min(int(clamped.y + clamped.h), target_h))

            if x1 >= x2 or y1 >= y2:
                continue
            mask[y1:y2, x1:x2] = True

        return mask

    def compute(self, sample: EvaluationSample, predictions: Sequence[UIElement]) -> MetricResult:
        size = self._maybe_read_image_size(sample)
        if size is None:
            return MetricResult(self.name, None, details={"reason": "missing_image"})

        width, height = size
        gt_boxes = [el.bbox for el in sample.ground_truth]
        pred_boxes = [el.bbox for el in predictions]

        if not gt_boxes and not pred_boxes:
            return MetricResult(self.name, 1.0, details={"gt": 0, "pred": 0})

        gt_mask = self._render_mask(gt_boxes, width, height)
        pred_mask = self._render_mask(pred_boxes, width, height)

        inter = np.logical_and(gt_mask, pred_mask).sum(dtype=np.int64)
        union = np.logical_or(gt_mask, pred_mask).sum(dtype=np.int64)

        if union == 0:
            value = 1.0 if not gt_boxes and not pred_boxes else 0.0
        else:
            value = float(inter) / float(union)

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "gt": len(gt_boxes),
                "pred": len(pred_boxes),
                "scale": self.max_resolution,
                "inter_pixels": int(inter),
                "union_pixels": int(union),
            },
        )
