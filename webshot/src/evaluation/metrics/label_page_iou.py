from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from PIL import Image

from ..types import BoundingBox, EvaluationSample, MetricResult, UIElement
from .base import Metric


class LabelAwarePageIoU(Metric):
    """
    PageIoU variant that requires label agreement per pixel.

    Each pixel is assigned the label of the smallest box covering it
    (both for GT and predictions). IoU is then computed on these
    per-pixel labels.
    """

    def __init__(self, max_resolution: int = 0, name: str = "label_page_iou"):
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

    @staticmethod
    def _label_id(label: str, mapping: Dict[str, int]) -> int:
        if label not in mapping:
            mapping[label] = len(mapping)
        return mapping[label]

    def _render_label_map(
        self,
        elements: Sequence[UIElement],
        width: int,
        height: int,
    ) -> np.ndarray:
        if width <= 0 or height <= 0:
            return np.full((1, 1), fill_value=-1, dtype=np.int32)

        scale = 1.0
        target_w, target_h = width, height
        longest = max(width, height)
        if self.max_resolution and longest > self.max_resolution:
            scale = float(self.max_resolution) / float(longest)
            target_w = max(1, int(round(width * scale)))
            target_h = max(1, int(round(height * scale)))

        label_map = np.full((target_h, target_w), fill_value=-1, dtype=np.int32)
        area_map = np.full((target_h, target_w), fill_value=np.inf, dtype=np.float64)
        label_ids: Dict[str, int] = {}

        for el in elements:
            if not el.label:
                continue
            label_id = self._label_id(el.label, label_ids)
            box = el.bbox.clamp(width, height).scaled(scale).rounded()
            if box.w <= 0 or box.h <= 0:
                continue

            x1 = max(0, min(int(box.x), target_w))
            y1 = max(0, min(int(box.y), target_h))
            x2 = max(0, min(int(box.x + box.w), target_w))
            y2 = max(0, min(int(box.y + box.h), target_h))
            if x1 >= x2 or y1 >= y2:
                continue

            area = float(box.w * box.h)
            region = area_map[y1:y2, x1:x2]
            update_mask = area < region
            if not np.any(update_mask):
                continue

            area_map[y1:y2, x1:x2] = np.where(update_mask, area, region)
            region_labels = label_map[y1:y2, x1:x2]
            region_labels[update_mask] = label_id
            label_map[y1:y2, x1:x2] = region_labels

        return label_map

    def compute(self, sample: EvaluationSample, predictions: Sequence[UIElement]) -> MetricResult:
        size = self._maybe_read_image_size(sample)
        if size is None:
            return MetricResult(self.name, None, details={"reason": "missing_image"})

        width, height = size
        gt_map = self._render_label_map(sample.ground_truth, width, height)
        pred_map = self._render_label_map(predictions, width, height)

        # Intersection where labels agree and are not background
        match = (gt_map == pred_map) & (gt_map != -1) & (pred_map != -1)
        inter = match.sum(dtype=np.int64)

        union_mask = (gt_map != -1) | (pred_map != -1)
        union = union_mask.sum(dtype=np.int64)

        if union == 0:
            value = 1.0 if not sample.ground_truth and not predictions else 0.0
        else:
            value = float(inter) / float(union)

        return MetricResult(
            name=self.name,
            value=value,
            details={
                "gt": len(sample.ground_truth),
                "pred": len(predictions),
                "scale": self.max_resolution,
                "inter_pixels": int(inter),
                "union_pixels": int(union),
            },
        )
