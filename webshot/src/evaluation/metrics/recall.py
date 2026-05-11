from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from .base import Metric
from ..types import BoundingBox, EvaluationSample, MetricResult, UIElement


def _iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ax0, ay0, ax1, ay1 = box_a.to_ltrb()
    bx0, by0, bx1, by1 = box_b.to_ltrb()
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    union = box_a.area() + box_b.area() - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class Recall(Metric):
    """
    Recall at a single IoU threshold.
    - label_aware=True: match predictions to GT only within the same label.
    - label_aware=False: ignore labels (best IoU match counts).
    """

    def __init__(self, iou_threshold: float = 0.5, label_aware: bool = True, name: str | None = None):
        if name is None:
            suffix = "label" if label_aware else "agnostic"
            name = f"recall_{suffix}_{int(iou_threshold * 100)}"
        super().__init__(name)
        self.iou_threshold = iou_threshold
        self.label_aware = label_aware

    def compute(self, sample: EvaluationSample, predictions: Sequence[UIElement]) -> MetricResult:
        per_class_gt: Dict[str, List[Tuple[BoundingBox, bool]]] = defaultdict(list)
        if self.label_aware:
            for gt in sample.ground_truth:
                if not gt.label:
                    continue
                per_class_gt[gt.label].append((gt.bbox, False))
        else:
            for gt in sample.ground_truth:
                per_class_gt["_all"].append((gt.bbox, False))

        per_class_tp: Dict[str, int] = defaultdict(int)
        total_gt = sum(len(v) for v in per_class_gt.values())
        if total_gt == 0:
            return MetricResult(self.name, None, details={"gt": 0, "tp": 0, "iou_threshold": self.iou_threshold, "label_aware": self.label_aware})

        preds_by_class: Dict[str, List[UIElement]] = defaultdict(list)
        for pred in predictions:
            key = pred.label if self.label_aware else "_all"
            preds_by_class[key].append(pred)

        for label, preds in preds_by_class.items():
            gt_pool = per_class_gt.get(label, [])
            if not gt_pool:
                continue
            preds_sorted = sorted(preds, key=lambda p: p.score if p.score is not None else 1.0, reverse=True)
            for pred in preds_sorted:
                best_iou = 0.0
                best_idx = -1
                for idx, (gt_box, matched) in enumerate(gt_pool):
                    if matched:
                        continue
                    iou = _iou(pred.bbox, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_iou >= self.iou_threshold and best_idx >= 0:
                    gt_box, _ = gt_pool[best_idx]
                    gt_pool[best_idx] = (gt_box, True)
                    per_class_tp[label] += 1

        tp_total = sum(per_class_tp.values())
        recall = tp_total / total_gt if total_gt > 0 else None
        return MetricResult(
            name=self.name,
            value=recall,
            details={
                "tp": tp_total,
                "gt": total_gt,
                "per_class_tp": dict(per_class_tp),
                "per_class_gt": {k: len(v) for k, v in per_class_gt.items()},
                "iou_threshold": self.iou_threshold,
                "label_aware": self.label_aware,
            },
        )

    def aggregate(self, results: List[MetricResult]) -> MetricResult:
        total_tp = 0
        total_gt = 0
        for res in results:
            det = res.details or {}
            total_tp += int(det.get("tp", 0))
            total_gt += int(det.get("gt", 0))
        value = None if total_gt == 0 else total_tp / total_gt
        return MetricResult(
            name=self.name,
            value=value,
            details={"tp": total_tp, "gt": total_gt, "iou_threshold": self.iou_threshold, "label_aware": self.label_aware},
        )
