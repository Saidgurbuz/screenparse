from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

from ..types import BoundingBox, EvaluationSample, MetricResult, UIElement
from .base import Metric


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


def _compute_ap(pairs: List[Tuple[float, bool]], total_gt: int) -> float:
    """
    pairs: list of (score, is_tp) for a single class, sorted descending by score.
    total_gt: number of GT boxes for this class across all samples.
    """
    if total_gt == 0 or not pairs:
        return 0.0 if total_gt > 0 else 1.0

    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    tps = []
    fps = []
    for _, is_tp in pairs:
        tps.append(1 if is_tp else 0)
        fps.append(0 if is_tp else 1)

    tp_cum = []
    fp_cum = []
    running_tp = 0
    running_fp = 0
    for t, f in zip(tps, fps):
        running_tp += t
        running_fp += f
        tp_cum.append(running_tp)
        fp_cum.append(running_fp)

    precisions = []
    recalls = []
    for tp, fp in zip(tp_cum, fp_cum):
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / total_gt if total_gt > 0 else 0.0)

    # VOC-style interpolation
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


class MeanAveragePrecision(Metric):
    """
    Label-aware mAP at a single IoU threshold (default: 0.5).
    """

    def __init__(self, iou_threshold: float = 0.5, name: str = "map_50"):
        super().__init__(name=name)
        self.iou_threshold = iou_threshold

    def compute(self, sample: EvaluationSample, predictions: Sequence[UIElement]) -> MetricResult:
        # Per-sample bookkeeping (used by aggregate to merge globally)
        per_class_pairs: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
        per_class_gt: Dict[str, int] = defaultdict(int)

        gt_by_class: Dict[str, List[Tuple[BoundingBox, bool]]] = defaultdict(list)
        for gt in sample.ground_truth:
            if not gt.label:
                continue
            gt_by_class[gt.label].append((gt.bbox, False))
            per_class_gt[gt.label] += 1

        # Sort predictions per class by score
        preds_by_class: Dict[str, List[UIElement]] = defaultdict(list)
        for pred in predictions:
            if not pred.label:
                continue
            preds_by_class[pred.label].append(pred)

        for label, preds in preds_by_class.items():
            preds_sorted = sorted(preds, key=lambda p: p.score if p.score is not None else 1.0, reverse=True)
            gt_pool = gt_by_class.get(label, [])

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
                is_tp = False
                if best_iou >= self.iou_threshold and best_idx >= 0:
                    gt_box, _ = gt_pool[best_idx]
                    gt_pool[best_idx] = (gt_box, True)
                    is_tp = True
                score = pred.score if pred.score is not None else 1.0
                per_class_pairs[label].append((float(score), is_tp))

        # Per-sample AP (not aggregated; provided for visibility)
        class_aps: Dict[str, float] = {}
        all_labels = set(per_class_gt.keys()) | set(per_class_pairs.keys())
        for label in all_labels:
            pairs = per_class_pairs.get(label, [])
            class_aps[label] = _compute_ap(pairs, per_class_gt.get(label, 0))
        sample_ap = (
            sum(class_aps.values()) / len(class_aps)
            if class_aps
            else (1.0 if not per_class_gt else 0.0)
        )

        return MetricResult(
            name=self.name,
            value=sample_ap,
            details={
                "per_class_pairs": {k: v for k, v in per_class_pairs.items()},
                "per_class_gt": dict(per_class_gt),
                "per_class_ap": class_aps,
                "iou_threshold": self.iou_threshold,
            },
        )

    def aggregate(self, results: List[MetricResult]) -> MetricResult:
        merged_pairs: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
        merged_gt: Dict[str, int] = defaultdict(int)
        iou_thr = self.iou_threshold

        for res in results:
            det = res.details or {}
            pairs = det.get("per_class_pairs", {})
            gts = det.get("per_class_gt", {})
            for label, items in pairs.items():
                merged_pairs[label].extend(items)
            for label, count in gts.items():
                merged_gt[label] += int(count)

        per_class_ap: Dict[str, float] = {}
        all_labels = set(merged_gt.keys()) | set(merged_pairs.keys())
        for label in all_labels:
            pairs = merged_pairs.get(label, [])
            per_class_ap[label] = _compute_ap(pairs, merged_gt.get(label, 0))

        valid_classes = [ap for lbl, ap in per_class_ap.items() if merged_gt.get(lbl, 0) > 0]
        map_value = (
            sum(valid_classes) / len(valid_classes)
            if valid_classes
            else (1.0 if sum(merged_gt.values()) == 0 else 0.0)
        )

        return MetricResult(
            name=self.name,
            value=map_value,
            details={
                "per_class_ap": per_class_ap,
                "per_class_gt": dict(merged_gt),
                "iou_threshold": iou_thr,
            },
        )
