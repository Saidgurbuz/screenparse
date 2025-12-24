from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from .base import Metric
from ..types import BoundingBox, EvaluationSample, MetricResult, UIElement
from Levenshtein import distance


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalized_edit_distance(a: str, b: str) -> float:
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    return distance(a, b) / max_len


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


def _hungarian(cost: List[List[float]]) -> List[int]:
    n = len(cost)
    m = len(cost[0]) if n else 0
    size = max(n, m)
    pad = 1.0
    matrix = [[pad for _ in range(size)] for _ in range(size)]
    for i in range(n):
        for j in range(m):
            matrix[i][j] = cost[i][j]

    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)

    for i in range(1, size + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, size + 1):
                if not used[j]:
                    cur = matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(size + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    assignment = [-1] * size
    for j in range(1, size + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment[:n]


class NormalizedEditDistance(Metric):
    """
    Normalized edit distance with Hungarian matching over text-bearing elements.
    Lower is better (0 = perfect match).
    """

    def __init__(self, iou_threshold: float = 0.0, name: str = "ned"):
        super().__init__(name)
        self.iou_threshold = iou_threshold

    def compute(self, sample: EvaluationSample, predictions: Sequence[UIElement]) -> MetricResult:
        gt_items: List[Tuple[BoundingBox, str]] = []
        for gt in sample.ground_truth:
            if gt.text and gt.text.strip():
                gt_items.append((gt.bbox, _normalize_text(gt.text)))

        pred_items: List[Tuple[BoundingBox, str]] = []
        for pred in predictions:
            if pred.text and pred.text.strip():
                pred_items.append((pred.bbox, _normalize_text(pred.text)))

        if not gt_items or not pred_items:
            return MetricResult(
                name=self.name,
                value=None,
                details={
                    "gt_text": len(gt_items),
                    "pred_text": len(pred_items),
                    "iou_threshold": self.iou_threshold,
                },
            )

        n = len(gt_items)
        m = len(pred_items)
        cost: List[List[float]] = []
        for gt_box, gt_text in gt_items:
            row = []
            for pred_box, pred_text in pred_items:
                if self.iou_threshold > 0.0 and _iou(gt_box, pred_box) < self.iou_threshold:
                    row.append(1.0)
                else:
                    row.append(_normalized_edit_distance(gt_text, pred_text))
            cost.append(row)

        assignment = _hungarian(cost)
        total_cost = 0.0
        for i in range(n):
            j = assignment[i]
            if j >= 0 and j < m:
                total_cost += cost[i][j]
            else:
                total_cost += 1.0

        mean_cost = total_cost / n if n > 0 else None
        return MetricResult(
            name=self.name,
            value=mean_cost,
            details={
                "gt_text": n,
                "pred_text": m,
                "sum_cost": total_cost,
                "iou_threshold": self.iou_threshold,
            },
        )

    def aggregate(self, results: List[MetricResult]) -> MetricResult:
        total_cost = 0.0
        total_gt = 0
        for res in results:
            det = res.details or {}
            if det.get("sum_cost") is None:
                continue
            total_cost += float(det.get("sum_cost", 0.0))
            total_gt += int(det.get("gt_text", 0))
        value = None if total_gt == 0 else total_cost / total_gt
        return MetricResult(
            name=self.name,
            value=value,
            details={"sum_cost": total_cost, "gt_text": total_gt, "iou_threshold": self.iou_threshold},
        )
