"""Lightweight evaluation toolkit for screen parsing models."""

from .datasets import build_raw_dataset, build_yolo_dataset, build_groundcua_dataset
from .label_mapping import LabelMapper, get_class_list, UI_ELEMENTS_55, GROUNDCUA_CATEGORIES
from .metrics.base import Metric
from .metrics.page_iou import PageIoU
from .metrics.label_page_iou import LabelAwarePageIoU
from .metrics.map import MeanAveragePrecision
from .metrics.recall import Recall
from .models.base import OfflinePredictionRunner
from .models.gemini import GeminiRunner
from .models.paddleocrvl import PaddleOCRVLRunner
from .models.yolo import YoloModelRunner
from .models.qwen3_vl import Qwen3VLRunner
from .runner import Evaluator
from .types import BoundingBox, EvaluationSample, MetricResult, SampleResult, UIElement

__all__ = [
    "BoundingBox",
    "UIElement",
    "EvaluationSample",
    "MetricResult",
    "SampleResult",
    "Metric",
    "PageIoU",
    "LabelAwarePageIoU",
    "MeanAveragePrecision",
    "Recall",
    "Evaluator",
    "build_raw_dataset",
    "build_yolo_dataset",
    "build_groundcua_dataset",
    "OfflinePredictionRunner",
    "GeminiRunner",
    "PaddleOCRVLRunner",
    "YoloModelRunner",
    "Qwen3VLRunner",
    "LabelMapper",
    "get_class_list",
    "UI_ELEMENTS_55",
    "GROUNDCUA_CATEGORIES",
]
