"""Lightweight evaluation toolkit for screen parsing models."""

from .datasets import build_raw_dataset, build_yolo_dataset, build_groundcua_dataset, build_screenspot_dataset
from .label_mapping import (
    LabelMapper,
    get_class_list,
    UI_ELEMENTS_55,
    GROUNDCUA_CATEGORIES,
    SCREENSPOT_CATEGORIES,
    custom_to_screenspot,
)
from .metrics.base import Metric
from .metrics.page_iou import PageIoU, PageIoURecall
from .metrics.label_page_iou import LabelAwarePageIoU
from .metrics.map import MeanAveragePrecision
from .metrics.recall import Recall
from .metrics.ned import NormalizedEditDistance
from .models.base import OfflinePredictionRunner

try:
    from .models.gemini import GeminiRunner
except ImportError:
    GeminiRunner = None

try:
    from .models.paddleocrvl import PaddleOCRVLRunner
except ImportError:
    PaddleOCRVLRunner = None

try:
    from .models.screenvlm import ScreenVLMRunner
except ImportError:
    ScreenVLMRunner = None

try:
    from .models.yolo import YoloModelRunner
except ImportError:
    YoloModelRunner = None

try:
    from .models.qwen3_vl import Qwen3VLRunner
except ImportError:
    Qwen3VLRunner = None

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
    "PageIoURecall",
    "LabelAwarePageIoU",
    "MeanAveragePrecision",
    "Recall",
    "NormalizedEditDistance",
    "Evaluator",
    "build_raw_dataset",
    "build_yolo_dataset",
    "build_groundcua_dataset",
    "build_screenspot_dataset",
    "OfflinePredictionRunner",
    "GeminiRunner",
    "PaddleOCRVLRunner",
    "ScreenVLMRunner",
    "YoloModelRunner",
    "Qwen3VLRunner",
    "LabelMapper",
    "get_class_list",
    "UI_ELEMENTS_55",
    "GROUNDCUA_CATEGORIES",
    "SCREENSPOT_CATEGORIES",
    "custom_to_screenspot",
]
