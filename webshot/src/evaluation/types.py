from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BoundingBox:
    """Axis-aligned bounding box using (x, y, w, h) with pixel units."""

    x: float
    y: float
    w: float
    h: float

    def to_ltrb(self) -> Tuple[float, float, float, float]:
        """Return left, top, right, bottom."""
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def clamp(self, width: int, height: int) -> "BoundingBox":
        """Clamp box to image bounds."""
        left = max(0.0, min(self.x, float(width)))
        top = max(0.0, min(self.y, float(height)))
        right = max(0.0, min(self.x + self.w, float(width)))
        bottom = max(0.0, min(self.y + self.h, float(height)))
        return BoundingBox(
            x=left,
            y=top,
            w=max(0.0, right - left),
            h=max(0.0, bottom - top),
        )

    def scaled(self, scale: float) -> "BoundingBox":
        return BoundingBox(self.x * scale, self.y * scale, self.w * scale, self.h * scale)

    def rounded(self) -> "BoundingBox":
        return BoundingBox(
            x=int(round(self.x)),
            y=int(round(self.y)),
            w=int(round(self.w)),
            h=int(round(self.h)),
        )

    def area(self) -> float:
        if self.w <= 0 or self.h <= 0:
            return 0.0
        return self.w * self.h

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class UIElement:
    bbox: BoundingBox
    label: Optional[str] = None
    text: Optional[str] = None
    score: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "bbox": self.bbox.to_dict(),
            "label": self.label,
            "text": self.text,
            "score": self.score,
        }
        if self.raw is not None:
            data["raw"] = self.raw
        return data


@dataclass
class EvaluationSample:
    image_path: str
    ground_truth: List[UIElement]
    image_size: Optional[Tuple[int, int]] = None  # (width, height)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sample_id: Optional[str] = None

    def id(self) -> str:
        if self.sample_id:
            return self.sample_id
        return self.metadata.get("stem") or self.image_path


@dataclass
class MetricResult:
    name: str
    value: Optional[float]
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class SampleResult:
    sample_id: str
    model_name: str
    metrics: Dict[str, MetricResult]
    predictions: List[UIElement] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "model": self.model_name,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "predictions": [p.to_dict() for p in self.predictions],
            "metadata": self.metadata,
        }
