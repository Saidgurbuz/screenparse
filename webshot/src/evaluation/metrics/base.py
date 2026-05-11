from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from ..types import EvaluationSample, MetricResult, UIElement


class Metric(ABC):
    """Base class for evaluation metrics."""

    def __init__(self, name: str | None = None):
        self.name = name or self.__class__.__name__

    @abstractmethod
    def compute(self, sample: EvaluationSample, predictions: Sequence[UIElement]) -> MetricResult:
        ...

    def aggregate(self, results: List[MetricResult]) -> MetricResult:
        values = [r.value for r in results if r.value is not None]
        if not values:
            return MetricResult(name=self.name, value=None, details={"count": 0})
        mean_value = float(sum(values) / len(values))
        return MetricResult(
            name=self.name,
            value=mean_value,
            details={"count": len(values)},
        )
