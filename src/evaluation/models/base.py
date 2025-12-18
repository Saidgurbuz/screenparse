from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from ..datasets import load_elements_file
from ..types import EvaluationSample, UIElement


class ModelRunner(ABC):
    """Base interface for model inference."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        ...

    def predict_batch(self, samples: Iterable[EvaluationSample]) -> List[Sequence[UIElement]]:
        return [self.predict(s) for s in samples]

    def close(self):
        """Optional cleanup hook for releasing resources."""
        return None


class CallableModelRunner(ModelRunner):
    """Wrap a simple callable into the ModelRunner interface."""

    def __init__(self, name: str, fn: Callable[[EvaluationSample], Sequence[UIElement]]):
        super().__init__(name)
        self.fn = fn

    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        return self.fn(sample)


class OfflinePredictionRunner(ModelRunner):
    """
    Reads precomputed predictions from disk.

    Expects files named <sample_id><suffix>, where sample_id comes from
    EvaluationSample.sample_id or the image stem.
    """

    def __init__(self, name: str, predictions_dir: str, suffix: str = ".pred.json"):
        super().__init__(name)
        self.pred_dir = Path(predictions_dir)
        self.suffix = suffix

    def _prediction_path(self, sample: EvaluationSample) -> Path:
        stem = sample.sample_id or Path(sample.image_path).stem
        return self.pred_dir / f"{stem}{self.suffix}"

    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        path = self._prediction_path(sample)
        if not path.exists():
            return []
        return load_elements_file(path, include_raw=False)
