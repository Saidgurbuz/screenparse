from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from tqdm import tqdm

from .datasets import serialize_elements
from .metrics.base import Metric
from .models.base import ModelRunner
from .types import EvaluationSample, MetricResult, SampleResult, UIElement
from .viz import VizConfig, save_viz


class Evaluator:
    """Runs metrics for a model over a dataset."""

    def __init__(self, metrics: Sequence[Metric], label_mapper=None):
        self.metrics = list(metrics)
        self.label_mapper = label_mapper

    def _write_predictions(self, out_dir: Path, sample: EvaluationSample, preds: Sequence[UIElement]):
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = sample.sample_id or Path(sample.image_path).stem
        path = out_dir / f"{stem}.pred.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(serialize_elements(preds), f, indent=2)

    def _write_viz(self, out_dir: Path, sample: EvaluationSample, preds: Sequence[UIElement]):
        save_viz(out_dir, sample, preds, VizConfig())

    def evaluate_model(
        self,
        samples: Iterable[EvaluationSample],
        model: ModelRunner,
        save_predictions_dir: str | None = None,
        batch_size: int | None = None,
        save_viz: bool = True,
    ) -> Dict[str, object]:
        sample_list = list(samples)
        sample_results: List[SampleResult] = []
        metric_collectors: Dict[str, List[MetricResult]] = {m.name: [] for m in self.metrics}
        pred_dir = Path(save_predictions_dir) if save_predictions_dir else None
        bs = batch_size or len(sample_list) or 1

        progress = tqdm(
            total=len(sample_list),
            desc=f"Evaluating {model.name}",
            unit="sample",
            dynamic_ncols=True,
        )

        for start in range(0, len(sample_list), bs):
            batch = sample_list[start : start + bs]
            try:
                batch_predictions = list(model.predict_batch(batch))
            except Exception as exc:
                batch_predictions = [[] for _ in batch]
                batch_errors = [str(exc)] * len(batch)
            else:
                batch_errors = [None] * len(batch)

            # Ensure alignment
            if len(batch_predictions) != len(batch):
                batch_errors = [batch_errors[0] or "prediction batch length mismatch"] * len(batch)
                batch_predictions = [[] for _ in batch]

            for sample, preds, err in zip(batch, batch_predictions, batch_errors):
                if self.label_mapper:
                    preds = [self.label_mapper.map_element(p) for p in preds]
                if err:
                    metric_results = {
                        m.name: MetricResult(m.name, None, details={"error": err})
                        for m in self.metrics
                    }
                else:
                    metric_results = {m.name: m.compute(sample, preds) for m in self.metrics}

                if pred_dir:
                    self._write_predictions(pred_dir, sample, preds)
                    if save_viz:
                        self._write_viz(pred_dir, sample, preds)

                for name, res in metric_results.items():
                    metric_collectors[name].append(res)

                sample_results.append(
                    SampleResult(
                        sample_id=sample.id(),
                        model_name=model.name,
                        metrics=metric_results,
                        predictions=list(preds),
                        metadata=sample.metadata,
                    )
                )
            progress.update(len(batch))

        progress.close()

        dataset_metrics = {
            name: metric.aggregate(results)
            for name, metric, results in (
                (m.name, m, metric_collectors[m.name]) for m in self.metrics
            )
        }

        try:
            model.close()
        except Exception:
            pass
        try:
            import torch, gc  # type: ignore

            torch.cuda.empty_cache()
            gc.collect()
        except Exception:
            pass

        return {
            "model": model.name,
            "num_samples": len(sample_results),
            "dataset_metrics": {k: v.to_dict() for k, v in dataset_metrics.items()},
            "samples": [s.to_dict() for s in sample_results],
        }
