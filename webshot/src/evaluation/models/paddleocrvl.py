from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from ..types import BoundingBox, EvaluationSample, UIElement
from .base import ModelRunner


LABEL_MAP: Dict[str, str] = {
    "text": "Text",
    "image": "Image",
    "header_image": "Image",
    "footer_image": "Image",
    "paragraph_title": "Heading",
    "header": "Heading",
    "aside_text": "Text",
    "footer": "Text",
    "footnote": "Text",
}


def _norm_label(label: str | None) -> str | None:
    if not label:
        return None
    if label in LABEL_MAP:
        return LABEL_MAP[label]
    return label.title()


def _bbox_from_ltrb(coords) -> BoundingBox | None:
    if not coords or len(coords) != 4:
        return None
    try:
        l, t, r, b = coords
        w = float(r) - float(l)
        h = float(b) - float(t)
        return BoundingBox(float(l), float(t), w, h)
    except Exception:
        return None


def _discover_site_packages(python_exe: str) -> List[str]:
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import site, json; print(json.dumps(site.getsitepackages()))"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        paths = json.loads(proc.stdout)
        return [p for p in paths if isinstance(p, str)]
    except Exception:
        return []


class PaddleOCRVLRunner(ModelRunner):
    """
    Runs PaddleOCRVL by injecting the paddle venv site-packages into sys.path.
    If import/init fails, this runner will return empty predictions with an init_error.
    """

    def __init__(
        self,
        python_path: str = ".venv-paddle/bin/python",
        name: str = "paddleocrvl",
    ):
        super().__init__(name)
        self.python_path = python_path
        self.pipeline = None
        self.init_error: str | None = None
        self._init_pipeline()

    def _init_pipeline(self):
        # Try to import paddleocr in-process by extending sys.path with the paddle venv
        site_paths = _discover_site_packages(self.python_path)
        added = False
        for p in site_paths:
            if p not in sys.path:
                sys.path.append(p)
                added = True
        try:
            from paddleocr import PaddleOCRVL  # type: ignore
        except Exception as exc:
            self.pipeline = None
            self.init_error = f"import_failed: {exc}"
            return
        try:
            self.pipeline = PaddleOCRVL()
            self.init_error = None
        except Exception as exc:
            self.pipeline = None
            self.init_error = f"init_failed: {exc}"

    def _convert_results(self, raw: List[dict]) -> List[UIElement]:
        elements: List[UIElement] = []
        for item in raw:
            layout = item.get("layout_det_res", {}) if isinstance(item, dict) else {}
            boxes = layout.get("boxes") or []
            for box in boxes:
                bbox = _bbox_from_ltrb(box.get("coordinate"))
                if not bbox:
                    continue
                label = _norm_label(box.get("label"))
                score = float(box.get("score")) if box.get("score") is not None else None
                elements.append(UIElement(bbox=bbox, label=label, score=score))

            for blk in item.get("parsing_res_list") or []:
                bbox = _bbox_from_ltrb(blk.get("block_bbox"))
                if not bbox:
                    continue
                label = _norm_label(blk.get("block_label"))
                text = blk.get("block_content") or ""
                elements.append(UIElement(bbox=bbox, label=label, text=text))
        return elements

    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        return self.predict_batch([sample])[0]

    def predict_batch(self, samples: Iterable[EvaluationSample]) -> List[Sequence[UIElement]]:
        sample_list = list(samples)
        outputs: List[Sequence[UIElement]] = []
        if self.pipeline:
            for sample in sample_list:
                raw_results = []
                try:
                    raw_outputs = self.pipeline.predict(sample.image_path)
                    for res in raw_outputs:
                        print(f'res: {res}')
                        res_dict = getattr(res, "res", None)
                        if res_dict:
                            raw_results.append(res_dict)
                except Exception as exc:
                    raw_results = []
                    self.init_error = f"inference_failed: {exc}"
                outputs.append(self._convert_results(raw_results))
            return outputs

        # No pipeline available; return empty predictions
        for _ in sample_list:
            outputs.append([])
        return outputs
