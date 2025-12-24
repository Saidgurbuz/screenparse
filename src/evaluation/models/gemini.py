from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from PIL import Image
from ..label_mapping import get_class_list
from ..types import BoundingBox, EvaluationSample, UIElement
from .base import ModelRunner

os.environ["GEMINI_API_KEY"] = "AIzaSyDd6uD_bq5Pz3X8jMk5hL1UNafO1nnXp3o"

def _classes_block(classes: List[str]) -> str:
    return "Allowed labels:\n- " + "\n- ".join(classes)


def _default_prompt(class_schema: str) -> str:
    classes = get_class_list(class_schema)
    return (
        "You are a UI parser. Given a screenshot image, extract all visible UI elements.\n"
        "Return JSON list with objects: "
        '{"bbox_tlbr":[t,l,b,r], "label": "<type>", "text": "<visible text>"}.\n'
        "The bbox_tlbr should be normalized to 0-1000. (t,l,b,r). Include all elements.\n"
        f"{_classes_block(classes)}"
    )


def _default_parser(output_text: str, width: int, height: int) -> List[UIElement]:
    return _parse_output(output_text, width, height)


def _scale_val(val: float, size: int, norm: int = 1000) -> float:
    if size <= 0:
        return 0.0
    v = max(0.0, min(float(val), float(norm)))
    return (v / float(norm)) * float(size)


def _parse_output(output_text: str, width: int, height: int) -> List[UIElement]:
    try:
        start = output_text.find("[")
        end = output_text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        payload = json.loads(output_text[start : end + 1])
        if not isinstance(payload, list):
            return []
        elements: List[UIElement] = []
        for obj in payload:
            label = obj.get("label") or obj.get("type") or obj.get("tag")
            text = obj.get("text") or obj.get("inner_text") or obj.get("own_text")
            score = obj.get("score") or obj.get("confidence")
            bbox = None
            if "bbox_tlbr" in obj:
                t, l, b, r = obj["bbox_tlbr"]
                bbox = (l, t, r, b)
            elif "bbox_ltrb" in obj:
                l, t, r, b = obj["bbox_ltrb"]
                bbox = (l, t, r, b)
            elif "bbox" in obj:
                l, t, r, b = obj["bbox"]
                bbox = (l, t, r, b)
            elif "bbox_xywh" in obj:
                x, y, w, h = obj["bbox_xywh"]
                bbox = (x, y, x + w, y + h)
            if bbox is None:
                continue
            l, t, r, b = bbox
            x1 = _scale_val(l, width)
            y1 = _scale_val(t, height)
            x2 = _scale_val(r, width)
            y2 = _scale_val(b, height)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            elements.append(
                UIElement(
                    bbox=BoundingBox(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)),
                    label=label,
                    text=text,
                    score=float(score) if score is not None else None,
                )
            )
        return elements
    except Exception:
        return []


class GeminiRunner(ModelRunner):
    """Calls Gemini via the google-genai SDK."""

    def __init__(
        self,
        model_id: str = "gemini-2.5-flash-lite",
        name: str | None = None,
        api_key: str | None = None,
        prompt: str | None = None,
        parser: Callable[[str], List[UIElement]] | None = None,
        class_schema: str = "custom55",
    ):
        runner_name = name or Path(model_id).name
        super().__init__(runner_name)
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as exc:
            raise ImportError("google-genai is required for GeminiRunner") from exc

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GeminiRunner requires an API key (env GEMINI_API_KEY/GOOGLE_API_KEY or api_key param).")

        self.prompt = prompt or _default_prompt(class_schema)
        self.parser = parser or _default_parser
        self.client = genai.Client(api_key=key)
        self.types = types
        self.model_id = model_id

    def _guess_mime(self, path: str) -> str:
        mime, _ = mimetypes.guess_type(path)
        return mime or "image/png"

    def _build_content(self, sample: EvaluationSample):
        with open(sample.image_path, "rb") as f:
            image_bytes = f.read()
        mime = self._guess_mime(sample.image_path)
        return [
            self.types.Part.from_bytes(data=image_bytes, mime_type=mime),
            self.prompt,
        ]

    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        return self.predict_batch([sample])[0]

    def predict_batch(self, samples: Iterable[EvaluationSample]) -> List[Sequence[UIElement]]:
        outputs: List[Sequence[UIElement]] = []
        for sample in samples:
            contents = self._build_content(sample)
            response = self.client.models.generate_content(model=self.model_id, contents=contents)
            text = getattr(response, "text", "") or ""
            try:
                with Image.open(sample.image_path) as im:
                    w, h = im.size
            except Exception:
                w, h = 0, 0
            try:
                outputs.append(self.parser(text, w, h))
            except TypeError:
                outputs.append(self.parser(text))
        return outputs
