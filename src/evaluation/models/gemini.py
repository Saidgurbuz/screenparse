from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

from ..datasets import element_from_obj
from ..types import EvaluationSample, UIElement
from .base import ModelRunner

os.environ["GEMINI_API_KEY"] = "AIzaSyDd6uD_bq5Pz3X8jMk5hL1UNafO1nnXp3o"

_CANON_CLASSES = [
    "Table",
    "Column/Browser",
    "Button",
    "Utility Button",
    "App Icon",
    "Navigation Bar",
    "Status Bar",
    "Search Field",
    "Toolbar",
    "Tooltip",
    "Video",
    "Tab Bar",
    "Side Bar",
    "Slider",
    "Picker",
    "ContextMenu",
    "DockMenu",
    "EditMenu",
    "Image",
    "Scroll",
    "Switch",
    "File Icon",
    "Chart",
    "Window",
    "Screen",
    "List",
    "List Item",
    "PopUp Menu",
    "Steppers",
    "Toggles",
    "Text Input",
    "Rating Indicator",
    "Checkbox",
    "Radiobox",
    "Select",
    "Avatar",
    "Badge",
    "Alert",
    "Progress bar",
    "Bottom navigation",
    "Breadcrumb",
    "Page control",
    "Link",
    "Menu",
    "Pagination",
    "Tab",
    "Search Bar",
    "Date-Time picker",
    "Calendar",
    "Text",
    "Heading",
    "Code snippet",
    "Carousel",
    "Notification",
    "Logo",
]


def _dedupe(seq: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


CANON_CLASSES = _dedupe(_CANON_CLASSES)


def _classes_block() -> str:
    return "Allowed labels:\n- " + "\n- ".join(CANON_CLASSES)


def _default_prompt() -> str:
    return (
        "You are a UI parser. Given a screenshot image, extract all visible UI elements.\n"
        "Return JSON list with objects: "
        '{"bbox_tlbr":[t,l,b,r], "label": "<type>", "text": "<visible text>"}.\n'
        "The bbox_tlbr should be normalized to 0-1000. (t,l,b,r). Include all elements.\n"
        f"{_classes_block()}"
    )


def _default_parser(output_text: str) -> List[UIElement]:
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
            el = element_from_obj(obj, include_raw=False)
            if el:
                elements.append(el)
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

        self.prompt = prompt or _default_prompt()
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
            outputs.append(self.parser(text))
        return outputs
