from __future__ import annotations

import multiprocessing as mp
import os
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from webshot.utils import parse_screentag
from ..types import BoundingBox, EvaluationSample, UIElement
from .base import ModelRunner


NORM_SIZE = 500
DEFAULT_PROMPT = "Generate the screen representation for this UI:"


def _clamp(val: float, low: float, high: float) -> float:
    return max(low, min(val, high))


def _normalize_coords(coords: Tuple[int, int, int, int], norm_size: int) -> Tuple[float, float, float, float]:
    l, t, r, b = coords
    l = _clamp(float(l), 0.0, float(norm_size))
    t = _clamp(float(t), 0.0, float(norm_size))
    r = _clamp(float(r), 0.0, float(norm_size))
    b = _clamp(float(b), 0.0, float(norm_size))
    if r < l:
        l, r = r, l
    if b < t:
        t, b = b, t
    return l, t, r, b


def _to_abs_bbox(coords: Tuple[float, float, float, float], width: int, height: int) -> BoundingBox:
    l, t, r, b = coords
    x1 = (l / NORM_SIZE) * width
    y1 = (t / NORM_SIZE) * height
    x2 = (r / NORM_SIZE) * width
    y2 = (b / NORM_SIZE) * height
    x1 = _clamp(x1, 0.0, float(width))
    x2 = _clamp(x2, 0.0, float(width))
    y1 = _clamp(y1, 0.0, float(height))
    y2 = _clamp(y2, 0.0, float(height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return BoundingBox(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def _flatten_nodes(nodes) -> List[Tuple[str, Tuple[int, int, int, int], str]]:
    flat = []
    stack = list(nodes)
    while stack:
        node = stack.pop()
        flat.append((node.tag, node.bbox, node.text))
        if node.children:
            stack.extend(node.children)
    return flat


def _parse_screentag_elements(text: str, width: int, height: int) -> List[UIElement]:
    elements: List[UIElement] = []
    try:
        nodes = parse_screentag(text)
    except Exception:
        nodes = []

    if nodes:
        for tag, bbox, node_text in _flatten_nodes(nodes):
            l, t, r, b = _normalize_coords(bbox, NORM_SIZE)
            abs_box = _to_abs_bbox((l, t, r, b), width, height)
            elements.append(UIElement(bbox=abs_box, label=tag, text=node_text))
        return elements

    # Fallback regex: <tag><loc_x><loc_y><loc_x><loc_y>optional_text
    pattern = re.compile(
        r"<(?P<tag>[a-zA-Z0-9_]+)>(?P<locs>(?:\\s*<loc_\\d+>){4})(?P<text>[^<]*)"
    )
    for match in pattern.finditer(text):
        tag_name = match.group("tag")
        coords = re.findall(r"<loc_(\\d+)>", match.group("locs"))
        if len(coords) != 4:
            continue
        l, t, r, b = _normalize_coords(tuple(int(c) for c in coords), NORM_SIZE)
        abs_box = _to_abs_bbox((l, t, r, b), width, height)
        node_text = (match.group("text") or "").strip()
        elements.append(UIElement(bbox=abs_box, label=tag_name, text=node_text or None))
    return elements


class ScreenVLMRunner(ModelRunner):
    """
    ScreenVLM vLLM runner that emits ScreenTag outputs.
    """

    def __init__(
        self,
        checkpoint: str,
        name: str | None = None,
        prompt: str = DEFAULT_PROMPT,
        max_new_tokens: int = 6192,
        temperature: float = 0.0,
        top_p: float = 0.9,
        top_k: int = 50,
    ):
        runner_name = name or Path(checkpoint).name
        super().__init__(runner_name)

        try:
            current = mp.get_start_method(allow_none=True)
            if current != "spawn":
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self.prompt = prompt
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            skip_special_tokens=False,
        )
        self.llm = LLM(
            model=checkpoint,
            revision="untied",
            limit_mm_per_prompt={"image": 1},
            trust_remote_code=True,
        )

    def _build_prompt(self) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        return self.predict_batch([sample])[0]

    def predict_batch(self, samples: Iterable[EvaluationSample]) -> List[Sequence[UIElement]]:
        vllm_inputs = []
        sizes: List[Tuple[int, int]] = []
        valid_idx = []
        for idx, sample in enumerate(samples):
            try:
                with Image.open(sample.image_path) as im:
                    img = im.convert("RGB")
                    sizes.append(img.size)
                    prompt = self._build_prompt()
                    vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
                    valid_idx.append(idx)
            except Exception:
                sizes.append((0, 0))

        outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params) if vllm_inputs else []
        parsed: List[Sequence[UIElement]] = [[] for _ in range(len(sizes))]
        for out, idx in zip(outputs, valid_idx):
            text = out.outputs[0].text if out.outputs else ""
            width, height = sizes[idx]
            parsed[idx] = _parse_screentag_elements(text, width, height)
        return parsed

    def close(self):
        try:
            if hasattr(self, "llm") and self.llm is not None:
                engine = getattr(self.llm, "llm_engine", None)
                if engine and hasattr(engine, "shutdown"):
                    engine.shutdown()
                if hasattr(self.llm, "engine_context"):
                    ctx = getattr(self.llm, "engine_context")
                    shutdown_fn = getattr(ctx, "shutdown", None)
                    if callable(shutdown_fn):
                        shutdown_fn()
                self.llm = None
        finally:
            try:
                import gc

                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
