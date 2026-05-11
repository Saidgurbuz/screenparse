from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import torch
from PIL import Image
from vllm import LLM, SamplingParams

from ..label_mapping import get_class_list
from ..types import BoundingBox, EvaluationSample, UIElement
from .base import ModelRunner


def _classes_block(schema_classes: List[str]) -> str:
    return "Allowed labels:\n- " + "\n- ".join(schema_classes)


def _default_prompt(class_schema: str) -> str:
    classes = get_class_list(class_schema)
    return (
        "You are a UI parser. Given a screenshot image, extract all visible UI elements.\n"
        "Return JSON list with objects: "
        '{"bbox_ltrb":[l,t,r,b], "label": "<type>", "text": "<visible text>"}.\n'
        "The bbox_ltrb should be normalized to 0-1000. (l,t,r,b). Include all elements.\n"
        f"{_classes_block(classes)}"
    )


def _default_parser(output_text: str, width: int, height: int) -> List[UIElement]:
    return _parse_output(output_text, width, height)


def _coerce_label(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                item = item.strip()
                if item:
                    return item
            else:
                text = str(item).strip()
                if text:
                    return text
        return None
    return str(value)


def _coerce_text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                item = item.strip()
                if item:
                    parts.append(item)
            else:
                parts.append(str(item))
        return " ".join(parts) if parts else None
    return str(value)


def _coerce_score(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)) and value:
        return _coerce_score(value[0])
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
            if "bbox_ltrb" in obj:
                l, t, r, b = obj["bbox_ltrb"]
                bbox = (l, t, r, b)
            elif "bbox_tlbr" in obj:
                t, l, b, r = obj["bbox_tlbr"]
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
                    label=_coerce_label(label),
                    text=_coerce_text(text),
                    score=_coerce_score(score),
                )
            )
        return elements
    except Exception:
        return []


class InternVL3Runner(ModelRunner):
    """
    OpenGVLab/InternVL3-2B via vLLM (vision-language).
    """

    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL3-2B",
        name: str | None = None,
        prompt: str | None = None,
        parser: Callable[[str], List[UIElement]] | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.9,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int | None = None,
        max_model_len: int = 4096,
        class_schema: str = "custom55",
    ):
        runner_name = name or Path(model_id).name
        super().__init__(runner_name)

        try:
            current = mp.get_start_method(allow_none=True)
            if current != "spawn":
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        self.prompt = prompt or _default_prompt(class_schema)
        self.parser = parser or _default_parser
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )

        tp = tensor_parallel_size if tensor_parallel_size is not None else torch.cuda.device_count()
        tp = max(1, tp)
        self.llm = LLM(
            model=model_id,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tp,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 1},
        )

    def _format_prompt(self, text: str) -> str:
        if "<image>" in text and "Assistant:" in text:
            return text
        if "<image>" in text:
            return f"User: {text}<|end|>\nAssistant:"
        return f"User: <image>\n{text}<|end|>\nAssistant:"

    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        return self.predict_batch([sample])[0]

    def predict_batch(self, samples: Iterable[EvaluationSample]) -> List[Sequence[UIElement]]:
        vllm_inputs = []
        sizes = []
        valid_idx = []
        for idx, sample in enumerate(samples):
            try:
                with Image.open(sample.image_path) as im:
                    img = im.convert("RGB")
                    sizes.append(img.size)
                    prompt = self._format_prompt(self.prompt)
                    vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
                    valid_idx.append(idx)
            except Exception:
                sizes.append((0, 0))

        outputs = self.llm.generate(vllm_inputs, sampling_params=self.sampling_params) if vllm_inputs else []
        parsed: List[Sequence[UIElement]] = [[] for _ in range(len(sizes))]
        for out, idx in zip(outputs, valid_idx):
            text = out.outputs[0].text if out.outputs else ""
            width, height = sizes[idx]
            try:
                parsed[idx] = self.parser(text, width, height)
            except TypeError:
                parsed[idx] = self.parser(text)
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
