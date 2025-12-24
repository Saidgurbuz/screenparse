from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor

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
                    label=label,
                    text=text,
                    score=float(score) if score is not None else None,
                )
            )
        return elements
    except Exception:
        return []


class Qwen3VLRunner(ModelRunner):
    """
    Qwen/Qwen3-VL-8B-Instruct via vLLM (vision-language).
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-8B-Instruct",
        name: str | None = None,
        prompt: str | None = None,
        parser: Callable[[str], List[UIElement]] | None = None,
        max_new_tokens: int = 4096,
        temperature: float = 0.0,
        top_p: float = 0.9,
        trust_remote_code: bool = True,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int | None = None,
        class_schema: str = "custom55",
    ):
        runner_name = name or Path(model_id).name
        super().__init__(runner_name)
        try:
            from qwen_vl_utils import process_vision_info  # type: ignore
        except Exception as exc:
            raise ImportError("qwen_vl_utils>=0.0.14 is required for Qwen3VLRunner") from exc

        # Ensure spawn start method for CUDA/vLLM workers
        try:
            current = mp.get_start_method(allow_none=True)
            if current != "spawn":
                mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as exc:
            raise ImportError("vllm is required for Qwen3VLRunner") from exc

        self.prompt = prompt or _default_prompt(class_schema)
        self.parser = parser or _default_parser
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.patch_size = getattr(self.processor.image_processor, "patch_size", None)
        self._process_vision_info = process_vision_info

        tp = tensor_parallel_size if tensor_parallel_size is not None else torch.cuda.device_count()
        self.llm = LLM(
            model=model_id,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tp,
        )

    def _prepare_inputs(self, messages):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            messages,
            image_patch_size=self.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        return {
            "prompt": text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }

    def _build_messages(self, sample: EvaluationSample):
        image = Image.open(sample.image_path).convert("RGB")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

    def predict(self, sample: EvaluationSample) -> Sequence[UIElement]:
        return self.predict_batch([sample])[0]

    def predict_batch(self, samples: Iterable[EvaluationSample]) -> List[Sequence[UIElement]]:
        inputs = []
        sizes = []
        for sample in samples:
            msgs = self._build_messages(sample)
            inputs.append(self._prepare_inputs(msgs))
            try:
                with Image.open(sample.image_path) as im:
                    sizes.append(im.size)
            except Exception:
                sizes.append((0, 0))

        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        parsed: List[Sequence[UIElement]] = []
        for out, (w, h) in zip(outputs, sizes):
            text = out.outputs[0].text if out.outputs else ""
            try:
                parsed.append(self.parser(text, w, h))
            except TypeError:
                parsed.append(self.parser(text))
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
                import torch

                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
