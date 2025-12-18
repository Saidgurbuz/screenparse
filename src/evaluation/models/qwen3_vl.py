from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor

from ..datasets import element_from_obj
from ..types import EvaluationSample, UIElement
from .base import ModelRunner


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
        '{"bbox_ltrb":[l,t,r,b], "label": "<type>", "text": "<visible text>"}.\n'
        "The bbox_ltrb should be normalized to 0-1000. (l,t,r,b). Include all elements.\n"
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

        self.prompt = prompt or _default_prompt()
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
        for sample in samples:
            msgs = self._build_messages(sample)
            inputs.append(self._prepare_inputs(msgs))

        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        parsed: List[Sequence[UIElement]] = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
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
