# -*- coding: utf-8 -*-
"""
VLM-driven UI element relabeling (Qwen3-VL via vLLM) — FIXED:
- Build prompts with the model's chat template (no manual "<image>").
- Optional visualization hooks are exposed via the CLI (see cli.py patch).
## smoke test on a few pages, with overlays
wsd vlm-label \
  --raw-dir old_data/raw \
  --crops-dir old_data/crops \
  --limit 50 \
  --batch-size 64 \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --viz-dir old_data/viz_vlm


## full run, tp=2 across two GPUs (example)
wsd vlm-label \
  --raw-dir old_data/raw \
  --crops-dir old_data/crops \
  --batch-size 32 \
  --tp 2 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --inplace-elements \
  --viz-dir data/viz_vlm
  
## multi-gpu with tp=4 (example)
# env that tends to help multi-GPU stability
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1          # or: export NCCL_P2P_LEVEL=NVL  (on NVLink boxes)
export NCCL_IB_DISABLE=1           # if you don't have IB or it's flaky
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENFORCE_EAGER=1
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
wsd vlm-label \
  --raw-dir old_data/raw \
  --crops-dir old_data/crops \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --tp 4 \
  --batch-size 2048 \
  --viz-dir old_data/viz_vlm \
  --viz-min-conf 0.3 \
  --inplace-elements


## multi-gpu sharded example (8 GPUs, each handling 1/8th of the data)
# Shard 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 wsd vlm-label \
  --raw-dir data/raw \
  --crops-dir data/crops_shard0 \
  --viz-dir data/viz_vlm_shard0 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 512 \
  --tp 1 \
  --num-shards 8 \
  --inplace-elements \
  --shard-index 0 \
  > logs/shard0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 wsd vlm-label \
  --raw-dir data/raw \
  --crops-dir data/crops_shard1 \
  --viz-dir data/viz_vlm_shard1 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 512 \
  --tp 1 \
  --num-shards 8 \
  --inplace-elements \
  --shard-index 1 \
  > logs/shard1.log 2>&1 &
  
CUDA_VISIBLE_DEVICES=2 wsd vlm-label \
  --raw-dir data/raw \
  --crops-dir data/crops_shard2 \
  --viz-dir data/viz_vlm_shard2 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 512 \
  --tp 1 \
  --num-shards 8 \
  --inplace-elements \
  --shard-index 2 \
  > logs/shard2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 wsd vlm-label \
  --raw-dir data/raw \
  --crops-dir data/crops_shard3 \
  --viz-dir data/viz_vlm_shard3 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 512 \
  --tp 1 \
  --num-shards 8 \
  --inplace-elements \
  --shard-index 3 \
  > logs/shard3.log 2>&1 &

CUDA_VISIBLE_DEVICES=4 wsd vlm-label \
  --raw-dir data/raw \
  --crops-dir data/crops_shard4 \
  --viz-dir data/viz_vlm_shard4 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 512 \
  --tp 1 \
  --num-shards 8 \
  --inplace-elements \
  --shard-index 4 \
  > logs/shard4.log 2>&1 &
  
CUDA_VISIBLE_DEVICES=5 wsd vlm-label \
  --raw-dir data/raw \
  --crops-dir data/crops_shard5 \
  --viz-dir data/viz_vlm_shard5 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 512 \
  --tp 1 \
  --num-shards 8 \
  --inplace-elements \
  --shard-index 5 \
  > logs/shard5.log 2>&1 &
  
CUDA_VISIBLE_DEVICES=6 wsd vlm-label \
  --raw-dir data/raw \
  --crops-dir data/crops_shard6 \
  --viz-dir data/viz_vlm_shard6 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 512 \
  --tp 1 \
  --num-shards 8 \
  --inplace-elements \
  --shard-index 6 \
  > logs/shard6.log 2>&1 &
  
CUDA_VISIBLE_DEVICES=7 wsd vlm-label \
  --raw-dir data/raw \
  --crops-dir data/crops_shard7 \
  --viz-dir data/viz_vlm_shard7 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --batch-size 512 \
  --tp 1 \
  --num-shards 8 \
  --inplace-elements \
  --shard-index 7 \
  > logs/shard7.log 2>&1 &

"""
from __future__ import annotations
import os, io, json, re, glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable
from PIL import Image
from tqdm import tqdm
import time

try:
    from vllm import LLM, SamplingParams  # type: ignore
    _HAS_VLLM = True
except Exception:
    _HAS_VLLM = False

# -------- Canonical classes (deduped, order-stable) --------
_CANON_CLASSES = [
    "Table", "Column/Browser", "Button", "Utility Button", "App Icon",
    "Navigation Bar", "Status Bar", "Search Field", "Toolbar", "Tooltip",
    "Video", "Tab Bar", "Side Bar", "Slider", "Picker", "ContextMenu",
    "DockMenu", "EditMenu", "Image", "Scroll", "Switch", "File Icon", "Chart",
    "Window", "Screen", "List", "List Item", "PopUp Menu", "Steppers",
    "Toggles", "Text Input", "Rating Indicator", "Checkbox", "Radiobox",
    "Select", "Avatar", "Badge", "Alert", "Progress bar", "Bottom navigation",
    "Breadcrumb", "Page control", "Link", "Menu", "Pagination",
    "Tab", "Search Bar", "Date-Time picker", "Calendar", "Text", "Heading",
    "Code snippet", "Carousel", "Notification", "Logo",
]
def _dedupe(seq: Sequence[str]) -> List[str]:
    seen = set(); out=[]
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out
CANON_CLASSES = _dedupe(_CANON_CLASSES)

_ALIAS: Dict[str, str] = {
    "progressbar": "Progress bar", "progress-bar": "Progress bar",
    "navbar": "Navigation Bar", "sidebar": "Side Bar",
    "bottom nav": "Bottom navigation", "breadcrumb": "Breadcrumb",
    "pathcontrol": "Breadcrumb", "pagecontrol": "Page control",
    "check box": "Checkbox", "checkbox button": "Checkbox",
    "radio": "Radiobox", "radio button": "Radiobox", "radiobutton": "Radiobox",
    "text field": "Text Input", "input": "Text Input",
    "search box": "Search Field", "search input": "Search Field",
    "searchbar": "Search Bar", "avatar image": "Avatar",
    "profile image": "Avatar", "user image": "Avatar",
    "icon": "File Icon", "logo image": "Logo", "video player": "Video",
    "image view": "Image", "picture": "Image",
}

_SYSTEM_PROMPT = (
    "You classify ONE UI element using:\n"
    " (1) the full-page screenshot,\n"
    " (2) a cropped image of the element,\n"
    " (3) a compact HTML-like snippet for the element.\n\n"
    "Labeling rule:\n"
    "- Choose exactly ONE label from the allowed list.\n"
    "- Prefer the most specific, functionally correct label that best matches the element's purpose in its context.\n"
    "- Favor function over appearance. If a more specific option exists, prefer it (e.g., 'Search Field' over 'Text Input' when it is clearly a search box; 'Link' over 'Text' if it navigates; 'Button' over 'Image' if it acts like a button).\n"
    "- Use HTML/ARIA (role, type, href, aria-*, classes) to break ties. If still ambiguous, choose the closest single label from the list.\n\n"
    "Interactability:\n"
    "- Set interactable=true if a typical end-user can directly act on this element itself (click/tap/select/type/drag/scroll within it). Otherwise false (pure content/decoration or a passive container).\n\n"
    "Output ONLY one JSON object on a single line:\n"
    "{\"label\":\"<one-of-allowed>\",\"interactable\":<true|false>}"
)


def _classes_block() -> str:
    return "Allowed labels:\n- " + "\n- ".join(CANON_CLASSES)

def build_user_prompt(element_html: str) -> str:
    html = (element_html or "").strip()
    if len(html) > 1500:
        html = html[:1400] + " … " + html[-90:]
    return (
        f"{_classes_block()}\n\n"
        "Element HTML snippet:\n"
        "---------------------\n"
        f"{html}\n\n"
        "Instructions:\n"
        "- Consider the crop FIRST to understand the element's visual identity.\n"
        "- Use the full-page screenshot for surrounding context and function.\n"
        "- Use HTML attributes (tag, role, type, aria-*, classes) as semantic hints.\n"
        "- Output strictly one JSON object as specified (no extra text)."
    )

def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _clip_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int,int,int,int]:
    x = max(0, min(int(x), max(0, W-1)))
    y = max(0, min(int(y), max(0, H-1)))
    w = max(1, int(min(w, W - x)))
    h = max(1, int(min(h, H - y)))
    return x, y, w, h

def element_to_compact_json(el: Dict[str, Any]) -> str:
    """
    Serialize the element almost 'as is', but:
    - use real JSON (double quotes, null, true/false),
    - remove our own VLM outputs (vlm_*),
    - and make it compact (no unnecessary spaces).
    """
    # Optional: don't feed previous VLM predictions back as input
    cleaned = {
        k: v for k, v in el.items()
        if not k.startswith("vlm_")
    }
    try:
        # separators=(",",":") removes spaces after commas/colons → more compact
        return json.dumps(cleaned, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        # Fallback to plain str if something goes wrong
        return str(cleaned)

def reconstruct_html(el: Dict[str, Any]) -> str:
    tag = (el.get("tag") or "div").lower()
    attrs = el.get("attrs") or {}
    bits = []
    for k in ["id","class","role","type","aria-label","aria-hidden","href","value","name","placeholder"]:
        v = attrs.get(k) if k in attrs else el.get(k.replace("-", "_"))
        if v:
            v = str(v)
            if len(v) > 120: v = v[:117] + "..."
            if k == "class": v = " ".join(v.split())
            bits.append(f'{k}="{v}"')
    inner = (el.get("inner_text") or "").strip()
    if len(inner) > 200:
        inner = inner[:170] + " ... " + inner[-20:]
    head = f"<{tag}" + ((" " + " ".join(bits)) if bits else "") + ">"
    return head + inner + f"</{tag}>"

def reconstruct_html_simple(el: Dict[str, Any]) -> str:
    """It will directly give the str of Dictionary element."""
    return element_to_compact_json(el)

def element_scale(meta: Dict[str, Any]) -> float:
    s = meta.get("css_to_image_scale", 1)
    try: s = float(s)
    except Exception: s = 1.0
    return max(0.1, min(16.0, s))

def crop_element_to_file(page_png: str, bbox_xywh: Dict[str,int], scale: float, out_path: str, padding: int = 5) -> Optional[str]:
    try:
        with Image.open(page_png) as im:
            W, H = im.size
            x = int(round(bbox_xywh["x"] * scale))
            y = int(round(bbox_xywh["y"] * scale))
            w = int(round(bbox_xywh["w"] * scale))
            h = int(round(bbox_xywh["h"] * scale))
            
            # Apply padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(W - x, w + 2 * padding)
            h = min(H - y, h + 2 * padding)
            
            x,y,w,h = _clip_box(x,y,w,h,W,H)
            if w < 2 or h < 2: return None
            im.crop((x,y,x+w,y+h)).save(out_path)
            return out_path
    except Exception:
        return None

@dataclass
class VLMConfig:
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    
    max_model_len: int | None = None
    limit_mm_per_prompt: dict | None = None
    
    # NEW: parallel CPU & multimodal tuning
    # mm_encoder_tp_mode: str = "weights"    # "data" when TP > 1
    # mm_processor_cache_gb: float = 0.0     # e.g. 4.0 to enable cache
    # mm_processor_cache_type: str = "shm"  # let vLLM choose (e.g. "shm")

class VLMEngine:
    """vLLM wrapper that uses the model's chat template for multimodal prompts."""
    def __init__(self, cfg: VLMConfig):
        if not _HAS_VLLM:
            raise RuntimeError("vllm is not installed. Install with: pip install vllm")
        self.cfg = cfg
        self.llm = LLM(
            model=cfg.model,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype=cfg.dtype,
            trust_remote_code=True,
            tokenizer_mode="auto",
            # mm_encoder_tp_mode=cfg.mm_encoder_tp_mode,
            # mm_processor_cache_gb=cfg.mm_processor_cache_gb,
            # mm_processor_cache_type=cfg.mm_processor_cache_type,
        )
        self.sampling = SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        # Prefer llm.apply_chat_template if present; fallback to tokenizer
        self._has_llm_act = hasattr(self.llm, "apply_chat_template")
        self._tokenizer = None if self._has_llm_act else self.llm.get_tokenizer()

    def _chat_template(self, messages: List[Dict[str, Any]]) -> str:
        if self._has_llm_act:
            return self.llm.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        # fallback
        return self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def build_request(self, page_img: str, crop_img: str, user_prompt: str) -> Dict[str, Any]:
        """
        Build a **chat-templated** prompt with 2 image placeholders in order:
        [1] full page, [2] element crop. Qwen processor will replace them.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},  # placeholder for full page
                {"type": "image"},  # placeholder for crop
                {"type": "text", "text": user_prompt},
            ]}
        ]
        prompt = self._chat_template(messages)
        return {
            "prompt": prompt,
            "multi_modal_data": {"image": [page_img, crop_img]},
        }

    def generate(self, requests: List[Dict[str, Any]]) -> List[str]:
        # vLLM offline API supports a list of "requests" dicts:
        # [{"prompt": ..., "multi_modal_data": {...}}, ...]
        outputs = self.llm.generate(requests, self.sampling)
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]

# --------- JSON parsing / normalization ----------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

def _extract_json_line(s: str) -> Optional[Dict[str, Any]]:
    if not s: return None
    s = s.strip()
    if s.startswith("```"): s = s.strip("`")
    m = _JSON_RE.search(s)
    if not m: return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def normalize_label(raw_label: str) -> Optional[str]:
    if not raw_label: return None
    k = raw_label.strip()
    low = re.sub(r"[\s\-_/]+", " ", k.lower()).strip()
    if low in _ALIAS: return _ALIAS[low]
    for c in CANON_CLASSES:
        if k.lower() == c.lower(): return c
    for c in CANON_CLASSES:
        cn = c.lower()
        if low == cn or low in cn or cn in low:
            return c
    return None

# --------- Public API ----------

def label_dir(
    raw_dir: str = "data/raw",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    crops_dir: str = "data/crops",
    out_suffix: str = ".vlm.labels.json",
    batch_size: int = 256,
    tensor_parallel_size: int = 1,
    limit_images: Optional[int] = None,
    min_elem_size: int = 3,
    inplace_elements: bool = True,
    viz_dir: Optional[str] = None,
    screens_per_pass: int = 64,   # outer chunking to cap memory
    shard_index: int = 0,         # NEW: which shard (0-based)
    num_shards: int = 1,          # NEW: total number of shards
) -> None:
    """
    Build requests in outer 'passes' over at most `screens_per_pass` screenshots,
    so a single vLLM micro-batch can still contain elements from multiple screens
    without materializing the entire corpus at once.

    Writes sidecars / inplace .elements.json / viz overlays **incrementally**
    after every vLLM micro-batch, so progress is durable if the run aborts.

    Sharding:
    - All screenshots are sorted into a list.
    - This job processes only bases[b_i] where b_i % num_shards == shard_index,
      implemented via Python slicing bases_all[shard_index::num_shards].
    """

    def _iter_screens(raw_dir: str) -> Iterable[str]:
        for p in sorted(glob.glob(os.path.join(raw_dir, "*.elements.json"))):
            yield p[:-len(".elements.json")]

    def _is_already_processed(elements_path: str) -> bool:
        """Check if any element has vlm_label attribute."""
        elements = _load_json(elements_path)
        if elements is None:
            return True
        try:
            if "vlm_label" in elements[0]:
                return True
        except Exception:
            return True
        return False

    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(
            f"shard_index must be in [0, {num_shards-1}], got {shard_index}"
        )

    # 1) Build full list once, then shard it
    bases_all = list(_iter_screens(raw_dir))
    if limit_images:
        bases_all = bases_all[:limit_images]

    # Slice to this shard's subset
    bases = bases_all[shard_index::num_shards]
    
    # # reverse the bases to iterate from end to start
    # bases = list(reversed(bases))

    print(
        f"[VLM] Total bases: {len(bases_all)} | "
        f"num_shards={num_shards} shard_index={shard_index} -> "
        f"{len(bases)} bases for this job"
    )

    cfg = VLMConfig(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=16384,
        limit_mm_per_prompt={"image": 2, "video": 0, "audio": 0},
        # mm_encoder_tp_mode="data" if tensor_parallel_size > 1 else "weights",
        # mm_processor_cache_gb=4.0,      # e.g. 4 GB for HF processor cache
        # mm_processor_cache_type="shm",
    )

    engine = VLMEngine(cfg)

    from .vlm_visualize import draw_vlm_overlay

    # Clean old artifacts ONLY for this shard's bases
    # if os.path.exists(crops_dir):
    #     for f in glob.glob(os.path.join(crops_dir, "*.png")):
    #         os.remove(f)
    #     # keep existing JPEGs if you switched crops to JPEG; adjust pattern as needed
    # if viz_dir and os.path.exists(viz_dir):
    #     for f in glob.glob(os.path.join(viz_dir, "*.vlm.viz.jpg")):
    #         os.remove(f)
    # for base in bases:
    #     p = base + out_suffix
    #     if os.path.exists(p):
    #         os.remove(p)

    _ensure_dir(crops_dir)

    # Accumulators across passes
    elements_by_base: Dict[str, List[Dict[str, Any]]] = {}
    page_png_by_base: Dict[str, str] = {}
    # store (elem_index, label, interactable)
    labels_by_base: Dict[str, List[Tuple[int, str, bool]]] = {}

    # Helper: fast crop using already opened PIL image
    def _crop_from_open_image(
        im: Image.Image,
        r: Dict[str, int],
        scale: float,
        out_path: str,
        padding: int = 5,
    ) -> bool:
        try:
            W, H = im.size
            x = int(round(r.get("x", 0) * scale))
            y = int(round(r.get("y", 0) * scale))
            w = int(round(r.get("w", 0) * scale))
            h = int(round(r.get("h", 0) * scale))
            # padding
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(W - x, w + 2 * padding)
            h = min(H - y, h + 2 * padding)
            if w < 2 or h < 2:
                return False
            
            aspect_ratio = max(w, h) / min(w, h)
            if aspect_ratio > 199:
                return False
            im.crop((x, y, x + w, y + h)).save(out_path)
            return True
        except Exception:
            return False

    # Flush helper: write sidecar / inplace / viz for a set of bases
    def _flush_bases(bases_to_flush: Iterable[str]) -> None:
        for base in bases_to_flush:
            triples = labels_by_base.get(base, [])
            triples.sort(key=lambda x: x[0])

            # Sidecar (no confidence anymore)
            sidecar = {
                "model": model,
                "classes": CANON_CLASSES,
                "elements": [
                    {"index": i, "label": lab, "interactable": inter}
                    for (i, lab, inter) in triples
                ],
            }
            with open(base + out_suffix, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, ensure_ascii=False, indent=2)

            # If we indexed this base, we can also write inplace + viz
            if base not in elements_by_base:
                continue

            elements = elements_by_base[base]

            if inplace_elements:
                elements_path = base + ".elements.json"
                for i_el, lab, inter in triples:
                    if 0 <= i_el < len(elements):
                        elements[i_el]["vlm_label"] = lab
                        elements[i_el]["vlm_interactable"] = bool(inter)
                with open(elements_path, "w", encoding="utf-8") as f:
                    json.dump(elements, f, ensure_ascii=False, indent=2)

            if viz_dir:
                _ensure_dir(viz_dir)
                out_img = os.path.join(
                    viz_dir, os.path.basename(base) + ".vlm.viz.jpg"
                )
                draw_vlm_overlay(
                    page_png_by_base.get(base, base + ".png"),
                    elements,
                    triples,   # (i, label, interactable)
                    out_img,
                )

    def _as_bool(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        s = str(v).strip().lower()
        return s in {"true", "yes", "y", "1", "t"}

    # Process in outer passes
    for start in tqdm(range(0, len(bases), screens_per_pass), desc="Passes over screenshots"):
        group = bases[start:start + screens_per_pass]

        all_reqs: List[Dict[str, Any]] = []
        all_keys: List[Tuple[str, int]] = []  # (base, element_index)

        loop_start = time.time()
        for base in tqdm(group, desc="Indexing elements for VLM (pass)", leave=False):
            page_png = base + ".png"
            elements_path = base + ".elements.json"
            meta_path = base + ".meta.json"
            if not (os.path.exists(page_png) and os.path.exists(elements_path) and os.path.exists(meta_path)):
                continue

            # Skip if already processed
            if _is_already_processed(elements_path):
                print(f"[VLM] Skipping already processed: {base}")
                continue

            elements = _load_json(elements_path)
            meta = _load_json(meta_path)
            if elements is None or meta is None or not elements:
                print(f"[VLM] Skipping invalid/missing data: {base}")
                continue
            scale = element_scale(meta)

            if base not in elements_by_base:
                elements_by_base[base] = elements
                page_png_by_base[base] = page_png

            with Image.open(page_png) as im:
                # Skip entire page if aspect ratio is too extreme
                W, H = im.size
                if W > 0 and H > 0:
                    page_aspect_ratio = max(W, H) / min(W, H)
                    if page_aspect_ratio > 199:
                        continue
                for idx, el in enumerate(elements):
                    r = el.get("rect") or {}
                    w = int(r.get("w", 0)); h = int(r.get("h", 0))
                    if w < min_elem_size or h < min_elem_size:
                        continue
                    if w > 0 and h > 0:
                        aspect_ratio = max(w, h) / min(w, h)
                        if aspect_ratio > 199:
                            continue

                    crop_name = f"{os.path.basename(base)}__el{idx}.png"
                    crop_path = os.path.join(crops_dir, crop_name)
                    if not os.path.exists(crop_path):
                        ok = _crop_from_open_image(im, r, scale, crop_path, padding=5)
                        if not ok:
                            continue
                    if os.path.getsize(crop_path) == 0:
                        continue

                    try:
                        html_snippet = reconstruct_html_simple(el)
                    except NameError:
                        html_snippet = reconstruct_html(el)

                    req = engine.build_request(
                        page_img=page_png,
                        crop_img=crop_path,
                        user_prompt=build_user_prompt(html_snippet),
                    )
                    all_reqs.append(req)
                    all_keys.append((base, idx))

        loop_elapsed = time.time() - loop_start
        print(f"Indexing loop completed in {loop_elapsed:.2f} seconds ({len(all_reqs)} requests)")

        # Track flushed bases across micro-batches
        flushed_bases = set()

        # Micro-batches inside this pass
        for i in tqdm(range(0, len(all_reqs), batch_size), desc="VLM batches (pass)", leave=False):
            texts = engine.generate(all_reqs[i:i + batch_size])
            for off, t in enumerate(texts):
                base, elem_index = all_keys[i + off]
                obj = _extract_json_line(t) or {}
                raw_label = (obj.get("label") or "").strip()
                label = normalize_label(raw_label) or "Unknown"
                inter = _as_bool(obj.get("interactable", False))
                labels_by_base.setdefault(base, []).append((elem_index, label, inter))

            # Flush everything touched in this micro-batch
            touched_bases = {b for (b, _) in all_keys[i:i + batch_size]}
            _flush_bases(touched_bases)
            flushed_bases.update(touched_bases)

        # Final sweep: flush any bases in this pass that weren't flushed
        unflushed = set(group) - flushed_bases
        _flush_bases(unflushed)
