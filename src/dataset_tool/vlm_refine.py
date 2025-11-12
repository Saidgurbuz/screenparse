# -*- coding: utf-8 -*-
"""
VLM-driven UI element relabeling (Qwen3-VL via vLLM) — FIXED:
- Build prompts with the model's chat template (no manual "<image>").
- Optional visualization hooks are exposed via the CLI (see cli.py patch).
# smoke test on a few pages, with overlays
wsd vlm-label \
  --raw-dir old_data/raw \
  --crops-dir old_data/crops \
  --limit 5 \
  --batch-size 256 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --viz-dir old_data/viz_vlm \
  --viz-min-conf 0.3

# full run, tp=2 across two GPUs (example)
wsd vlm-label \
  --raw-dir old_data/raw \
  --crops-dir old_data/crops \
  --batch-size 32 \
  --tp 2 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --inplace-elements \
  --viz-dir data/viz_vlm


"""
from __future__ import annotations
import os, io, json, re, glob
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable
from PIL import Image
from tqdm import tqdm

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
    "Breadcrumb/Pathcontrol", "Page control", "Link", "Menu", "Pagination",
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
    "bottom nav": "Bottom navigation", "breadcrumb": "Breadcrumb/Pathcontrol",
    "pathcontrol": "Breadcrumb/Pathcontrol", "pagecontrol": "Page control",
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
    "You are a meticulous UI element classifier. You receive:\n"
    " (1) the full-page screenshot of a webpage/app,\n"
    " (2) a cropped image of ONE element highlighted from that page, and\n"
    " (3) a compact HTML-like snippet for that element.\n\n"
    "Choose exactly ONE label from the allowed list of UI element types. "
    "Use BOTH visual context (full page + crop) and HTML/ARIA hints (role, input type, classes). "
    "Prefer the most specific label. If multiple are plausible, choose the one best aligned with purpose.\n\n"
    'Return ONLY a single JSON object on one line: '
    '{"label":"<one-of-allowed>","confidence":<0..1>,"reason":"<short rationale>"}'
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
        "- Output strictly one JSON object as specified."
    )

def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _clip_box(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int,int,int,int]:
    x = max(0, min(int(x), max(0, W-1)))
    y = max(0, min(int(y), max(0, H-1)))
    w = max(1, int(min(w, W - x)))
    h = max(1, int(min(h, H - y)))
    return x, y, w, h

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
    return str(el)

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
    max_new_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0

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

# --------- Public APIs ----------
def classify_ui_samples(
    samples: List[Dict[str, Any]],
    model: str = "Qwen/Qwen3-VL-3B-Instruct",
    tensor_parallel_size: int = 1,
    batch_size: int = 256,
    max_new_tokens: int = 1024,
) -> List[Dict[str, Any]]:
    """
    Each sample:
      {"page_image": "...", "crop_image": "...", "element_html": "...", "sample_id": "..."}
    Returns (aligned order):
      [{"label": "...", "confidence": 0.0..1.0, "raw": <json or raw_text>, "sample_id": "..."}]
    """
    cfg = VLMConfig(model=model, tensor_parallel_size=tensor_parallel_size, max_new_tokens=max_new_tokens)
    engine = VLMEngine(cfg)
    out: List[Dict[str, Any]] = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        reqs = []
        for s in batch:
            reqs.append(engine.build_request(
                page_img=s["page_image"],
                crop_img=s["crop_image"],
                user_prompt=build_user_prompt(s.get("element_html","")),
            ))
        texts = engine.generate(reqs)
        for s, t in zip(batch, texts):
            print('Debug output:', t)  # DEBUG
            obj = _extract_json_line(t) or {}
            raw_label = (obj.get("label") or "").strip()
            label = normalize_label(raw_label) or "Unknown"
            conf = obj.get("confidence", 0.0)
            try: conf = float(conf)
            except Exception: conf = 0.0
            out.append({
                "label": label,
                "confidence": max(0.0, min(1.0, conf)),
                "raw": obj or {"raw_text": t},
                "sample_id": s.get("sample_id"),
            })
    return out

def _iter_screens(raw_dir: str) -> Iterable[str]:
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.elements.json"))):
        yield p[:-len(".elements.json")]

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
    viz_min_conf: float = 0.0,
    screens_per_pass: int = 32,   # NEW: outer chunking to cap memory
) -> None:
    """
    Build requests in outer "passes" over at most `screens_per_pass` screenshots,
    so a single vLLM micro-batch can still contain elements from multiple screens
    without materializing the *entire* corpus at once.

    Also speeds up indexing: each screenshot PNG is opened once and reused for
    all element crops in that pass (avoids re-opening per element).
    """
    bases = list(_iter_screens(raw_dir))
    if limit_images:
        bases = bases[:limit_images]

    cfg = VLMConfig(model=model, tensor_parallel_size=tensor_parallel_size)
    engine = VLMEngine(cfg)

    from .vlm_visualize import draw_vlm_overlay

    # before start, remove all old crops and viz_vlm contents as well as all .vlm.labels.json files
    if os.path.exists(crops_dir):
        for f in glob.glob(os.path.join(crops_dir, "*.png")):
            os.remove(f)
    if viz_dir and os.path.exists(viz_dir):
        for f in glob.glob(os.path.join(viz_dir, "*.vlm.viz.jpg")):
            os.remove(f)
    for base in bases:
        sidecar_path = base + out_suffix
        if os.path.exists(sidecar_path):
            os.remove(sidecar_path)

    _ensure_dir(crops_dir)

    # Global accumulators we fill pass-by-pass (keeps memory bounded):
    elements_by_base: Dict[str, List[Dict[str, Any]]] = {}
    page_png_by_base: Dict[str, str] = {}
    labels_by_base: Dict[str, List[Tuple[int, str, float]]] = {}

    # Helper: fast crop using an already opened PIL image
    def _crop_from_open_image(im: Image.Image, r: Dict[str, int], scale: float, out_path: str, padding: int = 5) -> bool:
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
            im.crop((x, y, x + w, y + h)).save(out_path)
            return True
        except Exception:
            return False

    # Process in outer passes to cap memory
    for start in tqdm(range(0, len(bases), screens_per_pass), desc="Passes over screenshots"):
        group = bases[start:start + screens_per_pass]

        # Build requests for this pass only
        all_reqs: List[Dict[str, Any]] = []
        all_keys: List[Tuple[str, int]] = []  # (base, element_index)

        for base in tqdm(group, desc="Indexing elements for VLM (pass)", leave=False):
            page_png = base + ".png"
            elements_path = base + ".elements.json"
            meta_path = base + ".meta.json"
            if not (os.path.exists(page_png) and os.path.exists(elements_path) and os.path.exists(meta_path)):
                continue

            elements = _load_json(elements_path)
            meta = _load_json(meta_path)
            scale = element_scale(meta)

            # retain for writing later (once overall labeling is done)
            if base not in elements_by_base:
                elements_by_base[base] = elements
                page_png_by_base[base] = page_png

            # Open screenshot ONCE and crop all elements from this in-memory image
            with Image.open(page_png) as im:
                for idx, el in enumerate(elements):
                    r = el.get("rect") or {}
                    w = int(r.get("w", 0)); h = int(r.get("h", 0))
                    if w < min_elem_size or h < min_elem_size:
                        continue

                    crop_name = f"{os.path.basename(base)}__el{idx}.png"
                    crop_path = os.path.join(crops_dir, crop_name)
                    if not os.path.exists(crop_path):
                        ok = _crop_from_open_image(im, r, scale, crop_path, padding=5)
                        if not ok:
                            continue

                    # Prefer your light HTML serializer if present
                    try:
                        html_snippet = reconstruct_html_simple(el)  # your version
                    except NameError:
                        html_snippet = reconstruct_html(el)         # fallback

                    req = engine.build_request(
                        page_img=page_png,
                        crop_img=crop_path,
                        user_prompt=build_user_prompt(html_snippet),
                    )
                    all_reqs.append(req)
                    all_keys.append((base, idx))

        # Run vLLM for this pass in micro-batches and route outputs
        for i in tqdm(range(0, len(all_reqs), batch_size), desc="VLM batches (pass)", leave=False):
            texts = engine.generate(all_reqs[i:i + batch_size])
            for off, t in enumerate(texts):
                base, elem_index = all_keys[i + off]
                obj = _extract_json_line(t) or {}
                raw_label = (obj.get("label") or "").strip()
                label = normalize_label(raw_label) or "Unknown"
                conf = obj.get("confidence", 0.0)
                try:
                    conf = float(conf)
                except Exception:
                    conf = 0.0
                labels_by_base.setdefault(base, []).append(
                    (elem_index, label, max(0.0, min(1.0, conf)))
                )

    # After all passes: write sidecars / inplace / viz per base
    for base in tqdm(bases, desc="Writing VLM sidecars"):
        labels = labels_by_base.get(base, [])
        labels.sort(key=lambda x: x[0])

        sidecar = {
            "model": model,
            "classes": CANON_CLASSES,
            "elements": [{"index": i, "label": lab, "confidence": conf} for (i, lab, conf) in labels],
        }
        with open(base + out_suffix, "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)

        if base not in elements_by_base:
            continue

        elements = elements_by_base[base]

        if inplace_elements:
            elements_path = base + ".elements.json"
            for i_el, lab, conf in labels:
                if 0 <= i_el < len(elements):
                    elements[i_el]["vlm_label"] = lab
                    elements[i_el]["vlm_conf"] = conf
            with open(elements_path, "w", encoding="utf-8") as f:
                json.dump(elements, f, ensure_ascii=False, indent=2)

        if viz_dir:
            _ensure_dir(viz_dir)
            out_img = os.path.join(viz_dir, os.path.basename(base) + ".vlm.viz.jpg")
            draw_vlm_overlay(page_png_by_base[base], elements, labels, out_img, min_conf=viz_min_conf)