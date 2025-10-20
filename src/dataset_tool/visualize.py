# src/dataset_tool/visualize.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os, math, random
from PIL import Image, ImageDraw, ImageFont
from .utils import load_json, ensure_dir
from .config import Config


# Fallback font (macOS has HelveticaNeue/Arial; PIL default also OK)
def _load_font(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype("/System/Library/Fonts/SFNS.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Arial.ttf", size
            )
        except Exception:
            return ImageFont.load_default()


def _color_from_name(name: str):
    # Stable pseudo-random color from a label
    rnd = random.Random(hash(name) & 0xFFFFFFFF)
    # Avoid very light colors (hard to see)
    return tuple(rnd.randint(40, 220) for _ in range(3))


@dataclass
class VizOptions:
    draw_elements: bool = True
    draw_text_spans: bool = True
    draw_ocr: bool = True
    min_box_area: int = 12 * 12  # skip tiny noise
    label_elements: bool = True
    label_text_spans: bool = False
    opacity: int = 80  # box fill opacity (0..255)
    line_width: int = 2
    max_text_len: int = 120  # clamp label length


def _rect_to_xyxy(rect: Dict[str, int]):
    return (rect["x"], rect["y"], rect["x"] + rect["w"], rect["y"] + rect["h"])


def _should_skip(rect: Dict[str, int], min_area: int) -> bool:
    return rect["w"] * rect["h"] < min_area or rect["w"] < 2 or rect["h"] < 2


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple,
    txt: str,
    color: tuple,
    font: ImageFont.ImageFont,
):
    # Draw a filled label box with text at xy (x1,y1)
    margin_x, margin_y = 4, 2
    tw, th = draw.textbbox((0, 0), txt, font=font)[2:]
    x1, y1 = xy
    x2, y2 = x1 + tw + 2 * margin_x, y1 + th + 2 * margin_y
    # background
    draw.rectangle((x1, y1, x2, y2), fill=color)
    # text
    draw.text((x1 + margin_x, y1 + margin_y), txt, fill=(255, 255, 255), font=font)


def visualize_one(
    image_path: str,
    elements_path: Optional[str],
    texts_path: Optional[str],
    ocr_path: Optional[str],
    out_path: str,
    opts: VizOptions = VizOptions(),
):
    im = Image.open(image_path).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_small = _load_font(12)
    font_med = _load_font(14)

    # Elements layer
    if elements_path and os.path.exists(elements_path) and opts.draw_elements:
        elems: List[Dict[str, Any]] = load_json(elements_path)
        for e in elems:
            r = e["rect"]
            if _should_skip(r, opts.min_box_area):
                continue
            cls = e.get("type") or e.get("tag") or "unknown"
            color = _color_from_name(cls)
            x1, y1, x2, y2 = _rect_to_xyxy(r)
            # stroke
            draw.rectangle((x1, y1, x2, y2), outline=color, width=opts.line_width)
            # semi-transparent fill
            draw.rectangle((x1, y1, x2, y2), fill=(*color, opts.opacity))
            if opts.label_elements:
                lab = cls
                # Add hints if available
                tag = e.get("tag")
                role = e.get("role")
                if tag and tag != cls:
                    lab += f" | <{tag}>"
                if role:
                    lab += f" | {role}"
                if len(lab) > opts.max_text_len:
                    lab = lab[: opts.max_text_len - 1] + "…"
                _draw_label(draw, (x1 + 1, max(0, y1 - 18)), lab, color, font_small)

    # Text spans layer
    if texts_path and os.path.exists(texts_path) and opts.draw_text_spans:
        spans: List[Dict[str, Any]] = load_json(texts_path)
        for t in spans:
            r = t["rect"]
            if _should_skip(r, 6 * 6):  # allow smaller for text
                continue
            color = _color_from_name("text_span")
            x1, y1, x2, y2 = _rect_to_xyxy(r)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=1)
            if opts.label_text_spans:
                txt = (t.get("text") or "").strip().replace("\n", " ")
                if len(txt) > opts.max_text_len:
                    txt = txt[: opts.max_text_len - 1] + "…"
                if txt:
                    _draw_label(draw, (x1 + 1, max(0, y1 - 16)), txt, color, font_small)

    # OCR overlay (per-element)
    if ocr_path and os.path.exists(ocr_path) and opts.draw_ocr:
        try:
            ocr_items: List[Dict[str, Any]] = load_json(ocr_path)
            # Need element rects to place OCR near its element
            elem_rects = []
            if elements_path and os.path.exists(elements_path):
                elems = load_json(elements_path)
                elem_rects = [e["rect"] for e in elems]
            for o in ocr_items:
                idx = o.get("element_index")
                if idx is None or idx >= len(elem_rects):
                    continue
                r = elem_rects[idx]
                x1, y1, _, _ = _rect_to_xyxy(r)
                color = _color_from_name("ocr")
                txt = o.get("text", "").strip().replace("\n", " ")
                if txt:
                    _draw_label(
                        draw,
                        (x1 + 1, max(0, y1 - 18)),
                        f"OCR: {txt[:opts.max_text_len]}",
                        color,
                        font_med,
                    )
        except Exception:
            pass

    # Composite & save
    out = Image.alpha_composite(im, overlay).convert("RGB")
    ensure_dir(os.path.dirname(out_path))
    out.save(out_path, quality=95)


def visualize_record(
    record: Dict[str, Any], cfg: Config, opts: VizOptions = VizOptions()
) -> str:
    # record is what collect_one returns
    img = record["image_path"]
    elements = record.get("elements_path")
    texts = record.get("texts_path")
    ocr = record.get("ocr_path")
    stem = os.path.splitext(os.path.basename(img))[0]
    out_path = os.path.join(cfg.viz_dir, f"{stem}.viz.jpg")
    visualize_one(img, elements, texts, ocr, out_path, opts)
    return out_path
