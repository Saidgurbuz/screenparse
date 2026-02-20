# src/webshot/visualize.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import os, math, random
from PIL import Image, ImageDraw, ImageFont
from .utils import load_json, ensure_dir
from .config import Config
from docling_ibm_models.reading_order.reading_order_rb import (
    PageElement,
    ReadingOrderPredictor,
    DocItemLabel,
    Size,
)
from docling_core.types.doc.base import CoordOrigin


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
    draw_ocr: bool = False
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
    print(f"OCR path: {ocr_path}")
    print(f"opts: {opts}")
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
            cls = e.get("vlm_label") or e.get("type") or e.get("tag") or "unknown"
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
            print(f"Adding OCR overlay from {ocr_path}")
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


# ─────────────────────────────────────────────────────────────────────────────
# Reading order visualization
# ─────────────────────────────────────────────────────────────────────────────

def _center_of_rect(rect: Dict[str, int]) -> tuple[float, float]:
    return (
        rect["x"] + rect["w"] / 2.0,
        rect["y"] + rect["h"] / 2.0,
    )


def _sort_key(rect: Dict[str, int]):
    return (rect["y"], rect["x"], rect["h"], rect["w"])


def _build_reading_order(elements: List[Dict[str, Any]], page_size: Size) -> List[int]:
    """
    Build a reading-order traversal (preorder DFS).

    Preference order for hierarchy fields:
      1) parent_index / children_indices (reconstructed hierarchy)
      2) _parent_dom_index (raw DOM parent captured during crawl)
    Falls back to spatial ordering if hierarchy is missing/invalid.
    """

    n = len(elements)
    if n == 0:
        return []

    children: Dict[int, List[int]] = {i: [] for i in range(n)}
    roots: List[int] = []

    # Prefer reconstructed hierarchy if present
    has_parent_index = any("parent_index" in el for el in elements)

    for idx, el in enumerate(elements):
        parent = None
        if has_parent_index:
            p = el.get("parent_index")
            if isinstance(p, int) and 0 <= p < n and p != idx:
                parent = p

        if parent is not None:
            children[parent].append(idx)
        else:
            roots.append(idx)

    # If no roots detected (bad parents), fall back to all indices
    if not roots:
        roots = list(range(n))

    # Use ReadingOrderPredictor to sort siblings
    def _predict_order(indices: List[int]) -> List[int]:
        if not indices:
            return []
        
        # Use the actual image dimensions provided by visualize_reading_order
        max_w = float(page_size.width)
        max_h = float(page_size.height)
        if max_w <= 0:
            max_w = 100.0
        if max_h <= 0:
            max_h = 100.0
        
        page_elems = []
        local_map = {} # local_idx -> original_idx
        
        for i, idx in enumerate(indices):
            local_map[i] = idx
            el = elements[idx]
            r = el["rect"]
            
            # Use default label as placeholder
            label = DocItemLabel.TEXT
            
            # Convert to Bottom-Left origin manually to avoid BoundingBox conversion issue
            # In TL: y=0 is top. t < b.
            # In BL: y=0 is bottom. t > b.
            t_tl = r["y"]
            b_tl = r["y"] + r["h"]
            
            t_bl = float(max_h) - t_tl
            b_bl = float(max_h) - b_tl
            
            pe = PageElement(
                cid=i,
                label=label,
                page_no=1,
                page_size=page_size,
                text=el.get("inner_text", ""),
                l=r["x"],
                t=t_bl,
                r=r["x"]+r["w"],
                b=b_bl,
                coord_origin=CoordOrigin.BOTTOMLEFT
            )
            page_elems.append(pe)
            
        predictor = ReadingOrderPredictor()
        # Predict order
        sorted_pe = predictor.predict_reading_order(page_elems)
        
        return [local_map[pe.cid] for pe in sorted_pe]

    roots = _predict_order(roots)
    for k in children:
        children[k] = _predict_order(children[k])

    order: List[int] = []
    visited = set()

    def dfs(i: int):
        if i in visited:
            return  # guard against accidental cycles
        visited.add(i)
        order.append(i)  # Always include — viewport filtering is handled by the caller

        for c in children.get(i, []):
            dfs(c)

    for r in roots:
        dfs(r)

    # Repair any disconnected nodes (broken/missing hierarchy)
    if len(visited) != n:
        remaining = [i for i in range(n) if i not in visited]
        order.extend(_predict_order(remaining))

    return order


def _draw_arrow(draw: ImageDraw.ImageDraw, p0: tuple[float, float], p1: tuple[float, float], color: tuple, width: int = 2):
    draw.line((p0[0], p0[1], p1[0], p1[1]), fill=color, width=width)
    # Arrow head
    angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    head_len = 8 + width  # scale with width a bit
    head_angle = math.radians(24)
    left = (
        p1[0] - head_len * math.cos(angle - head_angle),
        p1[1] - head_len * math.sin(angle - head_angle),
    )
    right = (
        p1[0] - head_len * math.cos(angle + head_angle),
        p1[1] - head_len * math.sin(angle + head_angle),
    )
    draw.polygon([p1, left, right], fill=color)


def _render_reading_order_image(
    im: "Image.Image",
    elems: List[Dict[str, Any]],
    order: List[int],
    leaf_only: bool,
    min_box_area: int,
) -> "Image.Image":
    """Render reading-order boxes and arrows onto a copy of *im* and return it."""

    palette = [
        (255, 0, 0),
        (0, 128, 0),
        (0, 0, 255),
        (255, 140, 0),
        (128, 0, 128),
        (0, 191, 255),
        (255, 0, 255),
        (139, 69, 19),
        (46, 139, 87),
        (75, 0, 130),
    ]
    color_arrow = (220, 40, 60)
    line_width = 2

    # Identify which elements are leaves (have no children that are also in the list)
    is_leaf = [True] * len(elems)
    if leaf_only:
        has_parent_index = any("parent_index" in el for el in elems)
        if has_parent_index:
            for idx, el in enumerate(elems):
                p = el.get("parent_index")
                if isinstance(p, int) and 0 <= p < len(elems) and p != idx:
                    is_leaf[p] = False

    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(13)

    centers: List[Optional[tuple[float, float]]] = []
    drawn_idx = 0

    for el_idx in order:
        if leaf_only and not is_leaf[el_idx]:
            continue

        el = elems[el_idx]
        r = el["rect"]
        if _should_skip(r, min_box_area):
            centers.append(None)
            continue

        drawn_idx += 1
        color = palette[(drawn_idx - 1) % len(palette)]
        x1, y1, x2, y2 = _rect_to_xyxy(r)

        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        _draw_label(draw, (x1 + 1, max(0, y1 - 18)), str(drawn_idx), color, font)
        centers.append(_center_of_rect(r))

    # Arrows between consecutive drawn elements
    prev_center = None
    for c in centers:
        if c is None:
            continue
        if prev_center is not None:
            _draw_arrow(draw, prev_center, c, color_arrow, width=line_width)
        prev_center = c

    return Image.alpha_composite(im, overlay).convert("RGB")


def visualize_reading_order_flat(
    image_path: str,
    elements_path: str,
    out_path: str,
    min_box_area: int = 9,
):
    """Draw a reading-order visualization for a flat, pre-ordered element list.

    Unlike :func:`visualize_reading_order`, this does **not** re-compute the
    reading order from the element hierarchy.  Instead it uses the
    ``reading_order_index`` field already annotated on each element, producing
    a stable, correctly-ordered rendering even when the elements have no
    meaningful parent/child relationships (e.g. the leaf coverage set).

    Saves a single image at *out_path*.
    """
    if not (os.path.exists(image_path) and os.path.exists(elements_path)):
        return

    elems: List[Dict[str, Any]] = load_json(elements_path)
    if not elems:
        return

    # Order by reading_order_index; fall back to list position for any element
    # that lacks the annotation.
    order = sorted(
        range(len(elems)),
        key=lambda i: elems[i].get("reading_order_index", i),
    )

    im = Image.open(image_path).convert("RGBA")
    ensure_dir(os.path.dirname(out_path))

    result = _render_reading_order_image(
        im, elems, order, leaf_only=False, min_box_area=min_box_area
    )
    result.save(out_path, quality=95)


def visualize_reading_order(
    image_path: str,
    elements_path: str,
    out_path: str,
    min_box_area: int = 9,
):
    """Draw reading-order visualizations over the screenshot.

    Saves two images:
    - *out_path*               — leaf elements only (cleaner, less cluttered)
    - *out_path* with ``_all`` suffix — all elements including containers

    Both images are numbered in reading order and connected with red arrows.
    """
    if not (os.path.exists(image_path) and os.path.exists(elements_path)):
        return

    elems: List[Dict[str, Any]] = load_json(elements_path)
    if not elems:
        return

    im = Image.open(image_path).convert("RGBA")
    page_size = Size(width=im.width, height=im.height)

    order = _build_reading_order(elems, page_size)
    if not order:
        return

    ensure_dir(os.path.dirname(out_path))

    # Version 1: leaf elements only
    leaf_img = _render_reading_order_image(im, elems, order, leaf_only=True, min_box_area=min_box_area)
    leaf_img.save(out_path, quality=95)

    # Version 2: all elements (including containers)
    base, ext = os.path.splitext(out_path)
    all_path = f"{base}_all{ext}"
    all_img = _render_reading_order_image(im, elems, order, leaf_only=False, min_box_area=min_box_area)
    all_img.save(all_path, quality=95)


