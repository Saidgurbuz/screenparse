from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

from .types import EvaluationSample, UIElement


@dataclass(frozen=True)
class VizConfig:
    include_gt: bool = True
    include_pred: bool = True
    layout: str = "side_by_side"  # "side_by_side" or "overlay"
    style: str = "modern"  # "modern" or "legacy"
    gt_color: Tuple[int, int, int] = (0, 114, 178)
    pred_color: Tuple[int, int, int] = (213, 94, 0)
    line_width: int = 4
    fill_alpha: int = 48
    halo_color: Tuple[int, int, int] = (0, 0, 0)
    halo_width: int = 2
    label_max_len: int = 50
    label_font_size: int = 14
    label_pad: int = 3
    label_bg_alpha: int = 200
    label_text_color: Tuple[int, int, int] = (255, 255, 255)
    show_labels: bool = False
    title_font_size: int = 18
    title_pad: int = 6
    title_bg_alpha: int = 210
    title_text_color: Tuple[int, int, int] = (255, 255, 255)
    show_panel_titles: bool = True
    divider_color: Tuple[int, int, int] = (30, 30, 30)
    divider_width: int = 2
    jpeg_quality: int = 92


def legacy_defaults() -> VizConfig:
    return VizConfig(
        style="legacy",
        gt_color=(0, 160, 0),
        pred_color=(200, 30, 30),
        line_width=2,
        fill_alpha=0,
        halo_width=0,
        label_max_len=40,
        label_font_size=12,
        label_pad=2,
        label_bg_alpha=255,
        show_labels=True,
        show_panel_titles=False,
        divider_width=0,
        jpeg_quality=90,
    )


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def _text_size(draw, text: str, font) -> Tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return (max(0, bbox[2] - bbox[0]), max(0, bbox[3] - bbox[1]))
    except Exception:
        try:
            return draw.textsize(text, font=font)
        except Exception:
            return (max(1, len(text)) * 7, 12)


def _draw_elements(
    base,
    elements: Sequence[UIElement],
    color: Tuple[int, int, int],
    config: VizConfig,
):
    if not elements:
        return base

    from PIL import Image, ImageDraw

    base_rgba = base.convert("RGBA")
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(config.label_font_size)

    lw = max(1, int(config.line_width))
    halo = max(0, int(config.halo_width))
    r, g, b = color
    hr, hg, hb = config.halo_color
    for el in elements:
        x1, y1, x2, y2 = el.bbox.to_ltrb()
        x1 = max(0, min(int(x1), base_rgba.width - 1))
        y1 = max(0, min(int(y1), base_rgba.height - 1))
        x2 = max(0, min(int(x2), base_rgba.width - 1))
        y2 = max(0, min(int(y2), base_rgba.height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        if config.fill_alpha > 0:
            draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, config.fill_alpha))
        if halo > 0:
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=(hr, hg, hb, 255),
                width=lw + (2 * halo),
            )
        draw.rectangle([x1, y1, x2, y2], outline=(r, g, b, 255), width=lw)
        if config.show_labels:
            label = (el.label or "").strip()
            if label:
                label = label[: config.label_max_len]
                if font:
                    text_w, text_h = _text_size(draw, label, font)
                    pad = max(1, int(config.label_pad))
                    lx = x1
                    ly = y1 - (text_h + 2 * pad)
                    if ly < 0:
                        ly = y1
                    if lx + text_w + 2 * pad > base_rgba.width:
                        lx = max(0, base_rgba.width - (text_w + 2 * pad))
                    if ly + text_h + 2 * pad > base_rgba.height:
                        ly = max(0, base_rgba.height - (text_h + 2 * pad))
                    rect = (lx, ly, lx + text_w + 2 * pad, ly + text_h + 2 * pad)
                    draw.rectangle(rect, fill=(r, g, b, config.label_bg_alpha))
                    draw.rectangle(rect, outline=(hr, hg, hb, 255), width=1)
                    draw.text(
                        (lx + pad, ly + pad),
                        label,
                        fill=(*config.label_text_color, 255),
                        font=font,
                    )
    base_rgba.alpha_composite(overlay)
    return base_rgba.convert("RGB")


def _draw_panel_title(base, text: str, color: Tuple[int, int, int], config: VizConfig):
    from PIL import Image, ImageDraw

    base_rgba = base.convert("RGBA")
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(config.title_font_size, bold=True) or _load_font(config.title_font_size)
    pad = max(1, int(config.title_pad))
    r, g, b = color
    text_w, text_h = _text_size(draw, text, font)
    x = pad
    y = pad
    rect = (x - pad, y - pad, x + text_w + pad, y + text_h + pad)
    draw.rectangle(rect, fill=(r, g, b, config.title_bg_alpha))
    draw.rectangle(rect, outline=(0, 0, 0, 255), width=1)
    draw.text((x, y), text, fill=(*config.title_text_color, 255), font=font)
    base_rgba.alpha_composite(overlay)
    return base_rgba.convert("RGB")


def render_viz(
    image_path: str,
    ground_truth: Sequence[UIElement],
    predictions: Sequence[UIElement],
    config: VizConfig,
):
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    if config.style == "legacy":
        return _render_viz_legacy(img, ground_truth, predictions, config)

    include_gt = config.include_gt
    include_pred = config.include_pred
    if include_gt and include_pred and config.layout == "side_by_side":
        gt_img = _draw_elements(img.copy(), ground_truth, config.gt_color, config)
        pred_img = _draw_elements(img.copy(), predictions, config.pred_color, config)
        if config.show_panel_titles:
            gt_img = _draw_panel_title(gt_img, "Ground Truth", config.gt_color, config)
            pred_img = _draw_panel_title(pred_img, "Predictions", config.pred_color, config)
        combined = Image.new("RGB", (img.width * 2, img.height), (255, 255, 255))
        combined.paste(gt_img, (0, 0))
        combined.paste(pred_img, (img.width, 0))
        if config.divider_width > 0:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(combined)
            x = img.width
            half = max(0, int(config.divider_width) // 2)
            for dx in range(-half, half + 1):
                draw.line(
                    [(x + dx, 0), (x + dx, img.height)],
                    fill=config.divider_color,
                    width=1,
                )
        return combined

    out = img.copy()
    if include_gt:
        out = _draw_elements(out, ground_truth, config.gt_color, config)
    if include_pred:
        out = _draw_elements(out, predictions, config.pred_color, config)
    if config.show_panel_titles:
        if include_gt and include_pred:
            title = "GT + Pred"
            title_color = config.pred_color
        elif include_gt:
            title = "Ground Truth"
            title_color = config.gt_color
        else:
            title = "Predictions"
            title_color = config.pred_color
        out = _draw_panel_title(out, title, title_color, config)
    return out


def _render_viz_legacy(
    img,
    ground_truth: Sequence[UIElement],
    predictions: Sequence[UIElement],
    config: VizConfig,
):
    from PIL import Image, ImageDraw, ImageFont

    def _draw(base, elements: Sequence[UIElement], color: Tuple[int, int, int]):
        draw = ImageDraw.Draw(base)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        lw = max(1, int(config.line_width))
        for el in elements:
            x1, y1, x2, y2 = el.bbox.to_ltrb()
            x1 = max(0, min(int(x1), base.width - 1))
            y1 = max(0, min(int(y1), base.height - 1))
            x2 = max(0, min(int(x2), base.width - 1))
            y2 = max(0, min(int(y2), base.height - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            if config.show_labels:
                label = (el.label or "").strip()
                if label and font:
                    label = label[: config.label_max_len]
                    try:
                        text_bbox = draw.textbbox((x1, y1), label, font=font)
                    except Exception:
                        w, h = draw.textsize(label, font=font)
                        text_bbox = (x1, y1, x1 + w, y1 + h)
                    draw.rectangle(text_bbox, fill=color)
                    draw.text((x1, y1), label, fill="white", font=font)

    include_gt = config.include_gt
    include_pred = config.include_pred
    if include_gt and include_pred and config.layout == "side_by_side":
        gt_img = img.copy()
        pred_img = img.copy()
        _draw(gt_img, ground_truth, config.gt_color)
        _draw(pred_img, predictions, config.pred_color)
        combined = Image.new("RGB", (img.width * 2, img.height), (255, 255, 255))
        combined.paste(gt_img, (0, 0))
        combined.paste(pred_img, (img.width, 0))
        return combined

    out = img.copy()
    if include_gt:
        _draw(out, ground_truth, config.gt_color)
    if include_pred:
        _draw(out, predictions, config.pred_color)
    return out


def save_viz(
    out_dir: Path,
    sample: EvaluationSample,
    predictions: Sequence[UIElement],
    config: VizConfig,
    suffix: str = ".viz.jpg",
) -> Optional[Path]:
    if not config.include_gt and not config.include_pred:
        return None

    img = render_viz(sample.image_path, sample.ground_truth, predictions, config)
    if img is None:
        return None

    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    stem = sample.sample_id or Path(sample.image_path).stem
    out_path = viz_dir / f"{stem}{suffix}"
    img.save(out_path, quality=config.jpeg_quality)
    return out_path
