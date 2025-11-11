# -*- coding: utf-8 -*-
"""
Simple overlay for VLM labels on top of screenshot.
"""
from __future__ import annotations
import os, math, hashlib
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw, ImageFont

def _hash_color(text: str) -> Tuple[int,int,int]:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    # brighten a bit
    return (128 + r//2, 128 + g//2, 128 + b//2)

def _load_font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        # If you have DejaVuSans on your system this looks nice
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def draw_vlm_overlay(
    page_png: str,
    elements: List[Dict[str, Any]],
    labels: List[Tuple[int,str,float]],
    out_path: str,
    min_conf: float = 0.0,
    line_w: int = 2,
) -> None:
    # labels: list of (index, "Label", conf)
    idx2lab = {i:(lab,conf) for (i,lab,conf) in labels if conf >= min_conf}
    im = Image.open(page_png).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    font = _load_font(16)

    for i, el in enumerate(elements):
        if i not in idx2lab: continue
        lab, conf = idx2lab[i]
        r = el.get("rect") or {}
        x = int(r.get("x",0)); y = int(r.get("y",0)); w = int(r.get("w",0)); h = int(r.get("h",0))
        if w < 1 or h < 1: continue
        color = _hash_color(lab)
        # box
        draw.rectangle([x, y, x+w, y+h], outline=color + (255,), width=line_w)
        # label bg
        text = f"{lab} ({conf:.2f})"
        tw, th = draw.textbbox((0,0), text, font=font)[2:]
        pad = 3
        bg = (color[0], color[1], color[2], 180)
        draw.rectangle([x, y - th - 2*pad, x + tw + 2*pad, y], fill=bg)
        draw.text((x + pad, y - th - pad), text, font=font, fill=(0,0,0,255))
    im.save(out_path, quality=92)