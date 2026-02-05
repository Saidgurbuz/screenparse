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
    labels: List[Tuple[int,str,bool]],
    out_path: str,
    line_w: int = 2,
) -> None:
    # labels: list of (index, "Label", "interactable")
    idx2lab = {i:(lab,interactable) for (i,lab,interactable) in labels}
    im = Image.open(page_png).convert("RGB")
    draw = ImageDraw.Draw(im, "RGBA")
    font = _load_font(16)

    img_w, img_h = im.size
    for i, el in enumerate(elements):
        if i not in idx2lab: continue
        lab, interactable = idx2lab[i]
        r = el.get("rect") or {}
        x = int(r.get("x",0)); y = int(r.get("y",0)); w = int(r.get("w",0)); h = int(r.get("h",0))
        if w < 1 or h < 1: continue
        color = _hash_color(lab)
        
        # Add a subtle fill overlay for interactable elements
        if interactable:
            draw.rectangle([x, y, x+w, y+h], fill=color + (70,))
        
        # box - use thicker line for interactable elements
        outline_width = line_w + 2 if interactable else line_w
        draw.rectangle([x, y, x+w, y+h], outline=color + (255,), width=outline_width)
        
        # label bg
        label_text = f"[⚡] {lab}" if interactable else lab
        tw, th = draw.textbbox((0,0), label_text, font=font)[2:]
        pad = 3
        
        # Clamp label position to stay within image bounds
        label_x = max(0, min(x, img_w - tw - 2*pad))
        label_y = y - th - 2*pad
        # If label would go above image, place it below the box instead
        if label_y < 0:
            label_y = min(y + h, img_h - th - 2*pad)
        
        bg = (color[0], color[1], color[2], 180)
        draw.rectangle([label_x, label_y, label_x + tw + 2*pad, label_y + th + 2*pad], fill=bg)
        draw.text((label_x + pad, label_y + pad), label_text, font=font, fill=(0,0,0,255))
    im.save(out_path, quality=92)