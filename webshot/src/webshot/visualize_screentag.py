# src/webshot/visualize_screentag.py
"""
Visualization tool for ScreenTag representations.

This module provides functionality to render ScreenTag annotations
on screenshot images for visual inspection and debugging.

Usage:
    # Single file
    wsd viz-screentag --image path/to/image.png --screentag path/to/image.screentag.txt

    # Batch processing
    wsd viz-screentag --raw-dir data/raw --viz-dir data/viz_screentag
"""

import os
import re
import glob
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .utils import ensure_dir


# ─────────────────────────────────────────────────────────────────────────────
# Font loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a font, with fallbacks for different systems."""
    font_paths = [
        "/System/Library/Fonts/SFNS.ttf",  # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/usr/share/fonts/TTF/DejaVuSans.ttf",  # Arch Linux
        "C:/Windows/Fonts/arial.ttf",  # Windows
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ─────────────────────────────────────────────────────────────────────────────
# Color generation
# ─────────────────────────────────────────────────────────────────────────────

# Predefined color palette for common element types
ELEMENT_COLORS = {
    # Interactive elements - warm colors
    "button": (220, 80, 60),
    "utility_button": (200, 100, 80),
    "link": (60, 120, 220),
    "tab": (100, 160, 200),
    
    # Text elements - cool colors
    "text": (80, 80, 80),
    "heading": (40, 40, 120),
    "code_snippet": (60, 60, 60),
    
    # Input elements - green tones
    "input": (80, 160, 80),
    "text_input": (80, 160, 80),
    "search": (60, 140, 100),
    "search_field": (60, 140, 100),
    "dropdown": (100, 180, 120),
    "select": (100, 180, 120),
    "checkbox": (120, 180, 100),
    "radiobox": (120, 180, 100),
    "switch": (100, 200, 150),
    
    # Container elements - muted colors
    "navigation": (180, 140, 100),
    "menu": (160, 120, 80),
    "form": (140, 140, 180),
    "list": (140, 160, 140),
    "table": (160, 160, 160),
    
    # Media elements - purple tones
    "image": (160, 100, 180),
    "video": (140, 80, 160),
    "icon": (180, 120, 200),
    "logo": (200, 140, 220),
    "avatar": (180, 100, 180),
    
    # Misc
    "badge": (220, 160, 60),
    "tooltip": (100, 100, 100),
    "alert": (220, 120, 60),
    "notification": (220, 140, 80),
}


def _get_element_color(tag: str) -> Tuple[int, int, int]:
    """Get color for an element type, with fallback to hash-based color."""
    tag_lower = tag.lower().replace("-", "_").replace(" ", "_")
    
    if tag_lower in ELEMENT_COLORS:
        return ELEMENT_COLORS[tag_lower]
    
    # Hash-based fallback for unknown types
    import random
    rnd = random.Random(hash(tag) & 0xFFFFFFFF)
    return tuple(rnd.randint(60, 200) for _ in range(3))


# ─────────────────────────────────────────────────────────────────────────────
# ScreenTag parsing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScreenTagElement:
    """Parsed element from ScreenTag."""
    tag: str
    left: int
    top: int
    right: int
    bottom: int
    text: str
    children: List["ScreenTagElement"] = field(default_factory=list)
    depth: int = 0


def parse_screentag(screentag_content: str, viewport_w: int, viewport_h: int) -> List[ScreenTagElement]:
    """
    Parse ScreenTag content into a list of elements with absolute coordinates.
    
    Args:
        screentag_content: Raw ScreenTag string
        viewport_w: Viewport width for denormalization
        viewport_h: Viewport height for denormalization
    
    Returns:
        List of parsed ScreenTagElement objects (flat list)
    """
    elements = []
    
    # Pattern to match location tokens
    loc_pattern = re.compile(r'<loc_(\d+)>')
    
    # Pattern to match element tags with their content
    # This regex finds opening tags and their positions
    tag_pattern = re.compile(r'<([a-zA-Z0-9_-]+)>(<loc_\d+>){4}')
    
    def denormalize(loc_val: int, size: int, norm_size: int = 500) -> int:
        """Convert normalized location (0-500) back to pixel coordinates."""
        return int((loc_val / norm_size) * size)
    
    def parse_element(content: str, depth: int = 0) -> Tuple[Optional[ScreenTagElement], str]:
        """
        Parse a single element from the beginning of content.
        Returns the element and remaining content.
        """
        content = content.strip()
        if not content or content.startswith('</'):
            return None, content
        
        # Match opening tag
        match = re.match(r'<([a-zA-Z0-9_-]+)>', content)
        if not match:
            return None, content
        
        tag = match.group(1)
        
        # Skip screentag wrapper
        if tag == 'screentag':
            inner = content[len('<screentag>'):]
            if inner.endswith('</screentag>'):
                inner = inner[:-len('</screentag>')]
            # Parse all root elements
            root_elements = []
            while inner.strip():
                el, inner = parse_element(inner, depth)
                if el:
                    root_elements.append(el)
                elif not inner.strip():
                    break
                else:
                    # Skip unrecognized content
                    inner = inner[1:] if inner else ""
            return None, ""  # Return elements via side effect
        
        content = content[match.end():]
        
        # Extract 4 location tokens
        locs = []
        for _ in range(4):
            loc_match = loc_pattern.match(content)
            if not loc_match:
                return None, content
            locs.append(int(loc_match.group(1)))
            content = content[loc_match.end():]
        
        left = denormalize(locs[0], viewport_w)
        top = denormalize(locs[1], viewport_h)
        right = denormalize(locs[2], viewport_w)
        bottom = denormalize(locs[3], viewport_h)
        
        # Find the closing tag for this element
        closing_tag = f'</{tag}>'
        
        # Extract content until closing tag (handling nested tags)
        nesting = 1
        pos = 0
        while pos < len(content) and nesting > 0:
            # Check for opening tag of same type
            open_match = re.match(rf'<{re.escape(tag)}>', content[pos:])
            if open_match:
                nesting += 1
                pos += open_match.end()
                continue
            
            # Check for closing tag
            close_match = re.match(rf'</{re.escape(tag)}>', content[pos:])
            if close_match:
                nesting -= 1
                if nesting == 0:
                    break
                pos += close_match.end()
                continue
            
            pos += 1
        
        inner_content = content[:pos]
        remaining = content[pos + len(closing_tag):] if pos < len(content) else ""
        
        # Extract text (content before first child tag)
        text = ""
        text_match = re.match(r'^([^<]*)', inner_content)
        if text_match:
            text = text_match.group(1).strip()
        
        # Parse children
        children = []
        child_content = inner_content[len(text):] if text else inner_content
        while child_content.strip():
            child, child_content = parse_element(child_content, depth + 1)
            if child:
                children.append(child)
            elif not child_content.strip():
                break
            else:
                # Skip to next tag
                skip_match = re.search(r'<[a-zA-Z0-9_-]+>', child_content)
                if skip_match:
                    child_content = child_content[skip_match.start():]
                else:
                    break
        
        element = ScreenTagElement(
            tag=tag,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            text=text,
            children=children,
            depth=depth,
        )
        
        return element, remaining
    
    # Parse root elements from screentag
    content = screentag_content.strip()
    if content.startswith('<screentag>'):
        content = content[len('<screentag>'):]
    if content.endswith('</screentag>'):
        content = content[:-len('</screentag>')]
    
    # Parse all root elements
    while content.strip():
        el, content = parse_element(content, depth=0)
        if el:
            elements.append(el)
        elif not content.strip():
            break
        else:
            # Skip unrecognized content
            skip_match = re.search(r'<[a-zA-Z0-9_-]+>', content)
            if skip_match:
                content = content[skip_match.start():]
            else:
                break
    
    # Flatten the tree for drawing (depth-first)
    def flatten(el_list: List[ScreenTagElement]) -> List[ScreenTagElement]:
        result = []
        for el in el_list:
            result.append(el)
            result.extend(flatten(el.children))
        return result
    
    return flatten(elements)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScreenTagVizOptions:
    """Options for ScreenTag visualization."""
    draw_boxes: bool = True
    draw_labels: bool = True
    draw_text: bool = True
    fill_opacity: int = 15  # Box fill opacity (0-255), low for transparency
    outline_opacity: int = 200  # Outline opacity (0-255), high for visibility
    line_width: int = 2
    max_text_len: int = 50
    min_box_area: int = 10 * 10  # Skip tiny elements
    font_size: int = 11
    label_font_size: int = 10
    label_opacity: int = 100  # Label background opacity (semi-transparent)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    bg_color: Tuple[int, int, int],
    font: ImageFont.ImageFont,
    opacity: int = 100,
):
    """Draw a label with semi-transparent background at the specified position."""
    margin_x, margin_y = 2, 1
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    x1, y1 = xy
    x2, y2 = x1 + tw + 2 * margin_x, y1 + th + 2 * margin_y
    
    # Semi-transparent background so overlapping elements are visible
    draw.rectangle((x1, y1, x2, y2), fill=(*bg_color, opacity))
    # Text with slight transparency for softer look
    draw.text((x1 + margin_x, y1 + margin_y), text, fill=(255, 255, 255, 220), font=font)


def visualize_screentag(
    image_path: str,
    screentag_path: str,
    output_path: str,
    viewport: Optional[Dict[str, Any]] = None,
    opts: ScreenTagVizOptions = None,
) -> bool:
    """
    Visualize ScreenTag annotations on an image.
    
    Args:
        image_path: Path to the source PNG image
        screentag_path: Path to the .screentag.txt file
        output_path: Path for the output visualization
        viewport: Optional viewport dict with 'w' and 'h' keys.
                  If not provided, uses image dimensions.
        opts: Visualization options
    
    Returns:
        True if successful, False otherwise
    """
    if opts is None:
        opts = ScreenTagVizOptions()
    
    try:
        # Load image
        im = Image.open(image_path).convert("RGBA")
        img_w, img_h = im.size
        
        # Determine viewport dimensions
        if viewport:
            vp_w = viewport.get("w", img_w)
            vp_h = viewport.get("h", img_h)
        else:
            vp_w, vp_h = img_w, img_h
        
        # Load ScreenTag content
        with open(screentag_path, "r", encoding="utf-8") as f:
            screentag_content = f.read()
        
        # Parse ScreenTag
        elements = parse_screentag(screentag_content, vp_w, vp_h)
        
        if not elements:
            print(f"Warning: No elements parsed from {screentag_path}")
            return False
        
        # Create overlay
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Load fonts
        font_label = _load_font(opts.label_font_size)
        font_text = _load_font(opts.font_size)
        
        # Draw elements (in reverse order so parents are behind children)
        for el in reversed(elements):
            # Skip tiny elements
            area = (el.right - el.left) * (el.bottom - el.top)
            if area < opts.min_box_area:
                continue
            
            color = _get_element_color(el.tag)
            x1, y1, x2, y2 = el.left, el.top, el.right, el.bottom
            
            # Clamp to image bounds
            x1 = max(0, min(x1, img_w - 1))
            y1 = max(0, min(y1, img_h - 1))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            if opts.draw_boxes:
                # Draw very light semi-transparent fill first (so overlaps are visible)
                if opts.fill_opacity > 0:
                    draw.rectangle((x1, y1, x2, y2), fill=(*color, opts.fill_opacity))
                
                # Draw solid outline on top for visibility
                # Use multiple passes for thicker, more visible lines
                outline_color = (*color, opts.outline_opacity)
                for offset in range(opts.line_width):
                    # Ensure the offset rectangle is still valid
                    ox1, oy1 = x1 + offset, y1 + offset
                    ox2, oy2 = x2 - offset, y2 - offset
                    if ox2 > ox1 and oy2 > oy1:
                        draw.rectangle((ox1, oy1, ox2, oy2), outline=outline_color, width=1)
            
            if opts.draw_labels:
                # Build label
                label = el.tag
                if el.text and opts.draw_text:
                    text_preview = el.text[:opts.max_text_len]
                    if len(el.text) > opts.max_text_len:
                        text_preview += "…"
                    label = f"{el.tag}: {text_preview}"
                
                # Draw label above the box
                label_y = max(0, y1 - 14)
                _draw_label(draw, (x1, label_y), label, color, font_label, opts.label_opacity)
        
        # Composite and save
        out = Image.alpha_composite(im, overlay).convert("RGB")
        ensure_dir(os.path.dirname(output_path) or ".")
        out.save(output_path, quality=95)
        
        return True
    
    except Exception as e:
        print(f"Error visualizing {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_screentag_single(args: Tuple[str, str, str, Optional[Dict], ScreenTagVizOptions]) -> Tuple[str, bool, str]:
    """Worker function for parallel processing."""
    image_path, screentag_path, output_path, viewport, opts = args
    try:
        success = visualize_screentag(image_path, screentag_path, output_path, viewport, opts)
        if success:
            return (image_path, True, "Success")
        else:
            return (image_path, False, "Failed to visualize")
    except Exception as e:
        return (image_path, False, str(e))


def visualize_screentag_directory(
    raw_dir: str,
    viz_dir: str,
    workers: int = 1,
    opts: ScreenTagVizOptions = None,
) -> Dict[str, Any]:
    """
    Visualize ScreenTag for all samples in a directory.
    
    Args:
        raw_dir: Directory containing .png and .screentag.txt files
        viz_dir: Output directory for visualizations
        workers: Number of parallel workers
        opts: Visualization options
    
    Returns:
        Statistics dictionary
    """
    from .utils import load_json
    
    if opts is None:
        opts = ScreenTagVizOptions()
    
    ensure_dir(viz_dir)
    
    # Find all screentag files
    screentag_files = sorted(glob.glob(os.path.join(raw_dir, "*.screentag.txt")))
    
    if not screentag_files:
        print(f"No .screentag.txt files found in {raw_dir}")
        return {"total": 0, "processed": 0, "errors": 0}
    
    print(f"Found {len(screentag_files)} screentag files")
    
    # Prepare tasks
    tasks = []
    for screentag_path in screentag_files:
        base = screentag_path.replace(".screentag.txt", "")
        image_path = f"{base}.png"
        meta_path = f"{base}.meta.json"
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for {screentag_path}")
            continue
        
        # Load viewport from meta if available
        viewport = None
        if os.path.exists(meta_path):
            try:
                meta = load_json(meta_path)
                viewport = meta.get("viewport", {})
            except Exception:
                pass
        
        # Output path
        output_name = os.path.basename(base) + ".screentag_viz.jpg"
        output_path = os.path.join(viz_dir, output_name)
        
        tasks.append((image_path, screentag_path, output_path, viewport, opts))
    
    stats = {
        "total": len(tasks),
        "processed": 0,
        "errors": 0,
        "error_files": [],
    }
    
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(visualize_screentag_single, t): t[0] for t in tasks}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Visualizing"):
                path, success, message = future.result()
                if success:
                    stats["processed"] += 1
                else:
                    stats["errors"] += 1
                    stats["error_files"].append((path, message))
    else:
        for task in tqdm(tasks, desc="Visualizing"):
            path, success, message = visualize_screentag_single(task)
            if success:
                stats["processed"] += 1
            else:
                stats["errors"] += 1
                stats["error_files"].append((path, message))
    
    return stats


def print_viz_stats(stats: Dict[str, Any]) -> None:
    """Print visualization statistics."""
    print("\n" + "=" * 60)
    print("SCREENTAG VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Total files:     {stats['total']}")
    print(f"Processed:       {stats['processed']}")
    print(f"Errors:          {stats['errors']}")
    
    if stats.get("error_files"):
        print("\nErrors:")
        for path, msg in stats["error_files"][:10]:
            print(f"  {os.path.basename(path)}: {msg}")
        if len(stats["error_files"]) > 10:
            print(f"  ... and {len(stats['error_files']) - 10} more")
