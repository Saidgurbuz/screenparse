"""Export annotations in YOLO format."""

import os
import random
import shutil
from typing import List, Dict, Any, Tuple
from pathlib import Path
from .utils import ensure_dir, save_json, load_json


# Simplified YOLO class set - visually distinct UI elements
YOLO_CLASSES = [
    "button",
    "input",
    "text",
    "image",
    "icon",
    "link",
    "heading",
    "navigation",
    "card",
    "list",
    "table",
    "form",
    "video",
    "checkbox",
    "dropdown",
    "search",
    "menu",
    "footer",
    "header",
    "logo",
    "ad",
    "badge",
    "tooltip",
    "modal",
    "tab",
]


def map_to_yolo_class(
    element_type: str, tag: str, attrs: Dict[str, str], inner_text: str = ""
) -> str:
    """
    Map detailed element type to simplified YOLO class.
    Enhanced with better heuristics and text analysis.
    """
    element_type = (element_type or "").lower()
    tag = (tag or "").lower()
    classes = (attrs.get("class") or "").lower()
    elem_id = (attrs.get("id") or "").lower()
    text = (inner_text or "").lower().strip()

    # ============================================
    # INTERACTIVE ELEMENTS
    # ============================================
    if element_type == "button":
        return "button"

    if element_type == "link":
        # Check if it's styled as a button
        if "btn" in classes or "button" in classes:
            return "button"
        return "link"

    if element_type in {"input", "textarea"}:
        # Distinguish search from regular input
        if "search" in classes or "search" in elem_id or attrs.get("type") == "search":
            return "search"
        return "input"

    if element_type in {"checkbox", "radio", "switch"}:
        return "checkbox"

    if element_type == "select":
        return "dropdown"

    if element_type == "tab":
        return "tab"

    # ============================================
    # CONTENT ELEMENTS
    # ============================================
    if element_type == "title":
        return "heading"

    if element_type in {"paragraph", "code"}:
        return "text"

    if element_type == "image":
        # Check if it's a logo
        if "logo" in classes or "logo" in elem_id or "brand" in classes:
            return "logo"
        return "image"

    if element_type in {"icon", "svg"}:
        # Large SVGs might be images
        # This will be size-checked in filtering
        return "icon"

    if element_type in {"video", "audio"}:
        return "video"

    # ============================================
    # DATA STRUCTURES
    # ============================================
    if element_type == "table":
        return "table"

    if element_type in {"list", "list_item"}:
        return "list"

    if element_type == "form":
        return "form"

    # ============================================
    # STRUCTURAL
    # ============================================
    if element_type == "nav" or element_type == "breadcrumb":
        return "navigation"

    if element_type == "header":
        return "header"

    if element_type == "footer":
        return "footer"

    # ============================================
    # UI COMPONENTS
    # ============================================
    if element_type == "card":
        return "card"

    if element_type == "modal":
        return "modal"

    if element_type == "tooltip":
        return "tooltip"

    if element_type in {"badge", "chip"}:
        return "badge"

    if element_type == "ad":
        return "ad"

    # ============================================
    # GENERIC/CONTAINER -> Try to infer from context
    # ============================================
    if element_type in {"section", "article", "aside", "main", "div", "container"}:
        # Check classes/id for hints
        if "card" in classes or "panel" in classes or "tile" in classes:
            return "card"
        if "nav" in classes or "menu" in classes:
            return "navigation"
        if "header" in classes or "masthead" in classes:
            return "header"
        if "footer" in classes:
            return "footer"

        # Check if it contains button-like text
        if text and len(text) < 30:
            button_words = [
                "click",
                "submit",
                "send",
                "go",
                "search",
                "buy",
                "add",
                "delete",
            ]
            if any(word in text for word in button_words):
                return "button"

        # Default to text for content containers
        return "text"

    # ============================================
    # FALLBACK
    # ============================================
    if element_type == "figure":
        return "image"

    if element_type == "pagination":
        return "navigation"

    # Final fallback
    return "text"


def bbox_to_yolo_format(
    rect: Dict[str, int], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """
    Convert absolute bbox to YOLO format (normalized center_x, center_y, width, height).

    Returns: (center_x, center_y, width, height) all in [0, 1]
    """
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]

    # Convert to center coordinates
    center_x = x + w / 2.0
    center_y = y + h / 2.0

    # Normalize to [0, 1]
    center_x_norm = center_x / img_width
    center_y_norm = center_y / img_height
    width_norm = w / img_width
    height_norm = h / img_height

    # Clamp to valid range
    center_x_norm = max(0.0, min(1.0, center_x_norm))
    center_y_norm = max(0.0, min(1.0, center_y_norm))
    width_norm = max(0.0, min(1.0, width_norm))
    height_norm = max(0.0, min(1.0, height_norm))

    return center_x_norm, center_y_norm, width_norm, height_norm


def export_yolo_dataset(
    raw_dir: str,
    yolo_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Export collected data to YOLO format with train/val/test splits.

    Directory structure:
        yolo_dir/
            images/
                train/
                val/
                test/
            labels/
                train/
                val/
                test/
            data.yaml
            classes.txt
    """
    random.seed(seed)

    # Create directory structure
    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(yolo_dir, "images", split))
        ensure_dir(os.path.join(yolo_dir, "labels", split))

    # Find all collected pages
    records = []
    for fname in os.listdir(raw_dir):
        if fname.endswith(".meta.json"):
            stem = fname.replace(".meta.json", "")
            img_path = os.path.join(raw_dir, f"{stem}.png")
            elements_path = os.path.join(raw_dir, f"{stem}.elements.json")

            if os.path.exists(img_path) and os.path.exists(elements_path):
                records.append(
                    {
                        "stem": stem,
                        "image": img_path,
                        "elements": elements_path,
                        "meta": os.path.join(raw_dir, fname),
                    }
                )

    if not records:
        print("No records found in", raw_dir)
        return

    # Shuffle and split
    random.shuffle(records)
    n = len(records)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": records[:n_train],
        "val": records[n_train : n_train + n_val],
        "test": records[n_train + n_val :],
    }

    # Build class to ID mapping
    class_to_id = {cls: idx for idx, cls in enumerate(YOLO_CLASSES)}

    # Process each split
    stats = {"train": 0, "val": 0, "test": 0}
    class_counts = {cls: 0 for cls in YOLO_CLASSES}

    for split_name, split_records in splits.items():
        print(f"\nProcessing {split_name} split ({len(split_records)} images)...")

        for rec in split_records:
            try:
                # Load data
                elements = load_json(rec["elements"])
                meta = load_json(rec["meta"])

                # Get image dimensions (viewport-only now)
                img_w = meta["viewport"]["w"]
                img_h = meta["viewport"]["h"]

                # Copy image (convert PNG to JPG for YOLO)
                img_dest = os.path.join(
                    yolo_dir, "images", split_name, f"{rec['stem']}.jpg"
                )

                # Convert PNG to JPG
                from PIL import Image

                img = Image.open(rec["image"])
                if img.mode == "RGBA":
                    # Convert RGBA to RGB
                    img = img.convert("RGB")
                img.save(img_dest, "JPEG", quality=95)

                # Convert annotations
                label_lines = []
                for el in elements:
                    el_type = el.get("type", "unknown")
                    tag = el.get("tag", "")
                    attrs = el.get("attrs", {})
                    inner_text = el.get("inner_text", "")

                    yolo_class = map_to_yolo_class(el_type, tag, attrs, inner_text)

                    if yolo_class not in class_to_id:
                        continue

                    class_id = class_to_id[yolo_class]
                    cx, cy, w, h = bbox_to_yolo_format(el["rect"], img_w, img_h)

                    # Skip degenerate boxes
                    if w <= 0 or h <= 0:
                        continue

                    # YOLO format: class_id center_x center_y width height
                    label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    class_counts[yolo_class] += 1

                # Write label file
                label_dest = os.path.join(
                    yolo_dir, "labels", split_name, f"{rec['stem']}.txt"
                )
                with open(label_dest, "w") as f:
                    f.write("\n".join(label_lines))

                stats[split_name] += len(label_lines)

            except Exception as e:
                print(f"Error processing {rec['stem']}: {e}")

    # Write data.yaml
    yaml_content = f"""# YOLO Dataset Configuration - Viewport-Only Screenshots
path: {os.path.abspath(yolo_dir)}
train: images/train
val: images/val
test: images/test

# Classes
nc: {len(YOLO_CLASSES)}
names: {YOLO_CLASSES}
"""

    with open(os.path.join(yolo_dir, "data.yaml"), "w") as f:
        f.write(yaml_content)

    # Write classes.txt
    with open(os.path.join(yolo_dir, "classes.txt"), "w") as f:
        for cls in YOLO_CLASSES:
            f.write(f"{cls}\n")

    # Write detailed stats
    stats_content = "# Dataset Statistics\n\n"
    stats_content += f"Total images: {sum(len(s) for s in splits.values())}\n"
    stats_content += f"Total annotations: {sum(stats.values())}\n\n"

    stats_content += "## Split Distribution\n"
    for split, count in stats.items():
        stats_content += f"{split}: {len(splits[split])} images, {count} annotations\n"

    stats_content += "\n## Class Distribution\n"
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_classes:
        if count > 0:
            pct = 100 * count / sum(stats.values())
            stats_content += f"{cls}: {count} ({pct:.1f}%)\n"

    with open(os.path.join(yolo_dir, "STATS.txt"), "w") as f:
        f.write(stats_content)

    # Print summary
    print("\n" + "=" * 60)
    print("YOLO Dataset Export Complete!")
    print("=" * 60)
    print(f"Output directory: {yolo_dir}")
    print(f"Total images: {sum(len(s) for s in splits.values())}")
    print(f"  Train: {len(splits['train'])} images ({stats['train']} annotations)")
    print(f"  Val:   {len(splits['val'])} images ({stats['val']} annotations)")
    print(f"  Test:  {len(splits['test'])} images ({stats['test']} annotations)")
    print(f"Classes: {len(YOLO_CLASSES)}")
    print(f"\nTop 5 classes:")
    for cls, count in sorted_classes[:5]:
        pct = 100 * count / sum(stats.values()) if sum(stats.values()) > 0 else 0
        print(f"  {cls:15s}: {count:5d} ({pct:5.1f}%)")
    print(f"\nConfiguration saved to: {os.path.join(yolo_dir, 'data.yaml')}")
    print(f"Statistics saved to: {os.path.join(yolo_dir, 'STATS.txt')}")
