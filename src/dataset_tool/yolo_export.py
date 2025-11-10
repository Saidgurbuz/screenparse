"""Export annotations in YOLO format (parallel, streaming-friendly)."""

import os
import random
import hashlib
from typing import List, Dict, Any, Tuple, Iterable, Optional
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from .utils import ensure_dir, load_json

# ============================================================
# YOLO classes (unchanged)
# ============================================================
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
    """(unchanged)"""
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
        if "btn" in classes or "button" in classes:
            return "button"
        return "link"

    if element_type in {"input", "textarea"}:
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
        if "logo" in classes or "logo" in elem_id or "brand" in classes:
            return "logo"
        return "image"

    if element_type in {"icon", "svg"}:
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
        if "card" in classes or "panel" in classes or "tile" in classes:
            return "card"
        if "nav" in classes or "menu" in classes:
            return "navigation"
        if "header" in classes or "masthead" in classes:
            return "header"
        if "footer" in classes:
            return "footer"

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

        return "text"

    if element_type == "figure":
        return "image"

    if element_type == "pagination":
        return "navigation"

    return "text"


def bbox_to_yolo_format(
    rect: Dict[str, int], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """(unchanged)"""
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    center_x_norm = max(0.0, min(1.0, center_x / img_width))
    center_y_norm = max(0.0, min(1.0, center_y / img_height))
    width_norm = max(0.0, min(1.0, w / img_width))
    height_norm = max(0.0, min(1.0, h / img_height))
    return center_x_norm, center_y_norm, width_norm, height_norm


# ============================================================
# Parallel/export infrastructure
# ============================================================


def _choose_split(
    stem: str, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
) -> str:
    """
    Deterministic split selection using MD5 hash(seed + '/' + stem) -> [0,1).
    Avoids needing to load all records into memory.
    """
    assert (
        abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6
    ), "Split ratios must sum to 1.0"
    h = hashlib.md5(f"{seed}/{stem}".encode("utf-8")).hexdigest()
    # Take first 15 hex chars -> int -> normalize (good enough uniformity)
    r = int(h[:15], 16) / float(16**15)
    if r < train_ratio:
        return "train"
    elif r < train_ratio + val_ratio:
        return "val"
    else:
        return "test"


def _scan_records(raw_dir: str) -> Iterable[Dict[str, str]]:
    """
    Streaming, single-pass directory scan using os.scandir (fast, low memory).
    Yields dicts with paths for each valid record (meta+image+elements present).
    """
    # Expect files side-by-side: <stem>.meta.json, <stem>.png, <stem>.elements.json
    with os.scandir(raw_dir) as it:
        for entry in it:
            # Extremely cheap filter: suffix match only (no stat call)
            name = entry.name
            if not name.endswith(".meta.json"):
                continue
            stem = name[:-10]  # remove '.meta.json'
            img_path = os.path.join(raw_dir, f"{stem}.png")
            elements_path = os.path.join(raw_dir, f"{stem}.elements.json")
            meta_path = os.path.join(raw_dir, name)
            if os.path.exists(img_path) and os.path.exists(elements_path):
                yield {
                    "stem": stem,
                    "image": img_path,
                    "elements": elements_path,
                    "meta": meta_path,
                }


# Globals set in worker processes by _init_worker
_G = {
    "yolo_dir": None,
    "class_to_id": None,
    "seed": None,
    "ratios": None,
}


def _init_worker(
    yolo_dir: str,
    class_to_id: Dict[str, int],
    seed: int,
    ratios: Tuple[float, float, float],
):
    _G["yolo_dir"] = yolo_dir
    _G["class_to_id"] = class_to_id
    _G["seed"] = seed
    _G["ratios"] = ratios


def _process_one(
    rec: Dict[str, str],
) -> Tuple[bool, str, int, Dict[str, int], str, Optional[str]]:
    """
    Worker: converts one PNG to JPG, writes label file, returns stats.

    Returns:
        (ok, split_name, num_annotations, class_counts_dict, stem, error_message_if_any)
    """
    try:
        from PIL import Image  # import inside worker to avoid preload cost in parent

        # Decide split deterministically
        train_ratio, val_ratio, test_ratio = _G["ratios"]
        split = _choose_split(
            rec["stem"], train_ratio, val_ratio, test_ratio, _G["seed"]
        )

        # Load data
        elements = load_json(rec["elements"])
        meta = load_json(rec["meta"])
        img_w = meta["viewport"]["w"]
        img_h = meta["viewport"]["h"]

        # Convert image to JPG
        img_dest = os.path.join(_G["yolo_dir"], "images", split, f"{rec['stem']}.jpg")
        # Avoid partial writes
        tmp_img_dest = img_dest + ".tmp"

        with Image.open(rec["image"]) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(tmp_img_dest, "JPEG", quality=95, optimize=True)
        os.replace(tmp_img_dest, img_dest)

        # Convert annotations
        class_counts_local = {cls: 0 for cls in YOLO_CLASSES}
        label_lines: List[str] = []
        for el in elements:
            el_type = el.get("type", "unknown")
            tag = el.get("tag", "")
            attrs = el.get("attrs", {})
            inner_text = el.get("inner_text", "")

            yolo_class = map_to_yolo_class(el_type, tag, attrs, inner_text)
            if yolo_class not in _G["class_to_id"]:
                continue

            class_id = _G["class_to_id"][yolo_class]
            cx, cy, w, h = bbox_to_yolo_format(el["rect"], img_w, img_h)
            if w <= 0 or h <= 0:
                continue

            label_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            class_counts_local[yolo_class] += 1

        # Write label file (YOLO expects file even if empty)
        label_dest = os.path.join(_G["yolo_dir"], "labels", split, f"{rec['stem']}.txt")
        tmp_label_dest = label_dest + ".tmp"
        with open(tmp_label_dest, "w") as f:
            if label_lines:
                f.write("\n".join(label_lines))
        os.replace(tmp_label_dest, label_dest)

        return True, split, len(label_lines), class_counts_local, rec["stem"], None

    except Exception as e:
        return False, "unknown", 0, {}, rec["stem"], str(e)


def export_yolo_dataset(
    raw_dir: str,
    yolo_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    workers: Optional[int] = None,
    chunksize: int = 64,
):
    """
    Export collected data to YOLO format with train/val/test splits.

    ⚡ Parallel + streaming:
    - Uses os.scandir for fast, low-memory scanning (handles tens of millions).
    - Deterministic per-file split via hashing (no global shuffle needed).
    - Multiprocessing for image conversion and label writing.

    Directory structure:
        yolo_dir/
            images/{train,val,test}/
            labels/{train,val,test}/
            data.yaml
            classes.txt
            STATS.txt
    """
    # Seed only used for deterministic split hashing
    random.seed(seed)

    # Prepare output structure
    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(yolo_dir, "images", split))
        ensure_dir(os.path.join(yolo_dir, "labels", split))

    # Class mapping
    class_to_id = {cls: idx for idx, cls in enumerate(YOLO_CLASSES)}

    # Worker pool sizing
    if workers is None:
        workers = max(1, min(cpu_count() or 1, 128))
    if workers < 1:
        workers = 1

    print("============================================================")
    print("YOLO Export (parallel)")
    print("============================================================")
    print(f"Raw dir:            {raw_dir}")
    print(f"YOLO out dir:       {yolo_dir}")
    print(f"Workers:            {workers}")
    print(f"Task chunk size:    {chunksize}")
    print(f"Split ratios:       train={train_ratio} val={val_ratio} test={test_ratio}")
    print("Scanning and exporting...")

    # Stats
    split_image_counts = {"train": 0, "val": 0, "test": 0}
    split_ann_counts = {"train": 0, "val": 0, "test": 0}
    class_counts = {cls: 0 for cls in YOLO_CLASSES}
    total_processed = 0
    total_errors = 0

    # Stream scan -> process in parallel
    iterable = _scan_records(raw_dir)

    try:
        with Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(
                yolo_dir,
                class_to_id,
                seed,
                (train_ratio, val_ratio, test_ratio),
            ),
            maxtasksperchild=1000,
        ) as pool:
            # imap_unordered lazily consumes the generator; no giant queues
            for ok, split, n_anns, class_counts_local, stem, err in tqdm(
                pool.imap_unordered(_process_one, iterable, chunksize=chunksize),
                desc="Exporting",
                unit="file",
            ):
                if ok:
                    split_image_counts[split] += 1
                    split_ann_counts[split] += n_anns
                    for k, v in class_counts_local.items():
                        class_counts[k] += v
                else:
                    total_errors += 1
                    tqdm.write(f"[WARN] {stem}: {err}")
                total_processed += 1
    except KeyboardInterrupt:
        print("\nInterrupted by user, writing partial stats...")

    if total_processed == 0:
        print("No records found (meta+image+elements) in", raw_dir)
        return

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

    # Write STATS
    total_images = sum(split_image_counts.values())
    total_annotations = sum(split_ann_counts.values())
    stats_lines = []
    stats_lines.append("# Dataset Statistics\n")
    stats_lines.append(f"Total images: {total_images}\n")
    stats_lines.append(f"Total annotations: {total_annotations}\n")
    stats_lines.append(f"Total errors: {total_errors}\n\n")

    stats_lines.append("## Split Distribution\n")
    for s in ["train", "val", "test"]:
        stats_lines.append(
            f"{s}: {split_image_counts[s]} images, {split_ann_counts[s]} annotations\n"
        )

    stats_lines.append("\n## Class Distribution\n")
    nonzero = [(cls, c) for cls, c in class_counts.items() if c > 0]
    nonzero.sort(key=lambda x: x[1], reverse=True)
    for cls, c in nonzero:
        pct = 100.0 * c / total_annotations if total_annotations > 0 else 0.0
        stats_lines.append(f"{cls}: {c} ({pct:.1f}%)\n")

    with open(os.path.join(yolo_dir, "STATS.txt"), "w") as f:
        f.write("".join(stats_lines))

    # Console summary
    print("\n" + "=" * 60)
    print("YOLO Dataset Export Complete!")
    print("=" * 60)
    print(f"Output directory: {yolo_dir}")
    print(f"Total images: {total_images}")
    print(
        f"  Train: {split_image_counts['train']} images ({split_ann_counts['train']} annotations)"
    )
    print(
        f"  Val:   {split_image_counts['val']} images ({split_ann_counts['val']} annotations)"
    )
    print(
        f"  Test:  {split_image_counts['test']} images ({split_ann_counts['test']} annotations)"
    )
    print(f"Classes: {len(YOLO_CLASSES)}")
    if nonzero:
        print("\nTop 5 classes:")
        for cls, count in nonzero[:5]:
            pct = 100 * count / total_annotations if total_annotations > 0 else 0
            print(f"  {cls:15s}: {count:5d} ({pct:5.1f}%)")
    print(f"\nConfiguration saved to: {os.path.join(yolo_dir, 'data.yaml')}")
    print(f"Statistics saved to: {os.path.join(yolo_dir, 'STATS.txt')}")
