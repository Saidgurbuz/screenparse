"""Export annotations in YOLO format (parallel, streaming-friendly).

Now uses VLM labels as primary class source:

1. If element["vlm_label"] exists and is not "Unknown" (case-insensitive),
   we use that as the YOLO label (matched against YOLO_CLASSES).
2. Otherwise, we try to fall back to element["type"] with a small normalization.
3. If that also fails, we assign the generic class "Text".
"""

import os
import random
import hashlib
import numpy as np
from tqdm import tqdm
from pathlib import Path
from functools import partial
from .utils import ensure_dir, load_json
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Any, Tuple, Iterable, Optional

# ============================================================
# YOLO classes: VLM canonical classes
# ============================================================
YOLO_CLASSES = [
    "Table",
    "Column/Browser",
    "Button",
    "Utility Button",
    "App Icon",
    "Navigation Bar",
    "Status Bar",
    "Search Field",
    "Toolbar",
    "Tooltip",
    "Video",
    "Tab Bar",
    "Side Bar",
    "Slider",
    "Picker",
    "ContextMenu",
    "DockMenu",
    "EditMenu",
    "Image",
    "Scroll",
    "Switch",
    "File Icon",
    "Chart",
    "Window",
    "Screen",
    "List",
    "List Item",
    "PopUp Menu",
    "Steppers",
    "Toggles",
    "Text Input",
    "Rating Indicator",
    "Checkbox",
    "Radiobox",
    "Select",
    "Avatar",
    "Badge",
    "Alert",
    "Progress bar",
    "Bottom navigation",
    "Breadcrumb",
    "Page control",
    "Link",
    "Menu",
    "Pagination",
    "Tab",
    "Search Bar",
    "Date-Time picker",
    "Calendar",
    "Text",
    "Heading",
    "Code snippet",
    "Carousel",
    "Notification",
    "Logo",
]

# quick helpers for type->label fallback
_YC_LOWER = [c.lower() for c in YOLO_CLASSES]


# Which YOLO classes are considered layout/containers vs atomic UI elements
CONTAINER_CLASS_NAMES = {
    "Table",
    "Column/Browser",
    "Navigation Bar",
    "Status Bar",
    "Toolbar",
    "Tab Bar",
    "Side Bar",
    "ContextMenu",
    "DockMenu",
    "EditMenu",
    "Scroll",
    "Window",
    "Screen",
    "List",
    "PopUp Menu",
    "Alert",
    "Bottom navigation",
    "Breadcrumb",
    "Menu",
    "Pagination",
    "Search Bar",
    "Date-Time picker",
    "Calendar",
    "Carousel",
    "Notification",
}

ATOMIC_CLASS_NAMES = {
    "Button",
    "Utility Button",
    "App Icon",
    "Search Field",
    "Tooltip",
    "Video",
    "Slider",
    "Picker",
    "Image",
    "Switch",
    "File Icon",
    "Chart",
    "List Item",
    "Steppers",
    "Toggles",
    "Text Input",
    "Rating Indicator",
    "Checkbox",
    "Radiobox",
    "Select",
    "Avatar",
    "Badge",
    "Progress bar",
    "Page control",
    "Link",
    "Tab",
    "Text",
    "Heading",
    "Code snippet",
    "Logo",
}

def _match_vlm_label(raw: str) -> Optional[str]:
    """Case-insensitive exact match of a VLM label against YOLO_CLASSES."""
    if not raw:
        return None
    low = raw.strip().lower()
    if not low or low == "unknown":
        return None
    for cls, cls_low in zip(YOLO_CLASSES, _YC_LOWER):
        if low == cls_low:
            return cls
    return None


def _fallback_from_type(el_type: str) -> Optional[str]:
    """
    Try to recover a class from the element 'type' field.

    Strategy:
    1) Exact case-insensitive match against YOLO_CLASSES.
    2) Tiny keyword-based heuristics for a few common patterns.
    """
    if not el_type:
        return None
    s = el_type.strip()
    if not s or s.lower() == "unknown":
        return None

    low = s.lower()

    # 1) exact match against known classes
    for cls, cls_low in zip(YOLO_CLASSES, _YC_LOWER):
        if low == cls_low:
            return cls

    # 2) lightweight keyword mapping
    if "button" in low or "btn" in low:
        return "Button"
    if "input" in low or "field" in low:
        return "Text Input"
    if "search" in low:
        # prefer the generic search input
        return "Search Field"
    if "checkbox" in low:
        return "Checkbox"
    if "radio" in low:
        return "Radiobox"
    if "table" in low:
        return "Table"
    if "list" in low:
        return "List"
    if "tab" in low:
        return "Tab"
    if "menu" in low:
        return "Menu"
    if "nav" in low:
        return "Navigation Bar"
    if "image" in low or "img" in low or "picture" in low:
        return "Image"
    if "text" in low or "paragraph" in low or "label" in low:
        return "Text"
    if "logo" in low or "brand" in low:
        return "Logo"
    if "tooltip" in low:
        return "Tooltip"
    if "badge" in low or "chip" in low:
        return "Badge"
    if "video" in low:
        return "Video"
    if "slider" in low:
        return "Slider"
    if "calendar" in low:
        return "Calendar"

    return None


def map_to_yolo_class(el: Dict[str, Any]) -> str:
    """
    Decide YOLO class for an element.

    Priority:
    1. Use 'vlm_label' if present and not 'Unknown', mapped to YOLO_CLASSES.
    2. Else, try element['type'] as a fallback.
    3. Else, assign generic 'Element'.
    """
    # 1) VLM label (primary)
    vlm_label = (el.get("vlm_label") or "").strip()
    label = _match_vlm_label(vlm_label)
    if label is not None:
        return label

    # 2) Fallback: element type
    el_type = (el.get("type") or "").strip()
    label = _fallback_from_type(el_type)
    if label is not None:
        return label

    # 3) Final fallback: generic bucket
    return "Text"


def bbox_to_yolo_format(
    rect: Dict[str, int], img_width: int, img_height: int
) -> Tuple[float, float, float, float]:
    """Convert rect {x,y,w,h} in pixel coords to YOLO-normalized cx,cy,w,h."""
    x, y, w, h = rect["x"], rect["y"], rect["w"], rect["h"]
    center_x = x + w / 2.0
    center_y = y + h / 2.0
    center_x_norm = max(0.0, min(1.0, center_x / img_width))
    center_y_norm = max(0.0, min(1.0, center_y / img_height))
    width_norm = max(0.0, min(1.0, w / img_width))
    height_norm = max(0.0, min(1.0, h / img_height))
    return center_x_norm, center_y_norm, width_norm, height_norm


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two boxes [x1,y1,x2,y2] in normalized coords."""
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _clean_gt_boxes(labels: np.ndarray, container_ids: set[int], iou_thr: float = 0.85) -> np.ndarray:
    """
    Remove duplicate GTs per class using high IoU clustering.

    labels: [N,5] -> [cls, cx, cy, w, h] (normalized).
    For container classes -> keep largest box in a cluster.
    For atomic classes   -> keep smallest box in a cluster.
    """
    if labels.shape[0] <= 1:
        return labels

    cls_ids = labels[:, 0].astype(int)
    cxcywh = labels[:, 1:]

    # convert to xyxy (still normalized 0..1)
    xyxy = np.zeros_like(cxcywh)
    xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2.0  # x1
    xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2.0  # y1
    xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2.0  # x2
    xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2.0  # y2

    keep_indices: list[int] = []

    for c in np.unique(cls_ids):
        idxs = np.where(cls_ids == c)[0]
        if len(idxs) == 1:
            keep_indices.append(idxs[0])
            continue

        remaining = list(idxs)
        while remaining:
            i = remaining.pop(0)
            cluster = [i]
            to_remove = []
            for j in remaining:
                if _iou_xyxy(xyxy[i], xyxy[j]) > iou_thr:
                    cluster.append(j)
                    to_remove.append(j)
            remaining = [k for k in remaining if k not in to_remove]

            # choose representative from cluster
            areas = [
                (xyxy[k][2] - xyxy[k][0]) * (xyxy[k][3] - xyxy[k][1]) for k in cluster
            ]
            if c in container_ids:
                # for containers -> keep largest
                best = cluster[int(np.argmax(areas))]
            else:
                # for atomic -> keep smallest
                best = cluster[int(np.argmin(areas))]
            keep_indices.append(best)

    keep_indices = sorted(set(keep_indices))
    return labels[keep_indices]

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
    "container_ids": None,
}


def _init_worker(
    yolo_dir: str,
    class_to_id: Dict[str, int],
    seed: int,
    ratios: Tuple[float, float, float],
    container_ids: set[int]
):
    _G["yolo_dir"] = yolo_dir
    _G["class_to_id"] = class_to_id
    _G["seed"] = seed
    _G["ratios"] = ratios
    _G["container_ids"] = container_ids


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
        tmp_img_dest = img_dest + ".tmp"

        with Image.open(rec["image"]) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(tmp_img_dest, "JPEG", quality=95, optimize=True)
        os.replace(tmp_img_dest, img_dest)

        # Convert annotations
        class_counts_local = {cls: 0 for cls in YOLO_CLASSES}

        # Collect raw labels [cls, cx, cy, w, h] before dedup
        raw_labels: List[Tuple[float, float, float, float, float]] = []

        for el in elements:
            rect = el.get("rect")
            if not rect:
                continue

            yolo_class = map_to_yolo_class(el)
            if yolo_class not in _G["class_to_id"]:
                continue

            class_id = _G["class_to_id"][yolo_class]
            cx, cy, w, h = bbox_to_yolo_format(rect, img_w, img_h)
            if w <= 0 or h <= 0:
                continue

            raw_labels.append((float(class_id), cx, cy, w, h))

        label_lines: List[str] = []
        if raw_labels:
            labels_arr = np.array(raw_labels, dtype=float)  # [N,5]

            # Clean GT: remove duplicate overlapping boxes per class
            labels_arr = _clean_gt_boxes(labels_arr, _G["container_ids"], iou_thr=0.85)

            # Rebuild label_lines and class_counts after cleaning
            for cls_id, cx, cy, w, h in labels_arr:
                cls_id_int = int(cls_id)
                label_lines.append(
                    f"{cls_id_int} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )
                # find class name for stats
                for name, cid in _G["class_to_id"].items():
                    if cid == cls_id_int:
                        class_counts_local[name] += 1
                        break

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
    random.seed(seed)

    # Prepare output structure
    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(yolo_dir, "images", split))
        ensure_dir(os.path.join(yolo_dir, "labels", split))

    # Class mapping
    class_to_id = {cls: idx for idx, cls in enumerate(YOLO_CLASSES)}
    
    # Container class ids for GT cleaning
    container_ids = {
        class_to_id[name]
        for name in CONTAINER_CLASS_NAMES
        if name in class_to_id
    }

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
                container_ids,
            ),
            maxtasksperchild=1000,
        ) as pool:
            for ok, split, n_anns, class_counts_local, stem, err in tqdm(
                pool.imap_unordered(_process_one, iterable, chunksize=chunksize),
                desc="Exporting",
                unit="file",
            ):
                if ok:
                    if split in split_image_counts:
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
            print(f"  {cls:25s}: {count:7d} ({pct:5.1f}%)")
    print(f"\nConfiguration saved to: {os.path.join(yolo_dir, 'data.yaml')}")
    print(f"Statistics saved to: {os.path.join(yolo_dir, 'STATS.txt')}")