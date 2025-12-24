from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

from PIL import Image
from .types import BoundingBox, EvaluationSample, UIElement
from .label_mapping import LabelMapper


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_bbox(obj: dict) -> Optional[BoundingBox]:
    if obj is None:
        return None

    # Common representations
    if "bbox" in obj:
        val = obj["bbox"]
        if isinstance(val, dict):
            x, y, w, h = (
                val.get("x", 0.0),
                val.get("y", 0.0),
                val.get("w", 0.0),
                val.get("h", 0.0),
            )
            return BoundingBox(float(x), float(y), float(w), float(h))
        if isinstance(val, (list, tuple)) and len(val) == 4:
            x, y, w, h = val
            return BoundingBox(float(x), float(y), float(w), float(h))

    if "rect" in obj:
        rect = obj["rect"]
        if isinstance(rect, dict):
            x, y, w, h = (
                rect.get("x", 0.0),
                rect.get("y", 0.0),
                rect.get("w", 0.0),
                rect.get("h", 0.0),
            )
            return BoundingBox(float(x), float(y), float(w), float(h))

    if "bbox_ltrb" in obj:
        ltrb = obj["bbox_ltrb"]
        if isinstance(ltrb, (list, tuple)) and len(ltrb) == 4:
            x0, y0, x1, y1 = ltrb
            return BoundingBox(
                float(x0),
                float(y0),
                float(x1) - float(x0),
                float(y1) - float(y0),
            )
    
    if "bbox_tlbr" in obj:
        tlbr = obj["bbox_tlbr"]
        if isinstance(tlbr, (list, tuple)) and len(tlbr) == 4:
            y0, x0, y1, x1 = tlbr
            return BoundingBox(
                float(x0),
                float(y0),
                float(x1) - float(x0),
                float(y1) - float(y0),
            )
    if "bbox_xywh" in obj:
        xywh = obj["bbox_xywh"]
        if isinstance(xywh, (list, tuple)) and len(xywh) == 4:
            x, y, w, h = xywh
            return BoundingBox(float(x), float(y), float(w), float(h))

    return None


def element_from_obj(obj: dict, include_raw: bool = False) -> Optional[UIElement]:
    bbox = _extract_bbox(obj)
    if bbox is None:
        return None

    label = (
        obj.get("label")
        or obj.get("vlm_label")
        or obj.get("type")
        or obj.get("tag")
    )
    text = obj.get("text") or obj.get("inner_text") or obj.get("own_text")
    score = obj.get("score") or obj.get("confidence")

    raw_payload = obj if include_raw else None
    return UIElement(
        bbox=bbox,
        label=label,
        text=text,
        score=score if score is None else float(score),
        raw=raw_payload,
    )


def load_jsonl_elements(path: Path, include_raw: bool = False) -> List[UIElement]:
    elements: List[UIElement] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            el = element_from_obj(obj, include_raw=include_raw)
            if el:
                elements.append(el)
    return elements


def load_elements_file(path: Path, include_raw: bool = False) -> List[UIElement]:
    if not path.exists():
        return []
    if path.suffix.lower() in {".jsonl", ".jsonlines"}:
        return load_jsonl_elements(path, include_raw=include_raw)

    payload = _read_json(path)
    items = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        # Allow nested structure like {"elements": [...]}
        for key in ("elements", "items", "data"):
            if key in payload and isinstance(payload[key], list):
                items = payload[key]
                break
    elements: List[UIElement] = []
    for obj in items:
        el = element_from_obj(obj, include_raw=include_raw)
        if el:
            elements.append(el)
    return elements


def _find_ground_truth_file(stem: Path) -> Optional[Path]:
    candidates = [
        stem.with_suffix(".triplets.jsonl"),
        stem.with_suffix(".elements.json"),
        stem.with_suffix(".elements.unfiltered.json"),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _read_image_size(image_path: Path) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(image_path) as im:
            return im.size  # (width, height)
    except Exception:
        return None
    
def _load_annotations(ann_path: Path) -> List[Dict]:
    payload = _read_json(ann_path)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        # Some files might wrap the list
        for key in ("items", "data", "annotations"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
    return []


def _iter_images(image_dir: Path, exts: Sequence[str], max_count: Optional[int]) -> Iterable[Path]:
    count = 0
    for ext in exts:
        for path in image_dir.glob(f"*{ext}"):
            yield path
            count += 1
            if max_count is not None and count >= max_count:
                return


def build_raw_dataset(
    image_dir: str,
    include_raw: bool = False,
    image_exts: Sequence[str] = (".png", ".jpg", ".jpeg"),
    max_samples: Optional[int] = None,
    label_mapper: Optional[LabelMapper] = None,
) -> List[EvaluationSample]:
    base = Path(image_dir)
    samples: List[EvaluationSample] = []
    for image_path in _iter_images(base, image_exts, max_samples):
        stem = image_path.with_suffix("")
        gt_file = _find_ground_truth_file(stem)
        if not gt_file:
            continue
        gt_elements = load_elements_file(gt_file, include_raw=include_raw)
        if label_mapper:
            gt_elements = [label_mapper.map_element(e) for e in gt_elements]
        img_size = _read_image_size(image_path)
        samples.append(
            EvaluationSample(
                image_path=str(image_path),
                ground_truth=gt_elements,
                image_size=img_size,
                metadata={"stem": stem.name, "ground_truth_file": str(gt_file)},
                sample_id=stem.name,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def build_groundcua_dataset(
    root_dir: str,
    include_raw: bool = False,
    image_exts: Sequence[str] = (".png", ".jpg", ".jpeg"),
    max_samples: Optional[int] = None,
    label_mapper: Optional[LabelMapper] = None,
) -> List[EvaluationSample]:
    """
    Build GroundCUA dataset with uniform sampling across platform subfolders.

    Directory layout:
      root_dir/images/<Platform>/<hash>.png
      root_dir/data/<Platform>/<hash>.json
    """
    root = Path(root_dir)
    img_root = root / "images"
    data_root = root / "data"
    platforms = [p for p in img_root.iterdir() if p.is_dir()]
    if not platforms:
        return []

    per_platform = None
    if max_samples:
        per_platform = max(1, max_samples // max(len(platforms), 1))

    samples: List[EvaluationSample] = []
    for plat_dir in platforms:
        ann_dir = data_root / plat_dir.name
        if not ann_dir.exists():
            continue

        imgs = []
        for ext in image_exts:
            imgs.extend(sorted(plat_dir.glob(f"*{ext}")))
        if per_platform:
            import random

            random.Random(42).shuffle(imgs)
            imgs = imgs[:per_platform]

        for img_path in imgs:
            ann_path = ann_dir / (img_path.stem + ".json")
            if not ann_path.exists():
                continue
            ann_items = _load_annotations(ann_path)
            elements: List[UIElement] = []
            for obj in ann_items:
                bbox = _extract_bbox({"bbox_ltrb": obj.get("bbox")}) if obj.get("bbox") else None
                if not bbox:
                    continue
                label = obj.get("category") or obj.get("label") or None
                text = obj.get("text") or ""
                raw_payload = obj if include_raw else None
                elements.append(
                    UIElement(
                        bbox=bbox,
                        label=label,
                        text=text,
                        score=None,
                        raw=raw_payload,
                    )
                )
            if label_mapper:
                elements = [label_mapper.map_element(e) for e in elements]
            img_size = _read_image_size(img_path)
            samples.append(
                EvaluationSample(
                    image_path=str(img_path),
                    ground_truth=elements,
                    image_size=img_size,
                    metadata={
                        "stem": img_path.stem,
                        "platform": plat_dir.name,
                        "ground_truth_file": str(ann_path),
                        "format": "groundcua",
                    },
                    sample_id=img_path.stem,
                )
            )
            if max_samples and len(samples) >= max_samples:
                break
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def build_screenspot_dataset(
    root_dir: str,
    split: str = "all",
    include_raw: bool = False,
    image_exts: Sequence[str] = (".png", ".jpg", ".jpeg"),
    max_samples: Optional[int] = None,
    label_mapper: Optional[LabelMapper] = None,
) -> List[EvaluationSample]:
    """
    Build ScreenSpot dataset from preprocessed JSON/image folders.

    Layout:
      root_dir/images/{web,pc,mobile}/file_name.png
      root_dir/data/{web,pc,mobile}/file_name.json
    """
    root = Path(root_dir)
    data_root = root / "data"
    img_root = root / "images"
    splits = ["web", "pc", "mobile"] if split == "all" else [split]

    def _label_from_type(val: Optional[str]) -> Optional[str]:
        if not val:
            return None
        v = val.strip().lower()
        if v == "text":
            return "Text"
        if v in ("icon", "widget"):
            return "Image"
        return val

    per_split = None
    if max_samples:
        per_split = max(1, max_samples // max(len(splits), 1))

    samples: List[EvaluationSample] = []
    for sp in splits:
        ann_dir = data_root / sp
        img_dir = img_root / sp
        if not ann_dir.exists() or not img_dir.exists():
            continue

        ann_files = sorted(ann_dir.glob("*.json"))
        if per_split:
            import random

            random.shuffle(ann_files)
            ann_files = ann_files[:per_split]

        for ann_path in ann_files:
            ann_items = _load_annotations(ann_path)
            if not ann_items:
                continue
            stem = ann_path.stem
            img_path = img_dir / f"{stem}.png"
            if not img_path.exists():
                # Try to fall back to image_path in JSON
                img_path_str = ann_items[0].get("image_path")
                if img_path_str:
                    candidate = (root / img_path_str).resolve()
                    if candidate.exists():
                        img_path = candidate
            if not img_path.exists():
                continue

            img_size = _read_image_size(img_path)
            if img_size is None:
                continue
            img_w, img_h = img_size

            elements: List[UIElement] = []
            instructions = []
            for obj in ann_items:
                bbox = None
                if obj.get("bbox"):
                    bb = obj.get("bbox")
                    if isinstance(bb, (list, tuple)) and len(bb) == 4:
                        x0, y0, x1, y1 = bb
                        bbox = _extract_bbox(
                            {
                                "bbox_ltrb": (
                                    float(x0) * img_w,
                                    float(y0) * img_h,
                                    float(x1) * img_w,
                                    float(y1) * img_h,
                                )
                            }
                        )
                if not bbox:
                    continue
                label = _label_from_type(obj.get("data_type"))
                raw_payload = obj if include_raw else None
                elements.append(
                    UIElement(
                        bbox=bbox,
                        label=label,
                        text=None,
                        score=None,
                        raw=raw_payload,
                    )
                )
                if obj.get("instruction"):
                    instructions.append(obj.get("instruction"))

            if label_mapper:
                elements = [label_mapper.map_element(e) for e in elements]

            samples.append(
                EvaluationSample(
                    image_path=str(img_path),
                    ground_truth=elements,
                    image_size=img_size,
                    metadata={
                        "stem": stem,
                        "split": sp,
                        "ground_truth_file": str(ann_path),
                        "format": "screenspot",
                        "instructions": instructions,
                    },
                    sample_id=stem,
                )
            )
            if max_samples and len(samples) >= max_samples:
                break
        if max_samples and len(samples) >= max_samples:
            break

    return samples


def _load_class_names(classes_path: Path) -> Optional[List[str]]:
    if not classes_path.exists():
        return None
    names: List[str] = []
    with classes_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names or None


def _read_yolo_label_file(
    label_path: Path,
    image_size: Tuple[int, int],
    class_names: Optional[List[str]],
) -> List[UIElement]:
    width, height = image_size
    elements: List[UIElement] = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            abs_w = w * width
            abs_h = h * height
            x = (xc * width) - (abs_w / 2.0)
            y = (yc * height) - (abs_h / 2.0)
            label = str(cls_id)
            if class_names and 0 <= cls_id < len(class_names):
                label = class_names[cls_id]
            elements.append(
                UIElement(
                    bbox=BoundingBox(x, y, abs_w, abs_h),
                    label=label,
                    score=None,
                    raw=None,
                )
            )
    return elements


def build_yolo_dataset(
    image_dir: str,
    labels_dir: Optional[str] = None,
    classes_path: Optional[str] = None,
    include_raw: bool = False,
    image_exts: Sequence[str] = (".png", ".jpg", ".jpeg"),
    max_samples: Optional[int] = None,
    label_mapper: Optional[LabelMapper] = None,
) -> List[EvaluationSample]:
    image_root = Path(image_dir)
    label_root = Path(labels_dir) if labels_dir else image_root.parent / "labels" / image_root.name

    class_file = Path(classes_path) if classes_path else label_root.parent / "classes.txt"
    class_names = _load_class_names(class_file) if class_file.exists() else None

    samples: List[EvaluationSample] = []
    for image_path in _iter_images(image_root, image_exts, max_samples):
        label_path = label_root / (image_path.stem + ".txt")
        if not label_path.exists():
            continue
        img_size = _read_image_size(image_path)
        if img_size is None:
            continue
        gt_elements = _read_yolo_label_file(label_path, img_size, class_names)
        if label_mapper:
            gt_elements = [label_mapper.map_element(e) for e in gt_elements]
        samples.append(
            EvaluationSample(
                image_path=str(image_path),
                ground_truth=gt_elements,
                image_size=img_size,
                metadata={
                    "stem": image_path.stem,
                    "ground_truth_file": str(label_path),
                    "format": "yolo",
                },
                sample_id=image_path.stem,
            )
        )
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def serialize_elements(elements: Iterable[UIElement]) -> List[dict]:
    return [el.to_dict() for el in elements]
