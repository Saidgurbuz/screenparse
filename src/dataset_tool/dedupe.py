# src/dataset_tool/dedupe.py
import os, csv
from typing import Dict, List, Tuple
from PIL import Image
import imagehash
from .utils import ensure_dir


def _phash(path: str) -> imagehash.ImageHash:
    with Image.open(path) as im:
        return imagehash.phash(im)


def _hamming(a: imagehash.ImageHash, b: imagehash.ImageHash) -> int:
    return a - b


def find_duplicates(image_dir: str, distance_threshold: int = 8) -> List[List[str]]:
    # collect images
    imgs = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    hashes: Dict[str, imagehash.ImageHash] = {p: _phash(p) for p in imgs}
    # naive O(n^2) is fine for trials; switch to LSH later if needed
    visited = set()
    groups: List[List[str]] = []
    for i, p in enumerate(imgs):
        if p in visited:
            continue
        group = [p]
        visited.add(p)
        for q in imgs[i + 1 :]:
            if q in visited:
                continue
            if _hamming(hashes[p], hashes[q]) <= distance_threshold:
                group.append(q)
                visited.add(q)
        if len(group) > 1:
            groups.append(group)
    return groups


def write_groups_csv(groups: List[List[str]], out_csv: str):
    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group_id", "image_path"])
        for gid, g in enumerate(groups):
            for p in g:
                w.writerow([gid, p])
