# src/ui_postproc.py
from __future__ import annotations
from typing import Tuple, Set
import numpy as np



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

NON_NESTABLE_CLASS_NAMES = {
    "Text",
    "Heading",
    "Link",
    "Button",
    "Utility Button",
    "App Icon",
    "File Icon",
    "Image",
    "Logo",
    "Checkbox",
    "Radiobox",
    "Switch",
    "Slider",
    "Text Input",
    "Search Field",
    "Date-Time picker",
    "Progress bar",
    "Rating Indicator",
    "Avatar",
    "Badge",
}


def get_class_id_sets(model_names: dict[int, str]) -> Tuple[Set[int], Set[int], Set[int]]:
    """Map container/atomic/non-nestable name sets to class ID sets for a given YOLO model."""
    name_to_id = {v.lower(): int(k) for k, v in model_names.items()}
    container_ids = {name_to_id[n.lower()] for n in CONTAINER_CLASS_NAMES if n.lower() in name_to_id}
    atomic_ids = {name_to_id[n.lower()] for n in ATOMIC_CLASS_NAMES if n.lower() in name_to_id}
    non_nestable_ids = {name_to_id[n.lower()] for n in NON_NESTABLE_CLASS_NAMES if n.lower() in name_to_id}
    return container_ids, atomic_ids, non_nestable_ids


# --- IoU and geometry helpers ---

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
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


def iou_matrix(boxes: np.ndarray) -> np.ndarray:
    N = len(boxes)
    ious = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            val = iou_xyxy(boxes[i], boxes[j])
            ious[i, j] = ious[j, i] = val
    return ious


def area(box: np.ndarray) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def contains(parent: np.ndarray, child: np.ndarray, min_cover: float = 0.65) -> bool:
    """Return True if parent covers at least min_cover fraction of child."""
    inter_x1 = max(parent[0], child[0])
    inter_y1 = max(parent[1], child[1])
    inter_x2 = min(parent[2], child[2])
    inter_y2 = min(parent[3], child[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    child_area = area(child)
    if child_area <= 0:
        return False
    return inter / child_area >= min_cover


# --- 1) De-dup same-class overlapping predictions (Weighted Boxes Fusion-ish) ---

def _fuse_cluster(indices, boxes, scores):
    cluster_boxes = boxes[indices]
    cluster_scores = scores[indices]
    w = cluster_scores / (cluster_scores.sum() + 1e-6)
    fused = np.sum(cluster_boxes * w[:, None], axis=0)
    fused_score = cluster_scores.max()
    return fused, fused_score


def dedup_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    clss: np.ndarray,
    t_cluster: float = 0.75,
    non_nestable_ids: set[int] | None = None,
    contain_thr: float = 0.65,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Merge same-class boxes with high IoU into a single box (same element).
    Keeps semantic overlaps across *different* classes (e.g., Window + Button).
    """
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    clss = np.asarray(clss, dtype=int)
    new_boxes = []
    new_scores = []
    new_clss = []

    for c in np.unique(clss):
        idxs = np.where(clss == c)[0]
        if len(idxs) == 0:
            continue
        if len(idxs) == 1:
            i = idxs[0]
            new_boxes.append(boxes[i])
            new_scores.append(scores[i])
            new_clss.append(clss[i])
            continue

        sub_boxes = boxes[idxs]
        ious = iou_matrix(sub_boxes)

        remaining = list(range(len(idxs)))
        while remaining:
            root = remaining.pop(0)
            cluster = [root]
            to_remove = []
            for r in remaining:
                is_dup = ious[root, r] > t_cluster
                if (not is_dup) and non_nestable_ids and (c in non_nestable_ids):
                    # containment-based duplicate for non-nestable classes
                    box_root = boxes[idxs[root]]
                    box_r = boxes[idxs[r]]
                    if contains(box_root, box_r, contain_thr) or contains(
                        box_r, box_root, contain_thr
                    ):
                        is_dup = True
                if is_dup:
                    cluster.append(r)
                    to_remove.append(r)
            remaining = [r for r in remaining if r not in to_remove]

            cluster_global_idx = [idxs[k] for k in cluster]
            fused_box, fused_score = _fuse_cluster(cluster_global_idx, boxes, scores)
            new_boxes.append(fused_box)
            new_scores.append(fused_score)
            new_clss.append(c)

    return np.array(new_boxes), np.array(new_scores), np.array(new_clss, dtype=int)


# --- 2) Build parent-child relationships from geometry & container ids ---

def build_parent_child_graph(
    boxes: np.ndarray,
    clss: np.ndarray,
    container_ids: Set[int],
    contain_thr: float = 0.75,
):
    N = len(boxes)
    parents = [[] for _ in range(N)]
    children = [[] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if clss[j] in container_ids and contains(boxes[j], boxes[i], min_cover=contain_thr):
                parents[i].append(j)
                children[j].append(i)
    return parents, children


# --- 3) Filter containers (keep meaningful parents, drop junk wrappers) ---

def hierarchy_filter(
    boxes: np.ndarray,
    scores: np.ndarray,
    clss: np.ndarray,
    container_ids: Set[int],
    atomic_ids: Set[int],
    min_children_default: int = 1,
    contain_thr: float = 0.65,
    max_area_ratio_single_child: float = 1.3,
    iou_wrapper_thr: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    - Keep containers that have enough atomic children inside.
    - Drop containers that tightly hug a single child (wrapper divs).
    """
    N = len(boxes)
    if N == 0:
        return boxes, scores, clss

    parents, children = build_parent_child_graph(boxes, clss, container_ids, contain_thr)
    keep = np.ones(N, dtype=bool)

    # 1) Drop containers without atomic children
    for i in range(N):
        if clss[i] not in container_ids:
            continue
        kids = children[i]
        atomic_kids = [k for k in kids if clss[k] in atomic_ids]
        if len(atomic_kids) < min_children_default:
            keep[i] = False

    # 2) Drop wrapper containers hugging a single child
    for i in range(N):
        if not keep[i] or clss[i] not in container_ids:
            continue
        kids = [k for k in children[i] if keep[k]]
        if len(kids) != 1:
            continue
        k = kids[0]
        area_c = area(boxes[i])
        area_k = area(boxes[k])
        if area_k <= 0:
            continue
        ar = area_c / area_k
        if ar < max_area_ratio_single_child or iou_xyxy(boxes[i], boxes[k]) > iou_wrapper_thr:
            keep[i] = False

    return boxes[keep], scores[keep], clss[keep]


# --- 4) Optional: simple score reweighting by hierarchy ---

def reweight_scores(
    boxes: np.ndarray,
    scores: np.ndarray,
    clss: np.ndarray,
    container_ids: Set[int],
    atomic_ids: Set[int],
) -> np.ndarray:
    """Boost containers with many children, and atomics inside containers."""
    parents, children = build_parent_child_graph(boxes, clss, container_ids)
    scores = scores.copy()

    # upweight containers with children
    for i in range(len(boxes)):
        if clss[i] in container_ids:
            n_kids = len(children[i])
            if n_kids > 0:
                factor = 1.0 + min(0.5, 0.1 * n_kids)  # up to +50%
                scores[i] *= factor

    # upweight atomic elements that live inside containers
    for i in range(len(boxes)):
        if clss[i] in atomic_ids:
            if parents[i]:
                scores[i] *= 1.1

    return scores


# --- 5) Full post-processor: dedup + hierarchy filter ---

def postprocess_detections(
    boxes: np.ndarray,
    scores: np.ndarray,
    clss: np.ndarray,
    container_ids: Set[int],
    atomic_ids: Set[int],
    non_nestable_ids: Set[int],
    conf_thr: float = 0.25,
    cluster_iou: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """End-to-end: threshold -> dedup -> reweight -> hierarchy filter."""
    if len(boxes) == 0:
        return boxes, scores, clss

    # basic score threshold
    m = scores >= conf_thr
    boxes, scores, clss = boxes[m], scores[m], clss[m]
    if len(boxes) == 0:
        return boxes, scores, clss
    # 1) merge same-class duplicates
    boxes, scores, clss = dedup_per_class(
        boxes,
        scores,
        clss,
        t_cluster=cluster_iou,
        non_nestable_ids=non_nestable_ids,
        contain_thr=0.9,
    )

    # 2) optional: score reweighting
    # scores = reweight_scores(boxes, scores, clss, container_ids, atomic_ids)

    # 3) hierarchy-aware container filtering
    # boxes, scores, clss = hierarchy_filter(
    #     boxes,
    #     scores,
    #     clss,
    #     container_ids,
    #     atomic_ids,
    # )
    return boxes, scores, clss