"""Conservative element filtering - prioritize keeping valid UI elements."""

from typing import List, Dict, Any, Set, Tuple
import numpy as np


def compute_iou(box1: Dict[str, int], box2: Dict[str, int]) -> float:
    """Compute Intersection over Union between two boxes."""
    x1 = max(box1["x"], box2["x"])
    y1 = max(box1["y"], box2["y"])
    x2 = min(box1["x"] + box1["w"], box2["x"] + box2["w"])
    y2 = min(box1["y"] + box1["h"], box2["y"] + box2["h"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_containment(inner: Dict[str, int], outer: Dict[str, int]) -> float:
    """Check what fraction of inner box is contained in outer box."""
    x1 = max(inner["x"], outer["x"])
    y1 = max(inner["y"], outer["y"])
    x2 = min(inner["x"] + inner["w"], outer["x"] + outer["w"])
    y2 = min(inner["y"] + inner["h"], outer["y"] + inner["h"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    inner_area = inner["w"] * inner["h"]

    return inter / inner_area if inner_area > 0 else 0.0


def is_valid_box(rect: Dict[str, int], min_size: int = 4, max_size: int = 1000) -> bool:
    """
    Check if box meets size requirements.
    Very permissive - only filter obvious noise and body/html.
    """
    w, h = rect["w"], rect["h"]
    area = w * h

    # Only filter extremely small (likely rendering artifacts)
    if w < min_size or h < min_size:
        return False

    # Only filter unreasonably large (body/html/fullscreen overlays)
    if w > max_size or h > max_size:
        return False

    return True


def compute_viewport_overlap(rect: Dict[str, int], vp_w: int, vp_h: int) -> float:
    """Compute what fraction of box is within viewport."""
    x1 = max(0, rect["x"])
    y1 = max(0, rect["y"])
    x2 = min(vp_w, rect["x"] + rect["w"])
    y2 = min(vp_h, rect["y"] + rect["h"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    visible_area = (x2 - x1) * (y2 - y1)
    total_area = rect["w"] * rect["h"]

    return visible_area / total_area if total_area > 0 else 0.0


# Define protected element types that should NEVER be filtered aggressively
PROTECTED_TYPES = {
    # Interactive elements - MUST keep
    "button",
    "input",
    "select",
    "textarea",
    "checkbox",
    "radio",
    "switch",
    "link",
    "tab",
    "slider",
    # Important content
    "image",
    "video",
    "audio",
    "icon",
    "svg",
    "title",
    "form",
    # Structured data
    "table",
    "list",
    "list_item",
    # Important UI components
    "card",
    "modal",
    "tooltip",
    "badge",
    "chip",
}


# Container types that can be filtered if they contain protected children
FILTERABLE_CONTAINERS = {
    "div",
    "container",
    "section",
    "article",
    "aside",
    "main",
    "unknown",
}


def is_protected(element: Dict[str, Any]) -> bool:
    """
    Check if element should be protected from filtering.
    Protected elements are important UI components that should not be removed.
    """
    # Safely get values with None handling - use (value or "") pattern
    elem_type = (element.get("type") or "").lower()
    role = (element.get("role") or "").lower()
    tag = (element.get("tag") or "").lower()
    classes = (element.get("classes") or "").lower()
    attrs = element.get("attrs") or {}

    # Protected element types
    PROTECTED_TYPES = {
        "button",
        "input",
        "checkbox",
        "dropdown",
        "search",
        "link",
        "navigation",
        "menu",
        "form",
        "image",
    }

    # Protected roles
    PROTECTED_ROLES = {
        "button",
        "link",
        "textbox",
        "searchbox",
        "checkbox",
        "radio",
        "combobox",
        "navigation",
        "menu",
        "menuitem",
    }

    # Check type
    if elem_type in PROTECTED_TYPES:
        return True

    # Check role
    if role in PROTECTED_ROLES:
        return True

    # Check interactive tags
    if tag in (
        "button",
        "input",
        "a",
        "select",
        "textarea",
        "p",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "dt",
    ):
        return True

    # Check for interactive attributes
    onclick = (attrs.get("onclick") or "").lower()
    if onclick or "clickable" in classes or "interactive" in classes:
        return True

    return False


def has_interactive_indicators(element: Dict[str, Any]) -> bool:
    """Check if element has indicators of being interactive/clickable."""
    attrs = element.get("attrs", {})
    classes = (attrs.get("class") or "").lower()
    elem_id = (attrs.get("id") or "").lower()

    # Check for click handlers
    if any(k.startswith("on") for k in attrs.keys()):  # onclick, onmousedown, etc.
        return True

    # Check for common interactive class patterns
    interactive_keywords = [
        "btn",
        "button",
        "link",
        "click",
        "interactive",
        "action",
        "menu",
        "nav",
        "control",
        "input",
        "field",
        "form",
        "tab",
        "toggle",
        "switch",
        "select",
        "dropdown",
    ]

    if any(kw in classes or kw in elem_id for kw in interactive_keywords):
        return True

    # Check for cursor: pointer indicator
    # (We don't have computed styles, but we can check inline styles)
    style = (attrs.get("style") or "").lower()
    if "cursor" in style and "pointer" in style:
        return True

    return False


def is_pure_layout_container(element: Dict[str, Any]) -> bool:
    """
    Check if element is purely a layout container with no semantic meaning.
    Be VERY conservative - only mark obvious layout divs.
    """
    el_type = element.get("type", "unknown")

    # Only consider divs and generic containers
    if el_type not in {"div", "container"}:
        return False

    tag = element.get("tag", "").lower()
    if tag not in {"div", "span"}:
        return False

    attrs = element.get("attrs", {})
    classes = (attrs.get("class") or "").lower()
    elem_id = (attrs.get("id") or "").lower()

    # Check for pure layout class patterns
    pure_layout_keywords = [
        "container",
        "wrapper",
        "row",
        "col",
        "grid",
        "flex",
        "layout",
        "inner",
        "outer",
        "content-wrap",
    ]

    # Must have layout keywords AND no other semantic indicators
    has_layout = any(kw in classes for kw in pure_layout_keywords)

    if not has_layout:
        return False

    # Check it doesn't have semantic/interactive indicators
    semantic_keywords = [
        "card",
        "panel",
        "box",
        "item",
        "block",
        "section",
        "header",
        "footer",
        "nav",
        "menu",
        "sidebar",
        "btn",
        "button",
        "link",
        "modal",
        "popup",
    ]

    has_semantic = any(kw in classes or kw in elem_id for kw in semantic_keywords)

    # If it has both layout AND semantic indicators, keep it
    if has_semantic:
        return False

    # Check if it has any text content (if yes, might be meaningful)
    inner_text = (element.get("inner_text") or "").strip()
    if inner_text and len(inner_text) > 0:
        return False  # Has content, keep it

    return True


def filter_elements(
    elements: List[Dict[str, Any]],
    viewport_w: int,
    viewport_h: int,
    iou_threshold: float = 0.95,
    containment_threshold: float = 0.98,
    min_box_size: int = 4,
    max_box_size: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Filter elements to remove duplicates, tiny boxes, and hidden elements.

    Args:
        elements: List of element dictionaries with 'rect' and 'type' keys
        viewport_w: Viewport width in pixels
        viewport_h: Viewport height in pixels
        iou_threshold: IoU threshold for duplicate detection (0-1)
        containment_threshold: Containment threshold for parent/child (0-1)
        min_box_size: Minimum box dimension in pixels
        max_box_size: Maximum box dimension in pixels

    Returns:
        Filtered list of elements
    """
    if not elements:
        return []

    filtered = []

    for el in elements:
        rect = el.get("rect", {})
        x, y, w, h = (
            rect.get("x", 0),
            rect.get("y", 0),
            rect.get("w", 0),
            rect.get("h", 0),
        )

        # Skip invalid boxes
        if w <= 0 or h <= 0:
            continue

        # Skip tiny boxes (but allow protected elements)
        if (w * h < min_box_size) and not is_protected(el):
            continue

        # Skip overly large boxes (but allow protected elements)
        if (w * h > max_box_size) and not is_protected(el):
            continue

        # Skip boxes mostly outside viewport
        if x + w < 0 or y + h < 0 or x > viewport_w or y > viewport_h:
            continue

        # Skip if less than 1% visible
        visible_w = min(x + w, viewport_w) - max(x, 0)
        visible_h = min(y + h, viewport_h) - max(y, 0)
        if visible_w <= 0 or visible_h <= 0:
            continue

        visible_area = visible_w * visible_h
        total_area = w * h
        if total_area > 0 and visible_area / total_area < 0.01:
            continue

        # Skip hidden elements
        if (el.get("aria_hidden") or "").lower() == "true":
            continue

        filtered.append(el)

    # Remove duplicates based on IoU
    if iou_threshold < 1.0:
        filtered = _remove_duplicates_iou(filtered, iou_threshold)

    # Remove parent containers when child has same type
    filtered = _remove_parent_containers(filtered, containment_threshold)

    return filtered


def _remove_duplicates_iou(
    elements: List[Dict[str, Any]], iou_threshold: float
) -> List[Dict[str, Any]]:
    """Remove duplicate elements based on IoU overlap."""
    if not elements:
        return []

    # Sort by area (descending) to keep larger boxes
    sorted_elements = sorted(
        elements, key=lambda e: e["rect"]["w"] * e["rect"]["h"], reverse=True
    )

    keep = []
    for i, el_i in enumerate(sorted_elements):
        # Check if this element overlaps too much with any kept element
        is_duplicate = False
        protected_i = is_protected(el_i)

        for el_j in keep:
            iou = _compute_iou(el_i["rect"], el_j["rect"])
            if iou > iou_threshold:
                # If current element is protected but existing isn't, replace
                protected_j = is_protected(el_j)
                if protected_i and not protected_j:
                    keep.remove(el_j)
                    break
                else:
                    is_duplicate = True
                    break

        if not is_duplicate:
            keep.append(el_i)

    return keep


def _remove_parent_containers(
    elements: List[Dict[str, Any]], containment_threshold: float
) -> List[Dict[str, Any]]:
    """Remove parent containers when children have same type."""
    if not elements:
        return []

    keep = []

    for i, el_i in enumerate(elements):
        is_redundant = False
        type_i = (el_i.get("type") or "").lower()
        protected_i = is_protected(el_i)

        # Check if this element contains other elements of the same type
        for j, el_j in enumerate(elements):
            if i == j:
                continue

            type_j = (el_j.get("type") or "").lower()

            # Only consider same types
            if type_i != type_j:
                continue

            # Check containment
            containment = _compute_containment(el_j["rect"], el_i["rect"])

            if containment > containment_threshold:
                # el_j is contained in el_i, so el_i might be redundant parent
                protected_j = is_protected(el_j)

                # Keep parent if it's protected and child isn't
                if protected_i and not protected_j:
                    continue

                # Otherwise mark parent as redundant
                is_redundant = True
                break

        if not is_redundant:
            keep.append(el_i)

    return keep


def _compute_iou(rect1: Dict[str, int], rect2: Dict[str, int]) -> float:
    """Compute Intersection over Union of two rectangles."""
    x1 = max(rect1["x"], rect2["x"])
    y1 = max(rect1["y"], rect2["y"])
    x2 = min(rect1["x"] + rect1["w"], rect2["x"] + rect2["w"])
    y2 = min(rect1["y"] + rect1["h"], rect2["y"] + rect2["h"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = rect1["w"] * rect1["h"]
    area2 = rect2["w"] * rect2["h"]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _compute_containment(inner: Dict[str, int], outer: Dict[str, int]) -> float:
    """Compute how much of inner rectangle is contained in outer."""
    x1 = max(inner["x"], outer["x"])
    y1 = max(inner["y"], outer["y"])
    x2 = min(inner["x"] + inner["w"], outer["x"] + outer["w"])
    y2 = min(inner["y"] + inner["h"], outer["y"] + outer["h"])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    inner_area = inner["w"] * inner["h"]

    return intersection / inner_area if inner_area > 0 else 0.0


def analyze_filtering_impact(
    original: List[Dict[str, Any]], filtered: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze what was removed during filtering."""
    removed = len(original) - len(filtered)

    # Count by type
    original_types = {}
    filtered_types = {}

    for el in original:
        elem_type = (el.get("type") or "unknown").lower()
        original_types[elem_type] = original_types.get(elem_type, 0) + 1

    for el in filtered:
        elem_type = (el.get("type") or "unknown").lower()
        filtered_types[elem_type] = filtered_types.get(elem_type, 0) + 1

    removed_types = {}
    for elem_type, count in original_types.items():
        removed_count = count - filtered_types.get(elem_type, 0)
        if removed_count > 0:
            removed_types[elem_type] = removed_count

    # Check for protected elements that were removed
    protected_removed = 0
    protected_removed_types = {}

    filtered_set = set(id(el) for el in filtered)
    for el in original:
        if id(el) not in filtered_set and is_protected(el):
            protected_removed += 1
            elem_type = (el.get("type") or "unknown").lower()
            protected_removed_types[elem_type] = (
                protected_removed_types.get(elem_type, 0) + 1
            )

    return {
        "total_removed": removed,
        "removal_rate": removed / len(original) if original else 0,
        "removed_types": removed_types,
        "protected_removed": protected_removed,
        "protected_removed_types": protected_removed_types,
    }
