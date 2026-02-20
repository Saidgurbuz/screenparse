"""Conservative element filtering - prioritize keeping valid UI elements."""

from typing import List, Dict, Any, Set, Tuple
import numpy as np
from .utils import get_element_class


def remap_hierarchy_after_filtering(
    original_elements: List[Dict[str, Any]],
    filtered_elements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    After filtering, remap parent-child relationships.
    
    If an intermediate container was filtered out, children should point
    to the nearest surviving ancestor. This preserves the logical hierarchy
    even when redundant containers are removed.
    
    Args:
        original_elements: Full list with _dom_index, _parent_dom_index, _children_dom_indices
        filtered_elements: Subset that passed filtering
    
    Returns:
        filtered_elements with updated parent_index and children_indices fields
    """
    if not filtered_elements:
        return []
    
    # Build mapping from old DOM index to new filtered index
    old_to_new: Dict[int, int] = {}
    for new_idx, el in enumerate(filtered_elements):
        old_idx = el.get("_dom_index")
        if old_idx is not None:
            old_to_new[old_idx] = new_idx
    
    # Build a lookup of original elements by their DOM index
    original_by_dom_idx: Dict[int, Dict[str, Any]] = {}
    for el in original_elements:
        dom_idx = el.get("_dom_index")
        if dom_idx is not None:
            original_by_dom_idx[dom_idx] = el
    
    # For each filtered element, find its nearest surviving ancestor
    for el in filtered_elements:
        old_parent_idx = el.get("_parent_dom_index")
        
        # Walk up the chain until we find a parent that survived filtering
        new_parent_idx = None
        current_parent_dom_idx = old_parent_idx
        
        while current_parent_dom_idx is not None:
            if current_parent_dom_idx in old_to_new:
                # This parent survived filtering
                new_parent_idx = old_to_new[current_parent_dom_idx]
                break
            else:
                # This parent was filtered out, go to its parent
                parent_el = original_by_dom_idx.get(current_parent_dom_idx)
                if parent_el:
                    current_parent_dom_idx = parent_el.get("_parent_dom_index")
                else:
                    break
        
        el["parent_index"] = new_parent_idx
    
    # Build parent -> children mapping from the parent_index we just set
    parent_to_children: Dict[int, List[int]] = {}
    for new_idx, el in enumerate(filtered_elements):
        parent_idx = el.get("parent_index")
        if parent_idx is not None:
            parent_to_children.setdefault(parent_idx, []).append(new_idx)
    
    # Update children_indices based on who actually points to this element as parent
    for new_idx, el in enumerate(filtered_elements):
        el["children_indices"] = parent_to_children.get(new_idx, [])
    
    return filtered_elements


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
    y2 = min(inner["y"] + inner["h"], outer["y"] + outer["h"])

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
# Based on the 55 canonical classes from vlm_refine.py
PROTECTED_TYPES = {
    # Interactive controls
    "button",
    "utility button",
    "slider",
    "picker",
    "switch",
    "toggles",
    "steppers",
    "checkbox",
    "radiobox",
    "select",
    "text input",
    "search field",
    "search bar",
    "date-time picker",
    "rating indicator",
    "input",
    "textarea",
    "radio",
    # Navigation elements
    "navigation bar",
    "tab bar",
    "tab",
    "side bar",
    "breadcrumb",
    "bottom navigation",
    "page control",
    "pagination",
    "link",
    "menu",
    "contextmenu",
    "dockmenu",
    "editmenu",
    "popup menu",
    "toolbar",
    # Content elements
    "image",
    "video",
    "audio",
    "chart",
    "table",
    "list",
    "list item",
    "list_item",
    "avatar",
    "logo",
    "code snippet",
    "carousel",
    "calendar",
    # NOTE: "text" is intentionally excluded — it is the generic fallback type for any
    # unrecognised div/span/section. Protecting it would shield ALL generic containers
    # from filtering, which causes massive annotation clutter. Semantic text elements
    # (p, h1-h6, span) are already protected via their HTML tags below.
    "heading",
    "title",
    # Feedback/notification elements
    "tooltip",
    "alert",
    "notification",
    "badge",
    "progress bar",
    # Icons
    "app icon",
    "file icon",
    "icon",
    "svg",
    # UI components
    "card",
    "modal",
    "chip",
    "form",
    "scroll",
    "status bar",
    "window",
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

    Uses the module-level PROTECTED_TYPES constant based on the 55 canonical classes.
    """
    # Safely get values with None handling - use (value or "") pattern
    elem_type = (element.get("type") or "").lower()
    role = (element.get("role") or "").lower()
    tag = (element.get("tag") or "").lower()
    classes = (element.get("classes") or "").lower()
    attrs = element.get("attrs") or {}

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

    # Check interactive tags - HTML elements that are typically important
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


def _prune_wrapper_containers(
    elements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove generic div wrappers that are purely structural shells.

    A div is pruned when ALL of these hold:
    - tag is "div"
    - type is "text" (generic fallback — no semantic type detected)
    - no meaningful ARIA role
    - no interactive indicators (onClick, interactive class names, etc.)
    - has at least one child in the filtered set

    This aggressively removes layout shells (e.g. <div> wrapping a <nav>,
    <div> wrapping a single <button>) while the children themselves are kept.
    The benefit is much cleaner annotations without redundant containers.
    The risk (losing meaningful grouping divs) is handled by the IoU-dedup
    fix: we now preserve different-type element pairs, so text blocks are
    not lost when they overlap with link elements.
    """
    MEANINGFUL_ROLES = {
        "button", "link", "navigation", "banner", "contentinfo", "main",
        "search", "region", "complementary", "form", "alert", "dialog",
        "menu", "menubar", "tab", "tablist", "tabpanel", "listbox", "combobox",
    }

    keep = []
    for el in elements:
        tag = (el.get("tag") or "").lower()
        el_type = (el.get("type") or "").lower()
        children = el.get("children_indices", [])
        role = (el.get("role") or "").lower()

        if (
            tag == "div"                       # only plain divs (not section/article/main)
            and el_type == "text"              # generic fallback type only
            and role not in MEANINGFUL_ROLES
            and not has_interactive_indicators(el)
            and len(children) >= 1             # any wrapper with children is a structural shell
        ):
            continue  # drop: children already represent all content

        keep.append(el)
    return keep


# HTML tags that are semantically "text blocks" by definition.
# These can be safely recovered when they have own text that isn't
# represented in the filtered set. Layout containers (div, section,
# article, etc.) are explicitly excluded — they're structural, not textual.
_TEXT_SEMANTIC_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "dt", "dd", "blockquote"}

_MIN_OWN_TEXT_LEN = 15  # characters


def get_leaf_elements(
    filtered: List[Dict[str, Any]],
    original_elements: List[Dict[str, Any]],
    min_own_text: int = _MIN_OWN_TEXT_LEN,
) -> List[Dict[str, Any]]:
    """
    Extract the leaf-level coverage set from the filtered element list.

    Returns a flat, reading-order-sorted list that together provides complete
    UI coverage without redundant containers:

    - **True leaves**: elements with no children in the filtered hierarchy
      (buttons, links, images, icons, inputs, plain text nodes, …).
    - **Mixed-content text blocks**: elements that *do* have children but also
      carry their own direct text (prose text interspersed with inline links).
      Example: ``<p>Intro, see <a>link</a> and <a>link</a> for more.</p>``
      Their children are already included as leaves, but the surrounding prose
      text is only represented by the parent element.

    The list is sorted by ``reading_order_index`` (annotated in-place by the
    worker before saving), giving a flat sequence suitable for basic screen
    parsing, text extraction, or a leaf-only reading-order visualization.
    """
    original_by_dom: Dict[int, Dict[str, Any]] = {
        el["_dom_index"]: el
        for el in original_elements
        if el.get("_dom_index") is not None
    }

    leaf_elements: List[Dict[str, Any]] = []

    for el in filtered:
        children = el.get("children_indices") or []

        if not children:
            # True leaf: no children in the filtered hierarchy.
            leaf_elements.append(el)
            continue

        # Has children — include only if it contributes its own text content.
        child_dom_indices: List[int] = el.get("_children_dom_indices") or []
        inner_text = (el.get("inner_text") or "").strip()
        if not inner_text:
            continue

        own_text = inner_text
        for cdom in child_dom_indices:
            child = original_by_dom.get(cdom)
            if child:
                ct = (child.get("inner_text") or "").strip()
                if ct:
                    own_text = own_text.replace(ct, "", 1)
        own_text = own_text.strip()

        if len(own_text) >= min_own_text:
            leaf_elements.append(el)

    # Sort by reading order already annotated by the worker.
    # Elements without the annotation (shouldn't normally happen) go last.
    leaf_elements.sort(key=lambda e: e.get("reading_order_index", float("inf")))

    return leaf_elements


def _recover_mixed_content_text(
    filtered: List[Dict[str, Any]],
    original_elements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Recover text-semantic elements that were dropped by IoU deduplication
    despite having their own direct text content (text nodes mixed with
    inline children such as links).

    Classic case:
        <p>Intro text, see <a>link 1</a> and <a>link 2</a> for details.</p>

    After normal filtering the two <a> elements are kept but the <p> is
    dropped because one of its children has a high-IoU bounding box with it.
    The "Intro text ... for details." text is then invisible in the annotations.

    An element is recovered when ALL three conditions hold:
        1. tag is a text-semantic HTML element (p, h1-h6, li, dt, dd, blockquote)
           — this excludes layout containers (div, section, article, …) entirely.
        2. It has meaningful "own text" — direct text-node content that is NOT
           contributed by any of its children (computed by subtracting each
           child's inner_text from the element's inner_text).
        3. At least one of its direct DOM children is already in the filtered
           set — confirming this is the mixed-content case (children survived
           but the wrapping text element did not).
    """
    if not original_elements:
        return filtered

    filtered_dom_indices: Set[int] = {
        el["_dom_index"]
        for el in filtered
        if el.get("_dom_index") is not None
    }

    original_by_dom_idx: Dict[int, Dict[str, Any]] = {
        el["_dom_index"]: el
        for el in original_elements
        if el.get("_dom_index") is not None
    }

    to_recover: List[Dict[str, Any]] = []

    for el in original_elements:
        dom_idx = el.get("_dom_index")
        if dom_idx in filtered_dom_indices:
            continue  # already kept

        # Gate 1: text-semantic tag only
        tag = (el.get("tag") or "").lower()
        if tag not in _TEXT_SEMANTIC_TAGS:
            continue

        # Must have at least one child (otherwise it's a plain leaf — normal
        # filtering should have kept it; if it was dropped for another reason
        # we don't want to blindly re-add it here)
        child_dom_indices: List[int] = el.get("_children_dom_indices") or []
        if not child_dom_indices:
            continue

        # Gate 2: meaningful own text
        inner_text = (el.get("inner_text") or "").strip()
        if not inner_text:
            continue

        own_text = inner_text
        for cdom in child_dom_indices:
            child = original_by_dom_idx.get(cdom)
            if child:
                ct = (child.get("inner_text") or "").strip()
                if ct:
                    own_text = own_text.replace(ct, "", 1)
        own_text = own_text.strip()

        if len(own_text) < _MIN_OWN_TEXT_LEN:
            continue  # pure container or negligible own text

        # Gate 3: at least one direct child is in the filtered set
        if not any(c in filtered_dom_indices for c in child_dom_indices):
            continue  # children were also dropped — different issue, don't recover

        to_recover.append(el)

    if not to_recover:
        return filtered

    merged = filtered + to_recover
    return remap_hierarchy_after_filtering(original_elements, merged)


def _is_positional_orphan(
    element: Dict[str, Any],
    original_by_dom: Dict[int, Dict[str, Any]],
    tolerance: int = 5,
) -> bool:
    """
    Detect elements positioned completely outside their ancestor hierarchy's
    effective horizontal bounds — "carousel ghost" elements.

    These appear in sites that use horizontally-scrollable card decks / sliders
    where inactive cards are positioned to the right of the active slot via CSS
    absolute-positioning or transforms.  The cards are NOT clipped by
    overflow:hidden on their container (the JS isVisible check only blocks
    elements completely outside an overflow:hidden ancestor), so they pass DOM
    visibility but are invisible in the screenshot.

    Pattern (NYTimes reporter-thread-carddeck):
      <div slot  x=120 w=291>         ← the visible "slot"
        <div track x=451 w=291>       ← card 2's wrapper, moved to the right
          <a   card x=451 w=291>      ← the ghost element (outside slot)
            <div name x=501 w=84>     ← also ghost (nested inside ghost card)

    Algorithm — running x-intersection:
      Maintain the element's effective visible x-range starting as [el_x, el_x2].
      Walk up the _parent_dom_index chain.  For each ancestor at least as wide
      as the element (narrower ancestors are skipped — e.g. a tooltip is wider
      than its trigger button, and the containing ancestor is further up):
        • Intersect the running range with the ancestor's x-range.
        • If the intersection becomes empty → the element is outside that
          ancestor → positional orphan.
      This naturally catches nested ghosts (e.g. a name label inside a ghost
      card) without requiring special-casing.
    """
    el_rect = element["rect"]
    el_w = el_rect["w"]

    # Running visible x-interval; narrows as we walk up the ancestor chain.
    vis_x = el_rect["x"]
    vis_x2 = vis_x + el_w

    dom = element.get("_parent_dom_index")
    for _ in range(12):
        if dom is None:
            break
        ancestor = original_by_dom.get(dom)
        if not ancestor:
            break
        anc_rect = ancestor["rect"]
        anc_w = anc_rect["w"]
        anc_x = anc_rect["x"]
        anc_x2 = anc_x + anc_w

        # Skip ancestors narrower than the element — they cannot meaningfully
        # clip it (e.g. a tooltip wider than its trigger button).
        if anc_w < el_w - tolerance:
            dom = ancestor.get("_parent_dom_index")
            continue

        # Intersect running range with this ancestor's x-range.
        vis_x = max(vis_x, anc_x)
        vis_x2 = min(vis_x2, anc_x2)

        if vis_x2 <= vis_x:
            # Intersection is empty → element lies outside this ancestor.
            return True

        dom = ancestor.get("_parent_dom_index")

    return False


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
    
    Also remaps parent-child hierarchy after filtering so that children of
    removed parents point to their nearest surviving ancestor.

    Args:
        elements: List of element dictionaries with 'rect' and 'type' keys
        viewport_w: Viewport width in pixels
        viewport_h: Viewport height in pixels
        iou_threshold: IoU threshold for duplicate detection (0-1)
        containment_threshold: Containment threshold for parent/child (0-1)
        min_box_size: Minimum box dimension in pixels
        max_box_size: Maximum box dimension in pixels

    Returns:
        Filtered list of elements with updated parent_index and children_indices
    """
    if not elements:
        return []

    # Keep reference to original elements for hierarchy remapping
    original_elements = elements

    # Fast lookup by DOM index — needed for positional-orphan detection
    original_by_dom: Dict[int, Dict[str, Any]] = {
        el["_dom_index"]: el
        for el in original_elements
        if el.get("_dom_index") is not None
    }

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

        # Hard floor: always reject truly invisible elements regardless of protection.
        # width=1 or height=1 is the CSS sr-only trick (e.g. "Skip to content" links,
        # off-screen headings) — they have zero visual presence in the screenshot.
        if w <= 1 or h <= 1:
            continue

        # Skip tiny boxes (but allow protected elements)
        if (w * h < min_box_size) and not is_protected(el):
            continue


        element_type = (el.get("type") or "").lower()
        # Skip overly large boxes (but allow image elements)
        if (w * h > max_box_size) and not (element_type == "image"):
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

        
        # Skip tiny SVG paths but keep large ones (standalone icons/graphics)
        tag = (el.get("tag") or "").lower()
        if tag == "path":
            if rect.get("w", 0) * rect.get("h", 0) < 400:  # ~20x20 threshold
                continue

        # Skip hidden elements
        if (el.get("aria_hidden") or "").lower() == "true":
            continue

        # Skip carousel/slider ghost cards: elements positioned completely
        # outside their nearest meaningful DOM ancestor's horizontal bounds.
        if _is_positional_orphan(el, original_by_dom):
            continue

        filtered.append(el)

    # Remove duplicates based on IoU
    if iou_threshold < 1.0:
        filtered = _remove_duplicates_iou(filtered, iou_threshold)


    # Remap hierarchy after all filtering is done.
    # This ensures children of removed parents point to their nearest surviving ancestor.
    filtered = remap_hierarchy_after_filtering(original_elements, filtered)

    # Prune generic layout wrapper containers (div/section shells that only wrap children).
    # Now that children_indices are populated we can detect them reliably.
    pruned = _prune_wrapper_containers(filtered)
    if len(pruned) != len(filtered):
        # Remap hierarchy again so indices are valid for the pruned list.
        filtered = remap_hierarchy_after_filtering(original_elements, pruned)
    else:
        filtered = pruned

    # Recover text-semantic elements (p, h1-h6, li, …) that have their own
    # direct text content but were dropped by IoU deduplication because an
    # inline child (e.g. <a>) had an overlapping bounding box.
    filtered = _recover_mixed_content_text(filtered, original_elements)

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
            if get_element_class(el_i).lower() == get_element_class(el_j).lower():
                # if the elements are of the same type, we want to be more aggressive
                if iou > iou_threshold * 0.65:
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
        type_i = get_element_class(el_i).lower()
        protected_i = is_protected(el_i)

        # Check if this element contains other elements of the same type
        for j, el_j in enumerate(elements):
            if i == j:
                continue

            type_j = get_element_class(el_j).lower()

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
