"""
Reconstruct parent-child hierarchy from bounding box geometry.

This module provides functionality to infer parent-child relationships
between UI elements based on their bounding boxes when the DOM hierarchy
wasn't captured during rendering.

The algorithm:
1. For each element, find the smallest containing element (by area) that
   fully contains it - this becomes its parent.
2. Build children lists from the parent relationships.
3. Handle edge cases like overlapping elements, same-size elements, etc.

Usage:
    wsd reconstruct-hierarchy --raw-dir data/raw [--min-containment 0.95]
"""

import os
import glob
from typing import List, Dict, Any, Optional, Set, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from .utils import load_json, save_json, get_element_class


# Import class definitions for semantic awareness
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

# Elements that should never be parents (leaf nodes)
LEAF_ONLY_CLASSES = {
    "Text",
    "Heading", 
    "Image",
    "Logo",
    "Avatar",
    "Badge",
    "Checkbox",
    "Radiobox",
    "Switch",
    "Slider",
    "Progress bar",
    "Rating Indicator",
}

# Elements whose text should always be kept independently even if parent has it
# (clickable elements, interactive elements)
INTERACTIVE_CLASSES = {
    "Button",
    "Utility Button",
    "Link",
    "Tab",
    "Checkbox",
    "Radiobox",
    "Switch",
    "Select",
    "Text Input",
    "Search Field",
}

# Elements that typically contain icon/visual content with independent meaning
ICON_CLASSES = {
    "App Icon",
    "File Icon",
    "Logo",
    "Avatar",
    "Image",
}


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by collapsing whitespace and lowercasing.
    """
    import re
    if not text:
        return ""
    # Collapse all whitespace to single spaces and strip
    return re.sub(r'\s+', ' ', text.strip().lower())


def compute_own_text(
    element_idx: int,
    elements: List[Dict[str, Any]],
    children_indices: List[int],
) -> str:
    """
    Compute the 'own_text' for an element - text that belongs directly to this
    element and is not duplicated from its children.
    
    Algorithm:
    1. If no children or no children with text: own_text = inner_text
    2. Collect all children's inner_text
    3. Try to subtract children texts from parent text
    4. Handle special cases:
       - If child is interactive (link/button), keep its text separately
       - If texts are completely independent (no overlap), keep both
       - If parent text is just concatenation of children, parent gets empty
    
    Args:
        element_idx: Index of the element to compute own_text for
        elements: Full list of all elements
        children_indices: List of children indices for this element
    
    Returns:
        The own_text string for this element
    """
    element = elements[element_idx]
    inner_text = (element.get("inner_text") or "").strip()
    
    # No inner_text at all - easy case
    if not inner_text:
        return ""
    
    # No children - all text belongs to this element
    if not children_indices:
        return inner_text
    
    # Collect children's texts
    children_texts = []
    for c_idx in children_indices:
        if c_idx < len(elements):
            c_text = (elements[c_idx].get("inner_text") or "").strip()
            if c_text:
                c_type = get_element_class(elements[c_idx])
                c_tag = elements[c_idx].get("tag", "")
                children_texts.append({
                    "idx": c_idx,
                    "text": c_text,
                    "normalized": normalize_text(c_text),
                    "type": c_type,
                    "tag": c_tag,
                    "is_interactive": c_type in INTERACTIVE_CLASSES or c_tag == "a",
                    "is_icon": c_type in ICON_CLASSES,
                })
    
    # No children have text - all text belongs to this element
    if not children_texts:
        return inner_text
    
    # Normalize parent text for comparison
    parent_normalized = normalize_text(inner_text)
    
    # Check if parent text is exactly the concatenation of children texts
    # (order may vary, so we check all permutations for small sets or use containment)
    all_children_text_combined = " ".join(c["normalized"] for c in children_texts)
    
    # Case 1: Parent text is exactly children texts concatenated
    if parent_normalized == all_children_text_combined:
        return ""
    
    # Case 2: Check if children texts are substrings of parent
    # Special case: if a LINK's text is a SMALL SUBSTRING of parent text
    # (not the full text, just a few words), keep BOTH parent and child text.
    # This is because the link marks a clickable portion within a sentence.
    import re
    remaining_text = inner_text
    texts_to_remove = []
    
    for c in children_texts:
        c_text = c["text"]
        c_normalized = c["normalized"]
        c_tag = c["tag"]
        
        # Check if child is a link (anchor tag)
        is_link = c_tag == "a"
        
        # Check if child text matches parent text exactly (normalized)
        is_exact_match = c_normalized == parent_normalized
        
        # Check if this child's text is a substring of the parent
        is_substring = c_text in remaining_text
        if not is_substring:
            # Try case-insensitive match
            pattern = re.escape(c_normalized)
            match = re.search(pattern, remaining_text, re.IGNORECASE)
            is_substring = match is not None
        
        # Special case: Link with text that is a SMALL portion of parent text
        # If it's a link AND it's a proper substring (not exact match),
        # we keep BOTH the parent's full text and the link's text
        if is_link and is_substring and not is_exact_match:
            # Check if the link text is a "small" portion of parent text
            # (less than ~80% of parent text length means it's a partial link)
            ratio = len(c_normalized) / len(parent_normalized) if parent_normalized else 1
            if ratio < 0.3:
                # This is a link that's just a small part of the sentence
                # Don't remove it from parent - keep both
                continue
        
        # Normal case: remove child text from parent
        if c_text in remaining_text:
            texts_to_remove.append((c_text, c_text))
        else:
            pattern = re.escape(c_normalized)
            match = re.search(pattern, remaining_text, re.IGNORECASE)
            if match:
                texts_to_remove.append((match.group(), c_text))
    
    # Sort by length descending to remove longer texts first (greedy)
    texts_to_remove.sort(key=lambda x: len(x[0]), reverse=True)
    
    for pattern, _ in texts_to_remove:
        # Only remove the first occurrence to avoid over-removal
        remaining_text = re.sub(re.escape(pattern), " ", remaining_text, count=1, flags=re.IGNORECASE)
    
    # Clean up remaining text
    remaining_text = re.sub(r'\s+', ' ', remaining_text).strip()
    
    # If parent had some text but children covered it all, return empty
    if not remaining_text:
        return ""
    
    # If remaining text is very short (likely just punctuation/whitespace artifacts)
    if len(remaining_text) <= 2 and remaining_text in ".,;:!?-–—•·":
        return ""
    
    return remaining_text


def compute_own_text_for_all(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute own_text for all elements in the list.
    
    This should be called after hierarchy reconstruction so that
    children_indices is populated.
    
    Args:
        elements: List of elements with hierarchy info (children_indices)
    
    Returns:
        Elements with 'own_text' field added
    """
    for i, el in enumerate(elements):
        children_indices = el.get("children_indices", [])
        own_text = compute_own_text(i, elements, children_indices)
        el["own_text"] = own_text
    
    return elements


def process_single_file_own_text(
    elements_path: str,
    force: bool = False,
) -> Tuple[str, bool, str]:
    """
    Process a single elements.json file to add own_text field.
    
    Args:
        elements_path: Path to the elements.json file
        force: Recompute own_text even if it already exists
    
    Returns:
        Tuple of (path, success, message)
    """
    try:
        elements = load_json(elements_path)
        
        if not isinstance(elements, list):
            return (elements_path, False, "Invalid format: not a list")
        
        if not elements:
            return (elements_path, False, "Empty elements list")
        
        # Check if own_text already exists
        if "own_text" in elements[0] and not force:
            return (elements_path, True, "Already has own_text (skipped)")
        
        # Check if hierarchy exists (needed for own_text computation)
        if not has_hierarchy_info(elements):
            return (elements_path, False, "No hierarchy info - run reconstruct-hierarchy first")
        
        # Compute own_text
        elements = compute_own_text_for_all(elements)
        
        # Save back to file
        save_json(elements_path, elements)
        
        return (elements_path, True, f"Added own_text to {len(elements)} elements")
    
    except Exception as e:
        return (elements_path, False, f"Error: {str(e)}")


def add_own_text_for_directory(
    raw_dir: str,
    force: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    """
    Add own_text field to all elements.json files in a directory.
    
    Args:
        raw_dir: Directory containing elements.json files
        force: Recompute own_text even if it already exists
        workers: Number of parallel workers
    
    Returns:
        Statistics dictionary
    """
    # Find all elements.json files
    pattern = os.path.join(raw_dir, "*.elements.json")
    elements_files = sorted(glob.glob(pattern))
    
    if not elements_files:
        print(f"No elements.json files found in {raw_dir}")
        return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
    
    print(f"Found {len(elements_files)} elements.json files")
    print(f"Settings: force={force}")
    
    stats = {
        "total": len(elements_files),
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "error_files": [],
    }
    
    if workers > 1:
        # Parallel processing
        from functools import partial
        process_fn = partial(
            process_single_file_own_text,
            force=force,
        )
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_fn, f): f for f in elements_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Computing own_text"):
                path, success, message = future.result()
                
                if success:
                    if "skipped" in message.lower():
                        stats["skipped"] += 1
                    else:
                        stats["processed"] += 1
                else:
                    stats["errors"] += 1
                    stats["error_files"].append((path, message))
    else:
        # Sequential processing
        for elements_path in tqdm(elements_files, desc="Computing own_text"):
            path, success, message = process_single_file_own_text(
                elements_path,
                force=force,
            )
            
            if success:
                if "skipped" in message.lower():
                    stats["skipped"] += 1
                else:
                    stats["processed"] += 1
            else:
                stats["errors"] += 1
                stats["error_files"].append((path, message))
    
    return stats


def has_hierarchy_info(elements: List[Dict[str, Any]]) -> bool:
    """
    Check if elements already have hierarchy information.
    
    Returns True if the elements have valid _dom_index and parent_index fields.
    """
    if not elements:
        return False
    
    # Check first element for hierarchy fields
    first = elements[0]
    
    # If _dom_index exists and is a valid integer, hierarchy exists
    if "_dom_index" in first and isinstance(first.get("_dom_index"), int):
        return True
    
    # If parent_index exists (even if None for root), hierarchy exists
    if "parent_index" in first:
        return True
    
    return False


def is_container_class(class_name: Optional[str]) -> bool:
    """Check if a class is a container type."""
    if not class_name:
        return False
    return class_name in CONTAINER_CLASS_NAMES


def is_leaf_class(class_name: Optional[str]) -> bool:
    """Check if a class should be a leaf node (no children)."""
    if not class_name:
        return False
    return class_name in LEAF_ONLY_CLASSES


def compute_containment(inner: Dict[str, int], outer: Dict[str, int]) -> float:
    """
    Compute what fraction of inner box is contained within outer box.
    
    Returns a value between 0 and 1, where 1 means inner is fully contained.
    """
    inner_x1, inner_y1 = inner["x"], inner["y"]
    inner_x2, inner_y2 = inner_x1 + inner["w"], inner_y1 + inner["h"]
    
    outer_x1, outer_y1 = outer["x"], outer["y"]
    outer_x2, outer_y2 = outer_x1 + outer["w"], outer_y1 + outer["h"]
    
    # Intersection
    ix1 = max(inner_x1, outer_x1)
    iy1 = max(inner_y1, outer_y1)
    ix2 = min(inner_x2, outer_x2)
    iy2 = min(inner_y2, outer_y2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    
    intersection = (ix2 - ix1) * (iy2 - iy1)
    inner_area = inner["w"] * inner["h"]
    
    if inner_area <= 0:
        return 0.0
    
    return intersection / inner_area


def get_box_area(rect: Dict[str, int]) -> int:
    """Get the area of a bounding box."""
    return rect["w"] * rect["h"]


def boxes_equal(rect1: Dict[str, int], rect2: Dict[str, int], tolerance: int = 2) -> bool:
    """Check if two boxes are essentially the same (within tolerance)."""
    return (
        abs(rect1["x"] - rect2["x"]) <= tolerance and
        abs(rect1["y"] - rect2["y"]) <= tolerance and
        abs(rect1["w"] - rect2["w"]) <= tolerance and
        abs(rect1["h"] - rect2["h"]) <= tolerance
    )


def reconstruct_hierarchy(
    elements: List[Dict[str, Any]],
    min_containment: float = 0.85,
    use_semantic_hints: bool = False,
) -> List[Dict[str, Any]]:
    """
    Reconstruct parent-child hierarchy from bounding box geometry.
    
    Algorithm:
    1. For each element, find the smallest containing element that:
       - Contains at least min_containment fraction of the element
       - Has a larger area than the element
       - Is not a leaf-only class (if use_semantic_hints=True)
    2. That element becomes the parent
    3. Build children lists from parent relationships
    
    Args:
        elements: List of element dictionaries with 'rect' field
        min_containment: Minimum fraction of element that must be contained
        use_semantic_hints: Use VLM labels to inform hierarchy decisions
    
    Returns:
        Elements with added hierarchy fields:
        - _dom_index: Index in the list (for compatibility)
        - _parent_dom_index: Original parent's index (same as parent_index here)
        - _children_dom_indices: List of children's indices
        - _depth: Depth in the reconstructed tree
        - parent_index: Index of parent element (None for roots)
        - children_indices: List of children element indices
    """
    if not elements:
        return []
    
    n = len(elements)
    
    # Extract rect and compute areas
    rects = []
    areas = []
    for el in elements:
        rect = el.get("rect", {"x": 0, "y": 0, "w": 0, "h": 0})
        rects.append(rect)
        areas.append(get_box_area(rect))
    
    # Get semantic classes if available
    classes = [get_element_class(el) for el in elements]
    
    # Sort indices by area (largest first) - helps with parent finding
    sorted_indices = sorted(range(n), key=lambda i: -areas[i])
    
    # Find parent for each element
    parent_indices: List[Optional[int]] = [None] * n
    
    for i in range(n):
        rect_i = rects[i]
        area_i = areas[i]
        class_i = classes[i]
        
        # Skip elements with zero area
        if area_i <= 0:
            continue
        
        best_parent = None
        best_parent_area = float('inf')
        
        for j in range(n):
            if i == j:
                continue
            
            rect_j = rects[j]
            area_j = areas[j]
            class_j = classes[j]
            
            # Parent must be larger (or at least equal with different position)
            if area_j < area_i:
                continue
            
            # Skip if boxes are essentially the same
            if boxes_equal(rect_i, rect_j):
                continue
            
            # Check containment
            containment = compute_containment(rect_i, rect_j)
            if containment < min_containment:
                continue
            
            # Apply semantic hints
            if use_semantic_hints and class_j:
                # Leaf classes should not be parents
                if is_leaf_class(class_j):
                    continue
            
            # Check if this is a better (smaller) parent
            if area_j < best_parent_area:
                best_parent = j
                best_parent_area = area_j
        
        parent_indices[i] = best_parent
    
    # Build children lists from parent relationships
    children_indices: List[List[int]] = [[] for _ in range(n)]
    for i, parent_idx in enumerate(parent_indices):
        if parent_idx is not None:
            children_indices[parent_idx].append(i)
    
    # Compute depths
    depths = [0] * n
    
    def compute_depth(idx: int, visited: Set[int]) -> int:
        if idx in visited:
            # Cycle detected - break it
            return 0
        visited.add(idx)
        
        parent_idx = parent_indices[idx]
        if parent_idx is None:
            return 0
        return 1 + compute_depth(parent_idx, visited)
    
    for i in range(n):
        depths[i] = compute_depth(i, set())
    
    # Update elements with hierarchy information
    for i, el in enumerate(elements):
        el["_dom_index"] = i
        el["_parent_dom_index"] = parent_indices[i]
        el["_children_dom_indices"] = children_indices[i].copy()
        el["_depth"] = depths[i]
        el["parent_index"] = parent_indices[i]
        el["children_indices"] = children_indices[i].copy()
    
    return elements


def process_single_file(
    elements_path: str,
    min_containment: float = 0.90,
    use_semantic_hints: bool = False,
    force: bool = False,
) -> Tuple[str, bool, str]:
    """
    Process a single elements.json file.
    
    Args:
        elements_path: Path to the elements.json file
        min_containment: Minimum containment threshold
        use_semantic_hints: Use VLM labels for hierarchy decisions
        force: Reconstruct even if hierarchy already exists
    
    Returns:
        Tuple of (path, success, message)
    """
    try:
        elements = load_json(elements_path)
        
        if not isinstance(elements, list):
            return (elements_path, False, "Invalid format: not a list")
        
        if not elements:
            return (elements_path, False, "Empty elements list")
        
        # Check if hierarchy already exists
        if has_hierarchy_info(elements) and not force:
            # Even if skipping hierarchy, we should still compute own_text if missing
            if elements and "own_text" not in elements[0]:
                elements = compute_own_text_for_all(elements)
                save_json(elements_path, elements)
                return (elements_path, True, "Added own_text to existing hierarchy")
            return (elements_path, True, "Already has hierarchy (skipped)")
        
        # Reconstruct hierarchy
        elements = reconstruct_hierarchy(
            elements,
            min_containment=min_containment,
            use_semantic_hints=use_semantic_hints,
        )
        
        # Compute own_text for all elements
        elements = compute_own_text_for_all(elements)
        
        # Save back to file
        save_json(elements_path, elements)
        
        # Count roots and max depth for stats
        roots = sum(1 for el in elements if el.get("parent_index") is None)
        max_depth = max(el.get("_depth", 0) for el in elements) if elements else 0
        
        return (elements_path, True, f"Reconstructed: {len(elements)} elements, {roots} roots, depth={max_depth}")
    
    except Exception as e:
        return (elements_path, False, f"Error: {str(e)}")


def reconstruct_hierarchy_for_directory(
    raw_dir: str,
    min_containment: float = 0.95,
    use_semantic_hints: bool = False,
    force: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    """
    Reconstruct hierarchy for all elements.json files in a directory.
    
    Args:
        raw_dir: Directory containing elements.json files
        min_containment: Minimum containment threshold for parent-child
        use_semantic_hints: Use VLM labels to inform decisions
        force: Reconstruct even if hierarchy already exists
        workers: Number of parallel workers
    
    Returns:
        Statistics dictionary
    """
    # Find all elements.json files
    pattern = os.path.join(raw_dir, "*.elements.json")
    elements_files = sorted(glob.glob(pattern))
    
    if not elements_files:
        print(f"No elements.json files found in {raw_dir}")
        return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
    
    print(f"Found {len(elements_files)} elements.json files")
    print(f"Settings: min_containment={min_containment}, semantic_hints={use_semantic_hints}, force={force}")
    
    stats = {
        "total": len(elements_files),
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "error_files": [],
    }
    
    if workers > 1:
        # Parallel processing
        from functools import partial
        process_fn = partial(
            process_single_file,
            min_containment=min_containment,
            use_semantic_hints=use_semantic_hints,
            force=force,
        )
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_fn, f): f for f in elements_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Reconstructing hierarchy"):
                path, success, message = future.result()
                
                if success:
                    if "skipped" in message.lower():
                        stats["skipped"] += 1
                    else:
                        stats["processed"] += 1
                else:
                    stats["errors"] += 1
                    stats["error_files"].append((path, message))
    else:
        # Sequential processing
        for elements_path in tqdm(elements_files, desc="Reconstructing hierarchy"):
            path, success, message = process_single_file(
                elements_path,
                min_containment=min_containment,
                use_semantic_hints=use_semantic_hints,
                force=force,
            )
            
            if success:
                if "skipped" in message.lower():
                    stats["skipped"] += 1
                else:
                    stats["processed"] += 1
            else:
                stats["errors"] += 1
                stats["error_files"].append((path, message))
    
    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    """Print reconstruction statistics."""
    print("\n" + "=" * 60)
    print("HIERARCHY RECONSTRUCTION COMPLETE")
    print("=" * 60)
    print(f"Total files:     {stats['total']}")
    print(f"Processed:       {stats['processed']}")
    print(f"Skipped:         {stats['skipped']} (already had hierarchy)")
    print(f"Errors:          {stats['errors']}")
    
    if stats["error_files"]:
        print("\nErrors:")
        for path, msg in stats["error_files"][:10]:
            print(f"  {os.path.basename(path)}: {msg}")
        if len(stats["error_files"]) > 10:
            print(f"  ... and {len(stats['error_files']) - 10} more")


# ─────────────────────────────────────────────────────────────────────────────
# ScreenTag Export
# ─────────────────────────────────────────────────────────────────────────────

def process_single_file_screentag(
    elements_path: str,
    force: bool = False,
) -> Tuple[str, bool, str]:
    """
    Process a single elements.json file to generate ScreenTag representation.
    
    This function:
    1. Loads elements and ensures hierarchy exists (via process_single_file if needed)
    2. Filters elements to remove duplicates, tiny boxes, and hidden elements
    3. Reconstructs hierarchy on filtered elements
    4. Computes own_text for text deduplication
    5. Generates ScreenTag representation and saves to .screentag.txt
    
    Note: The original elements.json is only modified if hierarchy was missing.
    Filtering is applied in-memory only for screentag generation.
    
    Args:
        elements_path: Path to the elements.json file
        force: Regenerate screentag even if it already exists
    
    Returns:
        Tuple of (path, success, message)
    """
    from .utils import elements_to_screentag
    from .filtering import filter_elements
    
    try:
        # Derive related file paths
        base_path = elements_path.replace(".elements.json", "")
        meta_path = f"{base_path}.meta.json"
        screentag_path = f"{base_path}.screentag.txt"
        
        # Check if screentag already exists
        if os.path.exists(screentag_path) and not force:
            return (elements_path, True, "Already has screentag (skipped)")
        
        # Check if meta.json exists
        if not os.path.exists(meta_path):
            return (elements_path, False, "Missing meta.json file")
        
        # Load elements
        elements = load_json(elements_path)
        
        if not isinstance(elements, list):
            return (elements_path, False, "Invalid format: not a list")
        
        if not elements:
            return (elements_path, False, "Empty elements list")
        
        # Ensure hierarchy exists (needed for filter_elements to remap properly)
        # This will save hierarchy to the file if it was missing
        if not has_hierarchy_info(elements):
            _, success, message = process_single_file(
                elements_path,
                min_containment=0.95,
                use_semantic_hints=False,
                force=False,
            )
            if not success:
                return (elements_path, False, f"Failed to prepare hierarchy: {message}")
            
            # Reload elements after processing
            elements = load_json(elements_path)
        
        # Load meta.json for screen dimensions
        meta = load_json(meta_path)
        
        # Get viewport from meta
        viewport = meta.get("viewport", {})
        vp_w = viewport.get("w", 0)
        vp_h = viewport.get("h", 0)
        
        if not vp_w or not vp_h:
            return (elements_path, False, f"Invalid viewport in meta: {viewport}")
        
        # Track original count for stats
        original_count = len(elements)
        
        # Filter elements to remove duplicates, tiny boxes, hidden elements
        # This also remaps hierarchy after filtering using _dom_index/_parent_dom_index
        elements = filter_elements(
            elements=elements,
            viewport_w=vp_w,
            viewport_h=vp_h,
            iou_threshold=0.95,
            containment_threshold=0.98,
            min_box_size=4,
            max_box_size=vp_w * vp_h,  # Use viewport area as max
        )
        
        if not elements:
            return (elements_path, False, "No elements after filtering")
        
        filtered_count = len(elements)
        
        # Note: filter_elements already remaps hierarchy via remap_hierarchy_after_filtering
        # which preserves the DOM-based hierarchy from Playwright. We do NOT call
        # reconstruct_hierarchy here because that would overwrite the accurate DOM
        # hierarchy with a geometry-based approximation.
        
        # Compute own_text for all elements
        elements = compute_own_text_for_all(elements)
        
        # Generate ScreenTag representation
        screentag = elements_to_screentag(
            elements=elements,
            viewport=viewport,
        )
        
        # Save to file
        with open(screentag_path, "w", encoding="utf-8") as f:
            f.write(screentag)
        
        return (elements_path, True, f"Generated screentag ({filtered_count}/{original_count} elements, {vp_w}x{vp_h})")
    
    except Exception as e:
        import traceback
        return (elements_path, False, f"Error: {str(e)}\n{traceback.format_exc()}")


def export_screentag_for_directory(
    raw_dir: str,
    force: bool = False,
    workers: int = 1,
) -> Dict[str, Any]:
    """
    Export ScreenTag representations for all elements.json files in a directory.
    
    Args:
        raw_dir: Directory containing elements.json files
        force: Regenerate screentag even if it already exists
        workers: Number of parallel workers
    
    Returns:
        Statistics dictionary
    """
    # Find all elements.json files
    pattern = os.path.join(raw_dir, "*.elements.json")
    elements_files = sorted(glob.glob(pattern))
    
    if not elements_files:
        print(f"No elements.json files found in {raw_dir}")
        return {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
    
    print(f"Found {len(elements_files)} elements.json files")
    print(f"Settings: force={force}")
    
    stats = {
        "total": len(elements_files),
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "error_files": [],
    }
    
    if workers > 1:
        # Parallel processing
        from functools import partial
        process_fn = partial(
            process_single_file_screentag,
            force=force,
        )
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_fn, f): f for f in elements_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Exporting screentag"):
                path, success, message = future.result()
                
                if success:
                    if "skipped" in message.lower():
                        stats["skipped"] += 1
                    else:
                        stats["processed"] += 1
                else:
                    stats["errors"] += 1
                    stats["error_files"].append((path, message))
    else:
        # Sequential processing
        for elements_path in tqdm(elements_files, desc="Exporting screentag"):
            path, success, message = process_single_file_screentag(
                elements_path,
                force=force,
            )
            
            if success:
                if "skipped" in message.lower():
                    stats["skipped"] += 1
                else:
                    stats["processed"] += 1
            else:
                stats["errors"] += 1
                stats["error_files"].append((path, message))
    
    return stats


def print_screentag_stats(stats: Dict[str, Any]) -> None:
    """Print screentag export statistics."""
    print("\n" + "=" * 60)
    print("SCREENTAG EXPORT COMPLETE")
    print("=" * 60)
    print(f"Total files:     {stats['total']}")
    print(f"Processed:       {stats['processed']}")
    print(f"Skipped:         {stats['skipped']} (already had screentag)")
    print(f"Errors:          {stats['errors']}")
    
    if stats["error_files"]:
        print("\nErrors:")
        for path, msg in stats["error_files"][:10]:
            print(f"  {os.path.basename(path)}: {msg}")
        if len(stats["error_files"]) > 10:
            print(f"  ... and {len(stats['error_files']) - 10} more")

