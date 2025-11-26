import hashlib, re, json, os, orjson, unicodedata
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from slugify import slugify
from typing import List, Dict, Any, Optional, Tuple


def stable_hash(s: str, length: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:length]


def canonicalize_url(url: str) -> str:
    u = urlparse(url)
    q = sorted(
        (k, v)
        for k, v in parse_qsl(u.query, keep_blank_values=True)
        if not k.lower().startswith("utm_")
    )
    return urlunparse((u.scheme, u.netloc, u.path, u.params, urlencode(q), ""))


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_json(path: str, obj):
    with open(path, "wb") as f:
        f.write(orjson.dumps(obj, option=orjson.OPT_INDENT_2))


def load_json(path: str):
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def safe_stem_from_url(url: str) -> str:
    # human-ish but stable
    u = urlparse(url)
    base = f"{u.netloc}-{u.path}".strip("/")
    base = unicodedata.normalize("NFKC", base)
    base = slugify(base)[:80] or "page"
    return base


def rect_to_int(r):
    return {k: int(round(v)) for k, v in r.items()}


def get_processed_url_stems(out_dir: str) -> set:
    """
    Get set of URL stems for already-processed URLs.

    Scans output directory for .png files and extracts URL stems
    from filenames (format: stem-HASH.png).

    Args:
        out_dir: Directory containing processed files

    Returns:
        Set of URL stems (strings)
    """
    import os
    import glob

    if not os.path.exists(out_dir):
        return set()

    processed_stems = set()

    # Find all .png files (screenshot files)
    png_files = glob.glob(os.path.join(out_dir, "*.png"))

    for png_file in png_files:
        # Extract stem from filename: stem-HASH.png
        basename = os.path.basename(png_file)
        if basename.endswith(".png"):
            # Remove .png suffix
            name_without_ext = basename[:-4]  # len(".png") == 4
            # Stem is everything before the last hyphen
            parts = name_without_ext.rsplit("-", 1)
            if len(parts) == 2:
                url_stem = parts[0]
                processed_stems.add(url_stem)

    return processed_stems


def build_element_tree(elements: list) -> dict:
    """
    Convert flat element list with parent_index into a nested tree structure.
    
    Args:
        elements: List of elements with parent_index and children_indices fields
        
    Returns:
        Dict with:
            - roots: list of root element indices (those with no parent)
            - elements: the element list (for reference)
    """
    if not elements:
        return {"roots": [], "elements": []}
    
    # Find root elements (no parent)
    roots = []
    for idx, el in enumerate(elements):
        if el.get("parent_index") is None:
            roots.append(idx)
    
    return {
        "roots": roots,
        "elements": elements,
    }

def get_element_class(el: Dict[str, Any]) -> Optional[str]:
    """Get the VLM label or type of an element."""
    vlm_label = el.get("vlm_label", "").strip()
    if vlm_label and vlm_label.lower() != "unknown":
        return vlm_label
    
    el_type = el.get("type", "").strip()
    if el_type and el_type.lower() != "unknown":
        return el_type
    
    return None

# ─────────────────────────────────────────────────────────────────────────────
# ScreenTag utilities: location tokens and element rendering
# ─────────────────────────────────────────────────────────────────────────────

def get_location_token(val: float, rnorm: int = 500) -> str:
    """
    Convert a normalized coordinate value to a location token.
    
    Args:
        val: Normalized value between 0 and 1
        rnorm: Target range for normalization (default 500)
    
    Returns:
        Location token string like <loc_123>
    """
    # Clamp to [0, 1] and scale to [0, rnorm]
    val = max(0.0, min(1.0, val))
    loc_int = int(round(val * rnorm))
    # Ensure within bounds
    loc_int = max(0, min(rnorm, loc_int))
    return f"<loc_{loc_int}>"


def get_location_str(
    bbox: Tuple[float, float, float, float],
    page_w: float,
    page_h: float,
    xsize: int = 500,
    ysize: int = 500,
) -> str:
    """
    Get the location string given bbox (LTRB) and page dimensions.
    
    Args:
        bbox: Tuple of (left, top, right, bottom) in absolute coordinates
        page_w: Page/screen width
        page_h: Page/screen height
        xsize: X normalization size (default 500)
        ysize: Y normalization size (default 500)
    
    Returns:
        Location string like "<loc_10><loc_20><loc_100><loc_200>"
    """
    if page_w <= 0 or page_h <= 0:
        return "<loc_0><loc_0><loc_0><loc_0>"
    
    # Normalize to 0-1 range
    x0 = bbox[0] / page_w
    y0 = bbox[1] / page_h
    x1 = bbox[2] / page_w
    y1 = bbox[3] / page_h
    
    # Generate tokens (ensuring L <= R and T <= B)
    x0_tok = get_location_token(min(x0, x1), xsize)
    y0_tok = get_location_token(min(y0, y1), ysize)
    x1_tok = get_location_token(max(x0, x1), xsize)
    y1_tok = get_location_token(max(y0, y1), ysize)
    
    return f"{x0_tok}{y0_tok}{x1_tok}{y1_tok}"


def get_element_tag(el: Dict[str, Any]) -> str:
    """
    Get the appropriate tag name for an element in ScreenTag format.
    Uses vlm_label if available, otherwise falls back to type.
    
    Args:
        el: Element dictionary
    
    Returns:
        Tag name string (sanitized for use in XML-like format)
    """
    # Prefer vlm_label if it's valid
    vlm_label = (el.get("vlm_label") or "").strip()
    if vlm_label and vlm_label.lower() != "unknown":
        tag = vlm_label
    else:
        # Fall back to type
        tag = (el.get("type") or "").strip()
        if not tag or tag.lower() == "unknown":
            # Last resort: use HTML tag
            tag = el.get("tag", "element")
    
    # Sanitize: replace spaces with underscores, remove special chars
    tag = re.sub(r'[^a-zA-Z0-9_-]', '_', tag)
    tag = re.sub(r'_+', '_', tag).strip('_')
    
    return tag or "element"


def elements_to_screentag(
    elements: List[Dict[str, Any]],
    viewport: Dict[str, Any],
) -> str:
    """
    Convert elements with hierarchy into a ScreenTag representation.
    
    ScreenTag format:
    <screentag>
      <element-type><loc_L><loc_T><loc_R><loc_B>[text][children]</element-type>
    </screentag>
    
    Uses 'own_text' field if available (deduplicated text that excludes
    children's text), falling back to 'inner_text' for backwards compatibility.
    
    Traverses in depth-first order. Root elements are sorted by top-left
    coordinate (top first, then left for ties).
    
    Args:
        elements: List of elements with parent_index and children_indices
        viewport: Viewport dict with 'w' and 'h' keys for screen dimensions
    
    Returns:
        ScreenTag string representation
    """
    if not elements:
        return "<screentag></screentag>"
    
    # Get screen dimensions from viewport
    screen_width = viewport.get("w", 0)
    screen_height = viewport.get("h", 0)
    
    if screen_width <= 0 or screen_height <= 0:
        return "<screentag></screentag>"
    
    # Find root elements (those with no parent)
    root_indices = [i for i, el in enumerate(elements) if el.get("parent_index") is None]
    
    # Sort roots by top-left coordinate (top first, then left for ties)
    def get_top_left(idx: int) -> Tuple[int, int]:
        rect = elements[idx].get("rect", {})
        return (rect.get("y", 0), rect.get("x", 0))
    
    root_indices.sort(key=get_top_left)
    
    parts = []
    
    def _render_element(idx: int) -> str:
        """Render a single element and its children recursively (depth-first)."""
        if idx >= len(elements):
            return ""
        
        el = elements[idx]
        
        # Get element tag
        tag = get_element_tag(el)
        
        # Get bounding box as LTRB
        rect = el.get("rect", {})
        left = rect.get("x", 0)
        top = rect.get("y", 0)
        right = left + rect.get("w", 0)
        bottom = top + rect.get("h", 0)
        
        # Get location tokens
        loc_str = get_location_str(
            (left, top, right, bottom),
            screen_width,
            screen_height,
        )
        
        # Get text (prefer own_text for deduplication)
        text = ""
        if "own_text" in el:
            text = (el.get("own_text") or "").strip()
        else:
            text = (el.get("inner_text") or "").strip()
        
        # Get children and sort by top-left coordinate
        children_indices = el.get("children_indices", [])
        if children_indices:
            children_indices = sorted(children_indices, key=get_top_left)
        
        # Build element string
        # Format: <tag><loc_L><loc_T><loc_R><loc_B>[text][children]</tag>
        children_str = ""
        if children_indices:
            # Render children recursively (depth-first)
            children_parts = [_render_element(c_idx) for c_idx in children_indices]
            children_str = "".join(children_parts)
        
        return f"<{tag}>{loc_str}{text}{children_str}</{tag}>"
    
    # Render all roots
    for root_idx in root_indices:
        parts.append(_render_element(root_idx))
    
    content = "".join(parts)
    return f"<screentag>{content}</screentag>"


# ─────────────────────────────────────────────────────────────────────────────
# ScreenTag Parsing and Tree Visualization (Debug/QA Helper)
# ─────────────────────────────────────────────────────────────────────────────

def _truncate_text(text: str, max_len: int = 50) -> str:
    """
    Truncate text showing beginning and end for context.
    
    Shows first 20 chars + "..." + last 15 chars if text exceeds max_len.
    
    Args:
        text: Text to truncate
        max_len: Maximum length before truncation
    
    Returns:
        Truncated text string
    """
    if not text or len(text) <= max_len:
        return text
    
    # Show first 20 + "..." + last 15 = 38 chars minimum
    first_len = 20
    last_len = 15
    
    if len(text) <= first_len + last_len + 3:
        return text
    
    return f"{text[:first_len]}...{text[-last_len:]}"


class ScreenTagNode:
    """Represents a parsed node from ScreenTag format."""
    
    def __init__(
        self,
        tag: str,
        bbox: Tuple[int, int, int, int],  # (L, T, R, B) in 0-500 range
        text: str = "",
        children: Optional[List["ScreenTagNode"]] = None,
    ):
        self.tag = tag
        self.bbox = bbox  # (left, top, right, bottom)
        self.text = text
        self.children = children or []
    
    def __repr__(self):
        text_display = _truncate_text(self.text, 50)
        return f"ScreenTagNode(tag={self.tag!r}, bbox={self.bbox}, text={text_display!r}, children={len(self.children)})"


def parse_screentag(screentag_str: str) -> List[ScreenTagNode]:
    """
    Parse a ScreenTag string into a tree of ScreenTagNode objects.
    
    Args:
        screentag_str: ScreenTag string like "<screentag><button><loc_0>...</button></screentag>"
    
    Returns:
        List of root ScreenTagNode objects
    """
    # Remove outer <screentag> wrapper if present
    content = screentag_str.strip()
    if content.startswith("<screentag>"):
        content = content[len("<screentag>"):]
    if content.endswith("</screentag>"):
        content = content[:-len("</screentag>")]
    
    def _parse_location_tokens(s: str) -> Tuple[Tuple[int, int, int, int], str]:
        """Extract 4 location tokens from start of string, return (bbox, remaining)."""
        loc_pattern = r'^(<loc_(\d+)>){4}'
        match = re.match(loc_pattern, s)
        if match:
            # Extract all 4 loc values
            locs = re.findall(r'<loc_(\d+)>', match.group(0))
            if len(locs) == 4:
                bbox = (int(locs[0]), int(locs[1]), int(locs[2]), int(locs[3]))
                remaining = s[match.end():]
                return bbox, remaining
        return (0, 0, 0, 0), s
    
    def _parse_element(s: str, pos: int = 0) -> Tuple[Optional[ScreenTagNode], int]:
        """
        Parse a single element starting at position pos.
        Returns (node, new_position) or (None, pos) if no element found.
        """
        if pos >= len(s):
            return None, pos
        
        # Skip whitespace
        while pos < len(s) and s[pos].isspace():
            pos += 1
        
        if pos >= len(s) or s[pos] != '<':
            return None, pos
        
        # Check for closing tag (shouldn't happen at top level)
        if pos + 1 < len(s) and s[pos + 1] == '/':
            return None, pos
        
        # Find opening tag name
        tag_end = s.find('>', pos)
        if tag_end == -1:
            return None, pos
        
        tag_name = s[pos + 1:tag_end]
        
        # Skip location tokens (they look like tags but aren't)
        if tag_name.startswith('loc_'):
            return None, pos
        
        pos = tag_end + 1
        
        # Now we need to find the matching closing tag
        # But first, extract location tokens and text/children
        
        # Extract location tokens (4 of them: L, T, R, B)
        remaining_from_pos = s[pos:]
        bbox, after_locs = _parse_location_tokens(remaining_from_pos)
        pos += len(remaining_from_pos) - len(after_locs)
        
        # Now parse content until we hit the closing tag
        # Content can be: text, or child elements, or mix
        text_parts = []
        children = []
        
        closing_tag = f"</{tag_name}>"
        
        while pos < len(s):
            # Check for closing tag
            if s[pos:pos + len(closing_tag)] == closing_tag:
                pos += len(closing_tag)
                break
            
            # Check for child element (opening tag)
            if s[pos] == '<' and pos + 1 < len(s) and s[pos + 1] != '/':
                # Check if it's a loc token (skip it - shouldn't be here but just in case)
                loc_match = re.match(r'<loc_\d+>', s[pos:])
                if loc_match:
                    pos += loc_match.end()
                    continue
                
                # Try to parse child element
                child, new_pos = _parse_element(s, pos)
                if child:
                    children.append(child)
                    pos = new_pos
                else:
                    # Not a valid element, treat as text
                    text_parts.append(s[pos])
                    pos += 1
            else:
                # Text content
                text_parts.append(s[pos])
                pos += 1
        
        text = "".join(text_parts).strip()
        
        return ScreenTagNode(tag=tag_name, bbox=bbox, text=text, children=children), pos
    
    # Parse all root elements
    roots = []
    pos = 0
    while pos < len(content):
        node, new_pos = _parse_element(content, pos)
        if node:
            roots.append(node)
            pos = new_pos
        else:
            pos += 1  # Skip character and try again
    
    return roots


def screentag_to_tree_string(
    screentag_str: str,
    show_bbox: bool = True,
    show_text: bool = True,
    max_text_len: int = 50,
    indent_str: str = "  ",
) -> str:
    """
    Convert ScreenTag string to a human-readable tree representation.
    
    The tree starts with a virtual "screen" root node at depth 0 with
    bbox [0,0,500,500] representing the full screen.
    
    Args:
        screentag_str: ScreenTag string to parse
        show_bbox: Whether to show bounding box coordinates
        show_text: Whether to show text content
        max_text_len: Maximum text length to display (uses start...end truncation)
        indent_str: String to use for each indentation level
    
    Returns:
        Multi-line string showing the tree structure
    
    Example output:
        screen [0,0,500,500] (3 children)
        ├── navigation [10,20,400,80] (3 children)
        │   ├── button [15,25,100,75] "Home"
        │   ├── button [110,25,200,75] "About"
        │   └── button [210,25,300,75] "Contact"
        └── footer [10,450,490,495]
    """
    roots = parse_screentag(screentag_str)
    
    if not roots:
        return "(empty screentag)"
    
    lines = []
    
    def _format_text(text: str) -> str:
        """Format text for display with truncation."""
        if not text:
            return ""
        display_text = _truncate_text(text, max_text_len)
        # Escape newlines for display
        display_text = display_text.replace("\n", "\\n")
        return display_text
    
    def _format_node(node: ScreenTagNode, prefix: str = "", is_last: bool = True, is_root: bool = False) -> None:
        """Format a node and its children recursively."""
        # Build the node line
        if is_root:
            connector = ""
            child_prefix = ""
        else:
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
        
        # Tag name
        parts = [f"{prefix}{connector}{node.tag}"]
        
        # Bounding box
        if show_bbox:
            L, T, R, B = node.bbox
            parts.append(f" [{L},{T},{R},{B}]")
        
        # Text content
        if show_text and node.text:
            display_text = _format_text(node.text)
            parts.append(f' "{display_text}"')
        
        # Children count (if any and not showing them individually)
        if node.children:
            parts.append(f" ({len(node.children)} children)")
        
        lines.append("".join(parts))
        
        # Render children
        for i, child in enumerate(node.children):
            is_last_child = (i == len(node.children) - 1)
            _format_node(child, child_prefix, is_last_child, is_root=False)
    
    # Create virtual screen root node at depth 0
    screen_root = ScreenTagNode(
        tag="screen",
        bbox=(0, 0, 500, 500),
        text="",
        children=roots,
    )
    
    # Render from screen root
    _format_node(screen_root, "", is_last=True, is_root=True)
    
    return "\n".join(lines)


def print_screentag_tree(
    screentag_str: str,
    show_bbox: bool = True,
    show_text: bool = True,
    max_text_len: int = 50,
) -> None:
    """
    Print ScreenTag as a readable tree structure.
    
    Convenience wrapper around screentag_to_tree_string that prints to stdout.
    
    Args:
        screentag_str: ScreenTag string to parse and display
        show_bbox: Whether to show bounding box coordinates
        show_text: Whether to show text content
        max_text_len: Maximum text length to display
    """
    print(screentag_to_tree_string(
        screentag_str,
        show_bbox=show_bbox,
        show_text=show_text,
        max_text_len=max_text_len,
    ))


def screentag_to_dict(screentag_str: str) -> Dict[str, Any]:
    """
    Convert ScreenTag string to a nested dictionary structure.
    
    Useful for JSON serialization or programmatic inspection.
    
    Args:
        screentag_str: ScreenTag string to parse
    
    Returns:
        Dictionary with structure:
        {
            "roots": [
                {
                    "tag": "button",
                    "bbox": {"L": 10, "T": 20, "R": 100, "B": 80},
                    "text": "Click me",
                    "children": [...]
                },
                ...
            ],
            "stats": {
                "total_elements": 42,
                "max_depth": 5,
                "element_types": {"button": 10, "text": 15, ...}
            }
        }
    """
    roots = parse_screentag(screentag_str)
    
    def _node_to_dict(node: ScreenTagNode, depth: int = 0) -> Tuple[Dict[str, Any], int, Dict[str, int]]:
        """Convert node to dict, also return max_depth and type counts."""
        L, T, R, B = node.bbox
        result = {
            "tag": node.tag,
            "bbox": {"L": L, "T": T, "R": R, "B": B},
            "text": node.text if node.text else None,
            "children": [],
        }
        
        max_depth = depth
        type_counts = {node.tag: 1}
        
        for child in node.children:
            child_dict, child_depth, child_types = _node_to_dict(child, depth + 1)
            result["children"].append(child_dict)
            max_depth = max(max_depth, child_depth)
            for t, c in child_types.items():
                type_counts[t] = type_counts.get(t, 0) + c
        
        return result, max_depth, type_counts
    
    root_dicts = []
    total_elements = 0
    overall_max_depth = 0
    overall_type_counts: Dict[str, int] = {}
    
    for root in roots:
        root_dict, max_depth, type_counts = _node_to_dict(root)
        root_dicts.append(root_dict)
        overall_max_depth = max(overall_max_depth, max_depth)
        for t, c in type_counts.items():
            overall_type_counts[t] = overall_type_counts.get(t, 0) + c
            total_elements += c
    
    return {
        "roots": root_dicts,
        "stats": {
            "total_elements": total_elements,
            "max_depth": overall_max_depth + 1,  # 1-indexed depth
            "element_types": dict(sorted(overall_type_counts.items(), key=lambda x: -x[1])),
        },
    }


def screentag_stats(screentag_str: str) -> str:
    """
    Get a summary of statistics for a ScreenTag string.
    
    Args:
        screentag_str: ScreenTag string to analyze
    
    Returns:
        Multi-line string with statistics
    """
    data = screentag_to_dict(screentag_str)
    stats = data["stats"]
    
    lines = [
        "ScreenTag Statistics",
        "=" * 40,
        f"Total elements:  {stats['total_elements']}",
        f"Root elements:   {len(data['roots'])}",
        f"Max depth:       {stats['max_depth']}",
        "",
        "Element types (top 15):",
    ]
    
    for i, (tag, count) in enumerate(list(stats["element_types"].items())[:15]):
        lines.append(f"  {tag}: {count}")
    
    if len(stats["element_types"]) > 15:
        lines.append(f"  ... and {len(stats['element_types']) - 15} more types")
    
    return "\n".join(lines)


def export_screentag_trees(
    raw_dir: str,
    n: int = -1,
    show_bbox: bool = True,
    show_text: bool = True,
    max_text_len: int = 50,
    include_stats: bool = True,
) -> Dict[str, Any]:
    """
    Process screentag files in a directory and save tree representations.
    
    For each .screentag.txt file, creates a corresponding .screentag.tree.txt
    file with the human-readable tree representation.
    
    Args:
        raw_dir: Directory containing .screentag.txt files
        n: Number of files to process (-1 for all)
        show_bbox: Whether to show bounding box coordinates in tree
        show_text: Whether to show text content in tree
        max_text_len: Maximum text length to display
        include_stats: Whether to append statistics at the end
    
    Returns:
        Statistics dictionary with counts
    """
    import glob
    from tqdm import tqdm
    
    # Find all screentag files
    pattern = os.path.join(raw_dir, "*.screentag.txt")
    screentag_files = sorted(glob.glob(pattern))
    
    if not screentag_files:
        print(f"No .screentag.txt files found in {raw_dir}")
        return {"total": 0, "processed": 0, "errors": 0}
    
    # Limit to N files if specified
    if n > 0:
        screentag_files = screentag_files[:n]
    
    print(f"Processing {len(screentag_files)} screentag files...")
    
    stats = {
        "total": len(screentag_files),
        "processed": 0,
        "errors": 0,
        "error_files": [],
    }
    
    for screentag_path in tqdm(screentag_files, desc="Exporting trees"):
        try:
            # Read screentag file
            with open(screentag_path, "r", encoding="utf-8") as f:
                screentag_str = f.read()
            
            # Generate tree string
            tree_str = screentag_to_tree_string(
                screentag_str,
                show_bbox=show_bbox,
                show_text=show_text,
                max_text_len=max_text_len,
            )
            
            # Optionally append stats
            if include_stats:
                stats_str = screentag_stats(screentag_str)
                tree_str = f"{tree_str}\n\n{'=' * 60}\n{stats_str}"
            
            # Write tree file
            tree_path = screentag_path.replace(".screentag.txt", ".screentag.tree.txt")
            with open(tree_path, "w", encoding="utf-8") as f:
                f.write(tree_str)
            
            stats["processed"] += 1
            
        except Exception as e:
            stats["errors"] += 1
            stats["error_files"].append((screentag_path, str(e)))
    
    # Print summary
    print(f"\nProcessed: {stats['processed']}/{stats['total']}")
    if stats["errors"] > 0:
        print(f"Errors: {stats['errors']}")
        for path, msg in stats["error_files"][:5]:
            print(f"  {os.path.basename(path)}: {msg}")
    
    return stats

