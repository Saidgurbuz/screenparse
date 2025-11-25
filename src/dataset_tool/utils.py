import hashlib, re, json, os, orjson, unicodedata
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from slugify import slugify


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


def elements_to_screentag(
    elements: list,
    root_indices: list = None,
    indent: int = 0
) -> str:
    """
    Convert elements with hierarchy into a ScreenTag representation.
    
    ScreenTag is an LLM-friendly structured representation of UI elements,
    similar to HTML but optimized for screen understanding tasks.
    
    Args:
        elements: List of elements with parent_index and children_indices
        root_indices: Which elements to start from (None = all roots)
        indent: Current indentation level
    
    Returns:
        ScreenTag string representation
    
    Note:
        This is a stub function. The actual ScreenTag format will be 
        implemented collaboratively based on specific requirements.
    """
    # TODO: Implement ScreenTag format based on detailed specification
    # This is a placeholder that will be filled in collaboratively
    
    if not elements:
        return ""
    
    if root_indices is None:
        # Find roots
        root_indices = [i for i, el in enumerate(elements) if el.get("parent_index") is None]
    
    # Placeholder implementation - returns simple tree structure
    lines = []
    
    def _render_element(idx: int, depth: int):
        if idx >= len(elements):
            return
        el = elements[idx]
        indent_str = "  " * depth
        
        # Get element info
        tag = el.get("tag", "div")
        el_type = el.get("vlm_label") or el.get("type") or tag
        rect = el.get("rect", {})
        bbox_str = f"{rect.get('x',0)},{rect.get('y',0)},{rect.get('w',0)},{rect.get('h',0)}"
        
        children_indices = el.get("children_indices", [])
        
        if children_indices:
            lines.append(f"{indent_str}<{el_type} bbox=\"{bbox_str}\">")
            for child_idx in children_indices:
                _render_element(child_idx, depth + 1)
            lines.append(f"{indent_str}</{el_type}>")
        else:
            inner_text = (el.get("inner_text") or "").strip()
            if len(inner_text) > 50:
                inner_text = inner_text[:47] + "..."
            if inner_text:
                lines.append(f"{indent_str}<{el_type} bbox=\"{bbox_str}\">{inner_text}</{el_type}>")
            else:
                lines.append(f"{indent_str}<{el_type} bbox=\"{bbox_str}\"/>")
    
    for root_idx in root_indices:
        _render_element(root_idx, indent)
    
    return "\n".join(lines)
