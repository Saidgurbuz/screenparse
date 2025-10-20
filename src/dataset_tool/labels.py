from typing import Optional, Dict

# Canonical element types we’ll see in annotations
ELEMENT_TYPES = [
    "header",
    "footer",
    "nav",
    "section",
    "article",
    "aside",
    "link",
    "button",
    "input",
    "select",
    "textarea",
    "image",
    "video",
    "audio",
    "table",
    "list",
    "list_item",
    "breadcrumb",
    "card",
    "code",
    "form",
    "modal",
    "tooltip",
    "tab",
    "chip",
    "badge",
    "pagination",
    "icon",
    "svg",
    "canvas",
    "ad",
    "title",
    "paragraph",
    "figure",
    "label",
    "checkbox",
    "radio",
    "switch",
    "slider",
    "progress",
    "meter",
    "map",
    "unknown",
]


def guess_type(tag: str, role: Optional[str], attrs: Dict[str, str]) -> str:
    t = tag.lower() if tag else ""
    r = (role or "").lower()

    # landmarks & structure
    if t == "header" or r == "banner":
        return "header"
    if t == "footer" or r == "contentinfo":
        return "footer"
    if t == "nav" or r == "navigation":
        return "nav"
    if t in {"main", "section"} or r in {"region", "main"}:
        return "section"
    if t == "article" or r == "article":
        return "article"
    if t == "aside" or r == "complementary":
        return "aside"

    # interactive
    if t == "a" and attrs.get("href"):
        return "link"
    if t == "button" or r == "button":
        return "button"
    if t in {"input"} or r in {"textbox", "searchbox", "combobox", "spinbutton"}:
        itype = (attrs.get("type") or "").lower()
        if itype in {"checkbox"}:
            return "checkbox"
        if itype in {"radio"}:
            return "radio"
        return "input"
    if t == "select" or r == "combobox":
        return "select"
    if t == "textarea":
        return "textarea"
    if r in {"switch"}:
        return "switch"
    if r in {"slider"}:
        return "slider"
    if r in {"progressbar"}:
        return "progress"
    if r in {"tab"}:
        return "tab"

    # media/graphics
    if t == "img" or r == "img":
        return "image"
    if t == "video":
        return "video"
    if t == "audio":
        return "audio"
    if t == "svg":
        return "svg"
    if t == "canvas":
        return "canvas"
    if t == "figure":
        return "figure"
    if t == "i" and "icon" in (attrs.get("class") or ""):
        return "icon"

    # data/content containers
    if t == "table" or r == "table":
        return "table"
    if t in {"ul", "ol"} or r in {"list"}:
        return "list"
    if t == "li" or r == "listitem":
        return "list_item"
    if t in {"h1", "h2", "h3", "h4", "h5", "h6"} or r in {"heading"}:
        return "title"
    if t == "p":
        return "paragraph"
    if t == "pre" or t == "code":
        return "code"
    if t == "form" or r == "form":
        return "form"
    if "breadcrumb" in (attrs.get("class") or ""):
        return "breadcrumb"
    if any(cls in (attrs.get("class") or "") for cls in ["card", "tile", "panel"]):
        return "card"
    if any(cls in (attrs.get("class") or "") for cls in ["chip", "tag", "pill"]):
        return "chip"
    if "badge" in (attrs.get("class") or ""):
        return "badge"
    if "pagination" in (attrs.get("class") or ""):
        return "pagination"

    # simple ad heuristic
    if any(k in (attrs.get("id") or "") for k in ["ad", "ads", "advert", "sponsor"]):
        return "ad"

    return "unknown"
