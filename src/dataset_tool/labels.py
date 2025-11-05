"""Element type classification for UI elements."""

from typing import Dict, Any, Optional


def guess_type(tag: Optional[str], role: Optional[str], attrs: Dict[str, Any]) -> str:
    """
    Classify element into a semantic type based on tag, role, and attributes.

    Args:
        tag: HTML tag name (e.g., 'button', 'div')
        role: ARIA role attribute
        attrs: Dictionary of HTML attributes

    Returns:
        Semantic type string (e.g., 'button', 'input', 'text')
    """
    # Safely handle None values
    tag = (tag or "").lower().strip()
    role = (role or "").lower().strip()
    attrs = attrs or {}

    # Get common attributes
    type_attr = (attrs.get("type") or "").lower()
    class_str = (attrs.get("class") or "").lower()

    # Button detection
    if tag == "button" or role == "button":
        return "button"
    if tag == "input" and type_attr in ("submit", "button", "reset"):
        return "button"
    if "btn" in class_str or "button" in class_str:
        return "button"

    # Input detection
    if tag == "input" and type_attr in (
        "text",
        "email",
        "password",
        "tel",
        "url",
        "number",
    ):
        return "input"
    if tag == "textarea":
        return "input"
    if role in ("textbox", "searchbox"):
        return "input"

    # Search
    if tag == "input" and type_attr == "search":
        return "search"
    if role == "search" or "search" in class_str:
        return "search"

    # Checkbox/Radio
    if tag == "input" and type_attr in ("checkbox", "radio"):
        return "checkbox"
    if role in ("checkbox", "radio", "switch"):
        return "checkbox"

    # Dropdown/Select
    if tag == "select" or role in ("combobox", "listbox"):
        return "dropdown"

    # Link detection
    if tag == "a" or role == "link":
        return "link"

    # Heading detection
    if tag in ("h1", "h2", "h3", "h4", "h5", "h6") or role == "heading":
        return "heading"

    # Image detection
    if tag == "img" or role == "img":
        return "image"
    if tag == "picture" or tag == "figure":
        return "image"

    # Icon detection (heuristic)
    if "icon" in class_str or tag in ("i", "svg"):
        return "icon"

    # Video/Media
    if tag in ("video", "audio") or role == "video":
        return "video"

    # Navigation
    if tag == "nav" or role == "navigation":
        return "navigation"
    if "nav" in class_str or "menu" in class_str:
        return "navigation"

    # List
    if tag in ("ul", "ol", "dl") or role in ("list", "listitem"):
        return "list"

    # Table
    if tag == "table" or role in ("table", "grid"):
        return "table"

    # Form
    if tag == "form" or role == "form":
        return "form"

    # Card/Panel
    if role == "article" or tag == "article":
        return "card"
    if "card" in class_str or "panel" in class_str or "tile" in class_str:
        return "card"

    # Header
    if tag == "header" or role == "banner":
        return "header"
    if "header" in class_str and "page-header" in class_str:
        return "header"

    # Footer
    if tag == "footer" or role == "contentinfo":
        return "footer"

    # Modal/Dialog
    if role in ("dialog", "alertdialog") or "modal" in class_str:
        return "modal"

    # Tooltip
    if role == "tooltip" or "tooltip" in class_str or "popover" in class_str:
        return "tooltip"

    # Menu
    if role in ("menu", "menubar", "menuitem"):
        return "menu"

    # Tab
    if role in ("tab", "tablist", "tabpanel"):
        return "tab"

    # Badge/Label
    if "badge" in class_str or "label" in class_str or "chip" in class_str:
        return "badge"

    # Ad detection
    if "ad" in class_str or "advertisement" in class_str or "sponsor" in class_str:
        return "ad"

    # Logo
    if "logo" in class_str or (
        tag == "img" and "logo" in (attrs.get("alt") or "").lower()
    ):
        return "logo"

    # Text blocks
    if tag in ("p", "span", "div", "section", "main"):
        # Check if it's likely a text container
        if role == "text" or not role:
            return "text"

    # Default fallback
    return tag or "unknown"
