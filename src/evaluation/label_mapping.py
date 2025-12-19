from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .types import UIElement

UI_ELEMENTS_55 = [
    "Table",
    "Column/Browser",
    "Button",
    "Utility Button",
    "App Icon",
    "Navigation Bar",
    "Status Bar",
    "Search Field",
    "Toolbar",
    "Tooltip",
    "Video",
    "Tab Bar",
    "Side Bar",
    "Slider",
    "Picker",
    "ContextMenu",
    "DockMenu",
    "EditMenu",
    "Image",
    "Scroll",
    "Switch",
    "File Icon",
    "Chart",
    "Window",
    "Screen",
    "List",
    "List Item",
    "PopUp Menu",
    "Steppers",
    "Toggles",
    "Text Input",
    "Rating Indicator",
    "Checkbox",
    "Radiobox",
    "Select",
    "Avatar",
    "Badge",
    "Alert",
    "Progress bar",
    "Bottom navigation",
    "Breadcrumb",
    "Page control",
    "Link",
    "Menu",
    "Pagination",
    "Tab",
    "Search Bar",
    "Date-Time picker",
    "Calendar",
    "Text",
    "Heading",
    "Code snippet",
    "Carousel",
    "Notification",
    "Logo",
]

GROUNDCUA_CATEGORIES = [
    "Input Element",
    "Sidebar",
    "Information Display",
    "Button",
    "Navigation",
    "Visual Elements",
    "Menu",
    "Others",
]

groundcua_to_custom: Dict[str, List[str]] = {
    "Input Element": [
        "Search Field",
        "Slider",
        "Picker",
        "Switch",
        "Steppers",
        "Toggles",
        "Text Input",
        "Rating Indicator",
        "Checkbox",
        "Radiobox",
        "Select",
        "Search Bar",
        "Date-Time picker",
        "Calendar",
    ],
    "Sidebar": [
        "Side Bar",
        "Toolbar",
        "Column/Browser",
    ],
    "Information Display": [
        "Table",
        "Status Bar",
        "Tooltip",
        "List",
        "List Item",
        "Alert",
        "Text",
        "Heading",
        "Code snippet",
        "Notification",
    ],
    "Button": [
        "Button",
        "Utility Button",
    ],
    "Navigation": [
        "Navigation Bar",
        "Tab Bar",
        "Scroll",
        "Bottom navigation",
        "Breadcrumb",
        "Page control",
        "Link",
        "Pagination",
        "Tab",
    ],
    "Visual Elements": [
        "App Icon",
        "Video",
        "Image",
        "File Icon",
        "Chart",
        "Avatar",
        "Badge",
        "Progress bar",
        "Carousel",
        "Logo",
    ],
    "Menu": [
        "ContextMenu",
        "DockMenu",
        "EditMenu",
        "PopUp Menu",
        "Menu",
    ],
    "Others": [
        "Window",
        "Screen",
    ],
}

custom_to_groundcua: Dict[str, str] = {
    element: category for category, elements in groundcua_to_custom.items() for element in elements
}

groundcua_representative_map: Dict[str, str] = {
    "Input Element": "Text Input",
    "Sidebar": "Side Bar",
    "Information Display": "Text",
    "Button": "Button",
    "Navigation": "Navigation Bar",
    "Visual Elements": "Image",
    "Menu": "Menu",
    "Others": "Link",
}


def get_class_list(schema: str) -> List[str]:
    if schema == "groundcua":
        return GROUNDCUA_CATEGORIES
    return UI_ELEMENTS_55


@dataclass
class LabelMapper:
    target_schema: str = "custom55"

    def map_label(self, label: Optional[str]) -> Optional[str]:
        if not label:
            return label
        lbl = label.strip()

        if self.target_schema == "groundcua":
            # Map custom -> groundcua
            if lbl in custom_to_groundcua:
                return custom_to_groundcua[lbl]
            if lbl in GROUNDCUA_CATEGORIES:
                return lbl
            return lbl

        # target custom55
        if lbl in groundcua_representative_map:
            return groundcua_representative_map[lbl]
        return lbl

    def map_element(self, el: UIElement) -> UIElement:
        mapped_label = self.map_label(el.label)
        if mapped_label == el.label:
            return el
        return UIElement(
            bbox=el.bbox,
            label=mapped_label,
            text=el.text,
            score=el.score,
            raw=el.raw,
        )
