"""Debug tool to compare filtered vs unfiltered elements."""

import os
from typing import Dict, Any, List
from .utils import load_json
from collections import Counter


def compare_filtered_elements(base_path: str):
    """
    Compare filtered vs unfiltered elements to see what was removed.

    Usage: python -m dataset_tool.debug_filtering data/raw/example-com-abc123
    """
    elements_path = f"{base_path}.elements.json"
    unfiltered_path = f"{base_path}.elements.unfiltered.json"

    if not os.path.exists(elements_path):
        print(f"❌ Not found: {elements_path}")
        return

    if not os.path.exists(unfiltered_path):
        print(f"❌ Not found: {unfiltered_path}")
        print("   Run crawl with --save-unfiltered to generate this file")
        return

    filtered = load_json(elements_path)
    unfiltered = load_json(unfiltered_path)

    print("\n" + "=" * 70)
    print("FILTERING COMPARISON")
    print("=" * 70)

    print(f"\nUnfiltered: {len(unfiltered)} elements")
    print(f"Filtered:   {len(filtered)} elements")
    print(
        f"Removed:    {len(unfiltered) - len(filtered)} elements ({100*(len(unfiltered)-len(filtered))/len(unfiltered):.1f}%)"
    )

    # Find what was removed
    filtered_rects = {
        (el["rect"]["x"], el["rect"]["y"], el["rect"]["w"], el["rect"]["h"])
        for el in filtered
    }

    removed = []
    for el in unfiltered:
        rect_key = (el["rect"]["x"], el["rect"]["y"], el["rect"]["w"], el["rect"]["h"])
        if rect_key not in filtered_rects:
            removed.append(el)

    # Analyze removed elements
    removed_types = Counter(el.get("type", "unknown") for el in removed)

    print(f"\n📊 Removed by Type:")
    for el_type, count in removed_types.most_common(10):
        pct = 100 * count / len(removed) if removed else 0
        print(f"  {el_type:20s}: {count:4d} ({pct:5.1f}%)")

    # Check for removed interactive elements
    from .filtering import PROTECTED_TYPES

    removed_protected = [el for el in removed if el.get("type") in PROTECTED_TYPES]

    if removed_protected:
        print(f"\n⚠️  ALERT: {len(removed_protected)} PROTECTED elements were removed!")
        print(f"\nProtected elements removed:")
        for el in removed_protected[:10]:
            print(
                f"  - {el.get('type'):15s} <{el.get('tag')}> at ({el['rect']['x']}, {el['rect']['y']}) "
                f"{el['rect']['w']}x{el['rect']['h']} text: {el.get('inner_text', '')[:50]}"
            )
    else:
        print(f"\n✅ No protected elements were removed")

    # Show what was kept
    kept_types = Counter(el.get("type", "unknown") for el in filtered)

    print(f"\n📊 Kept by Type:")
    for el_type, count in kept_types.most_common(10):
        pct = 100 * count / len(filtered) if filtered else 0
        print(f"  {el_type:20s}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m dataset_tool.debug_filtering <base_path>")
        print(
            "Example: python -m dataset_tool.debug_filtering data/raw/github-com-abc123"
        )
    else:
        compare_filtered_elements(sys.argv[1])
