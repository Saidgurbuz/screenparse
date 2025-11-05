"""Dataset quality validation and statistics."""

import os
import json
from typing import Dict, List, Any
from collections import Counter
from .utils import load_json
from .yolo_export import YOLO_CLASSES


def validate_yolo_dataset(yolo_dir: str) -> Dict[str, Any]:
    """
    Validate YOLO dataset and return statistics.

    Checks:
    - All images have corresponding labels
    - All labels have valid format
    - Class distribution
    - Bounding box validity
    """
    stats = {
        "total_images": 0,
        "total_annotations": 0,
        "missing_labels": [],
        "invalid_labels": [],
        "class_distribution": Counter(),
        "splits": {},
        "bbox_issues": [],
    }

    for split in ["train", "val", "test"]:
        img_dir = os.path.join(yolo_dir, "images", split)
        label_dir = os.path.join(yolo_dir, "labels", split)

        if not os.path.exists(img_dir):
            continue

        split_stats = {
            "images": 0,
            "annotations": 0,
            "avg_boxes_per_image": 0,
        }

        images = [f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))]
        split_stats["images"] = len(images)
        stats["total_images"] += len(images)

        for img_file in images:
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(label_dir, label_file)

            # Check if label exists
            if not os.path.exists(label_path):
                stats["missing_labels"].append(os.path.join(split, img_file))
                continue

            # Validate label format
            try:
                with open(label_path, "r") as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 5:
                        stats["invalid_labels"].append(
                            f"{split}/{label_file}:{line_num} - Expected 5 values, got {len(parts)}"
                        )
                        continue

                    try:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])

                        # Validate ranges
                        if class_id < 0 or class_id >= len(YOLO_CLASSES):
                            stats["invalid_labels"].append(
                                f"{split}/{label_file}:{line_num} - Invalid class_id {class_id}"
                            )

                        if not (
                            0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1
                        ):
                            stats["bbox_issues"].append(
                                f"{split}/{label_file}:{line_num} - Box out of bounds: {cx},{cy},{w},{h}"
                            )

                        # Count class
                        stats["class_distribution"][YOLO_CLASSES[class_id]] += 1
                        split_stats["annotations"] += 1
                        stats["total_annotations"] += 1

                    except ValueError as e:
                        stats["invalid_labels"].append(
                            f"{split}/{label_file}:{line_num} - Parse error: {e}"
                        )

            except Exception as e:
                stats["invalid_labels"].append(
                    f"{split}/{label_file} - Read error: {e}"
                )

        if split_stats["images"] > 0:
            split_stats["avg_boxes_per_image"] = (
                split_stats["annotations"] / split_stats["images"]
            )

        stats["splits"][split] = split_stats

    return stats


def print_validation_report(stats: Dict[str, Any]):
    """Print human-readable validation report."""
    print("\n" + "=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)

    print(f"\n📊 Overview:")
    print(f"  Total Images: {stats['total_images']}")
    print(f"  Total Annotations: {stats['total_annotations']}")
    print(
        f"  Avg Boxes/Image: {stats['total_annotations'] / max(1, stats['total_images']):.2f}"
    )

    print(f"\n📁 Split Distribution:")
    for split, split_stats in stats["splits"].items():
        print(
            f"  {split.upper():5s}: {split_stats['images']:4d} images, "
            f"{split_stats['annotations']:5d} boxes "
            f"({split_stats['avg_boxes_per_image']:.1f} avg)"
        )

    print(f"\n🏷️  Class Distribution:")
    sorted_classes = sorted(
        stats["class_distribution"].items(), key=lambda x: x[1], reverse=True
    )
    for cls, count in sorted_classes[:20]:  # Top 20
        pct = 100 * count / max(1, stats["total_annotations"])
        print(f"  {cls:15s}: {count:5d} ({pct:5.1f}%)")
    if len(sorted_classes) > 20:
        print(f"  ... and {len(sorted_classes) - 20} more classes")

    # Issues
    issues_found = False

    if stats["missing_labels"]:
        issues_found = True
        print(f"\n⚠️  Missing Labels ({len(stats['missing_labels'])}):")
        for path in stats["missing_labels"][:5]:
            print(f"  - {path}")
        if len(stats["missing_labels"]) > 5:
            print(f"  ... and {len(stats['missing_labels']) - 5} more")

    if stats["invalid_labels"]:
        issues_found = True
        print(f"\n❌ Invalid Labels ({len(stats['invalid_labels'])}):")
        for issue in stats["invalid_labels"][:5]:
            print(f"  - {issue}")
        if len(stats["invalid_labels"]) > 5:
            print(f"  ... and {len(stats['invalid_labels']) - 5} more")

    if stats["bbox_issues"]:
        issues_found = True
        print(f"\n⚠️  Bounding Box Issues ({len(stats['bbox_issues'])}):")
        for issue in stats["bbox_issues"][:5]:
            print(f"  - {issue}")
        if len(stats["bbox_issues"]) > 5:
            print(f"  ... and {len(stats['bbox_issues']) - 5} more")

    if not issues_found:
        print(f"\n✅ No issues found! Dataset looks good.")

    print("\n" + "=" * 70)


def cmd_validate(args):
    """CLI command for validation."""
    stats = validate_yolo_dataset(args.yolo_dir)
    print_validation_report(stats)

    # Save JSON report
    if args.output:

        # Convert Counter to dict for JSON serialization
        stats["class_distribution"] = dict(stats["class_distribution"])
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nDetailed report saved to: {args.output}")
