#!/usr/bin/env python3
"""
Parallel removal of YOLO samples with empty label files.
Deletes both label and corresponding image efficiently using multiple CPUs.
"""

import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def _is_empty_label(label_path: Path) -> bool:
    """Check if a YOLO label file is empty or whitespace-only."""
    try:
        if label_path.stat().st_size == 0:
            return True
        with label_path.open("r", encoding="utf-8", errors="ignore") as f:
            chunk = f.read(128)
        return len(chunk.strip()) == 0
    except Exception:
        # If unreadable (e.g., broken/corrupt), treat as problematic
        return True


def _process_label(
    label_path: str, *, labels_root: str, images_root: str, dry_run: bool
) -> tuple[bool, str, str | None]:
    """
    Worker task: check one label file, delete it (and its image) if empty.
    Returns (was_deleted, label_path, image_path or None)
    """
    lp = Path(label_path)
    if not lp.exists():
        return False, str(lp), None

    if not _is_empty_label(lp):
        return False, str(lp), None

    # Compute image path (mirror directory structure under images/)
    rel = lp.relative_to(labels_root)
    ip = (Path(images_root) / rel).with_suffix(".jpg")

    if not dry_run:
        try:
            lp.unlink(missing_ok=True)
        except Exception:
            pass
        if ip.exists():
            try:
                ip.unlink()
            except Exception:
                pass

    return True, str(lp), str(ip) if ip else None


def remove_empty_yolo_samples(
    yolo_dir: str,
    dry_run: bool = False,
    workers: int | None = None,
    chunksize: int = 128,
):
    """
    Remove problematic YOLO samples with empty label files using multiple CPUs.

    Args:
        yolo_dir: Path to YOLO dataset root (contains 'images/' and 'labels/')
        dry_run: If True, only report files that would be deleted
        workers: Number of worker processes (default: all CPUs)
        chunksize: How many files to process per batch
    """
    yolo_dir = Path(yolo_dir).resolve()
    labels_root = yolo_dir / "labels"
    images_root = yolo_dir / "images"

    if not labels_root.exists():
        print(f"❌ Labels directory not found: {labels_root}")
        return

    label_files = [str(p) for p in labels_root.rglob("*.txt")]
    total_files = len(label_files)
    if total_files == 0:
        print(f"No label files found in {labels_root}")
        return

    if workers is None:
        workers = max(1, min(cpu_count() or 1, 128))

    print("============================================================")
    print("Parallel YOLO Cleanup")
    print("============================================================")
    print(f"YOLO directory:    {yolo_dir}")
    print(f"Workers:           {workers}")
    print(f"Chunksize:         {chunksize}")
    print(f"Dry run:           {dry_run}")
    print(f"Total label files: {total_files:,}")
    print("============================================================")

    deleted_labels = 0
    deleted_images = 0

    # Bind constant args so imap_unordered passes only label_path
    worker_fn = partial(
        _process_label,
        labels_root=str(labels_root),
        images_root=str(images_root),
        dry_run=dry_run,
    )

    # maxtasksperchild avoids any slow memory growth during very long runs
    with Pool(processes=workers, maxtasksperchild=1000) as pool:
        for was_deleted, label_path, image_path in tqdm(
            pool.imap_unordered(worker_fn, label_files, chunksize=chunksize),
            total=total_files,
            desc="Checking labels",
            unit="file",
        ):
            if was_deleted:
                deleted_labels += 1
                if image_path:
                    deleted_images += 1

    print("\n" + "=" * 60)
    print("Cleanup complete!")
    print("=" * 60)
    print(f"Total label files checked: {total_files:,}")
    print(f"Labels deleted:            {deleted_labels:,}")
    print(f"Images deleted:            {deleted_images:,}")
    print(f"Remaining labels:          {total_files - deleted_labels:,}")
    print(f"Dry run:                   {dry_run}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Remove YOLO samples with empty label files (parallelized)"
    )
    parser.add_argument("yolo_dir", help="Path to YOLO dataset root")
    parser.add_argument(
        "--dry-run", action="store_true", help="Only list files, do not delete"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: all CPUs)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=128,
        help="Number of files per worker task (default: 128)",
    )
    args = parser.parse_args()

    remove_empty_yolo_samples(
        yolo_dir=args.yolo_dir,
        dry_run=args.dry_run,
        workers=args.workers,
        chunksize=args.chunksize,
    )
