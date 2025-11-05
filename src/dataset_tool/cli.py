import argparse, os, glob, json
from typing import Optional
from .config import Config
from .crawl import crawl
from .visualize import visualize_one, VizOptions
from .yolo_export import export_yolo_dataset  # NEW
from .utils import load_json, ensure_dir


def _default_cfg(args) -> Config:
    return Config(
        headless=not args.headed,
        do_ocr=args.ocr,
        out_dir=args.out,
        viz_dir=args.viz,
    )


def cmd_crawl(args):
    cfg = _default_cfg(args)
    cfg.capture_full_page = args.full_page
    cfg.filter_config.save_unfiltered = args.save_unfiltered
    crawl(
        args.urls,
        out_dir=cfg.out_dir,
        do_ocr=cfg.do_ocr,
        headless=cfg.headless,
        workers=args.workers,
    )


def cmd_validate(args):
    from .validator import validate_yolo_dataset, print_validation_report

    stats = validate_yolo_dataset(args.yolo_dir)
    print_validation_report(stats)

    if args.output:
        import json

        stats["class_distribution"] = dict(stats["class_distribution"])
        with open(args.output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nDetailed report saved to: {args.output}")


def cmd_viz(args):
    cfg = _default_cfg(args)
    ensure_dir(cfg.viz_dir)

    def _maybe(path: Optional[str]) -> Optional[str]:
        return path if (path and os.path.exists(path)) else None

    opts = VizOptions(
        draw_elements=not args.no_elements,
        draw_text_spans=not args.no_texts,
        draw_ocr=args.ocr_overlay,
        label_elements=not args.no_labels,
        label_text_spans=args.label_text_spans,
        opacity=args.opacity,
        line_width=args.line_width,
    )

    # Modes:
    #  - base pattern: expects "...<stem>.(elements|texts|ocr).json" next to image
    #  - or explicit paths
    targets = []
    if args.image and os.path.exists(args.image):
        # Explicit single
        targets = [(args.image, args.elements, args.texts, args.ocr)]
    else:
        # Batch: find all images in out dir
        images = sorted(glob.glob(os.path.join(cfg.out_dir, "*.png")))
        for img in images:
            stem = os.path.splitext(img)[0]
            elements = stem + ".elements.json"
            texts = stem + ".texts.json"
            ocr = stem + ".ocr.json"
            targets.append((img, _maybe(elements), _maybe(texts), _maybe(ocr)))

    for img, elements, texts, ocr in targets:
        out_path = os.path.join(
            cfg.viz_dir, os.path.basename(img).replace(".png", ".viz.jpg")
        )
        visualize_one(img, elements, texts, ocr, out_path, opts)
        print("Wrote", out_path)


def cmd_dedupe(args):
    from .dedupe import find_duplicates, write_groups_csv

    groups = find_duplicates(args.image_dir, distance_threshold=args.threshold)
    if args.csv:
        write_groups_csv(groups, args.csv)
        print(f"Saved groups CSV: {args.csv}")
    print(f"Found {sum(len(g)>1 for g in groups)} duplicate groups")


def cmd_yolo(args):
    """Export dataset to YOLO format."""
    export_yolo_dataset(
        raw_dir=args.raw_dir,
        yolo_dir=args.yolo_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


def cmd_pipeline(args):
    """Run full pipeline: crawl -> visualize -> export to YOLO."""
    print("\n" + "=" * 60)
    print("STARTING FULL PIPELINE")
    print("=" * 60)

    cfg = _default_cfg(args)
    cfg.capture_full_page = args.full_page
    cfg.filter_config.save_unfiltered = args.save_unfiltered

    # Step 1: Crawl
    print("\n[1/4] Crawling URLs...")
    crawl(
        args.urls,
        out_dir=cfg.out_dir,
        do_ocr=cfg.do_ocr,
        headless=cfg.headless,
        workers=args.workers,
    )

    # Step 2: Deduplicate (optional)
    if not args.skip_dedupe:
        print("\n[2/4] Detecting duplicates...")
        from .dedupe import find_duplicates, write_groups_csv

        groups = find_duplicates(cfg.out_dir, distance_threshold=args.threshold)
        if groups:
            dedupe_csv = os.path.join(cfg.out_dir, "../dupes", "dupe_groups.csv")
            write_groups_csv(groups, dedupe_csv)
            print(
                f"Found {sum(len(g)>1 for g in groups)} duplicate groups -> {dedupe_csv}"
            )
    else:
        print("\n[2/4] Skipping deduplication")

    # Step 3: Visualize
    print("\n[3/4] Generating visualizations...")
    ensure_dir(cfg.viz_dir)
    opts = VizOptions(
        draw_elements=True,
        draw_text_spans=False,
        draw_ocr=args.ocr,
        label_elements=True,
        opacity=80,
        line_width=2,
    )
    images = sorted(glob.glob(os.path.join(cfg.out_dir, "*.png")))

    # Parallel visualization
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for img in images:
                stem = os.path.splitext(img)[0]
                elements = (
                    stem + ".elements.json"
                    if os.path.exists(stem + ".elements.json")
                    else None
                )
                texts = (
                    stem + ".texts.json"
                    if os.path.exists(stem + ".texts.json")
                    else None
                )
                ocr = stem + ".ocr.json" if os.path.exists(stem + ".ocr.json") else None
                out_path = os.path.join(
                    cfg.viz_dir, os.path.basename(img).replace(".png", ".viz.jpg")
                )
                futures.append(
                    executor.submit(
                        visualize_one, img, elements, texts, ocr, out_path, opts
                    )
                )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Visualization error: {e}")
    else:
        for img in images:
            stem = os.path.splitext(img)[0]
            elements = (
                stem + ".elements.json"
                if os.path.exists(stem + ".elements.json")
                else None
            )
            texts = (
                stem + ".texts.json" if os.path.exists(stem + ".texts.json") else None
            )
            ocr = stem + ".ocr.json" if os.path.exists(stem + ".ocr.json") else None
            out_path = os.path.join(
                cfg.viz_dir, os.path.basename(img).replace(".png", ".viz.jpg")
            )
            visualize_one(img, elements, texts, ocr, out_path, opts)

    print(f"Visualizations saved to: {cfg.viz_dir}")

    # Step 4: Export to YOLO
    print("\n[4/4] Exporting to YOLO format...")
    export_yolo_dataset(
        raw_dir=cfg.out_dir,
        yolo_dir=args.yolo_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"Raw data:        {cfg.out_dir}")
    print(f"Visualizations:  {cfg.viz_dir}")
    print(f"YOLO dataset:    {args.yolo_dir}")
    print("\nYou can now train YOLO with:")
    print(
        f"  yolo train data={os.path.join(args.yolo_dir, 'data.yaml')} model=yolov8n.pt epochs=100"
    )


def main():
    p = argparse.ArgumentParser(
        prog="wsd", description="Web screenshot dataset toolkit"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # crawl
    pc = sub.add_parser("crawl", help="Collect pages listed in urls.csv")
    pc.add_argument("--urls", default="urls.csv")
    pc.add_argument("--out", default="data/raw")
    pc.add_argument("--viz", default="data/viz")
    pc.add_argument(
        "--ocr", action="store_true", help="Run Tesseract OCR per element (optional)"
    )
    pc.add_argument("--headed", action="store_true", help="Run browser headed")
    pc.add_argument(
        "--full-page",
        action="store_true",
        help="Capture full page with scrolling (default: viewport only)",
    )
    pc.add_argument(
        "--save-unfiltered",
        action="store_true",
        help="Save unfiltered elements for debugging",
    )
    pc.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    pc.set_defaults(func=cmd_crawl)

    # viz
    pv = sub.add_parser("viz", help="Visualize annotations over screenshots")
    pv.add_argument("--image", help="Path to a single image.png (optional)")
    pv.add_argument("--elements", help="Path to elements.json (optional)")
    pv.add_argument("--texts", help="Path to texts.json (optional)")
    pv.add_argument("--ocr", help="Path to ocr.json (optional)")
    pv.add_argument("--out", default="data/raw")
    pv.add_argument("--viz", default="data/viz")
    pv.add_argument(
        "--ocr-overlay", action="store_true", help="Render OCR labels if available"
    )
    pv.add_argument("--no-elements", action="store_true")
    pv.add_argument("--no-texts", action="store_true")
    pv.add_argument("--no-labels", action="store_true")
    pv.add_argument("--label-text-spans", action="store_true")
    pv.add_argument("--opacity", type=int, default=80)
    pv.add_argument("--line-width", type=int, default=2)
    pv.add_argument("--headed", action="store_true")
    pv.add_argument("--ocr-flag", dest="ocr", action="store_true")
    pv.set_defaults(func=cmd_viz)

    # validate
    pval = sub.add_parser("validate", help="Validate YOLO dataset quality")
    pval.add_argument("--yolo-dir", default="data/yolo", help="YOLO dataset directory")
    pval.add_argument("--output", help="Save JSON report to file")
    pval.set_defaults(func=cmd_validate)

    # dedupe
    pd = sub.add_parser("dedupe", help="Perceptual dedupe over images")
    pd.add_argument("--image-dir", default="data/raw")
    pd.add_argument(
        "--threshold",
        type=int,
        default=8,
        help="Hamming distance threshold (lower is stricter)",
    )
    pd.add_argument("--csv", default="data/dupes/dupe_groups.csv")
    pd.set_defaults(func=cmd_dedupe)

    # yolo export (NEW)
    py = sub.add_parser("yolo", help="Export dataset to YOLO format")
    py.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    py.add_argument("--yolo-dir", default="data/yolo", help="YOLO output directory")
    py.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    py.add_argument("--val-ratio", type=float, default=0.2, help="Val split ratio")
    py.add_argument("--test-ratio", type=float, default=0.1, help="Test split ratio")
    py.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    py.set_defaults(func=cmd_yolo)

    # pipeline (NEW)
    pp = sub.add_parser("pipeline", help="Run full pipeline (crawl + viz + yolo)")
    pp.add_argument("--urls", default="urls.csv", help="URLs CSV file")
    pp.add_argument("--out", default="data/raw", help="Raw output directory")
    pp.add_argument("--viz", default="data/viz", help="Visualization directory")
    pp.add_argument("--yolo-dir", default="data/yolo", help="YOLO dataset directory")
    pp.add_argument("--ocr", action="store_true", help="Enable OCR")
    pp.add_argument("--headed", action="store_true", help="Run browser headed")
    pp.add_argument(
        "--skip-dedupe", action="store_true", help="Skip deduplication step"
    )
    pp.add_argument("--threshold", type=int, default=8, help="Dedupe threshold")
    pp.add_argument("--train-ratio", type=float, default=0.7)
    pp.add_argument("--val-ratio", type=float, default=0.2)
    pp.add_argument("--test-ratio", type=float, default=0.1)
    pp.add_argument("--seed", type=int, default=42)
    pp.add_argument(
        "--full-page",
        action="store_true",
        help="Capture full page with scrolling (default: viewport only)",
    )
    pp.add_argument(
        "--save-unfiltered",
        action="store_true",
        help="Save unfiltered elements for debugging",
    )
    pp.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    pp.set_defaults(func=cmd_pipeline)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
