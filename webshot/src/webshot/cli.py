import argparse, os, glob, json
import sys
from typing import Optional
from .config import Config
from .crawl import crawl
from .visualize import visualize_one, VizOptions
from .yolo_export import export_yolo_dataset
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

    # Validate worker count
    if args.workers < 1:
        print("ERROR: --workers must be >= 1")
        sys.exit(1)

    if args.workers > 1:
        print(f"NOTE: Using {args.workers} worker processes with persistent browsers")
        print(f"      Each worker maintains its own browser instance")
        print(
            f"      Optimal for: {args.workers * 10}-{args.workers * 50} URLs per worker"
        )

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

    groups = find_duplicates(
        args.image_dir,
        distance_threshold=args.threshold,
        workers=args.dedupe_workers,
        chunksize=args.dedupe_chunksize,
    )
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
        workers=args.export_workers,
        chunksize=args.export_chunksize,
    )


def cmd_pipeline(args):
    """Run full pipeline: crawl -> visualize -> export to YOLO."""
    print("\n" + "=" * 60)
    print("STARTING FULL PIPELINE")
    print("=" * 60)

    cfg = _default_cfg(args)
    cfg.capture_full_page = args.full_page
    cfg.filter_config.save_unfiltered = args.save_unfiltered

    # Validate worker count
    if args.workers < 1:
        print("ERROR: --workers must be >= 1")
        sys.exit(1)

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
    if args.viz_workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"Using {args.viz_workers} threads for visualization")
        with ThreadPoolExecutor(max_workers=args.viz_workers) as executor:
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
        workers=args.export_workers,
        chunksize=args.export_chunksize,
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


def cmd_train(args):
    """Train YOLO model with specified configuration."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Install with:")
        print("  pip install ultralytics")
        sys.exit(1)

    # Validate data.yaml
    if not os.path.exists(args.data):
        print(f"ERROR: Data file not found: {args.data}")
        print("\nRun the pipeline first:")
        print("  wsd pipeline --urls urls.csv")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STARTING YOLO TRAINING")
    print("=" * 60)
    print(f"Model:      {args.model}")
    print(f"Dataset:    {args.data}")
    print(f"Epochs:     {args.epochs}")
    print(f"Batch:      {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device:     {args.device}")
    print("=" * 60 + "\n")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        pretrained=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        patience=args.patience,
        save=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Weights saved to: {args.project}/{args.name}/weights/")


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
        default=4,
        help="Number of parallel workers (default: 4)",
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
    pd.add_argument(
        "--dedupe-workers",
        type=int,
        default=os.cpu_count(),
        help="Processes for dedupe (default: all CPUs)",
    )
    pd.add_argument(
        "--dedupe-chunksize",
        type=int,
        default=64,
        help="Task chunk size per worker (default: 64)",
    )
    pd.set_defaults(func=cmd_dedupe)

    # reconstruct-hierarchy — infer parent-child relationships from bounding boxes
    ph = sub.add_parser(
        "reconstruct-hierarchy",
        help="Reconstruct parent-child hierarchy from bounding box geometry",
    )
    ph.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing *.elements.json files",
    )
    ph.add_argument(
        "--min-containment",
        type=float,
        default=0.95,
        help="Minimum fraction of element that must be contained in parent (default: 0.95)",
    )
    ph.add_argument(
        "--no-semantic-hints",
        action="store_true",
        help="Disable using VLM labels to inform hierarchy decisions",
    )
    ph.add_argument(
        "--force",
        action="store_true",
        help="Reconstruct even if hierarchy already exists",
    )
    ph.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: all CPUs)",
    )

    def _cmd_reconstruct_hierarchy(args):
        from .hierarchy_reconstruct import (
            reconstruct_hierarchy_for_directory,
            print_stats,
        )

        stats = reconstruct_hierarchy_for_directory(
            raw_dir=args.raw_dir,
            min_containment=args.min_containment,
            use_semantic_hints=not args.no_semantic_hints,
            force=args.force,
            workers=args.workers,
        )
        print_stats(stats)

    ph.set_defaults(func=_cmd_reconstruct_hierarchy)

    # compute-own-text — compute own_text field for all elements
    pot = sub.add_parser(
        "compute-own-text",
        help="Compute own_text field for elements (removes duplicated text from children)",
    )
    pot.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing *.elements.json files",
    )
    pot.add_argument(
        "--force",
        action="store_true",
        help="Recompute own_text even if it already exists",
    )
    pot.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: all CPUs)",
    )

    def _cmd_compute_own_text(args):
        from .hierarchy_reconstruct import (
            add_own_text_for_directory,
            print_stats,
        )

        stats = add_own_text_for_directory(
            raw_dir=args.raw_dir,
            force=args.force,
            workers=args.workers,
        )
        print_stats(stats)

    pot.set_defaults(func=_cmd_compute_own_text)

    # screentag-export — export ScreenTag representation from elements
    pst = sub.add_parser(
        "screentag-export",
        help="Export ScreenTag representation from elements.json files",
    )
    pst.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing *.elements.json and *.meta.json files",
    )
    pst.add_argument(
        "--force",
        action="store_true",
        help="Regenerate screentag even if it already exists",
    )
    pst.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: all CPUs)",
    )

    def _cmd_screentag_export(args):
        from .hierarchy_reconstruct import (
            export_screentag_for_directory,
            print_screentag_stats,
        )

        stats = export_screentag_for_directory(
            raw_dir=args.raw_dir,
            force=args.force,
            workers=args.workers,
        )
        print_screentag_stats(stats)

    pst.set_defaults(func=_cmd_screentag_export)

    # viz-screentag — visualize ScreenTag annotations on images
    pvs = sub.add_parser(
        "viz-screentag",
        help="Visualize ScreenTag annotations on images",
    )
    pvs.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing *.png and *.screentag.txt files",
    )
    pvs.add_argument(
        "--viz-dir",
        default="data/viz_screentag",
        help="Output directory for visualizations",
    )
    pvs.add_argument(
        "--image",
        help="Single image path (for single-file mode)",
    )
    pvs.add_argument(
        "--screentag",
        help="Single screentag path (for single-file mode)",
    )
    pvs.add_argument(
        "--output",
        help="Output path (for single-file mode)",
    )
    pvs.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for batch mode",
    )
    pvs.add_argument(
        "--no-labels",
        action="store_true",
        help="Don't draw element labels",
    )
    pvs.add_argument(
        "--no-text",
        action="store_true",
        help="Don't show text content in labels",
    )
    pvs.add_argument(
        "--fill-opacity",
        type=int,
        default=15,
        help="Box fill opacity (0-255, default 15 for transparency)",
    )
    pvs.add_argument(
        "--outline-opacity",
        type=int,
        default=200,
        help="Outline opacity (0-255, default 200 for visibility)",
    )
    pvs.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Outline line width (default 2)",
    )

    def _cmd_viz_screentag(args):
        from .visualize_screentag import (
            visualize_screentag,
            visualize_screentag_directory,
            print_viz_stats,
            ScreenTagVizOptions,
        )
        
        opts = ScreenTagVizOptions(
            draw_labels=not args.no_labels,
            draw_text=not args.no_text,
            fill_opacity=args.fill_opacity,
            outline_opacity=args.outline_opacity,
            line_width=args.line_width,
        )
        
        if args.image and args.screentag:
            # Single file mode
            output = args.output or args.image.replace(".png", ".screentag_viz.jpg")
            success = visualize_screentag(
                image_path=args.image,
                screentag_path=args.screentag,
                output_path=output,
                opts=opts,
            )
            if success:
                print(f"Saved visualization to: {output}")
            else:
                print("Failed to create visualization")
        else:
            # Batch mode
            stats = visualize_screentag_directory(
                raw_dir=args.raw_dir,
                viz_dir=args.viz_dir,
                workers=args.workers,
                opts=opts,
            )
            print_viz_stats(stats)

    pvs.set_defaults(func=_cmd_viz_screentag)

    # vlm-label — relabel elements with a VLM
    pl = sub.add_parser("vlm-label", help="Relabel UI elements with a VLM (Qwen3-VL via vLLM)")
    pl.add_argument("--raw-dir", default="data/raw", help="Directory with *.png/*.elements.json/*.meta.json")
    pl.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="vLLM model id")
    pl.add_argument("--batch-size", type=int, default=16, help="VLM micro-batch size")
    pl.add_argument("--tp", dest="tensor_parallel_size", type=int, default=1, help="Tensor-parallel size")
    pl.add_argument("--crops-dir", default="data/crops", help="Where to write element crops")
    pl.add_argument("--limit", type=int, help="Limit number of screenshots (for smoke testing)")
    pl.add_argument("--min-elem-size", type=int, default=3, help="Skip elements smaller than this (CSS px)")
    pl.add_argument("--inplace-elements", action="store_true", help="Write vlm_label/vlm_conf back into *.elements.json")
    pl.add_argument("--viz-dir", help="If set, write visualizations with predicted labels")
    pl.add_argument("--shard-index", type=int, default=0, help="Shard index (0-based)")
    pl.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    def _cmd_vlm_label(args):
        from .vlm_refine import label_dir
        label_dir(
            raw_dir=args.raw_dir,
            model=args.model,
            crops_dir=args.crops_dir,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            limit_images=args.limit,
            min_elem_size=args.min_elem_size,
            inplace_elements=args.inplace_elements,
            viz_dir=args.viz_dir,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        )
    pl.set_defaults(func=_cmd_vlm_label)

    # vlm-score — score annotation quality with a VLM
    ps = sub.add_parser("vlm-score", help="Score annotation quality with a VLM (Qwen3-VL via vLLM)")
    ps.add_argument("--viz-dir", default="data/viz_screentag", help="Directory with visualization images (boxes/labels overlaid)")
    ps.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="vLLM model id")
    ps.add_argument("--batch-size", type=int, default=256, help="VLM micro-batch size")
    ps.add_argument("--tp", dest="tensor_parallel_size", type=int, default=1, help="Tensor-parallel size")
    ps.add_argument("--limit", type=int, help="Limit number of images (for smoke testing)")
    ps.add_argument("--threshold", type=float, default=70.0, help="Quality threshold; images below this are filtered (0-100)")
    ps.add_argument("--output", default="data/filtered_low_quality.txt", help="Output file for filtered sample paths")
    ps.add_argument("--scores-json", help="If set, write all scores to this JSON file")
    ps.add_argument("--images-per-pass", type=int, default=512, help="Number of images per outer pass (memory tuning)")
    ps.add_argument("--shard-index", type=int, default=0, help="Shard index (0-based)")
    ps.add_argument("--num-shards", type=int, default=1, help="Total number of shards")
    ps.add_argument("--viz-pattern", default="*.jpg", help="Glob pattern for visualization files")
    def _cmd_vlm_score(args):
        from .vlm_quality_score import score_visualizations
        score_visualizations(
            viz_dir=args.viz_dir,
            model=args.model,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            limit=args.limit,
            threshold=args.threshold,
            output_file=args.output,
            scores_json=args.scores_json,
            images_per_pass=args.images_per_pass,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            viz_pattern=args.viz_pattern,
        )
    ps.set_defaults(func=_cmd_vlm_score)

    # yolo export
    py = sub.add_parser("yolo", help="Export dataset to YOLO format")
    py.add_argument("--raw-dir", default="data/raw", help="Raw data directory")
    py.add_argument("--yolo-dir", default="data/yolo", help="YOLO output directory")
    py.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio")
    py.add_argument("--val-ratio", type=float, default=0.05, help="Val split ratio")
    py.add_argument("--test-ratio", type=float, default=0.05, help="Test split ratio")
    py.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    py.add_argument(
        "--export-workers",
        type=int,
        default=os.cpu_count(),
        help="Processes for YOLO export (default: all CPUs)",
    )
    py.add_argument(
        "--export-chunksize",
        type=int,
        default=64,
        help="Records per task handed to each worker (default: 16)",
    )
    py.set_defaults(func=cmd_yolo)

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
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    pp.add_argument(
        "--viz-workers",
        type=int,
        default=4,  # Visualization is less intensive
        help="Number of parallel threads for visualization (default: 4)",
    )
    pp.add_argument(
        "--export-workers",
        type=int,
        default=os.cpu_count(),
        help="Processes for YOLO export (default: all CPUs)",
    )
    pp.add_argument(
        "--export-chunksize",
        type=int,
        default=64,
        help="Records per task handed to each worker (default: 64)",
    )
    pp.set_defaults(func=cmd_pipeline)

    # train command
    pt = sub.add_parser("train", help="Train YOLO model")
    pt.add_argument("--data", default="data/yolo/data.yaml", help="Path to data.yaml")
    pt.add_argument("--model", default="yolo11l.pt", help="Model to train")
    pt.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    pt.add_argument("--batch", type=int, default=16, help="Batch size")
    pt.add_argument("--imgsz", type=int, default=640, help="Image size")
    pt.add_argument("--device", default="0", help="Device (0, cpu, 0,1,2,3)")
    pt.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    pt.add_argument("--project", default="runs/detect", help="Project directory")
    pt.add_argument("--name", default="webshot_ui", help="Experiment name")
    pt.add_argument("--optimizer", default="auto", help="Optimizer")
    pt.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    pt.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    pt.add_argument(
        "--exist-ok", action="store_true", help="Overwrite existing project"
    )
    pt.set_defaults(func=cmd_train)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
