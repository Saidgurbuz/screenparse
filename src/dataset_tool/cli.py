# src/dataset_tool/cli.py
import argparse, os, glob, json
from typing import Optional
from .config import Config
from .crawl import crawl
from .visualize import visualize_one, VizOptions
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
    crawl(args.urls, out_dir=cfg.out_dir, do_ocr=cfg.do_ocr, headless=cfg.headless)


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
    pv.add_argument(
        "--headed", action="store_true"
    )  # not used here but kept for symmetry
    pv.add_argument("--ocr-flag", dest="ocr", action="store_true")  # ignore; symmetry
    pv.set_defaults(func=cmd_viz)

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

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
