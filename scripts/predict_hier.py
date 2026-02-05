# scripts/predict_hier.py
"""
Hierarchy-aware YOLO prediction for UI detection.

Usage:
    python scripts/predict_hier.py \
        --weights path/to/weights.pt \
        --source path/to/images \
        --conf 0.21
"""

import argparse
import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset_tool.ui_postproc import get_class_id_sets, postprocess_detections


def auto_device():
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def xyxy_to_yolo_norm(box, img_w, img_h):
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", default="images/val")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--project", type=str, default="viz/preds_hier")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=None, help="max number of images")
    ap.add_argument("--max-det", type=int, default=700)
    ap.add_argument("--cluster-iou", type=float, default=0.75)
    args = ap.parse_args()

    # dynamic run name if not provided
    if args.name is None:
        weights_path = Path(args.weights)
        parent_folder = weights_path.parent.parent.name
        conf_str = str(args.conf).replace(".", "")
        args.name = f"hier_{parent_folder}_c{conf_str}_imgsz{args.imgsz}_md{args.max_det}"

    dev = args.device or auto_device()
    model = YOLO(args.weights)
    names = model.names
    container_ids, atomic_ids, non_nestable_ids = get_class_id_sets(names)

    base_dir = Path(args.project) / args.name
    out_dir = base_dir
    counter = 1
    while out_dir.exists():
        out_dir = Path(str(base_dir) + str(counter))
        counter += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    # optional limit of images
    source = args.source
    files = None
    if args.limit is not None:
        source_path = Path(source)
        if source_path.is_dir():
            files = sorted(
                p
                for p in source_path.iterdir()
                if p.suffix.lower()
                in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
            )
            random.shuffle(files)
            files = files[: args.limit]
            source = [str(p) for p in files]

    # run inference with stream=True to control postprocessing
    results = model(
        source,
        imgsz=args.imgsz,
        conf=args.conf,  # we'll still re-apply in postprocess
        device=dev,
        max_det=args.max_det,
        stream=True,
        verbose=True,
    )

    for res in results:
        # original image
        img = res.orig_img.copy()
        h, w = img.shape[:2]

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)

        # Apply our smart postprocessing
        boxes_pp, scores_pp, clss_pp = postprocess_detections(
            boxes,
            scores,
            clss,
            container_ids,
            atomic_ids,
            non_nestable_ids,
            conf_thr=args.conf,
            cluster_iou=args.cluster_iou,
        )

        # # Draw overlays
        # vis = img.copy()
        # for box, s, c in zip(boxes_pp, scores_pp, clss_pp):
        #     x1, y1, x2, y2 = map(int, box)
        #     cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     label = f"{names[int(c)]} {float(s):.2f}"
        #     cv2.putText(
        #         vis,
        #         label,
        #         (x1, max(0, y1 - 5)),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 255, 0),
        #         1,
        #         cv2.LINE_AA,
        #     )

        # # save image + YOLO-style label file for debugging
        # stem = Path(res.path).stem
        # out_img_path = out_dir / f"{stem}.jpg"
        # cv2.imwrite(str(out_img_path), vis)

        # label_path = labels_dir / f"{stem}.txt"
        # with open(label_path, "w") as f:
        #     for box, s, c in zip(boxes_pp, scores_pp, clss_pp):
        #         cx, cy, bw, bh = xyxy_to_yolo_norm(box, w, h)
        #         f.write(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {float(s):.4f}\n")
        # If nothing left after postprocessing, optionally still write an empty txt
        stem = Path(res.path).stem
        out_img_path = out_dir / f"{stem}.jpg"
        label_path = labels_dir / f"{stem}.txt"

        if len(boxes_pp) == 0:
            # optional: touch an empty label file to match previous behavior
            # label_path.touch()
            continue

        # --- push postprocessed boxes back into YOLO Results ---

        # Results.update expects boxes as (N, 6): x1, y1, x2, y2, conf, cls   [oai_citation:3‡Ultralytics Docs](https://docs.ultralytics.com/reference/engine/results/)
        dets_np = np.column_stack(
            [boxes_pp, scores_pp.astype(np.float32), clss_pp.astype(np.float32)]
        )  # shape: (N, 6)

        dets = torch.as_tensor(
            dets_np,
            dtype=torch.float32,
            device=res.boxes.data.device if hasattr(res.boxes, "data") else "cpu",
        )

        # Update the existing Results object with your filtered boxes
        res.update(boxes=dets)

        # --- use YOLO's own visualization & label saving ---

        # Save annotated image with YOLO’s standard look
        res.save(
            filename=str(out_img_path),
            line_width=2,       # e.g., thickness 2
            font_size=8        # e.g., font size 12
        )

        # Save YOLO-formatted labels (class, x_center, y_center, w, h, conf)
        # `save_conf=True` appends the confidence at the end   [oai_citation:4‡Ultralytics Docs](https://docs.ultralytics.com/reference/engine/results/)
        res.save_txt(txt_file=str(label_path), save_conf=True)

    print(f"\nHierarchy-aware overlays & labels saved under {out_dir}")


if __name__ == "__main__":
    main()