#!/usr/bin/env python3
"""
End-to-end pipeline: YOLO Inference -> OCR -> Hierarchy Reconstruction -> ScreenTag

# Process a single image
python scripts/yolo_to_screentag.py \
    --weights path/to/best.pt \
    --source path/to/image.png \
    --output path/to/output.screentag.txt

# Process a directory of images
python scripts/yolo_to_screentag.py \
    --weights /proj/docling-vision/users/said/webshot-dataset/runs/detect/webshot_ui_refined_labels_filtered/weights/best.pt \
    --source /proj/docling-vision/users/said/webshot-dataset/all_images

"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path to allow imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent / "src"
sys.path.append(str(src_dir))

from dataset_tool.inference_pipeline import InferencePipeline, process_directory

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO output to ScreenTag format end-to-end")
    parser.add_argument("--weights", required=True, help="Path to YOLO weights (.pt)")
    parser.add_argument("--source", required=True, help="Input directory containing images or single image path")
    parser.add_argument("--output", default="output/screentag", help="Output directory for ScreenTag files")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.17, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IOU threshold")
    parser.add_argument("--device", default=None, help="Device to run on (cpu, 0, mps)")
    parser.add_argument("--ocr-engine", default="easyocr", choices=["easyocr", "tesseract"], help="OCR engine to use")
    parser.add_argument("--no-json", action="store_true", help="Disable saving elements JSON file")
    parser.add_argument("--no-viz", action="store_true", help="Disable saving visualization image")
    
    args = parser.parse_args()
    
    pipeline = InferencePipeline(
        weights_path=args.weights,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        ocr_engine=args.ocr_engine,
        imgsz=args.imgsz
    )
    
    save_json = not args.no_json
    save_viz = not args.no_viz
    
    if os.path.isfile(args.source):
        # Single file
        print(f"Processing single file: {args.source}")
        if args.output.endswith(".txt"):
            out_path = args.output
        else:
            os.makedirs(args.output, exist_ok=True)
            out_name = os.path.splitext(os.path.basename(args.source))[0] + ".screentag.txt"
            out_path = os.path.join(args.output, out_name)
            
        pipeline.process_single(args.source, out_path, save_json_elements=save_json, save_viz=save_viz)
        print(f"Saved to {out_path}")
        
    elif os.path.isdir(args.source):
        # Directory
        print(f"Processing directory: {args.source}")
        process_directory(pipeline, args.source, args.output, save_json_elements=save_json, save_viz=save_viz)
        print(f"All done. Results in {args.output}")
        
    else:
        print(f"Error: Source {args.source} not found")

if __name__ == "__main__":
    main()
