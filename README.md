# Webshot: Automated Dataset Generation for Screen Parsing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/Paper-ICML%202026-red.svg)](https://arxiv.org/abs/XXXX.XXXXX)

**Webshot** is the automated dataset generation pipeline behind [ScreenParse](https://arxiv.org/abs/XXXX.XXXXX), a large-scale dataset for complete screen parsing with dense UI element annotations. This repository provides tools for:

- 🌐 **Automated web crawling** with Playwright for diverse screenshot collection
- 🎯 **Dense UI annotation** with 55 element classes and bounding boxes
- 🤖 **VLM-based refinement** using Qwen3-VL for improved label quality
- 📊 **Quality filtering** with VLM-based scoring
- 🔄 **Hierarchy reconstruction** from geometric relationships
- 📦 **YOLO export** with configurable train/val/test splits
- 🧪 **Evaluation framework** for benchmarking screen parsing models

## Overview

ScreenParse is a large-scale dataset for complete screen parsing, with dense annotations of all visible UI elements (boxes, 55-class types, and text) across 771K web screenshots (21M elements). Webshot is the scalable pipeline that generates this dataset through automated rendering, annotation extraction, VLM-based relabeling, and quality filtering.

## Features

### Dataset Generation Pipeline
- **Multi-process crawling**: Efficient parallel processing with persistent browsers
- **Dense annotations**: 55 UI element classes with bounding boxes, text, and hierarchy
- **VLM refinement**: Qwen3-VL-based relabeling for improved annotation quality
- **Quality scoring**: Automated filtering of low-quality annotations
- **Deduplication**: Perceptual hashing to remove duplicate screenshots
- **Flexible export**: YOLO format with customizable splits

### Evaluation Framework
- **Multi-dataset support**: YOLO, GroundCUA, ScreenSpot, and raw formats
- **Pluggable metrics**: PageIoU, Label-aware PageIoU, mAP, Recall, NED
- **Multiple models**: Support for ScreenVLM, RT-DETR, Qwen3-VL, InternVL3, Gemini, and more
- **Batch processing**: Efficient evaluation with configurable batch sizes
- **Extensible design**: Easy to add new metrics and model runners

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/webshot-dataset.git
cd webshot-dataset

# Install dependencies
pip install -e .

# Install Playwright browsers
playwright install chromium

# Optional: Install Tesseract for OCR support
# macOS: brew install tesseract
# Ubuntu: apt-get install tesseract-ocr
```

## Quick Start

### Generate a Dataset

```bash
# Create a URL list
cat > urls.csv << EOF
https://github.com
https://stackoverflow.com
https://reddit.com
EOF

# Run the complete pipeline (crawl → visualize → export)
wsd pipeline --urls urls.csv --workers 4

# Output structure:
# data/
# ├── raw/              # Screenshots + annotations
# ├── viz/              # Visualizations
# └── yolo/             # YOLO training dataset
```

### Train a Model

```bash
# Train YOLO model
wsd train --data data/yolo/data.yaml --epochs 100 --batch 16

# Or use ultralytics directly
yolo train data=data/yolo/data.yaml model=yolov8n.pt epochs=100
```

### Evaluate Models

```bash
# Evaluate a YOLO model
wsd-eval \
  --format yolo \
  --image-dir data/yolo/images/val \
  --labels-dir data/yolo/labels/val \
  --classes data/yolo/classes.txt \
  --yolo-model yolov8n.pt \
  --metrics page_iou,label_page_iou,map

# Evaluate a VLM (requires vLLM)
wsd-eval \
  --format yolo \
  --image-dir data/yolo/images/val \
  --labels-dir data/yolo/labels/val \
  --classes data/yolo/classes.txt \
  --qwen3-vl Qwen/Qwen3-VL-8B-Instruct \
  --metrics page_iou,label_page_iou,map \
  --batch-size 32
```

## Documentation

- **[USAGE.md](USAGE.md)**: Comprehensive usage guide with examples
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Guidelines for contributing
- **[examples/](examples/)**: Sample configurations and scripts

## UI Element Classes

Webshot annotates 55 UI element types optimized for screen parsing:

**Interactive Elements**: `button`, `input`, `link`, `checkbox`, `dropdown`, `search`, `tab`

**Content Elements**: `text`, `heading`, `image`, `icon`, `video`, `table`, `list`

**Layout Elements**: `navigation`, `header`, `footer`, `card`, `form`, `menu`, `modal`

**And more**: See [USAGE.md](USAGE.md) for the complete list.

## CLI Commands

<details>
<summary><b>📋 Dataset Generation Commands</b> (click to expand)</summary>

### `wsd crawl` - Crawl websites and collect annotations
```bash
# Basic crawl
wsd crawl --urls urls.csv --out data/raw

# Parallel crawling with 4 workers
wsd crawl --urls urls.csv --workers 4

# With OCR enabled
wsd crawl --urls urls.csv --ocr

# Full-page capture (with scrolling)
wsd crawl --urls urls.csv --full-page

# Headed mode (see browser)
wsd crawl --urls urls.csv --headed
```

### `wsd viz` - Visualize annotations
```bash
# Visualize all images in directory
wsd viz --out data/raw --viz data/viz

# Visualize single image
wsd viz --image data/raw/example.png --elements data/raw/example.elements.json

# With OCR overlay
wsd viz --ocr-overlay

# Customize appearance
wsd viz --opacity 120 --line-width 3 --no-labels
```

### `wsd dedupe` - Deduplicate images
```bash
# Find duplicates
wsd dedupe --image-dir data/raw --csv data/dupes/dupe_groups.csv

# Adjust threshold (lower = stricter)
wsd dedupe --image-dir data/raw --threshold 5

# Parallel processing
wsd dedupe --image-dir data/raw --dedupe-workers 8
```

### `wsd yolo` - Export to YOLO format
```bash
# Basic export
wsd yolo --raw-dir data/raw --yolo-dir data/yolo

# Custom splits
wsd yolo --raw-dir data/raw --yolo-dir data/yolo \
         --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05

# Parallel export
wsd yolo --raw-dir data/raw --export-workers 16
```

### `wsd validate` - Validate dataset quality
```bash
# Validate YOLO dataset
wsd validate --yolo-dir data/yolo

# Save report to JSON
wsd validate --yolo-dir data/yolo --output report.json
```

### `wsd pipeline` - Run complete pipeline
```bash
# Full pipeline with defaults
wsd pipeline --urls urls.csv

# Custom configuration
wsd pipeline --urls urls.csv \
             --workers 4 \
             --train-ratio 0.7 \
             --val-ratio 0.2 \
             --test-ratio 0.1

# Skip deduplication
wsd pipeline --urls urls.csv --skip-dedupe

# With OCR and full-page capture
wsd pipeline --urls urls.csv --ocr --full-page
```

### `wsd train` - Train YOLO model
```bash
# Basic training
wsd train --data data/yolo/data.yaml --epochs 100

# Custom configuration
wsd train --data data/yolo/data.yaml \
          --model yolov8m.pt \
          --epochs 200 \
          --batch 32 \
          --imgsz 1280 \
          --device 0

# Multi-GPU training
wsd train --data data/yolo/data.yaml --device 0,1,2,3
```

</details>

<details>
<summary><b>🤖 VLM Pipeline Commands</b> (click to expand)</summary>

### `wsd reconstruct-hierarchy` - Rebuild parent-child relationships
```bash
# Reconstruct hierarchy from bounding boxes
wsd reconstruct-hierarchy --raw-dir data/raw

# Custom containment threshold
wsd reconstruct-hierarchy --raw-dir data/raw --min-containment 0.90

# Force reconstruction
wsd reconstruct-hierarchy --raw-dir data/raw --force

# Parallel processing
wsd reconstruct-hierarchy --raw-dir data/raw --workers 16
```

### `wsd compute-own-text` - Compute own_text field
```bash
# Compute own_text (removes duplicated text from children)
wsd compute-own-text --raw-dir data/raw

# Force recomputation
wsd compute-own-text --raw-dir data/raw --force

# Parallel processing
wsd compute-own-text --raw-dir data/raw --workers 16
```

### `wsd screentag-export` - Export ScreenTag representation
```bash
# Export ScreenTag format
wsd screentag-export --raw-dir data/raw

# Force regeneration
wsd screentag-export --raw-dir data/raw --force

# Parallel processing
wsd screentag-export --raw-dir data/raw --workers 16
```

### `wsd viz-screentag` - Visualize ScreenTag annotations
```bash
# Visualize all ScreenTag files
wsd viz-screentag --raw-dir data/raw --viz-dir data/viz_screentag

# Single file
wsd viz-screentag --image data/raw/example.png \
                  --screentag data/raw/example.screentag.txt \
                  --output viz.jpg

# Customize appearance
wsd viz-screentag --no-labels --fill-opacity 30 --line-width 3
```

### `wsd vlm-label` - VLM-based relabeling
```bash
# Basic VLM relabeling
wsd vlm-label --raw-dir data/raw --crops-dir data/crops

# With specific model
wsd vlm-label --raw-dir data/raw \
              --model Qwen/Qwen3-VL-8B-Instruct \
              --batch-size 64

# Multi-GPU with tensor parallelism
wsd vlm-label --raw-dir data/raw --tp 2 --batch-size 32

# Update elements in-place
wsd vlm-label --raw-dir data/raw --inplace-elements

# With visualization
wsd vlm-label --raw-dir data/raw --viz-dir data/viz_vlm

# Smoke test (limit samples)
wsd vlm-label --raw-dir data/raw --limit 50

# Sharded processing (8 GPUs)
CUDA_VISIBLE_DEVICES=0 wsd vlm-label --raw-dir data/raw \
  --num-shards 8 --shard-index 0 &
CUDA_VISIBLE_DEVICES=1 wsd vlm-label --raw-dir data/raw \
  --num-shards 8 --shard-index 1 &
# ... repeat for shards 2-7
```

### `wsd vlm-score` - VLM-based quality scoring
```bash
# Score annotation quality
wsd vlm-score --viz-dir data/viz --threshold 50

# With specific model
wsd vlm-score --viz-dir data/viz \
              --model Qwen/Qwen3-VL-8B-Instruct \
              --batch-size 256

# Save scores to JSON
wsd vlm-score --viz-dir data/viz \
              --threshold 60 \
              --output filtered.txt \
              --scores-json scores.json

# Multi-GPU with tensor parallelism
wsd vlm-score --viz-dir data/viz --tp 2

# Smoke test
wsd vlm-score --viz-dir data/viz --limit 100

# Sharded processing
CUDA_VISIBLE_DEVICES=0 wsd vlm-score --viz-dir data/viz \
  --num-shards 8 --shard-index 0 \
  --output filtered_shard0.txt &
# ... repeat for shards 1-7
```

</details>

<details>
<summary><b>🧪 Evaluation Commands</b> (click to expand)</summary>

### Basic Evaluation
```bash
# Evaluate YOLO model
wsd-eval \
  --format yolo \
  --image-dir data/yolo/images/val \
  --labels-dir data/yolo/labels/val \
  --classes data/yolo/classes.txt \
  --yolo-model yolov8n.pt \
  --metrics page_iou,label_page_iou,map

# Evaluate VLM (Qwen3-VL)
wsd-eval \
  --format yolo \
  --image-dir data/yolo/images/val \
  --labels-dir data/yolo/labels/val \
  --classes data/yolo/classes.txt \
  --qwen3-vl Qwen/Qwen3-VL-8B-Instruct \
  --metrics page_iou,label_page_iou,map \
  --batch-size 32

# Evaluate InternVL3
wsd-eval \
  --format yolo \
  --image-dir data/yolo/images/val \
  --labels-dir data/yolo/labels/val \
  --classes data/yolo/classes.txt \
  --internvl3 OpenGVLab/InternVL3-2B \
  --metrics page_iou,label_page_iou,map \
  --batch-size 32
```

### Multi-Dataset Evaluation
```bash
# Evaluate on multiple benchmarks
wsd-eval \
  --dataset format=screenspot,root=/path/to/ScreenSpot,split=web,class_schema=screenspot \
  --dataset format=screenspot,root=/path/to/ScreenSpot,split=pc,class_schema=screenspot \
  --dataset format=groundcua,root=/path/to/GroundCUA,max_samples=870,class_schema=groundcua \
  --dataset format=yolo,image_dir=data/yolo/images/test,labels_dir=data/yolo/labels/test,classes=data/yolo/classes.txt,max_samples=1000,class_schema=custom55 \
  --yolo-model yolov8n.pt \
  --metrics page_iou,label_page_iou,map,recall_label,recall_agnostic \
  --batch-size 64
```

### Save Predictions
```bash
# Save predictions for analysis
wsd-eval \
  --format yolo \
  --image-dir data/yolo/images/val \
  --labels-dir data/yolo/labels/val \
  --classes data/yolo/classes.txt \
  --yolo-model yolov8n.pt \
  --metrics page_iou,map \
  --save-preds output/predictions \
  --output output/results.json
```

</details>

## Architecture

```
URLs → Crawl → Raw Screenshots + Annotations
                ↓
         VLM Refinement (optional)
                ↓
         Quality Filtering
                ↓
         Deduplication
                ↓
         YOLO Export → Training Dataset
                ↓
         Model Training
                ↓
         Evaluation Framework → Metrics
```

## Performance

- **Sequential**: ~10-15 seconds per page
- **Parallel (4 workers)**: ~2-4 seconds per page
- **Parallel (8 workers)**: ~1-2 seconds per page

Recommended: 4 workers for balanced speed/memory usage.

## Citation

If you use Webshot or ScreenParse in your research, please cite:

```bibtex
@inproceedings{screenparse2026,
  title={ScreenParse: Large-Scale Dataset for Complete Screen Parsing},
  author={Your Name and Others},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- Built with [Playwright](https://playwright.dev/) for web automation
- VLM refinement powered by [vLLM](https://github.com/vllm-project/vllm)
- Evaluation metrics inspired by [GroundCUA](https://github.com/google-research/google-research/tree/master/groundcua) and [ScreenSpot](https://github.com/google-research-datasets/screen_spot)

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com](mailto:your-email@example.com).
