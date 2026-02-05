# Webshot: Scalable Dataset Generation for Complete Screen Parsing

This repository contains **Webshot**, the automated dataset generation pipeline for [ScreenParse](https://arxiv.org/abs/XXXX.XXXXX), introduced in:

> **Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision**
> *International Conference on Machine Learning (ICML) 2026*

## Overview

Modern computer-use agents must perceive screens as structured states—identifying what elements are visible, where they are located, and what text they contain—before reliably grounding instructions. Existing grounding datasets provide only sparse supervision with limited label diversity, annotating small subsets of elements per screen.

**ScreenParse** addresses this limitation with dense annotations of all visible UI elements across 771K web screenshots (21M elements total), featuring:
- Bounding boxes for precise localization
- 55-class semantic type labels
- Extracted text content
- Hierarchical parent-child relationships

**Webshot** is the scalable pipeline that generates ScreenParse through:
1. Automated web rendering of diverse URLs
2. DOM-based annotation extraction
3. VLM-based label refinement (Qwen3-VL)
4. Quality filtering and deduplication

## Installation

```bash
git clone https://github.com/your-org/webshot-dataset.git
cd webshot-dataset
pip install -e .
playwright install chromium
```

Optional OCR support:
```bash
# Ubuntu
apt-get install tesseract-ocr
# macOS
brew install tesseract
```

## Quick Start

### Dataset Generation

```bash
# Prepare URL list
echo -e "https://github.com\nhttps://stackoverflow.com" > urls.csv

# Run full pipeline
wsd pipeline --urls urls.csv --workers 4

# Output: data/raw/ (screenshots), data/viz/ (visualizations), data/yolo/ (training data)
```

### Individual Pipeline Steps

```bash
# 1. Crawl websites
wsd crawl --urls urls.csv --workers 4 --out data/raw

# 2. Visualize annotations
wsd viz --out data/raw --viz data/viz

# 3. Export to YOLO format
wsd yolo --raw-dir data/raw --yolo-dir data/yolo
```

### VLM-based Refinement

```bash
# Relabel elements using Qwen3-VL
wsd vlm-label --raw-dir data/raw --model Qwen/Qwen3-VL-8B-Instruct --batch-size 64

# Quality filtering
wsd vlm-score --viz-dir data/viz --threshold 50
```

### Model Training

```bash
wsd train --data data/yolo/data.yaml --epochs 100 --batch 16
```

### Evaluation

```bash
# Evaluate detection model
wsd-eval \
  --format yolo \
  --image-dir data/yolo/images/val \
  --labels-dir data/yolo/labels/val \
  --classes data/yolo/classes.txt \
  --yolo-model weights.pt \
  --metrics page_iou,label_page_iou,map

# Evaluate VLM
wsd-eval \
  --qwen3-vl Qwen/Qwen3-VL-8B-Instruct \
  --format yolo \
  --image-dir data/yolo/images/val \
  --labels-dir data/yolo/labels/val \
  --classes data/yolo/classes.txt \
  --metrics page_iou,label_page_iou,map \
  --batch-size 32
```

## Pipeline Architecture

```
URLs → Web Crawling → Raw Annotations → VLM Refinement → Quality Filtering → YOLO Export
         (Playwright)    (DOM extraction)   (Qwen3-VL)      (VLM scoring)     (train/val/test)
```

## Element Taxonomy

Webshot annotates 55 UI element classes organized into categories:

| Category | Classes |
|----------|---------|
| Interactive | button, input, link, checkbox, radio, select, slider, switch, textarea |
| Navigation | navigation bar, tab bar, tab, breadcrumb, pagination, menu, toolbar |
| Content | text, heading, image, icon, video, table, list, card, code snippet |
| Feedback | tooltip, alert, notification, badge, progress bar |
| Layout | form, modal, sidebar, header, footer, scroll, window |

See [USAGE.md](USAGE.md) for the complete taxonomy.

## Supported Evaluation Metrics

| Metric | Description |
|--------|-------------|
| PageIoU | Pixel-level intersection-over-union of rendered boxes |
| Label PageIoU | Class-aware PageIoU |
| mAP | Mean average precision at IoU threshold |
| Recall | Detection recall (label-aware and agnostic variants) |
| NED | Normalized edit distance for text extraction |

## Supported Models

The evaluation framework supports:
- **Detection models**: YOLOv8, RT-DETR, OmniParser
- **Vision-language models**: Qwen3-VL, InternVL3, Gemini, ScreenVLM

## Documentation

- [USAGE.md](USAGE.md) — Detailed usage guide
- [CONTRIBUTING.md](CONTRIBUTING.md) — Contribution guidelines
- [examples/](examples/) — Example configurations

## Citation

```bibtex
@inproceedings{screenparse2026,
  title={Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision},
  author={...},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
