# Webshot: Scalable Dataset Generation for Complete Screen Parsing

This directory contains **Webshot**, the automated dataset generation pipeline for [ScreenParse](https://arxiv.org/abs/2602.14276), introduced in:

> **ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision**

## Overview

Modern computer-use agents must perceive screens as structured states, identifying what elements are visible, where they are located, and what text they contain, before reliably grounding instructions. Existing grounding datasets provide only sparse supervision with limited label diversity, annotating small subsets of elements per screen.

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
git clone <screenparse-repo-url>
cd screenparse/webshot
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
wsd vlm-score --viz-dir data/viz --threshold 70
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
URLs -> Web Crawling -> Raw Annotations -> VLM Refinement -> Quality Filtering -> YOLO Export
         (Playwright)    (DOM extraction)   (Qwen3-VL)      (VLM scoring)     (train/val/test)
```

## Element Taxonomy

Webshot annotates 55 UI element classes organized into the following categories:

| Category | Classes |
|----------|---------|
| Global Interface Elements | Status Bar, Navigation Bar, Tab Bar, Toolbar, Side Bar, Bottom navigation, DockMenu, EditMenu, ContextMenu |
| Navigation | Link, Breadcrumb, Pagination, Tab, Page control, Menu, PopUp Menu, Search Bar, Search Field |
| Inputs & Controls | Button, Utility Button, Text Input, Select, Checkbox, Radiobox, Switch, Slider, Steppers, Toggles, Picker, Date-Time picker, Calendar, Rating Indicator |
| Content & Media | Text, Heading, Image, Video, Carousel, Code snippet, Chart, Table, List, List Item, Column/Browser, File Icon, App Icon, Logo, Avatar |
| Feedback & Status | Tooltip, Alert, Notification, Badge, Progress bar |
| Layout & Viewport | Window, Screen, Scroll |

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
- **Detection models**: YOLOv11, RT-DETRv2, OmniParser
- **Vision-language models**: Qwen3-VL, InternVL3, Gemini, ScreenVLM

## Documentation

- [USAGE.md](USAGE.md) - Detailed usage guide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [examples/](examples/) - Example configurations

## Citation

```bibtex
@misc{gurbuz2026movingsparsegroundingcomplete,
      title={ScreenParse: Moving Beyond Sparse Grounding with Complete Screen Parsing Supervision},
      author={A. Said Gurbuz and Sunghwan Hong and Ahmed Nassar and Marc Pollefeys and Peter Staar},
      year={2026},
      eprint={2602.14276},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.14276},
      note={Accepted to ICML 2026},
}
```

## License

MIT License. See [../LICENSE](../LICENSE) for details.
