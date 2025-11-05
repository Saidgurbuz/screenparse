# Webshot Dataset Toolkit - Usage Guide

A Python toolkit for collecting web screenshots with rich annotations and exporting to YOLO format for training object detection models.

## Installation

```bash
# Clone the repository
cd webshot-dataset

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .

# Install Playwright browsers
playwright install chromium

# Optional: Install Tesseract for OCR support
# macOS: brew install tesseract
# Ubuntu: apt-get install tesseract-ocr
```

## Quick Start - Full Pipeline

The easiest way to use the toolkit is with the `pipeline` command that runs everything:

```bash
# Complete pipeline: crawl -> visualize -> export to YOLO
wsd pipeline --urls urls.csv --out data/raw --yolo-dir data/yolo

# With multithreading (4 workers)
wsd pipeline --urls urls.csv --workers 4

# With OCR enabled
wsd pipeline --urls urls.csv --ocr

# Custom train/val/test splits
wsd pipeline --urls urls.csv --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

**Output:**

```
data/
├── raw/              # Raw screenshots and annotations
├── viz/              # Visualizations with bounding boxes
└── yolo/             # YOLO-format dataset
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── data.yaml     # YOLO configuration
    └── classes.txt   # Class names
```

## Step-by-Step Usage

### 1. Prepare URLs

Create `urls.csv` with one URL per line:

```csv
https://example.com
https://www.wikipedia.org
https://github.com
```

### 2. Individual Commands

#### Crawl Websites

```bash
# Basic crawl
wsd crawl --urls urls.csv --out data/raw

# With multithreading (4 parallel workers)
wsd crawl --urls urls.csv --workers 4

# With OCR
wsd crawl --urls urls.csv --out data/raw --ocr

# Headed mode (see browser)
wsd crawl --urls urls.csv --headed

# Maximum parallelism (8 workers)
wsd crawl --urls urls.csv --workers 8
```

**Performance Tips:**

- Use `--workers 4` to crawl 4 pages in parallel
- Recommended: 2-8 workers depending on your CPU
- More workers = faster but higher memory usage
- OCR is per-worker, so `--workers 4 --ocr` runs 4 OCR processes

#### Visualize Annotations

```bash
# Visualize all
wsd viz --out data/raw --viz data/viz

# Single image
wsd viz --image data/raw/example.png \
        --elements data/raw/example.elements.json

# With OCR overlay
wsd viz --ocr-overlay

# Customize appearance
wsd viz --opacity 120 --line-width 3 --no-labels
```

#### Deduplicate Images

```bash
wsd dedupe --image-dir data/raw --csv data/dupes/dupe_groups.csv
```

#### Export to YOLO

```bash
wsd yolo --raw-dir data/raw --yolo-dir data/yolo

# Custom splits
wsd yolo --raw-dir data/raw --yolo-dir data/yolo \
         --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05
```

## YOLO Classes

The toolkit exports **25 visually distinct UI element types**:

| Class | Description |
|-------|-------------|
| button | Buttons, submit inputs |
| input | Text inputs, textareas |
| text | Paragraphs, text blocks |
| image | Images, photos |
| icon | Icons, small graphics |
| link | Hyperlinks |
| heading | H1-H6 headings |
| navigation | Nav bars, breadcrumbs |
| card | Cards, panels, tiles |
| list | Lists (ul, ol) |
| table | Data tables |
| form | Forms |
| video | Video/audio players |
| checkbox | Checkboxes, radios, switches |
| dropdown | Select dropdowns |
| search | Search inputs |
| menu | Dropdown menus |
| footer | Page footers |
| header | Page headers |
| logo | Logos, brand images |
| ad | Advertisements |
| badge | Badges, labels, chips |
| tooltip | Tooltips, popovers |
| modal | Modals, dialogs |
| tab | Tab controls |

## YOLO Format

Each `.txt` file in `labels/` contains one line per annotation:

```
class_id center_x center_y width height
```

All coordinates are normalized to [0, 1].

Example `labels/train/page1.txt`:

```
0 0.5000 0.2500 0.1500 0.0500
2 0.3000 0.4000 0.2000 0.1000
5 0.7500 0.1500 0.1000 0.0300
```

## Training with YOLOv8

```bash
# Install ultralytics
pip install ultralytics

# Train
yolo train data=data/yolo/data.yaml model=yolov8n.pt epochs=100 imgsz=640

# Validate
yolo val model=runs/detect/train/weights/best.pt data=data/yolo/data.yaml

# Predict
yolo predict model=runs/detect/train/weights/best.pt source=test_image.png
```

## Advanced Configuration

### Filtering Parameters

The toolkit automatically filters annotations to remove:

- Duplicate/overlapping boxes (IoU > 0.85)
- Boxes too small (< 8x8 pixels)
- Boxes mostly outside viewport
- Hidden elements (aria-hidden, display:none)
- Parent containers when children have same type

Adjust in code via `filtering.py`:

```python
filter_elements(
    elements,
    iou_threshold=0.85,        # Duplicate threshold
    containment_threshold=0.95, # Parent/child threshold
    min_box_size=8,            # Minimum dimension
    max_box_size=1000          # Maximum dimension
)
```

### Custom Viewport

Modify `src/dataset_tool/config.py`:

```python
@dataclass
class Viewport:
    width: int = 1920           # Change viewport width
    height: int = 1080          # Change viewport height
    device_scale_factor: float = 1.0  # Change DPR
```

### Custom Element Mapping

Edit `src/dataset_tool/yolo_export.py` to modify `YOLO_CLASSES` or `map_to_yolo_class()`.

## Programmatic Usage

```python
from dataset_tool.config import Config
from dataset_tool.crawl import crawl
from dataset_tool.yolo_export import export_yolo_dataset

# Crawl
cfg = Config(out_dir="data/raw", do_ocr=False, headless=True)
crawl("urls.csv", out_dir="data/raw", headless=True)

# Export
export_yolo_dataset(
    raw_dir="data/raw",
    yolo_dir="data/yolo",
    train_ratio=0.7,
    val_ratio=0.2,
    test_ratio=0.1,
)
```

## Annotation Quality

The toolkit includes several quality improvements:

1. **Visibility Filtering**: Removes invisible elements (display:none, opacity:0, etc.)
2. **Deduplication**: Removes overlapping boxes using IoU
3. **Size Filtering**: Removes tiny noise and unreasonably large boxes
4. **Viewport Clipping**: Only includes elements at least 50% visible
5. **Parent/Child Filtering**: Removes redundant parent containers

## Tips & Best Practices

1. **Start Small**: Test with 5-10 URLs first
2. **Review Visualizations**: Check `data/viz/` to verify annotations
3. **Adjust Classes**: Modify `YOLO_CLASSES` for your specific use case
4. **Consistent Viewport**: Use same viewport size for all crawls
5. **OCR**: Only enable if you need text from images (slower)
6. **Deduplication**: Run to identify redundant captures before training

## Troubleshooting

**No annotations exported:**

- Check that elements.json files exist in raw_dir
- Verify filtering isn't too aggressive (check min_box_size)

**Too many overlapping boxes:**

- Decrease `iou_threshold` in filtering.py (e.g., 0.7)
- Increase `containment_threshold` (e.g., 0.98)

**Missing elements:**

- Check visualizations to see what was captured
- Adjust visibility and size thresholds
- Some dynamic content may not load (increase network_idle_wait_ms)

**Memory issues:**

- Process URLs in batches
- Reduce viewport size
- Disable OCR

## Example Workflow

```bash
# 1. Prepare URLs
cat > urls.csv << EOF
https://github.com
https://stackoverflow.com
https://reddit.com
EOF

# 2. Run complete pipeline
wsd pipeline --urls urls.csv --out data/raw --yolo-dir data/yolo

# 3. Review visualizations
open data/viz/*.viz.jpg

# 4. Check dataset
ls -R data/yolo/

# 5. Train YOLO
yolo train data=data/yolo/data.yaml model=yolov8n.pt epochs=50
```

## Command Reference

```bash
# Full pipeline
wsd pipeline [--urls FILE] [--out DIR] [--yolo-dir DIR] [--ocr] [--headed]
             [--train-ratio FLOAT] [--val-ratio FLOAT] [--test-ratio FLOAT]
             [--workers INT]

# Individual commands
wsd crawl [--urls FILE] [--out DIR] [--ocr] [--headed] [--workers INT]
wsd viz [--out DIR] [--viz DIR] [--ocr-overlay] [--opacity N] [--line-width N]
wsd dedupe [--image-dir DIR] [--threshold N] [--csv FILE]
wsd yolo [--raw-dir DIR] [--yolo-dir DIR] [--train-ratio F] [--val-ratio F]
```

## Performance Optimization

### Multithreading

The toolkit supports parallel crawling for faster data collection:

```bash
# Sequential (default)
wsd crawl --urls urls.csv

# 4 parallel workers (recommended)
wsd crawl --urls urls.csv --workers 4

# 8 workers for large datasets
wsd crawl --urls urls.csv --workers 8
```

**Guidelines:**

- **2-4 workers**: Good for most machines
- **4-8 workers**: High-end machines with 16GB+ RAM
- **8+ workers**: Server environments only

**Memory usage**: ~500MB per worker (more with OCR)

### Pipeline Optimization

```bash
# Fast pipeline with parallelism
wsd pipeline --urls urls.csv --workers 6 --skip-dedupe

# Balanced (recommended)
wsd pipeline --urls urls.csv --workers 4
```
