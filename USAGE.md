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

#### Reconstruct Element Hierarchy

For older datasets that don't have parent-child hierarchy information, you can reconstruct it from bounding box geometry:

```bash
# Reconstruct hierarchy for all elements.json files
wsd reconstruct-hierarchy --raw-dir data/raw

# With custom containment threshold (default: 0.95)
wsd reconstruct-hierarchy --raw-dir data/raw --min-containment 0.90

# Force reconstruction even if hierarchy exists
wsd reconstruct-hierarchy --raw-dir data/raw --force

# Disable semantic hints from VLM labels
wsd reconstruct-hierarchy --raw-dir data/raw --no-semantic-hints

# Parallel processing
wsd reconstruct-hierarchy --raw-dir data/raw --workers 8
```

This adds `parent_index`, `children_indices`, `_dom_index`, `_parent_dom_index`, `_children_dom_indices`, and `_depth` fields to each element in the elements.json files.

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

### Quick Start

```bash
# Install ultralytics
pip install ultralytics

# Train with default settings
python scripts/train.py

# Train with custom parameters
python scripts/train.py --epochs 200 --batch 32 --imgsz 1280
```

### Training Configuration

Edit `configs/train_config.yaml` to customize all training parameters:

```yaml
# Model selection
model: yolov8n.pt  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

# Training parameters
epochs: 100
batch: 16
imgsz: 640
device: 0  # GPU device or 'cpu'

# Optimization
optimizer: auto
lr0: 0.01
momentum: 0.937
weight_decay: 0.0005

# Augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
```

### Training Examples

```bash
# Basic training (100 epochs, batch 16, 640px)
python scripts/train.py

# Larger model, more epochs
python scripts/train.py --model yolov8m.pt --epochs 200

# Higher resolution
python scripts/train.py --imgsz 1280 --batch 8

# Train on CPU
python scripts/train.py --device cpu

# Multi-GPU training
python scripts/train.py --device 0,1,2,3

# Resume interrupted training
python scripts/train.py --resume

# Custom configuration file
python scripts/train.py --config configs/my_config.yaml

# Check configuration without training
python scripts/train.py --dry-run
```

### Model Selection

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| yolov8n | 3MB | Fastest | Lowest | Real-time, edge devices |
| yolov8s | 11MB | Very fast | Low | Mobile, embedded |
| yolov8m | 26MB | Fast | Medium | General purpose |
| yolov8l | 44MB | Moderate | High | High accuracy |
| yolov8x | 68MB | Slow | Highest | Maximum accuracy |

### Training Tips

1. **Start with yolov8n**: Fast iterations for debugging
2. **Increase batch size**: Use largest batch that fits in GPU memory
3. **Higher resolution**: Use 1280 for better small object detection
4. **Monitor training**: Check `runs/detect/webshot_ui/` for metrics
5. **Early stopping**: Training stops if no improvement for 50 epochs
6. **Resume training**: Use `--resume` if interrupted

### Output Structure

After training, check `runs/detect/webshot_ui/`:

```
runs/detect/webshot_ui/
├── weights/
│   ├── best.pt      # Best checkpoint
│   └── last.pt      # Last checkpoint
├── results.csv      # Training metrics
├── results.png      # Training curves
├── confusion_matrix.png
├── val_batch0_pred.jpg  # Validation predictions
└── ...
```

### Validation

```bash
# Validate best model
yolo val model=runs/detect/webshot_ui/weights/best.pt data=data/yolo/data.yaml

# Validate on specific split
yolo val model=runs/detect/webshot_ui/weights/best.pt data=data/yolo/data.yaml split=test
```

### Prediction

```bash
# Single image
yolo predict model=runs/detect/webshot_ui/weights/best.pt source=test.png

# Directory of images
yolo predict model=runs/detect/webshot_ui/weights/best.pt source=test_images/

# With confidence threshold
yolo predict model=runs/detect/webshot_ui/weights/best.pt source=test.png conf=0.5

# Save results
yolo predict model=runs/detect/webshot_ui/weights/best.pt source=test.png save=true
```

### Advanced Training

#### Transfer Learning

```bash
# Fine-tune pre-trained model
python scripts/train.py --model yolov8n.pt --epochs 50 --freeze 10
```

#### Custom Augmentation

Edit `configs/train_config.yaml`:

```yaml
# Aggressive augmentation
degrees: 5.0      # rotation
translate: 0.2    # translation
scale: 0.9        # scaling
shear: 2.0        # shearing
mosaic: 1.0       # mosaic
mixup: 0.1        # mixup
```

#### Learning Rate Scheduling

```yaml
lr0: 0.01         # initial LR
lrf: 0.01         # final LR
cos_lr: true      # cosine LR scheduler
```

### Troubleshooting Training

**Out of memory:**
```bash
# Reduce batch size
python scripts/train.py --batch 8

# Reduce image size
python scripts/train.py --imgsz 416
```

**Training too slow:**
```bash
# More dataloader workers
python scripts/train.py --workers 16

# Cache dataset to RAM
python scripts/train.py --cache ram
```

**Poor accuracy:**
- Train longer (200-300 epochs)
- Use larger model (yolov8m or yolov8l)
- Increase image size (1280)
- Check class balance with `wsd validate`
- Review visualizations for annotation quality

**Model not converging:**
- Lower learning rate: `--lr0 0.001`
- Check dataset quality
- Verify data.yaml paths are correct
- Ensure sufficient training data (100+ images)

### CLI Training Command

You can also use the built-in CLI:

```bash
# Basic training through CLI
wsd train --data data/yolo/data.yaml --epochs 100

# With all options
wsd train --data data/yolo/data.yaml \
          --model yolov8m.pt \
          --epochs 200 \
          --batch 32 \
          --imgsz 1280 \
          --device 0
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
wsd reconstruct-hierarchy [--raw-dir DIR] [--min-containment F] [--force] [--workers N]
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
