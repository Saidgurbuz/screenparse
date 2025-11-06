# Webshot Dataset Toolkit

Automated toolkit for creating YOLO object detection datasets from web screenshots.

## Features

- 🌐 Automated web crawling with Playwright
- 🎯 Smart element detection and classification
- 🧹 Intelligent deduplication and filtering
- 📦 YOLO-format export with train/val/test splits
- 🎨 Visualization for quality checking
- 🔧 Configurable and extensible

## Quick Start

```bash
# Install
pip install -e .
playwright install chromium

# Create dataset (with 4 parallel workers)
echo "https://github.com" > urls.csv
wsd pipeline --urls urls.csv --workers 4

# Train YOLO
yolo train data=data/yolo/data.yaml model=yolov8n.pt epochs=100

# Or use the training script
python scripts/train.py --epochs 100 --batch 16
```

## Installation

```bash
git clone <your-repo>
cd webshot-dataset
pip install -e .
playwright install chromium
```

## Usage

See [USAGE.md](USAGE.md) for detailed documentation.

### Single Command Pipeline

```bash
# Sequential
wsd pipeline --urls urls.csv --out data/raw --yolo-dir data/yolo

# Parallel (4 workers - recommended)
wsd pipeline --urls urls.csv --workers 4
```

### Individual Steps

```bash
# 1. Crawl websites (with parallelism)
wsd crawl --urls urls.csv --out data/raw --workers 4

# 2. Visualize annotations
wsd viz --out data/raw --viz data/viz

# 3. Deduplicate
wsd dedupe --image-dir data/raw

# 4. Export to YOLO
wsd yolo --raw-dir data/raw --yolo-dir data/yolo

# 5. Validate dataset
wsd validate --yolo-dir data/yolo

# 6. Train model
python scripts/train.py
# or
wsd train --data data/yolo/data.yaml --epochs 100
```

## Performance

- **Sequential**: ~10-15 seconds per page
- **Parallel (4 workers)**: ~2-4 seconds per page
- **Parallel (8 workers)**: ~1-2 seconds per page

Recommended: 4 workers for balanced speed/memory usage.

## Output Structure

```
data/
├── raw/              # Raw screenshots + annotations
├── viz/              # Annotated visualizations
└── yolo/             # YOLO training dataset
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
```

## Element Classes

25 UI element types optimized for object detection:

`button`, `input`, `text`, `image`, `icon`, `link`, `heading`, `navigation`, `card`, `list`, `table`, `form`, `video`, `checkbox`, `dropdown`, `search`, `menu`, `footer`, `header`, `logo`, `ad`, `badge`, `tooltip`, `modal`, `tab`

## Configuration

Edit `src/dataset_tool/config.py` to customize:

- Viewport size and DPR
- Filtering thresholds (IoU, size limits)
- Network idle wait time
- OCR settings

## Project Structure

```
webshot-dataset/
├── src/
│   └── dataset_tool/
│       ├── cli.py              # Command-line interface
│       ├── collector.py        # Web scraping logic
│       ├── filtering.py        # Element filtering/dedup
│       ├── yolo_export.py      # YOLO format export
│       ├── visualize.py        # Annotation visualization
│       ├── labels.py           # Element classification
│       ├── validator.py        # Dataset validation
│       ├── config.py           # Configuration
│       └── utils.py            # Utilities
├── USAGE.md                    # Detailed documentation
├── README.md                   # This file
└── pyproject.toml             # Dependencies
```

## Requirements

- Python 3.8+
- Playwright
- Pillow
- NumPy
- (Optional) Tesseract for OCR

## Tips

1. **Start small**: Test with 5-10 URLs first
2. **Check visualizations**: Review `data/viz/` before training
3. **Validate**: Run `wsd validate` to check dataset quality
4. **Adjust filters**: Modify thresholds in `config.py` if needed
5. **Use consistent viewport**: Same size for all crawls

## Troubleshooting

**No annotations:**

- Check that URLs loaded successfully
- Verify filtering isn't too strict
- Look at visualizations

**Too many overlapping boxes:**

- Decrease `iou_threshold` in `config.py`

**Training fails:**

- Run `wsd validate` to check dataset
- Ensure class distribution is balanced

## License

See LICENSE file.

## Contributing

Contributions welcome! Please open an issue first to discuss changes.
