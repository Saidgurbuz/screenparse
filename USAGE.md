# Webshot Dataset Toolkit - Usage Guide

A Python toolkit for collecting web screenshots with rich annotations including DOM elements, text spans, accessibility trees, and optional OCR.

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

## Quick Start

### 1. Prepare URLs

Create a CSV file with URLs to capture (one per line):

```csv
https://example.com
https://www.wikipedia.org
https://news.ycombinator.com/
```

### 2. Crawl Websites

Collect screenshots and annotations:

```bash
# Basic crawl
wsd crawl --urls urls.csv --out data/raw

# With OCR enabled
wsd crawl --urls urls.csv --out data/raw --ocr

# Run browser in headed mode (visible window)
wsd crawl --urls urls.csv --headed
```

**Output Structure:**

```
data/raw/
├── example-com-a1b2c3d4.png              # Screenshot
├── example-com-a1b2c3d4.meta.json        # Metadata
├── example-com-a1b2c3d4.elements.json    # DOM elements
├── example-com-a1b2c3d4.texts.json       # Text spans
├── example-com-a1b2c3d4.ax.json          # Accessibility tree
├── example-com-a1b2c3d4.ocr.json         # OCR results (if --ocr)
└── example-com-a1b2c3d4.triplets.jsonl   # Training-ready triplets
```

### 3. Visualize Annotations

Generate annotated images with bounding boxes:

```bash
# Visualize all collected data
wsd viz --out data/raw --viz data/viz

# Visualize single image
wsd viz --image data/raw/example.png \
        --elements data/raw/example.elements.json \
        --texts data/raw/example.texts.json

# With OCR overlay
wsd viz --ocr-overlay --out data/raw --viz data/viz

# Customize appearance
wsd viz --opacity 120 --line-width 3 --no-labels
```

**Visualization Options:**

- `--no-elements` - Hide element boxes
- `--no-texts` - Hide text span boxes
- `--no-labels` - Hide element type labels
- `--label-text-spans` - Show text content labels
- `--opacity <0-255>` - Box fill transparency
- `--line-width <pixels>` - Border thickness

### 4. Deduplicate Images

Find and group visually similar screenshots:

```bash
# Find duplicates
wsd dedupe --image-dir data/raw --csv data/dupes/dupe_groups.csv

# Adjust sensitivity (lower = stricter)
wsd dedupe --image-dir data/raw --threshold 5
```

## Data Formats

### Elements JSON

Each element contains:

```json
{
  "tag": "button",
  "type": "button",
  "role": "button",
  "rect": {"x": 100, "y": 200, "w": 80, "h": 40},
  "inner_text": "Click me",
  "attrs": {"class": "btn-primary", "id": "submit"},
  "frame_index": 0,
  "z": 100
}
```

**Supported Types:** header, footer, nav, button, input, link, image, table, list, card, modal, icon, and 30+ more.

### Text Spans JSON

Fine-grained text layout:

```json
{
  "text": "Hello World",
  "rect": {"x": 50, "y": 100, "w": 120, "h": 18},
  "font_family": "Arial, sans-serif",
  "font_size": "16px",
  "font_weight": "400"
}
```

### Triplets JSONL

Training-ready format (one JSON object per line):

```json
{"element_index": 0, "bbox_ltrb": [100, 200, 180, 240], "type": "button", "text": "Submit", "tag": "button", "role": null}
{"element_index": 1, "bbox_ltrb": [50, 300, 250, 380], "type": "image", "text": "", "tag": "img", "role": "img"}
```

## Advanced Usage

### Custom Configuration

Modify viewport, locale, or other settings:

```python
from dataset_tool.config import Config, Viewport
from dataset_tool.crawl import crawl

cfg = Config(
    viewport=Viewport(width=1920, height=1080, device_scale_factor=1.0),
    locale="de-DE",
    color_scheme="dark",
    headless=True,
    do_ocr=True
)

results = crawl("urls.csv", out_dir="data/custom", 
                do_ocr=True, headless=True)
```

### Programmatic Collection

```python
from dataset_tool.collector import collect_one
from dataset_tool.config import Config

cfg = Config(out_dir="data/raw", do_ocr=False)
record = collect_one("https://example.com", cfg)

print(f"Saved to: {record['image_path']}")
print(f"Found {len(record)} annotation files")
```

### Batch Visualization

```python
from dataset_tool.visualize import visualize_record, VizOptions
from dataset_tool.config import Config

cfg = Config(viz_dir="data/viz")
opts = VizOptions(
    draw_elements=True,
    draw_text_spans=False,
    opacity=100,
    line_width=2
)

# Assuming 'record' from collect_one()
viz_path = visualize_record(record, cfg, opts)
print(f"Visualization saved to: {viz_path}")
```

## Tips & Best Practices

1. **Performance**: Crawling is slow (~5-10s per page). Run in parallel if needed.
2. **OCR**: Only enable if you need text from images/canvas. Adds significant time.
3. **Headless**: Always use headless mode in production (`--headed` is for debugging).
4. **Deduplication**: Run after crawling to identify redundant captures.
5. **Storage**: Each page generates ~500KB-5MB of data (varies by page complexity).

## Troubleshooting

**Playwright not found:**

```bash
playwright install chromium
```

**Tesseract errors (OCR):**

```bash
# Ensure it's installed and in PATH
which tesseract
brew install tesseract  # macOS
```

**Memory issues on large batches:**

- Process URLs in smaller chunks
- Close browser between batches
- Increase system swap space

## Example Workflow

```bash
# 1. Prepare URLs
echo "https://github.com" > urls.csv
echo "https://stackoverflow.com" >> urls.csv

# 2. Collect data
wsd crawl --urls urls.csv --out data/raw --ocr

# 3. Visualize
wsd viz --out data/raw --viz data/viz --ocr-overlay

# 4. Check for duplicates
wsd dedupe --image-dir data/raw --csv data/dupes/groups.csv

# 5. Review
open data/viz/*.viz.jpg
```

## Command Reference

```bash
# Crawl
wsd crawl [--urls FILE] [--out DIR] [--ocr] [--headed]

# Visualize
wsd viz [--image FILE] [--out DIR] [--viz DIR] [--ocr-overlay]
        [--no-elements] [--no-texts] [--no-labels]
        [--opacity N] [--line-width N]

# Deduplicate
wsd dedupe [--image-dir DIR] [--threshold N] [--csv FILE]
```

## License & Citation

See LICENSE file for terms. If using this toolkit in research, please cite appropriately.
