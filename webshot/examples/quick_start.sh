#!/bin/bash
# Quick Start Script for Webshot Dataset Generation
# This script demonstrates the simplest way to generate a dataset

set -e  # Exit on error

echo "==================================="
echo "Webshot Quick Start"
echo "==================================="

# Configuration
URLS_FILE="examples/urls_sample.csv"
OUTPUT_DIR="data"
WORKERS=4

# Check if URLs file exists
if [ ! -f "$URLS_FILE" ]; then
    echo "Error: URLs file not found: $URLS_FILE"
    echo "Creating sample URLs file..."
    cat > "$URLS_FILE" << EOF
https://github.com
https://stackoverflow.com
https://reddit.com
https://python.org
https://arxiv.org
EOF
fi

echo ""
echo "Step 1: Running complete pipeline..."
echo "  - Crawling websites"
echo "  - Generating visualizations"
echo "  - Exporting to YOLO format"
echo ""

wsd pipeline \
    --urls "$URLS_FILE" \
    --out "$OUTPUT_DIR/raw" \
    --viz "$OUTPUT_DIR/viz" \
    --yolo-dir "$OUTPUT_DIR/yolo" \
    --workers $WORKERS \
    --train-ratio 0.7 \
    --val-ratio 0.2 \
    --test-ratio 0.1

echo ""
echo "==================================="
echo "Pipeline Complete!"
echo "==================================="
echo ""
echo "Output directories:"
echo "  - Raw data:        $OUTPUT_DIR/raw/"
echo "  - Visualizations:  $OUTPUT_DIR/viz/"
echo "  - YOLO dataset:    $OUTPUT_DIR/yolo/"
echo ""
echo "Next steps:"
echo "  1. Review visualizations: ls $OUTPUT_DIR/viz/"
echo "  2. Validate dataset: wsd validate --yolo-dir $OUTPUT_DIR/yolo"
echo "  3. Train model: wsd train --data $OUTPUT_DIR/yolo/data.yaml --epochs 100"
echo ""