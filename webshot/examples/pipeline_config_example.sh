#!/bin/bash
# Complete Pipeline Example with Custom Configuration
# This script shows all available options for the dataset generation pipeline

set -e

echo "==================================="
echo "Webshot Pipeline - Full Configuration"
echo "==================================="

# ============================================
# CONFIGURATION
# ============================================

# Input/Output
URLS_FILE="examples/urls_sample.csv"
RAW_DIR="data/raw"
VIZ_DIR="data/viz"
YOLO_DIR="data/yolo"

# Parallelization
CRAWL_WORKERS=4          # Parallel browser instances
VIZ_WORKERS=4            # Parallel visualization threads
EXPORT_WORKERS=8         # Parallel YOLO export processes

# Dataset Splits
TRAIN_RATIO=0.7
VAL_RATIO=0.2
TEST_RATIO=0.1
RANDOM_SEED=42

# Optional Features
ENABLE_OCR=false         # Set to true to enable OCR
FULL_PAGE=false          # Set to true for full-page capture with scrolling
SKIP_DEDUPE=false        # Set to true to skip deduplication
SAVE_UNFILTERED=false    # Set to true to save unfiltered elements for debugging

# ============================================
# BUILD COMMAND
# ============================================

CMD="wsd pipeline"
CMD="$CMD --urls $URLS_FILE"
CMD="$CMD --out $RAW_DIR"
CMD="$CMD --viz $VIZ_DIR"
CMD="$CMD --yolo-dir $YOLO_DIR"
CMD="$CMD --workers $CRAWL_WORKERS"
CMD="$CMD --viz-workers $VIZ_WORKERS"
CMD="$CMD --export-workers $EXPORT_WORKERS"
CMD="$CMD --train-ratio $TRAIN_RATIO"
CMD="$CMD --val-ratio $VAL_RATIO"
CMD="$CMD --test-ratio $TEST_RATIO"
CMD="$CMD --seed $RANDOM_SEED"

# Add optional flags
if [ "$ENABLE_OCR" = true ]; then
    CMD="$CMD --ocr"
fi

if [ "$FULL_PAGE" = true ]; then
    CMD="$CMD --full-page"
fi

if [ "$SKIP_DEDUPE" = true ]; then
    CMD="$CMD --skip-dedupe"
fi

if [ "$SAVE_UNFILTERED" = true ]; then
    CMD="$CMD --save-unfiltered"
fi

# ============================================
# EXECUTE
# ============================================

echo ""
echo "Configuration:"
echo "  URLs:           $URLS_FILE"
echo "  Raw output:     $RAW_DIR"
echo "  Visualizations: $VIZ_DIR"
echo "  YOLO dataset:   $YOLO_DIR"
echo "  Workers:        $CRAWL_WORKERS crawl, $VIZ_WORKERS viz, $EXPORT_WORKERS export"
echo "  Splits:         ${TRAIN_RATIO}/${VAL_RATIO}/${TEST_RATIO}"
echo "  OCR:            $ENABLE_OCR"
echo "  Full page:      $FULL_PAGE"
echo ""
echo "Running command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "==================================="
echo "Pipeline Complete!"
echo "==================================="
echo ""
echo "Dataset statistics:"
wsd validate --yolo-dir $YOLO_DIR
echo ""