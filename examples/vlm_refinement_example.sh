#!/bin/bash
# VLM-Based Refinement Example
# This script demonstrates VLM-based relabeling and quality scoring

set -e

echo "==================================="
echo "VLM Refinement Pipeline"
echo "==================================="

# ============================================
# CONFIGURATION
# ============================================

RAW_DIR="data/raw"
CROPS_DIR="data/crops"
VIZ_DIR="data/viz_screentag"
VLM_MODEL="Qwen/Qwen3-VL-8B-Instruct"
BATCH_SIZE=64
TENSOR_PARALLEL=1  # Set to 2 or 4 for multi-GPU

# Quality scoring
QUALITY_THRESHOLD=50  # Filter out samples with score < 50
FILTERED_OUTPUT="data/filtered_low_quality.txt"
SCORES_JSON="data/quality_scores.json"

# ============================================
# STEP 1: Reconstruct Hierarchy
# ============================================

echo ""
echo "Step 1: Reconstructing element hierarchy..."
echo ""

wsd reconstruct-hierarchy \
    --raw-dir $RAW_DIR \
    --min-containment 0.95 \
    --workers 8

# ============================================
# STEP 2: Compute Own Text
# ============================================

echo ""
echo "Step 2: Computing own_text fields..."
echo ""

wsd compute-own-text \
    --raw-dir $RAW_DIR \
    --workers 8

# ============================================
# STEP 3: Export ScreenTag Format
# ============================================

echo ""
echo "Step 3: Exporting ScreenTag representation..."
echo ""

wsd screentag-export \
    --raw-dir $RAW_DIR \
    --workers 8

# ============================================
# STEP 4: Visualize ScreenTag
# ============================================

echo ""
echo "Step 4: Creating ScreenTag visualizations..."
echo ""

wsd viz-screentag \
    --raw-dir $RAW_DIR \
    --viz-dir $VIZ_DIR \
    --workers 4

# ============================================
# STEP 5: VLM-Based Relabeling
# ============================================

echo ""
echo "Step 5: VLM-based element relabeling..."
echo "  Model: $VLM_MODEL"
echo "  Batch size: $BATCH_SIZE"
echo "  Tensor parallel: $TENSOR_PARALLEL"
echo ""

wsd vlm-label \
    --raw-dir $RAW_DIR \
    --crops-dir $CROPS_DIR \
    --model $VLM_MODEL \
    --batch-size $BATCH_SIZE \
    --tp $TENSOR_PARALLEL \
    --inplace-elements \
    --viz-dir data/viz_vlm

# ============================================
# STEP 6: Quality Scoring
# ============================================

echo ""
echo "Step 6: VLM-based quality scoring..."
echo "  Threshold: $QUALITY_THRESHOLD"
echo ""

wsd vlm-score \
    --viz-dir $VIZ_DIR \
    --model $VLM_MODEL \
    --batch-size 256 \
    --tp $TENSOR_PARALLEL \
    --threshold $QUALITY_THRESHOLD \
    --output $FILTERED_OUTPUT \
    --scores-json $SCORES_JSON

echo ""
echo "==================================="
echo "VLM Refinement Complete!"
echo "==================================="
echo ""
echo "Outputs:"
echo "  - Element crops:      $CROPS_DIR/"
echo "  - ScreenTag viz:      $VIZ_DIR/"
echo "  - VLM viz:            data/viz_vlm/"
echo "  - Filtered samples:   $FILTERED_OUTPUT"
echo "  - Quality scores:     $SCORES_JSON"
echo ""
echo "Next steps:"
echo "  1. Review filtered samples: cat $FILTERED_OUTPUT"
echo "  2. Remove low-quality samples from dataset"
echo "  3. Re-export to YOLO format"
echo ""