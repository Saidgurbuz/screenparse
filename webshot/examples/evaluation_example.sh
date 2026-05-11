#!/bin/bash
# Evaluation Examples
# This script demonstrates various evaluation scenarios

set -e

echo "==================================="
echo "Webshot Evaluation Examples"
echo "==================================="

# ============================================
# CONFIGURATION
# ============================================

# Dataset paths (adjust to your setup)
YOLO_IMAGES="data/yolo/images/val"
YOLO_LABELS="data/yolo/labels/val"
YOLO_CLASSES="data/yolo/classes.txt"

# Model paths (examples - adjust to your models)
YOLO_MODEL="yolov8n.pt"
QWEN_MODEL="Qwen/Qwen3-VL-8B-Instruct"
INTERNVL_MODEL="OpenGVLab/InternVL3-2B"

# Evaluation settings
BATCH_SIZE=32
MAX_SAMPLES=100  # Set to null for full evaluation

# ============================================
# EXAMPLE 1: Basic YOLO Evaluation
# ============================================

echo ""
echo "Example 1: Evaluating YOLO model..."
echo ""

wsd-eval \
    --format yolo \
    --image-dir $YOLO_IMAGES \
    --labels-dir $YOLO_LABELS \
    --classes $YOLO_CLASSES \
    --yolo-model $YOLO_MODEL \
    --metrics page_iou,label_page_iou,map \
    --max-samples $MAX_SAMPLES \
    --output results/yolo_eval.json

# ============================================
# EXAMPLE 2: VLM Evaluation (Qwen3-VL)
# ============================================

echo ""
echo "Example 2: Evaluating Qwen3-VL..."
echo "  (Requires vLLM and model weights)"
echo ""

# Uncomment to run:
# wsd-eval \
#     --format yolo \
#     --image-dir $YOLO_IMAGES \
#     --labels-dir $YOLO_LABELS \
#     --classes $YOLO_CLASSES \
#     --qwen3-vl $QWEN_MODEL \
#     --metrics page_iou,label_page_iou,map \
#     --batch-size $BATCH_SIZE \
#     --max-samples $MAX_SAMPLES \
#     --output results/qwen_eval.json

# ============================================
# EXAMPLE 3: Multi-Dataset Evaluation
# ============================================

echo ""
echo "Example 3: Multi-dataset evaluation..."
echo "  (Requires ScreenSpot and GroundCUA datasets)"
echo ""

# Uncomment and adjust paths to run:
# wsd-eval \
#     --dataset format=yolo,image_dir=$YOLO_IMAGES,labels_dir=$YOLO_LABELS,classes=$YOLO_CLASSES,max_samples=100,tag=yolo100 \
#     --dataset format=screenspot,root=/path/to/ScreenSpot,split=web,class_schema=screenspot \
#     --dataset format=groundcua,root=/path/to/GroundCUA,max_samples=870,class_schema=groundcua \
#     --yolo-model $YOLO_MODEL \
#     --metrics page_iou,label_page_iou,map,recall_label,recall_agnostic \
#     --batch-size $BATCH_SIZE \
#     --output results/multi_dataset_eval.json

# ============================================
# EXAMPLE 4: Save Predictions for Analysis
# ============================================

echo ""
echo "Example 4: Saving predictions for analysis..."
echo ""

wsd-eval \
    --format yolo \
    --image-dir $YOLO_IMAGES \
    --labels-dir $YOLO_LABELS \
    --classes $YOLO_CLASSES \
    --yolo-model $YOLO_MODEL \
    --metrics page_iou,map \
    --max-samples $MAX_SAMPLES \
    --save-preds results/predictions \
    --output results/eval_with_preds.json

# ============================================
# EXAMPLE 5: Compare Multiple Models
# ============================================

echo ""
echo "Example 5: Comparing multiple models..."
echo ""

# Evaluate model 1
wsd-eval \
    --format yolo \
    --image-dir $YOLO_IMAGES \
    --labels-dir $YOLO_LABELS \
    --classes $YOLO_CLASSES \
    --yolo-model yolov8n.pt \
    --metrics page_iou,label_page_iou,map \
    --max-samples $MAX_SAMPLES \
    --output results/yolov8n_eval.json

# Evaluate model 2
wsd-eval \
    --format yolo \
    --image-dir $YOLO_IMAGES \
    --labels-dir $YOLO_LABELS \
    --classes $YOLO_CLASSES \
    --yolo-model yolov8m.pt \
    --metrics page_iou,label_page_iou,map \
    --max-samples $MAX_SAMPLES \
    --output results/yolov8m_eval.json

echo ""
echo "==================================="
echo "Evaluation Complete!"
echo "==================================="
echo ""
echo "Results saved to results/ directory"
echo ""
echo "Compare results:"
echo "  cat results/yolov8n_eval.json | jq '.aggregated'"
echo "  cat results/yolov8m_eval.json | jq '.aggregated'"
echo ""