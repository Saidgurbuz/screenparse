"""PYTHONPATH=src .venv/bin/python -m evaluation.cli \
  --format yolo \
  --image-dir /proj/docling-vision/users/said/data/yolo_filtered/images/test \
  --labels-dir /proj/docling-vision/users/said/data/yolo_filtered/labels/test \
  --groundcua-root /proj/docling-vision/users/said/GroundCUA/GroundCUA \
  --classes /proj/docling-vision/users/said/data/yolo_filtered/classes.txt \
  --max-samples 870 \
  --yolo-model /proj/docling-vision/users/said/webshot-dataset/runs/detect/webshot_ui_refined_labels_filtered/weights/best.pt \
  --qwen3-vl Qwen/Qwen3-VL-8B-Instruct \
  --omniparser-weights /proj/docling-vision/users/said/webshot-dataset/runs/omniparser/model.pt \
  --gemini gemini-2.5-flash-lite \
  --class-schema groundcua \
  --metrics page_iou,label_page_iou,map,recall,recall_agnostic \
  --batch-size 32 \
  --save-preds evaluation_results_groundcua/preds \
  --output evaluation_results_groundcua/report.json"""


from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple

from .datasets import build_raw_dataset, build_yolo_dataset, build_groundcua_dataset
from .label_mapping import LabelMapper
from .metrics.label_page_iou import LabelAwarePageIoU
from .metrics.map import MeanAveragePrecision
from .metrics.recall import Recall
from .metrics.page_iou import PageIoU
from .models.base import OfflinePredictionRunner
from .models.gemini import GeminiRunner
from .models.paddleocrvl import PaddleOCRVLRunner
from .models.qwen3_vl import Qwen3VLRunner
from .models.screenvlm import ScreenVLMRunner
from .models.yolo import YoloModelRunner
from .runner import Evaluator


def _metric_specs(metric_names: List[str], args) -> List[Tuple[str, Dict]]:
    specs = []
    for name in metric_names:
        if name == "page_iou":
            specs.append(("page_iou", {"max_resolution": args.pageiou_resolution}))
        elif name in ("label_page_iou", "labelawarepageiou"):
            specs.append(("label_page_iou", {"max_resolution": args.pageiou_resolution}))
        elif name in ("map", "map_50", "map50"):
            specs.append(("map", {"iou_threshold": args.map_iou_thr}))
        elif name in ("recall", "recall_label", "recall_labelaware"):
            specs.append(("recall", {"iou_threshold": args.recall_iou_thr, "label_aware": True}))
        elif name in ("recall_agnostic", "recall_nolabel", "recall_any"):
            specs.append(("recall", {"iou_threshold": args.recall_iou_thr, "label_aware": False}))
        else:
            print(f"Warning: unknown metric '{name}' - skipping.")
    return specs


def _build_metrics(specs: List[Tuple[str, Dict]]):
    metrics = []
    for name, kwargs in specs:
        if name == "page_iou":
            metrics.append(PageIoU(**kwargs))
        elif name == "label_page_iou":
            metrics.append(LabelAwarePageIoU(**kwargs))
        elif name == "map":
            metrics.append(MeanAveragePrecision(**kwargs))
        elif name == "recall":
            metrics.append(Recall(**kwargs))
    return metrics


def _build_runner_from_spec(spec: Tuple[str, Dict]):
    kind, kwargs = spec
    if kind == "yolo":
        return YoloModelRunner(**kwargs)
    if kind == "offline":
        return OfflinePredictionRunner(**kwargs)
    if kind == "qwen3_vl":
        return Qwen3VLRunner(**kwargs)
    if kind == "gemini":
        return GeminiRunner(**kwargs)
    if kind == "paddleocrvl":
        return PaddleOCRVLRunner(**kwargs)
    if kind == "screenvlm":
        return ScreenVLMRunner(**kwargs)
    raise ValueError(f"Unknown model kind {kind}")


def _eval_worker(dataset, metric_specs, model_spec, pred_dir, batch_size, queue, class_schema):
    try:
        metrics = _build_metrics(metric_specs)
        mapper = LabelMapper(class_schema)
        evaluator = Evaluator(metrics, label_mapper=mapper)
        model = _build_runner_from_spec(model_spec)
        report = evaluator.evaluate_model(
            dataset,
            model,
            save_predictions_dir=pred_dir,
            batch_size=batch_size,
        )
        queue.put({"ok": True, "report": report})
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc)})


def _parse_name_path(val: str) -> Tuple[str, str]:
    if "=" in val:
        name, path = val.split("=", 1)
        return name.strip(), path.strip()
    path = val.strip()
    return Path(path).stem, path


def _build_dataset(args, label_mapper):
    fmt = args.format
    if args.groundcua_root:
        return build_groundcua_dataset(
            root_dir=args.groundcua_root,
            max_samples=args.max_samples,
            label_mapper=label_mapper,
        )

    if fmt == "auto":
        fmt = "yolo" if args.labels_dir else "raw"

    if fmt == "yolo":
        return build_yolo_dataset(
            image_dir=args.image_dir,
            labels_dir=args.labels_dir,
            classes_path=args.classes,
            max_samples=args.max_samples,
            label_mapper=label_mapper,
        )
    return build_raw_dataset(image_dir=args.image_dir, max_samples=args.max_samples, label_mapper=label_mapper)


def main(argv: List[str] | None = None):
    parser = argparse.ArgumentParser(description="Evaluate screen parsing models.")
    parser.add_argument("--image-dir", required=True, help="Directory with evaluation images.")
    parser.add_argument("--labels-dir", help="YOLO labels directory (only for --format yolo/auto).")
    parser.add_argument("--classes", help="Optional classes.txt for YOLO labels.")
    parser.add_argument("--format", choices=["auto", "raw", "yolo"], default="auto", help="Dataset format.")
    parser.add_argument("--groundcua-root", help="GroundCUA root directory (contains data/ and images/).")
    parser.add_argument("--max-samples", type=int, help="Cap number of samples for quick runs.")

    parser.add_argument("--yolo-model", action="append", default=[], help="YOLO weights (name=path or just path).")
    parser.add_argument("--yolo-conf", type=float, default=0.10)
    parser.add_argument("--yolo-iou", type=float, default=0.10)
    parser.add_argument("--yolo-imgsz", type=int, default=1280)
    parser.add_argument("--device", help="Optional inference device hint.")
    parser.add_argument(
        "--omniparser-weights",
        help="Path to OmniParser YOLOv8 weights (treated as a YOLO model).",
    )

    parser.add_argument(
        "--qwen3-vl",
        action="append",
        default=[],
        help="Use Qwen/Qwen3-VL-8B-Instruct via vLLM (name=model_id or just model_id).",
    )
    parser.add_argument("--qwen-prompt", help="Custom prompt for Qwen3-VL.")
    parser.add_argument("--qwen-max-new-tokens", type=int, default=4096)
    parser.add_argument("--qwen-temperature", type=float, default=0.0)
    parser.add_argument("--qwen-top-p", type=float, default=0.9)
    parser.add_argument(
        "--screenvlm",
        action="append",
        default=[],
        help="ScreenVLM checkpoint (name=path or just path).",
    )
    parser.add_argument("--screenvlm-processor", help="Optional processor path for ScreenVLM.")
    parser.add_argument("--screenvlm-revision", help="Optional model revision for ScreenVLM.")
    parser.add_argument("--screenvlm-max-new-tokens", type=int, default=6192)
    parser.add_argument("--screenvlm-temperature", type=float, default=0.0)
    parser.add_argument("--screenvlm-top-p", type=float, default=0.9)
    parser.add_argument("--screenvlm-top-k", type=int, default=50)
    parser.add_argument("--screenvlm-tp", type=int, default=1, help="Tensor parallel size for ScreenVLM.")
    parser.add_argument("--screenvlm-gpu-mem", type=float, default=0.9)
    parser.add_argument("--screenvlm-max-model-len", type=int, default=262144)
    parser.add_argument(
        "--gemini",
        action="append",
        default=[],
        help="Use Gemini via google-genai (name=model_id or just model_id).",
    )
    parser.add_argument("--gemini-api-key", help="Gemini API key (or set GEMINI_API_KEY/GOOGLE_API_KEY).")
    parser.add_argument(
        "--paddle-ocrvl",
        action="store_true",
        help="Use PaddleOCRVL via separate Python env (.venv-paddle/bin/python).",
    )
    parser.add_argument(
        "--paddle-python",
        default=".venv-paddle/bin/python",
        help="Python executable for PaddleOCRVL.",
    )
    parser.add_argument(
        "--paddle-bridge",
        default=str((Path(__file__).resolve().parent / "models" / "paddle_ocrvl_bridge.py")),
        help="Bridge script path for PaddleOCRVL.",
    )

    parser.add_argument(
        "--offline-pred",
        action="append",
        default=[],
        help="Directory with precomputed predictions (name=dir or just dir).",
    )

    parser.add_argument("--pageiou-resolution", type=int, default=0, help="Downscale long side before PageIoU.")
    parser.add_argument("--map-iou-thr", type=float, default=0.5, help="IoU threshold for mAP.")
    parser.add_argument("--recall-iou-thr", type=float, default=0.5, help="IoU threshold for recall.")
    parser.add_argument(
        "--metrics",
        default="page_iou,label_page_iou,map,recall_label,recall_agnostic",
        help="Comma-separated metrics: page_iou,label_page_iou,map,recall_label,recall_agnostic",
    )
    parser.add_argument(
        "--class-schema",
        choices=["custom55", "groundcua"],
        default="custom55",
        help="Target class schema for labels.",
    )
    parser.add_argument("--batch-size", type=int, help="Batch size for predict_batch.")
    parser.add_argument("--save-preds", help="Base directory to dump model predictions.")
    parser.add_argument("--output", help="Path to write JSON report.")

    args = parser.parse_args(argv)

    mapper = LabelMapper(args.class_schema)

    dataset = _build_dataset(args, label_mapper=mapper)

    if not dataset:
        raise SystemExit("No evaluation samples found for given paths.")
    print(f"Loaded {len(dataset)} samples from {args.image_dir} (format={args.format}).")

    metric_names = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    metric_specs = _metric_specs(metric_names, args)
    if not metric_specs:
        raise SystemExit("No valid metrics selected.")

    model_specs: List[Tuple[str, Dict]] = []
    
    for entry in args.offline_pred:
        name, path = _parse_name_path(entry)
        model_specs.append(("offline", dict(name=name, predictions_dir=path)))
    
    for entry in args.yolo_model:
        name, path = _parse_name_path(entry)
        model_specs.append(
            (
                "yolo",
                dict(
                    weights_path=path,
                    name=name,
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                    imgsz=args.yolo_imgsz,
                    device=args.device,
                ),
            )
        )
        
    if args.omniparser_weights:
        model_specs.append(
            (
                "yolo",
                dict(
                    weights_path=args.omniparser_weights,
                    name="omniparser",
                    conf=args.yolo_conf,
                    iou=args.yolo_iou,
                    imgsz=args.yolo_imgsz,
                    device=args.device,
                ),
            )
        )

    for entry in args.qwen3_vl:
        name, model_id = _parse_name_path(entry)
        model_specs.append(
            (
                "qwen3_vl",
                dict(
                    model_id=model_id,
                    name=name,
                    prompt=args.qwen_prompt,
                    max_new_tokens=args.qwen_max_new_tokens,
                    temperature=args.qwen_temperature,
                    top_p=args.qwen_top_p,
                    class_schema=args.class_schema,
                ),
            )
        )

    for entry in args.screenvlm:
        name, checkpoint = _parse_name_path(entry)
        model_specs.append(
            (
                "screenvlm",
                dict(
                    checkpoint=checkpoint,
                    name=name,
                    processor_path=args.screenvlm_processor,
                    revision=args.screenvlm_revision,
                    max_new_tokens=args.screenvlm_max_new_tokens,
                    temperature=args.screenvlm_temperature,
                    top_p=args.screenvlm_top_p,
                    top_k=args.screenvlm_top_k,
                    tensor_parallel_size=args.screenvlm_tp,
                    gpu_memory_utilization=args.screenvlm_gpu_mem,
                    max_model_len=args.screenvlm_max_model_len,
                ),
            )
        )

    for entry in args.gemini:
        name, model_id = _parse_name_path(entry)
        model_specs.append(
            (
                "gemini",
                dict(
                    model_id=model_id,
                    name=name,
                    api_key=args.gemini_api_key,
                    prompt=args.qwen_prompt,
                    class_schema=args.class_schema,
                ),
            )
        )

    if args.paddle_ocrvl:
        model_specs.append(
            (
                "paddleocrvl",
                dict(
                    python_path=args.paddle_python,
                    name="paddleocrvl",
                ),
            )
        )

    if not model_specs:
        raise SystemExit("No models specified. Use --yolo-model or --offline-pred.")

    print(f"Running {len(model_specs)} model(s): {[cfg[1].get('name') for cfg in model_specs]}")

    reports = []
    base_pred_dir = Path(args.save_preds) if args.save_preds else None
    summary_rows = []
    ctx = mp.get_context("spawn")
    out_path = Path(args.output) if args.output else None
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for kind, cfg in model_specs:
        model_name = cfg.get("name") or cfg.get("model_id") or kind
        pred_dir = base_pred_dir / model_name if base_pred_dir else None
        print(f"\n[Model] Starting isolated process for {model_name}")
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_eval_worker,
            args=(
                dataset,
                metric_specs,
                (kind, cfg),
                str(pred_dir) if pred_dir else None,
                args.batch_size,
                queue,
                args.class_schema,
            ),
        )
        proc.start()
        result = queue.get()  # Wait for report or error
        proc.join()

        if not result.get("ok"):
            err = result.get("error", "Unknown error")
            raise SystemExit(f"Model {model_name} failed: {err}")

        report = result["report"]
        reports.append(report)
        print(f"Model {model_name}:")
        for metric_name, data in report["dataset_metrics"].items():
            print(f"  {metric_name}: {data.get('value')}")
        summary_rows.append((model_name, report["dataset_metrics"]))

        if out_path:
            existing = []
            if out_path.exists():
                with out_path.open("r", encoding="utf-8") as f:
                    try:
                        loaded = json.load(f)
                        existing = loaded if isinstance(loaded, list) else [loaded]
                    except json.JSONDecodeError:
                        existing = []
            merged = existing + [report]
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2)
            print(f"  Appended report to {out_path}")

    if out_path and summary_rows:
        import csv

        csv_path = out_path.parent / "summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        metric_names = sorted({m for _, metrics in summary_rows for m in metrics.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["model"] + metric_names)
            for model_name, metrics in summary_rows:
                row = [model_name]
                for m in metric_names:
                    val = metrics.get(m, {}).get("value")
                    if isinstance(val, float):
                        row.append(f"{val:.3f}")
                    elif val is None:
                        row.append("")
                    else:
                        row.append(val)
                writer.writerow(row)
        print(f"\nWrote summary CSV to {csv_path}")


if __name__ == "__main__":
    main()
