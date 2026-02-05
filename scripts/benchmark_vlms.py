#!/usr/bin/env python3
"""
VLM Inference Speed Benchmark Script

This script benchmarks inference speed for VLM models (ScreenVLM, Qwen3-VL, InternVL3)
using VLLM and outputs a formatted table for paper ablation study.

Usage:
    cd /proj/docling-vision/users/said/webshot-dataset
    PYTHONPATH=src .venv/bin/python scripts/benchmark_vlms.py

Output:
    - Console table with latency (mean ± std) and throughput
    - Optional CSV/JSON export
"""

# IMPORTANT: Set multiprocessing start method BEFORE importing torch/CUDA
# This must happen before any CUDA initialization
import multiprocessing as mp
import os

# Set spawn method before any CUDA imports
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # Already set

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# Disable NCCL if only using 1 GPU to avoid distributed issues
os.environ.setdefault("NCCL_DEBUG", "WARN")

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image


# ============================================================================
# Configuration
# ============================================================================

# Prompts matching the evaluation framework
SCREENVLM_PROMPT = "Generate the screen representation for this UI:"

QWEN3_INTERNVL3_PROMPT = (
    "You are a UI parser. Given a screenshot image, extract all visible UI elements.\n"
    "Return JSON list with objects: "
    '{"bbox_ltrb":[l,t,r,b], "label": "<type>", "text": "<visible text>"}.\n'
    "The bbox_ltrb should be normalized to 0-1000. (l,t,r,b). Include all elements.\n"
    "Allowed labels:\n- Text\n- Image\n- Icon\n- Button\n- Link\n- Input\n- Checkbox\n- Radio\n- Dropdown\n- Toggle\n- Slider\n- Tab\n- Menu\n- Modal\n- Card\n- Table\n- List\n- Form\n- Navigation\n- Header\n- Footer\n- Sidebar\n- Container"
)


@dataclass
class ModelConfig:
    """Configuration for a model to benchmark."""
    name: str
    model_id: str
    model_type: str  # "screenvlm", "qwen3", "internvl3"
    # Optional overrides
    revision: Optional[str] = None
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9


# Default models to benchmark
DEFAULT_MODELS = [
    ModelConfig(
        name="ScreenVLM",
        model_id="/proj/docling-vision/users/said/nanoVLM/GraniteDoclingV1-stage1-init/nanoVLM_siglip2-base-patch16-512_2048_mp4_GraniteDoclingV1-stage1-init_16xGPU_full_ds_bs64_287500_lr_vision_0.002-language_0.002-0.0212_0109-164905_lsf361208/converted_untied/step_150000",
        model_type="screenvlm",
        revision="untied",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    ),
    ModelConfig(
        name="Qwen3-VL-2B-Instruct",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        model_type="qwen3",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    ),
    ModelConfig(
        name="InternVL3-2B",
        model_id="OpenGVLab/InternVL3-2B",
        model_type="internvl3",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
    ),
]


# ============================================================================
# Benchmark Results
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results from benchmarking a single model."""
    model_name: str
    model_memory_gb: float  # Memory after loading model
    latencies_ms: List[float]  # Per-sample latencies
    num_samples: int
    batch_size: int
    
    @property
    def mean_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0
    
    @property
    def median_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0
        sorted_latencies = sorted(self.latencies_ms)
        n = len(sorted_latencies)
        if n % 2 == 0:
            return (sorted_latencies[n // 2 - 1] + sorted_latencies[n // 2]) / 2
        return sorted_latencies[n // 2]
    
    @property
    def std_latency_ms(self) -> float:
        if len(self.latencies_ms) < 2:
            return 0
        mean = self.mean_latency_ms
        variance = sum((x - mean) ** 2 for x in self.latencies_ms) / (len(self.latencies_ms) - 1)
        return variance ** 0.5
    
    @property
    def throughput_samples_per_sec(self) -> float:
        if self.mean_latency_ms == 0:
            return 0
        # mean_latency_ms is already per-sample, so throughput = 1000 / latency
        return 1000.0 / self.mean_latency_ms


# ============================================================================
# Model Loaders (simplified from evaluation models)
# ============================================================================

def load_screenvlm(config: ModelConfig) -> Tuple[any, any, any]:
    """Load ScreenVLM model with VLLM."""
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    
    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        top_k=50,
        max_tokens=8192,  # Limit for fair benchmarking
        skip_special_tokens=False,
    )
    
    llm_kwargs = {
        "model": config.model_id,
        "limit_mm_per_prompt": {"image": 1},
        "trust_remote_code": True,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "tensor_parallel_size": 1,  # Force single GPU
    }
    if config.revision:
        llm_kwargs["revision"] = config.revision
    
    llm = LLM(**llm_kwargs)
    return llm, processor, sampling_params


def load_qwen3(config: ModelConfig) -> Tuple[any, any, any]:
    """Load Qwen3-VL model with VLLM."""
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from qwen_vl_utils import process_vision_info
    
    processor = AutoProcessor.from_pretrained(config.model_id, trust_remote_code=True)
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=8192,  # Limit for fair benchmarking
    )
    
    llm = LLM(
        model=config.model_id,
        trust_remote_code=True,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=1,  # Force single GPU
    )
    return llm, (processor, process_vision_info), sampling_params


def load_internvl3(config: ModelConfig) -> Tuple[any, any, any]:
    """Load InternVL3 model with VLLM."""
    from vllm import LLM, SamplingParams
    
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=0.9,
        max_tokens=8192,  # Limit for fair benchmarking
    )
    
    llm_kwargs = {
        "model": config.model_id,
        "trust_remote_code": True,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "tensor_parallel_size": 1,  # Force single GPU
        "limit_mm_per_prompt": {"image": 1},
    }
    if config.max_model_len:
        llm_kwargs["max_model_len"] = config.max_model_len
    
    llm = LLM(**llm_kwargs)
    return llm, None, sampling_params


# ============================================================================
# Input Preparation
# ============================================================================

def prepare_screenvlm_input(processor, image: Image.Image, prompt: str) -> dict:
    """Prepare input for ScreenVLM."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    return {"prompt": text, "multi_modal_data": {"image": image}}


def prepare_qwen3_input(processor_tuple, image: Image.Image, prompt: str) -> dict:
    """Prepare input for Qwen3-VL."""
    processor, process_vision_info = processor_tuple
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    patch_size = getattr(processor.image_processor, "patch_size", None)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def prepare_internvl3_input(processor, image: Image.Image, prompt: str) -> dict:
    """Prepare input for InternVL3."""
    formatted_prompt = f"User: <image>\n{prompt}<|end|>\nAssistant:"
    return {"prompt": formatted_prompt, "multi_modal_data": {"image": image}}


# ============================================================================
# Benchmarking
# ============================================================================

def get_sample_images(image_dir: str, num_samples: int) -> List[Path]:
    """Get a list of sample images for benchmarking."""
    image_dir = Path(image_dir)
    images = sorted(image_dir.glob("*.jpg"))[:num_samples]
    if len(images) < num_samples:
        print(f"Warning: Only found {len(images)} images, requested {num_samples}")
    return images


def get_gpu_memory_usage_mb() -> float:
    """Get current GPU memory usage in MB using nvidia-smi (works with VLLM workers)."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            # Sum memory across all GPUs (in case of multi-GPU, though we use 1)
            memory_mb = sum(float(x.strip()) for x in result.stdout.strip().split('\n') if x.strip())
            return memory_mb
    except Exception:
        pass
    return 0.0


def benchmark_model(
    config: ModelConfig,
    image_paths: List[Path],
    batch_size: int = 1,
    warmup_batches: int = 3,
) -> BenchmarkResult:
    """Benchmark a single model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config.name}")
    print(f"{'='*60}")
    
    # Select prompt based on model type (matching evaluation framework)
    if config.model_type == "screenvlm":
        prompt = SCREENVLM_PROMPT
    else:
        prompt = QWEN3_INTERNVL3_PROMPT
    
    # Get baseline GPU memory before loading
    gc.collect()
    baseline_memory_mb = get_gpu_memory_usage_mb()
    print(f"Baseline GPU memory: {baseline_memory_mb:.0f} MB")
    
    # Load model and measure memory
    print(f"Loading model: {config.model_id}")
    load_start = time.perf_counter()
    
    if config.model_type == "screenvlm":
        llm, processor, sampling_params = load_screenvlm(config)
        prepare_input = lambda img: prepare_screenvlm_input(processor, img, prompt)
    elif config.model_type == "qwen3":
        llm, processor, sampling_params = load_qwen3(config)
        prepare_input = lambda img: prepare_qwen3_input(processor, img, prompt)
    elif config.model_type == "internvl3":
        llm, processor, sampling_params = load_internvl3(config)
        prepare_input = lambda img: prepare_internvl3_input(processor, img, prompt)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    load_time = time.perf_counter() - load_start
    
    # Measure memory after loading using nvidia-smi (captures VLLM worker memory)
    current_memory_mb = get_gpu_memory_usage_mb()
    model_memory_gb = (current_memory_mb - baseline_memory_mb) / 1024
    print(f"Model loaded in {load_time:.2f}s, GPU memory: {model_memory_gb:.2f} GB (total: {current_memory_mb:.0f} MB)")
    
    # Load all images
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    
    if not images:
        raise RuntimeError("No valid images loaded")
    
    print(f"Loaded {len(images)} images")
    
    # Prepare batches
    batches = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_inputs = [prepare_input(img) for img in batch_images]
        batches.append(batch_inputs)
    
    # Warmup - VLLM's generate() is blocking, no need for torch.cuda.synchronize()
    # which conflicts with VLLM's worker processes
    print(f"Warming up ({warmup_batches} batches)...")
    for i in range(min(warmup_batches, len(batches))):
        _ = llm.generate(batches[i % len(batches)], sampling_params=sampling_params)
    
    # Benchmark - just use wall-clock time, generate() blocks until done
    print(f"Running benchmark ({len(batches)} batches)...")
    latencies_ms = []
    
    for batch_idx, batch_inputs in enumerate(batches):
        start = time.perf_counter()
        _ = llm.generate(batch_inputs, sampling_params=sampling_params)
        end = time.perf_counter()
        
        # Per-sample latency (divide by batch size)
        batch_latency_ms = (end - start) * 1000
        per_sample_latency = batch_latency_ms / len(batch_inputs)
        latencies_ms.append(per_sample_latency)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{len(batches)}: {per_sample_latency:.1f} ms/sample")
    
    # Cleanup
    print("Cleaning up model...")
    try:
        engine = getattr(llm, "llm_engine", None)
        if engine and hasattr(engine, "shutdown"):
            engine.shutdown()
    except:
        pass
    del llm
    torch.cuda.empty_cache()
    gc.collect()
    
    # print the current results for this benchmark
    print(f"Completed benchmarking {config.name}:")
    print(f"  Mean latency: {sum(latencies_ms) / len(latencies_ms):.1f} ms/sample")
    print(f"  Std latency: {BenchmarkResult('', 0, latencies_ms, 0, 0).std_latency_ms:.1f} ms")
    print(f"  Throughput: {BenchmarkResult('', 0, latencies_ms, 0, batch_size).throughput_samples_per_sec:.2f} samples/s")
    print(f"  Memory usage: {model_memory_gb:.2f} GB")
    
    return BenchmarkResult(
        model_name=config.name,
        model_memory_gb=model_memory_gb,
        latencies_ms=latencies_ms,
        num_samples=len(images),
        batch_size=batch_size,
    )


def format_results_table(results: List[BenchmarkResult]) -> str:
    """Format results as a nice ASCII table."""
    # Header
    lines = []
    lines.append("")
    lines.append("=" * 95)
    lines.append("BENCHMARK RESULTS")
    lines.append("=" * 95)
    lines.append("")
    lines.append(f"{'Model':<20} {'Memory (GB)':<12} {'Latency (ms)':<25} {'Median (ms)':<12} {'Throughput':<15}")
    lines.append(f"{'':<20} {'':<12} {'(mean ± std)':<25} {'':<12} {'(samples/s)':<15}")
    lines.append("-" * 95)
    
    for r in results:
        latency_str = f"{r.mean_latency_ms:.1f} ± {r.std_latency_ms:.1f}"
        throughput_str = f"{r.throughput_samples_per_sec:.2f}"
        lines.append(f"{r.model_name:<20} {r.model_memory_gb:<12.2f} {latency_str:<25} {r.median_latency_ms:<12.1f} {throughput_str:<15}")
    
    lines.append("-" * 95)
    lines.append("")
    
    return "\n".join(lines)


def format_latex_table(results: List[BenchmarkResult]) -> str:
    """Format results as LaTeX table for paper."""
    lines = []
    lines.append("% LaTeX table for paper")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lccc}")
    lines.append("\\toprule")
    lines.append("Model & Memory (GB) & Latency (ms) & Throughput (samples/s) \\\\")
    lines.append("\\midrule")
    
    for r in results:
        latency_str = f"${r.mean_latency_ms:.1f} \\pm {r.std_latency_ms:.1f}$"
        lines.append(f"{r.model_name} & {r.model_memory_gb:.2f} & {latency_str} & {r.throughput_samples_per_sec:.2f} \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Inference speed comparison}")
    lines.append("\\label{tab:inference_speed}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def save_results(results: List[BenchmarkResult], output_dir: str):
    """Save results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON
    json_data = []
    for r in results:
        json_data.append({
            "model_name": r.model_name,
            "model_memory_gb": r.model_memory_gb,
            "mean_latency_ms": r.mean_latency_ms,
            "std_latency_ms": r.std_latency_ms,
            "throughput_samples_per_sec": r.throughput_samples_per_sec,
            "num_samples": r.num_samples,
            "batch_size": r.batch_size,
            "all_latencies_ms": r.latencies_ms,
        })
    
    json_path = output_dir / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON: {json_path}")
    
    # LaTeX
    latex_path = output_dir / "benchmark_table.tex"
    with open(latex_path, "w") as f:
        f.write(format_latex_table(results))
    print(f"Saved LaTeX: {latex_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VLM inference speed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-dir",
        default="/proj/docling-vision/users/said/data/yolo_filtered/images/test",
        help="Directory containing sample images",
    )
    parser.add_argument(
        "--num-samples", "-n",
        type=int,
        default=640,
        help="Number of samples to use for benchmarking (should be multiple of batch_size)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=64,
        help="Batch size for inference (larger = more robust per-sample latency)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup batches",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="/proj/docling-vision/users/said/webshot-dataset/benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["screenvlm", "qwen3", "internvl3"],
        default=["screenvlm", "qwen3", "internvl3"],
        help="Models to benchmark",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for VLLM (use higher values for larger batch sizes)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate per sample",
    )
    args = parser.parse_args()
    
    # Multiprocessing already set up at module level (before CUDA init)
    
    print("=" * 60)
    print("VLM Inference Speed Benchmark")
    print("=" * 60)
    print(f"Image directory: {args.image_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Warmup batches: {args.warmup}")
    print(f"Models: {args.models}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    
    # Get sample images
    image_paths = get_sample_images(args.image_dir, args.num_samples)
    if not image_paths:
        print("Error: No images found")
        sys.exit(1)
    print(f"Found {len(image_paths)} images")
    
    # Filter models based on selection
    model_type_map = {
        "screenvlm": "screenvlm",
        "qwen3": "qwen3",
        "internvl3": "internvl3",
    }
    selected_configs = []
    for m in DEFAULT_MODELS:
        if m.model_type in args.models:
            m.gpu_memory_utilization = args.gpu_memory_utilization
            selected_configs.append(m)
    
    # Run benchmarks
    results = []
    for config in selected_configs:
        try:
            result = benchmark_model(
                config=config,
                image_paths=image_paths,
                batch_size=args.batch_size,
                warmup_batches=args.warmup,
            )
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {config.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results
    print(format_results_table(results))
    
    # Save results
    if results:
        save_results(results, args.output_dir)
        print(f"\nResults saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
