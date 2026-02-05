#!/usr/bin/env python3
"""
Benchmark VLM inference speed using HuggingFace Transformers.

This script measures:
- GPU memory usage (actual model footprint)
- Inference latency (ms per sample)
- Throughput (samples per second)

Uses native Transformers inference for accurate measurements.
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Callable

import torch
from PIL import Image


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for a model to benchmark."""
    name: str
    model_id: str
    model_type: str  # "screenvlm", "qwen3", "internvl3"
    revision: Optional[str] = None
    torch_dtype: torch.dtype = torch.bfloat16


# Default models to benchmark
DEFAULT_MODELS = [
    ModelConfig(
        name="ScreenVLM",
        model_id="/proj/docling-vision/users/said/nanoVLM/GraniteDoclingV1-stage1-init/nanoVLM_siglip2-base-patch16-512_2048_mp4_GraniteDoclingV1-stage1-init_16xGPU_full_ds_bs64_287500_lr_vision_0.002-language_0.002-0.0212_0109-164905_lsf361208/converted_untied/step_150000",
        model_type="screenvlm",
        revision="untied",
    ),
    ModelConfig(
        name="Qwen3-VL-2B",
        model_id="/proj/docling-vision/users/said/Qwen3-VL/qwen-vl-finetune/checkpoints/Qwen3-VL-2B-Instruct_8GPU_Full_272335/checkpoint-80000",
        model_type="qwen3",
    ),
    ModelConfig(
        name="InternVL3-2B",
        model_id="/proj/docling-vision/users/said/InternVL/InternVL/internvl_chat/work_dirs/internvl_sft/internvl3_2B_full_train_ScreenVLM_Prod_AnnotationV2_SynthDocs_512px-ibm-granite-lsf375632_8GPU/checkpoint-9200",
        model_type="internvl3",
    ),
]


# ============================================================================
# Benchmark Results
# ============================================================================

@dataclass
class BenchmarkResult:
    """Results from benchmarking a single model."""
    model_name: str
    model_memory_gb: float
    latencies_ms: List[float]
    num_samples: int
    num_params_billion: float = 0.0
    
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
        return 1000.0 / self.mean_latency_ms


# ============================================================================
# Model Loaders
# ============================================================================

def count_parameters(model) -> float:
    """Count model parameters in billions."""
    total = sum(p.numel() for p in model.parameters())
    return total / 1e9


def load_screenvlm(config: ModelConfig, device: str) -> Tuple:
    """Load ScreenVLM model with Transformers."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    
    processor = AutoProcessor.from_pretrained(
        config.model_id, 
        trust_remote_code=True,
        revision=config.revision,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        torch_dtype=config.torch_dtype,
        revision=config.revision,
    ).to(device)
    model.eval()
    
    num_params = count_parameters(model)
    
    def prepare_inputs(image: Image.Image, prompt: str):
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
        inputs = processor(images=image, text=text, return_tensors="pt")
        return {k: v.to(device) for k, v in inputs.items()}
    
    return model, processor, prepare_inputs, num_params


def load_qwen3(config: ModelConfig, device: str) -> Tuple:
    """Load Qwen3-VL model with Transformers."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    processor = AutoProcessor.from_pretrained(
        config.model_id,
        trust_remote_code=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        torch_dtype=config.torch_dtype,
    ).to(device)
    model.eval()
    
    num_params = count_parameters(model)
    
    def prepare_inputs(image: Image.Image, prompt: str):
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
        # Process with image
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        return {k: v.to(device) for k, v in inputs.items()}
    
    return model, processor, prepare_inputs, num_params


def load_internvl3(config: ModelConfig, device: str) -> Tuple:
    """Load InternVL3 model with Transformers."""
    from transformers import AutoModel, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        config.model_id,
        trust_remote_code=True,
        torch_dtype=config.torch_dtype,
    ).to(device)
    model.eval()
    
    num_params = count_parameters(model)
    
    # InternVL3 has its own chat method, but for fair comparison we'll use generate
    def prepare_inputs(image: Image.Image, prompt: str):
        # InternVL uses a specific format
        # For basic inference, we need to prepare pixel values and input_ids
        from torchvision import transforms
        
        # Get image processor from model config if available
        if hasattr(model, 'img_context_token_id'):
            img_context_token_id = model.img_context_token_id
        else:
            img_context_token_id = tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        
        # Simple image preprocessing - InternVL expects 448x448 by default
        img_size = getattr(model.config, 'force_image_size', 448)
        if isinstance(img_size, (list, tuple)):
            img_size = img_size[0]
        
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(image).unsqueeze(0).to(device, dtype=config.torch_dtype)
        
        # Prepare text with image placeholder
        formatted_prompt = f"<image>\n{prompt}"
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(device)
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }
    
    return model, tokenizer, prepare_inputs, num_params


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


def benchmark_model(
    config: ModelConfig,
    image_paths: List[Path],
    device: str = "cuda",
    warmup_samples: int = 5,
    max_new_tokens: int = 256,
    prompt: str = "Generate the screen representation for this UI:",
) -> BenchmarkResult:
    """Benchmark a single model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {config.name}")
    print(f"{'='*60}")
    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Load model
    print(f"Loading model: {config.model_id}")
    load_start = time.perf_counter()
    
    if config.model_type == "screenvlm":
        model, processor, prepare_inputs, num_params = load_screenvlm(config, device)
    elif config.model_type == "qwen3":
        model, processor, prepare_inputs, num_params = load_qwen3(config, device)
    elif config.model_type == "internvl3":
        model, processor, prepare_inputs, num_params = load_internvl3(config, device)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    load_time = time.perf_counter() - load_start
    
    # Measure GPU memory after loading
    torch.cuda.synchronize()
    model_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"Model loaded in {load_time:.2f}s")
    print(f"Parameters: {num_params:.2f}B")
    print(f"GPU memory: {model_memory_gb:.2f} GB")
    
    # Load images
    print(f"Loading {len(image_paths)} images...")
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
    
    if not images:
        raise RuntimeError("No valid images loaded")
    
    # Warmup
    print(f"Warming up ({warmup_samples} samples)...")
    for i in range(min(warmup_samples, len(images))):
        with torch.no_grad():
            inputs = prepare_inputs(images[i], prompt)
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running benchmark ({len(images)} samples)...")
    latencies_ms = []
    
    for idx, img in enumerate(images):
        # Prepare inputs (not timed - we're measuring inference only)
        with torch.no_grad():
            inputs = prepare_inputs(img, prompt)
            
            # Time the generation
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            torch.cuda.synchronize()
            end = time.perf_counter()
        
        latency_ms = (end - start) * 1000
        latencies_ms.append(latency_ms)
        
        if (idx + 1) % 10 == 0:
            print(f"  Sample {idx + 1}/{len(images)}: {latency_ms:.1f} ms")
    
    # Cleanup
    print("Cleaning up...")
    del model
    if 'processor' in dir():
        del processor
    gc.collect()
    torch.cuda.empty_cache()
    
    return BenchmarkResult(
        model_name=config.name,
        model_memory_gb=model_memory_gb,
        latencies_ms=latencies_ms,
        num_samples=len(images),
        num_params_billion=num_params,
    )


# ============================================================================
# Output Formatting
# ============================================================================

def format_results_table(results: List[BenchmarkResult]) -> str:
    """Format results as a nice ASCII table."""
    lines = []
    lines.append("")
    lines.append("=" * 100)
    lines.append("BENCHMARK RESULTS (HuggingFace Transformers)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'Model':<20} {'Params (B)':<12} {'Memory (GB)':<12} {'Latency (ms)':<22} {'Median (ms)':<12} {'Throughput':<12}")
    lines.append(f"{'':<20} {'':<12} {'':<12} {'(mean ± std)':<22} {'':<12} {'(samp/s)':<12}")
    lines.append("-" * 100)
    
    for r in results:
        latency_str = f"{r.mean_latency_ms:.1f} ± {r.std_latency_ms:.1f}"
        lines.append(
            f"{r.model_name:<20} "
            f"{r.num_params_billion:<12.2f} "
            f"{r.model_memory_gb:<12.2f} "
            f"{latency_str:<22} "
            f"{r.median_latency_ms:<12.1f} "
            f"{r.throughput_samples_per_sec:<12.2f}"
        )
    
    lines.append("-" * 100)
    lines.append("")
    
    return "\n".join(lines)


def format_latex_table(results: List[BenchmarkResult]) -> str:
    """Format results as LaTeX table for paper."""
    lines = []
    lines.append("% LaTeX table for paper")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Model & Params (B) & Memory (GB) & Latency (ms) & Throughput \\\\")
    lines.append("\\midrule")
    
    for r in results:
        latency_str = f"${r.median_latency_ms:.1f}$"
        lines.append(
            f"{r.model_name} & "
            f"{r.num_params_billion:.2f} & "
            f"{r.model_memory_gb:.2f} & "
            f"{latency_str} & "
            f"{r.throughput_samples_per_sec:.2f} \\\\"
        )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Inference speed comparison of VLM models.}")
    lines.append("\\label{tab:inference_speed}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def format_markdown_table(results: List[BenchmarkResult]) -> str:
    """Format results as Markdown table."""
    lines = []
    lines.append("## Benchmark Results")
    lines.append("")
    lines.append("| Model | Params (B) | Memory (GB) | Latency (ms) | Throughput (samp/s) |")
    lines.append("|-------|------------|-------------|--------------|---------------------|")
    
    for r in results:
        lines.append(
            f"| {r.model_name} | "
            f"{r.num_params_billion:.2f} | "
            f"{r.model_memory_gb:.2f} | "
            f"{r.median_latency_ms:.1f} | "
            f"{r.throughput_samples_per_sec:.2f} |"
        )
    
    lines.append("")
    return "\n".join(lines)


def save_results(results: List[BenchmarkResult], output_dir: str):
    """Save benchmark results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = output_dir / f"benchmark_hf_{timestamp}.json"
    json_data = []
    for r in results:
        json_data.append({
            "model_name": r.model_name,
            "num_params_billion": r.num_params_billion,
            "model_memory_gb": r.model_memory_gb,
            "mean_latency_ms": r.mean_latency_ms,
            "median_latency_ms": r.median_latency_ms,
            "std_latency_ms": r.std_latency_ms,
            "throughput_samples_per_sec": r.throughput_samples_per_sec,
            "num_samples": r.num_samples,
            "latencies_ms": r.latencies_ms,
        })
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON results to: {json_path}")
    
    # Save LaTeX
    latex_path = output_dir / f"benchmark_hf_{timestamp}.tex"
    with open(latex_path, "w") as f:
        f.write(format_latex_table(results))
    print(f"Saved LaTeX table to: {latex_path}")
    
    # Save Markdown
    md_path = output_dir / f"benchmark_hf_{timestamp}.md"
    with open(md_path, "w") as f:
        f.write(format_markdown_table(results))
    print(f"Saved Markdown table to: {md_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark VLM inference speed using HuggingFace Transformers",
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
        default=30,
        help="Number of samples to use for benchmarking",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup samples",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help="Maximum new tokens to generate per sample",
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
        "--device",
        default="cuda",
        help="Device to run on",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("VLM Inference Benchmark (HuggingFace Transformers)")
    print("=" * 60)
    print(f"Image directory: {args.image_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Warmup samples: {args.warmup}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Models: {args.models}")
    print(f"Device: {args.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Get sample images
    image_paths = get_sample_images(args.image_dir, args.num_samples)
    if not image_paths:
        print("Error: No images found")
        sys.exit(1)
    print(f"Found {len(image_paths)} images")
    
    # Filter models
    selected_configs = [m for m in DEFAULT_MODELS if m.model_type in args.models]
    
    # Run benchmarks
    results = []
    for config in selected_configs:
        try:
            result = benchmark_model(
                config=config,
                image_paths=image_paths,
                device=args.device,
                warmup_samples=args.warmup,
                max_new_tokens=args.max_new_tokens,
            )
            results.append(result)
        except Exception as e:
            print(f"Error benchmarking {config.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print results
    if results:
        print(format_results_table(results))
        save_results(results, args.output_dir)
    else:
        print("No results to display")


if __name__ == "__main__":
    main()
