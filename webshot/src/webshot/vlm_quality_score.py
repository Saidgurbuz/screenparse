# -*- coding: utf-8 -*-
"""
VLM-driven annotation quality scoring using Qwen3-VL via vLLM.

This module scores the quality of UI bounding box annotations by analyzing
visualization images (screenshots with overlaid bounding boxes and labels).
It evaluates coverage, false positives, duplication, and box precision to
filter low-quality annotations from the dataset.

For usage examples, see examples/vlm_refinement_example.sh
"""
from __future__ import annotations

import os
import json
import re
import glob
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Iterable
from PIL import Image
from tqdm import tqdm

try:
    from vllm import LLM, SamplingParams  # type: ignore
    _HAS_VLLM = True
except Exception:
    _HAS_VLLM = False


# ================== QUALITY SCORING PROMPT ==================

_QUALITY_SYSTEM_PROMPT = '''You are a rigorous judge of UI bounding-box annotations.

You will be given:
- ONE image: a screenshot of a user interface with bounding boxes and text labels drawn on top of it. Each box corresponds to a single annotated UI element.
- The labels near the boxes show the element class (for example: Text, Button, Link, Image, Navigation Bar, etc.).

Your task is to evaluate **only the quality of the bounding box annotations**, NOT the correctness of the class names.

When judging, **treat the class label text as low priority**:
- Assume the class labels are mostly correct.
- Only use the labels to distinguish boxes or understand their intent.
- Do NOT heavily penalize minor class mistakes (e.g., Text vs Heading).
- Focus on: "Is there a sensible box here?" and "Are obvious UI elements missing or duplicated?"

---------------------------
EVALUATION DIMENSIONS
---------------------------

You must score the annotation quality using FOUR sub-scores and ONE overall score.

All scores are from 0 to 100, where:
- 100 = perfect in that dimension
- 80–99 = very good, only minor issues
- 60–79 = usable, but several noticeable issues
- 40–59 = poor, many issues
- 0–39 = extremely bad

1) COVERAGE / MISSING ELEMENTS (0–100)  
   - Look for visually obvious, distinct UI elements that **should** be annotated:
     - buttons, main text blocks, headings, input fields, icons, major images, cards, menu items, etc.
   - Large and important items that should be covered:
     - main hero images, large central text, clearly clickable buttons or tabs, prominent fields.
   - Ignore tiny decorative details (small icons, background textures) unless they clearly look like clickable controls.
   - Don't forget that we are referring to the ui elements and text coverage, if there are empty areas, white/blank parts of the screen, it is correct that there shouldn't be boxes there.
   - High score (90–100): Almost every clear, medium-to-large UI element has a box.
   - Medium score (60–80): Some obvious UI elements are missing (e.g., a few main buttons or text blocks are unannotated).
   - Low score (<60): Many large, obvious UI elements have NO bounding boxes.

2) FALSE POSITIVES / SPURIOUS BOXES (0–100)  
   - Penalize boxes that are not aligned with any visible UI element, such as:
     - Boxes in completely blank areas.
     - Boxes that repeat the same position but shifted somewhere else on the screen where nothing exists.
     - Boxes over pure background images or whitespace where there is no clear object or control.
   - Do NOT treat "container + child" as false positive if the container corresponds to a real region (e.g., window, navbar, card) and the child is a real button or text.
   - High score (90–100): Almost all boxes correspond to real UI elements or meaningful regions.
   - Medium score (60–80): Some boxes are obviously floating over empty space or nonsense regions.
   - Low score (<60): Many boxes are on empty background, duplicated far from any element, or clearly not aligned with any UI structure.

3) DUPLICATION / REDUNDANCY (0–100)  
   Focus especially on SAME-CLASS duplications:

   - For NON-NESTABLE classes (for example: Text, Heading, Button, Checkbox, Radiobox, Switch, Slider, Text Input, Search Field, Image, Logo, Icon, etc.):
     - Two boxes of the same class that heavily overlap OR where one box is completely inside another usually indicate a problem.
     - Examples of bad cases:
       - A "Text" box entirely inside another "Text" box for the same text block.
       - A button annotated two or three times with slightly shifted boxes around the same visual button.
   - For container vs child classes (for example: Window + Button, Navigation Bar + Text, Card + Image):
     - It is OK for a container and its children to overlap (this is NOT a duplication).
   - High score (90–100): Almost no obviously redundant same-class boxes; each UI element is annotated once.
   - Medium score (60–80): Some elements appear to have 2 overlapping same-class boxes.
   - Low score (<60): Many same-class boxes stacked or nested on top of each other in the same location, especially for Text and Button.

4) LOCALIZATION / ALIGNMENT (0–100)  
   - Evaluate how well each bounding box fits its intended UI element.
   - Good annotation:
     - The box tightly covers the element, with small margins.
     - It does not cut off major parts of the element.
   - Penalize:
     - Boxes that are much larger than the element and include large amounts of unrelated background.
     - Boxes that cut off significant parts of the text/button/icon.
   - High score (90–100): Boxes match element boundaries closely.
   - Medium score (60–80): Some boxes are too loose or slightly misaligned, but the element is still clear.
   - Low score (<60): Many boxes are badly misaligned, cutting off elements or covering very wrong region sizes.

---------------------------
OVERALL SCORE
---------------------------

Combine the four dimensions into a single **overall_quality** score (0–100).  
Use approximately this weighting:
- Coverage / Missing: 40%
- False Positives:   25%
- Duplication:       20%
- Localization:      15%

Guidance:
- 90–100: Excellent annotation; safe to use as high-quality training data.
- 70–89: Good annotation; usable without manual review.
- 50–69: Borderline; has significant issues, might be filtered out if we want very clean data.
- 0–49: Bad annotation; should be excluded.

---------------------------
SPECIAL FAILURE PATTERNS TO PENALIZE HARD
---------------------------

Please penalize strongly when you see:

- Repeated "ghost" boxes:  
  The same bounding box shape appears multiple times in unrelated places in the screenshot (especially when those copies are not aligned with any visible element). This is a strong indicator of broken coordinates.

- Repeated same-class boxes over one element:  
  For example, several "Text" labels stacked over the same text string, or multiple overlapping "Button" boxes for one button.

- Missing entire regions:  
  For example, a large hero image, main headline, or obvious call-to-action button with no annotation at all.

- Small irrelevant repeated boxes:
    Many tiny boxes of the same class (for example, small "Icon" or "Image" boxes) scattered around the screenshot that do not correspond to any real UI elements.

When such failure patterns are frequent across the image, the **overall_quality** should usually be below 40.

---------------------------
OUTPUT FORMAT
---------------------------

Return your answer as **pure JSON**, with no extra commentary.

Use this schema:

{
  "short_summary":         "<one-line summary of annotation quality, e.g. 'Good coverage but many duplicate Text boxes in irrelevant areas'>",
  "coverage_score":        <number 0-100>,
  "false_positive_score":  <number 0-100>,
  "duplication_score":     <number 0-100>,
  "localization_score":    <number 0-100>,
  "overall_quality":       <number 0-100>
}

Start with "short_summary" first — briefly describe the main quality issues (or lack thereof) before scoring.

Do not include any other keys or text outside this JSON.'''

_QUALITY_USER_PROMPT = "Please evaluate the annotation quality of this visualization image."


# ================== UTILITIES ==================

def _load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ================== VLM ENGINE ==================

@dataclass
class QualityVLMConfig:
    """Configuration for quality scoring VLM."""
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_new_tokens: int = 512  # Quality JSON needs more tokens
    temperature: float = 0.0
    top_p: float = 1.0
    max_model_len: int | None = None
    limit_mm_per_prompt: dict | None = None


class QualityVLMEngine:
    """vLLM wrapper for quality scoring with single-image prompts."""

    def __init__(self, cfg: QualityVLMConfig):
        if not _HAS_VLLM:
            raise RuntimeError("vllm is not installed. Install with: pip install vllm")
        self.cfg = cfg
        self.llm = LLM(
            model=cfg.model,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype=cfg.dtype,
            trust_remote_code=True,
            tokenizer_mode="auto",
        )
        self.sampling = SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        # Prefer llm.apply_chat_template if present; fallback to tokenizer
        self._has_llm_act = hasattr(self.llm, "apply_chat_template")
        self._tokenizer = None if self._has_llm_act else self.llm.get_tokenizer()

    def _chat_template(self, messages: List[Dict[str, Any]]) -> str:
        if self._has_llm_act:
            return self.llm.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        return self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

    def build_request(self, viz_img: str) -> Dict[str, Any]:
        """
        Build a chat-templated prompt with a single image placeholder.
        """
        messages = [
            {"role": "system", "content": _QUALITY_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image"},  # placeholder for visualization image
                {"type": "text", "text": _QUALITY_USER_PROMPT},
            ]}
        ]
        prompt = self._chat_template(messages)
        return {
            "prompt": prompt,
            "multi_modal_data": {"image": [viz_img]},
        }

    def generate(self, requests: List[Dict[str, Any]]) -> List[str]:
        outputs = self.llm.generate(requests, self.sampling)
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]


# ================== JSON PARSING ==================

# More permissive regex that finds JSON-like objects (handles strings with special chars)
_JSON_RE = re.compile(r"\{[^{}]*(?:\"[^\"]*\"[^{}]*)*\}", re.DOTALL)


def _extract_quality_json(s: str) -> Optional[Dict[str, Any]]:
    """Extract and parse the quality scores JSON from model output."""
    if not s:
        return None
    s = s.strip()
    
    # Remove markdown code blocks if present
    if s.startswith("```"):
        s = re.sub(r"```(?:json)?", "", s).strip()
    
    # Try to find the JSON object - first attempt: look for balanced braces
    start_idx = s.find("{")
    if start_idx == -1:
        return None
    
    # Find matching closing brace
    depth = 0
    end_idx = -1
    for i, c in enumerate(s[start_idx:], start=start_idx):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    
    if end_idx == -1:
        # Fallback to regex
        m = _JSON_RE.search(s)
        if not m:
            return None
        json_str = m.group(0)
    else:
        json_str = s[start_idx:end_idx]
    
    try:
        obj = json.loads(json_str)
        if not isinstance(obj, dict):
            return None
        return obj
    except Exception:
        return None


def _safe_score(obj: Dict[str, Any], key: str, default: float = 0.0) -> float:
    """Safely extract a numeric score from the parsed JSON."""
    try:
        val = obj.get(key, default)
        return float(val)
    except (TypeError, ValueError):
        return default


# ================== MAIN SCORING FUNCTION ==================

@dataclass
class QualityScore:
    """Container for quality scores of a single visualization."""
    image_path: str
    short_summary: str
    coverage_score: float
    false_positive_score: float
    duplication_score: float
    localization_score: float
    overall_quality: float
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "short_summary": self.short_summary,
            "coverage_score": self.coverage_score,
            "false_positive_score": self.false_positive_score,
            "duplication_score": self.duplication_score,
            "localization_score": self.localization_score,
            "overall_quality": self.overall_quality,
        }


def score_visualizations(
    viz_dir: str = "../data/viz",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    batch_size: int = 256,
    tensor_parallel_size: int = 1,
    limit: Optional[int] = None,
    threshold: float = 70.0,
    output_file: Optional[str] = None,
    scores_json: Optional[str] = None,
    images_per_pass: int = 512,
    shard_index: int = 0,
    num_shards: int = 1,
    viz_pattern: str = "*.jpg",
) -> List[QualityScore]:
    """
    Score annotation quality for all visualization images in a directory.
    
    Args:
        viz_dir: Directory containing visualization images (with boxes/labels overlaid)
        model: VLM model to use for scoring
        batch_size: Number of images per VLM batch
        tensor_parallel_size: Tensor parallel size for multi-GPU
        limit: Limit number of images to process (for testing)
        threshold: Samples with overall_quality < threshold are marked as filtered
        output_file: Path to write filtered sample paths (one per line)
        scores_json: Path to write all scores as JSON
        images_per_pass: Number of images to process in each outer pass
        shard_index: Which shard this job processes (0-based)
        num_shards: Total number of shards
        viz_pattern: Glob pattern for visualization files (default: *.jpg)
    
    Returns:
        List of QualityScore objects for all processed images
    """
    
    def _iter_viz_images(viz_dir: str, pattern: str) -> Iterable[str]:
        for p in sorted(glob.glob(os.path.join(viz_dir, pattern))):
            yield p

    # Validate sharding
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not (0 <= shard_index < num_shards):
        raise ValueError(
            f"shard_index must be in [0, {num_shards-1}], got {shard_index}"
        )

    # Build full list and apply sharding
    all_images = list(_iter_viz_images(viz_dir, viz_pattern))
    if limit:
        all_images = all_images[:limit]
    
    # Slice to this shard's subset
    images = all_images[shard_index::num_shards]

    print(
        f"[Quality Score] Total images: {len(all_images)} | "
        f"num_shards={num_shards} shard_index={shard_index} -> "
        f"{len(images)} images for this job"
    )
    print(f"[Quality Score] Threshold: {threshold}")

    if not images:
        print("[Quality Score] No images found to process.")
        return []

    # Initialize VLM engine
    cfg = QualityVLMConfig(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=16384,
        limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
    )
    engine = QualityVLMEngine(cfg)

    # Accumulators
    all_scores: List[QualityScore] = []
    filtered_paths: List[str] = []

    # Process in outer passes
    for start in tqdm(range(0, len(images), images_per_pass), desc="Passes over images"):
        group = images[start:start + images_per_pass]

        all_reqs: List[Dict[str, Any]] = []
        all_paths: List[str] = []

        loop_start = time.time()
        for img_path in tqdm(group, desc="Building VLM requests (pass)", leave=False):
            # Validate image
            if not os.path.exists(img_path):
                continue
            try:
                with Image.open(img_path) as im:
                    W, H = im.size
                    if W < 10 or H < 10:
                        continue
                    # Skip extreme aspect ratios
                    if max(W, H) / min(W, H) > 50:
                        continue
            except Exception:
                continue

            req = engine.build_request(viz_img=img_path)
            all_reqs.append(req)
            all_paths.append(img_path)

        loop_elapsed = time.time() - loop_start
        print(f"Request building completed in {loop_elapsed:.2f}s ({len(all_reqs)} requests)")

        # Micro-batches inside this pass
        for i in tqdm(range(0, len(all_reqs), batch_size), desc="VLM batches (pass)", leave=False):
            batch_reqs = all_reqs[i:i + batch_size]
            batch_paths = all_paths[i:i + batch_size]
            
            texts = engine.generate(batch_reqs)
            
            for img_path, raw_text in zip(batch_paths, texts):
                obj = _extract_quality_json(raw_text)
                
                if obj is None:
                    # Failed to parse - assign low scores
                    score = QualityScore(
                        image_path=img_path,
                        short_summary="Failed to parse VLM response",
                        coverage_score=0.0,
                        false_positive_score=0.0,
                        duplication_score=0.0,
                        localization_score=0.0,
                        overall_quality=0.0,
                        raw_response=raw_text,
                    )
                else:
                    score = QualityScore(
                        image_path=img_path,
                        short_summary=str(obj.get("short_summary", "")).strip(),
                        coverage_score=_safe_score(obj, "coverage_score"),
                        false_positive_score=_safe_score(obj, "false_positive_score"),
                        duplication_score=_safe_score(obj, "duplication_score"),
                        localization_score=_safe_score(obj, "localization_score"),
                        overall_quality=_safe_score(obj, "overall_quality"),
                        raw_response=raw_text,
                    )
                
                all_scores.append(score)
                
                # Check against threshold
                if score.overall_quality < threshold:
                    filtered_paths.append(img_path)

            # Incremental save after each batch
            if output_file:
                _ensure_dir(os.path.dirname(output_file) or ".")
                with open(output_file, "w", encoding="utf-8") as f:
                    for p in filtered_paths:
                        f.write(p + "\n")
            
            if scores_json:
                _ensure_dir(os.path.dirname(scores_json) or ".")
                with open(scores_json, "w", encoding="utf-8") as f:
                    json.dump(
                        [s.to_dict() for s in all_scores],
                        f, ensure_ascii=False, indent=2
                    )

    # Final summary
    print("\n" + "=" * 60)
    print("QUALITY SCORING COMPLETE")
    print("=" * 60)
    print(f"Total images processed: {len(all_scores)}")
    print(f"Filtered (below threshold {threshold}): {len(filtered_paths)}")
    
    if all_scores:
        avg_overall = sum(s.overall_quality for s in all_scores) / len(all_scores)
        avg_coverage = sum(s.coverage_score for s in all_scores) / len(all_scores)
        avg_fp = sum(s.false_positive_score for s in all_scores) / len(all_scores)
        avg_dup = sum(s.duplication_score for s in all_scores) / len(all_scores)
        avg_loc = sum(s.localization_score for s in all_scores) / len(all_scores)
        
        print(f"\nAverage Scores:")
        print(f"  Overall Quality:    {avg_overall:.1f}")
        print(f"  Coverage:           {avg_coverage:.1f}")
        print(f"  False Positives:    {avg_fp:.1f}")
        print(f"  Duplication:        {avg_dup:.1f}")
        print(f"  Localization:       {avg_loc:.1f}")
    
    if output_file:
        print(f"\nFiltered samples saved to: {output_file}")
    if scores_json:
        print(f"All scores saved to: {scores_json}")

    return all_scores


def get_raw_image_paths_from_filtered(
    filtered_file: str,
    viz_suffix: str = ".viz.jpg",
    raw_suffix: str = ".png",
) -> List[str]:
    """
    Convert filtered visualization paths to corresponding raw image paths.
    
    This is useful when you want to get the original screenshot paths
    from the filtered visualization list.
    
    Args:
        filtered_file: Path to file with filtered visualization paths
        viz_suffix: Suffix used in visualization filenames
        raw_suffix: Suffix to use for raw image paths
    
    Returns:
        List of raw image paths
    """
    if not os.path.exists(filtered_file):
        return []
    
    raw_paths = []
    with open(filtered_file, "r", encoding="utf-8") as f:
        for line in f:
            viz_path = line.strip()
            if not viz_path:
                continue
            
            # Extract base name and convert suffix
            basename = os.path.basename(viz_path)
            if basename.endswith(viz_suffix):
                base = basename[:-len(viz_suffix)]
                raw_name = base + raw_suffix
                # Assume raw is in sibling 'raw' directory
                viz_dir = os.path.dirname(viz_path)
                parent = os.path.dirname(viz_dir)
                raw_path = os.path.join(parent, "raw", raw_name)
                raw_paths.append(raw_path)
    
    return raw_paths
