#!/usr/bin/env python3
"""Measure a Mistral-7B dual-tower VLM on Arm64 for resource comparison.

This is intentionally a deployment-resource comparator, not a quality or
caption-equivalence benchmark. LLaVA and the SUVI Phase 3 checkpoint were
trained independently. Both are measured in fresh processes on the same VM,
with the same local image manifest, user text, generation cap, and repetition
policy.

Author: Nakul Vyas
Organization: Heysuvi Labs, LLC
Contact: nvyas@heysuvi.com
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaNextForConditionalGeneration

MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
DEVICE = torch.device("cpu")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise ValueError("Image directory contains no supported images.")
    return paths


def peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def load(dtype: torch.dtype):
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(DEVICE).eval()
    return model, processor, time.perf_counter() - start


def infer(model, processor, path: Path, prompt: str, max_new_tokens: int) -> dict:
    image = Image.open(path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    start = time.perf_counter()
    inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)
    preprocess_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    generation_seconds = time.perf_counter() - start
    generated_ids = generated[0, inputs.input_ids.shape[1]:]
    return {
        "image": path.name,
        "preprocess_seconds": preprocess_seconds,
        "generation_seconds": generation_seconds,
        "total_seconds": preprocess_seconds + generation_seconds,
        "generated_token_ids": generated_ids.tolist(),
        "generated_text": processor.decode(generated_ids, skip_special_tokens=True).strip(),
    }


def summary(runs: list[dict]) -> dict:
    def values(key):
        sorted_values = sorted(run[key] for run in runs)
        return {"median_seconds": statistics.median(sorted_values), "p95_seconds": sorted_values[-1]}
    return {key: values(key) for key in ("preprocess_seconds", "generation_seconds", "total_seconds")}


def main() -> None:
    config = args()
    if config.warmups < 0 or config.repetitions < 1 or config.threads < 1:
        raise ValueError("warmups >= 0, repetitions >= 1, and threads >= 1 are required")
    dtype = getattr(torch, config.dtype)
    torch.set_num_threads(config.threads)
    paths = image_paths(config.image_dir)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    model, processor, load_seconds = load(dtype)
    for _ in range(config.warmups):
        for path in paths:
            infer(model, processor, path, config.prompt, config.max_new_tokens)
    runs = []
    for repeat in range(config.repetitions):
        for path in paths:
            runs.append({**infer(model, processor, path, config.prompt, config.max_new_tokens), "repetition": repeat})
    deterministic = all(
        len({tuple(run["generated_token_ids"]) for run in runs if run["image"] == path.name}) == 1
        for path in paths
    )
    output = {
        "role": "dual_tower_resource_comparator",
        "model": MODEL_ID,
        "architecture": "CLIP vision encoder plus multimodal projector plus Mistral-7B decoder",
        "comparison_limit": "Resource comparison only; not a quality-equivalent benchmark against independently trained SUVI.",
        "hardware": {"architecture": platform.machine(), "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "threads": config.threads, "dtype": config.dtype},
        "prompt": config.prompt,
        "max_new_tokens": config.max_new_tokens,
        "warmups": config.warmups,
        "repetitions": config.repetitions,
        "model_load_seconds": load_seconds,
        "peak_rss_bytes": peak_rss_bytes(),
        "deterministic_repeats": deterministic,
        "summary": summary(runs),
        "runs": runs,
    }
    (config.artifact_dir / "dual_tower_metrics.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
