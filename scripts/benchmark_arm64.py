#!/usr/bin/env python3
"""Benchmark uncached and integer-FSQ-cache SUVI Phase 3 caption inference.

Author: Nakul Vyas
Organization: Heysuvi Labs, LLC
Contact: nvyas@heysuvi.com
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from suvi_arm.phase3 import CHANNEL_LEVELS, VISUAL_TOKEN_COUNT, build_prefix, image_to_codes, load_bridge

LLM_ID = "mistralai/Mistral-7B-v0.1"
VAE_ID = "stabilityai/sd-vae-ft-mse"
DEVICE = torch.device("cpu")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--prompt", default="View:")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--llm-dtype", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


def digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def images(image_dir: Path) -> list[Path]:
    result = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not result:
        raise ValueError("Image directory contains no JPG or PNG files.")
    return result


def pixels(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((256, 256))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return (torch.from_numpy(array).permute(2, 0, 1).sub(0.5).div(0.5).unsqueeze(0))


def cache_path(cache_dir: Path, image: Path) -> Path:
    return cache_dir / f"{image.stem}.fsq-u8"


def validate_codes(codes: torch.Tensor) -> None:
    if tuple(codes.shape) != (1, VISUAL_TOKEN_COUNT, 4):
        raise ValueError(f"Unexpected FSQ shape: {tuple(codes.shape)}")
    for channel, levels in enumerate(CHANNEL_LEVELS):
        if codes[..., channel].min().item() < 0 or codes[..., channel].max().item() >= levels:
            raise ValueError(f"Invalid ID in FSQ channel {channel}")


def create_cache(vae, paths: list[Path], cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in paths:
        start = time.perf_counter()
        codes = image_to_codes(vae, pixels(path).to(DEVICE))
        elapsed = time.perf_counter() - start
        validate_codes(codes)
        payload = codes.squeeze(0).to(torch.uint8).cpu().numpy().tobytes()
        target = cache_path(cache_dir, path)
        target.write_bytes(payload)
        # Fresh derivation is mandatory: never compare an entry only to itself.
        fresh = image_to_codes(vae, pixels(path).to(DEVICE))
        restored = torch.from_numpy(np.frombuffer(target.read_bytes(), dtype=np.uint8).copy()).reshape(1, 256, 4).long()
        if not torch.equal(restored, fresh.cpu()):
            raise ValueError(f"Cache mismatch for {path.name}")
        records.append({"image": path.name, "image_sha256": file_digest(path), "codes_sha256": digest(fresh), "cache_file": target.name, "cache_bytes": target.stat().st_size, "cache_creation_seconds": elapsed})
    return records


def cached_codes(cache_dir: Path, path: Path) -> torch.LongTensor:
    raw = cache_path(cache_dir, path).read_bytes()
    if len(raw) != 1024:
        raise ValueError(f"Malformed cache size for {path.name}: {len(raw)}")
    codes = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy()).reshape(1, 256, 4).long()
    validate_codes(codes)
    return codes


def load_runtime(adapter: Path, dtype: torch.dtype):
    llm = AutoModelForCausalLM.from_pretrained(LLM_ID, torch_dtype=dtype, low_cpu_mem_usage=True).to(DEVICE)
    llm = PeftModel.from_pretrained(llm, adapter, is_trainable=False).eval()
    # Phase 3 does not extend Mistral's vocabulary. Loading the canonical
    # base tokenizer avoids relying on stale adapter-side tokenizer metadata.
    tokenizer = AutoTokenizer.from_pretrained(LLM_ID, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float32).to(DEVICE).eval()
    bridge, checksum = load_bridge(adapter, DEVICE)
    return llm, tokenizer, vae, bridge, checksum


def generate(llm, tokenizer, bridge, codes: torch.Tensor, prompt: str, max_tokens: int, dtype: torch.dtype) -> dict:
    start = time.perf_counter()
    prefix = build_prefix(codes.to(DEVICE), bridge)
    bridge_seconds = time.perf_counter() - start
    prompt_tokens = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").to(DEVICE)
    start = time.perf_counter()
    embeds = torch.cat([prefix, llm.get_input_embeddings()(prompt_tokens.input_ids).float()], dim=1).to(dtype=dtype)
    mask = torch.cat([torch.ones(prefix.shape[:2], dtype=prompt_tokens.attention_mask.dtype), prompt_tokens.attention_mask], dim=1)
    with torch.inference_mode():
        generated = llm.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=max_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    generation_seconds = time.perf_counter() - start
    token_ids = generated[0, -max_tokens:].tolist()
    if tokenizer.eos_token_id in token_ids:
        token_ids = token_ids[:token_ids.index(tokenizer.eos_token_id)]
    return {"prefix_sha256": digest(prefix), "bridge_seconds": bridge_seconds, "generation_seconds": generation_seconds, "generated_token_ids": token_ids, "generated_text": tokenizer.decode(token_ids, skip_special_tokens=True).strip()}


def one_run(condition, llm, tokenizer, vae, bridge, cache_dir, path, config):
    preprocess_seconds = tokenization_seconds = cache_read_seconds = 0.0
    if condition == "uncached":
        start = time.perf_counter(); image = pixels(path).to(DEVICE); preprocess_seconds = time.perf_counter() - start
        start = time.perf_counter(); codes = image_to_codes(vae, image); tokenization_seconds = time.perf_counter() - start
    else:
        start = time.perf_counter(); codes = cached_codes(cache_dir, path); cache_read_seconds = time.perf_counter() - start
    result = generate(llm, tokenizer, bridge, codes, config.prompt, config.max_new_tokens, config.dtype)
    return {"condition": condition, "image": path.name, "preprocess_seconds": preprocess_seconds, "tokenization_seconds": tokenization_seconds, "cache_read_seconds": cache_read_seconds, **result, "total_seconds": preprocess_seconds + tokenization_seconds + cache_read_seconds + result["bridge_seconds"] + result["generation_seconds"]}


def distributions(runs: list[dict]) -> dict:
    def summarise(field):
        values = sorted(run[field] for run in runs)
        return {"median_seconds": statistics.median(values), "p95_seconds": values[math.ceil(len(values) * .95) - 1]}
    return {field: summarise(field) for field in ("preprocess_seconds", "tokenization_seconds", "cache_read_seconds", "bridge_seconds", "generation_seconds", "total_seconds")}


def main() -> None:
    config = args()
    if config.warmups < 0 or config.repetitions < 1 or config.threads < 1:
        raise ValueError("warmups >= 0, repetitions >= 1, and threads >= 1 are required")
    config.dtype = getattr(torch, config.llm_dtype)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = images(config.image_dir)
    llm, tokenizer, vae, bridge, bridge_checksum = load_runtime(config.adapter_path, config.dtype)
    cache = create_cache(vae, paths, config.artifact_dir / "fsq_cache")
    for _ in range(config.warmups):
        for path in paths:
            one_run("uncached", llm, tokenizer, vae, bridge, config.artifact_dir / "fsq_cache", path, config)
            one_run("cached", llm, tokenizer, vae, bridge, config.artifact_dir / "fsq_cache", path, config)
    uncached, cached = [], []
    for repeat in range(config.repetitions):
        for path in paths:
            uncached.append({**one_run("uncached", llm, tokenizer, vae, bridge, config.artifact_dir / "fsq_cache", path, config), "repetition": repeat})
            cached.append({**one_run("cached", llm, tokenizer, vae, bridge, config.artifact_dir / "fsq_cache", path, config), "repetition": repeat})
    equivalent = all(a["image"] == b["image"] and a["prefix_sha256"] == b["prefix_sha256"] and a["generated_token_ids"] == b["generated_token_ids"] for a, b in zip(uncached, cached))
    result = {"contract": "phase3_factorized_fsq_inputs_embeds", "bridge_checksum": bridge_checksum, "hardware": {"architecture": platform.machine(), "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "threads": config.threads, "llm_dtype": config.llm_dtype}, "prompt": config.prompt, "max_new_tokens": config.max_new_tokens, "warmups": config.warmups, "repetitions": config.repetitions, "cache": cache, "cache_fresh_id_equivalence": True, "cached_matches_uncached": equivalent, "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "uncached_summary": distributions(uncached), "cached_summary": distributions(cached), "uncached_runs": uncached, "cached_runs": cached}
    (config.artifact_dir / "benchmark_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    if not equivalent:
        raise RuntimeError("Cached and uncached paths diverged; metrics are not valid.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
