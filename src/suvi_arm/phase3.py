"""Frozen Phase 3 visual-tokenizer and bridge contract used by the benchmark.

Author: Nakul Vyas
Organization: Heysuvi Labs, LLC
Contact: nvyas@heysuvi.com
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

CHANNEL_LEVELS = (8, 8, 16, 16)
VISUAL_TOKEN_COUNT = 256
BRIDGE_CONTRACT = {
    "schema_version": 3,
    "interface": "factorized_fsq_inputs_embeds",
    "vae_dtype": "float32",
    "pooling_dtype": "float32",
    "quantization_dtype": "float32",
    "rounding": "torch.round (ties-to-even)",
    "latent_patch_size": 2,
    "channel_levels": list(CHANNEL_LEVELS),
    "visual_token_count": VISUAL_TOKEN_COUNT,
    "position_grid": [16, 16],
    "caption_eos": "explicit",
    "multimodal_input_dtype": "float32",
}


def state_checksum(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_bridge(adapter_path: Path, device: torch.device) -> tuple[dict[str, torch.Tensor], str]:
    contract = json.loads((adapter_path / "visual_bridge_contract.json").read_text())
    observed = {key: contract.get(key) for key in BRIDGE_CONTRACT}
    if observed != BRIDGE_CONTRACT:
        raise ValueError("Checkpoint visual-bridge contract is not Phase 3.")
    state = torch.load(adapter_path / "visual_bridge.pt", map_location="cpu", weights_only=True)
    checksum = state_checksum(state)
    if checksum != contract.get("sha256"):
        raise ValueError("Visual-bridge checksum mismatch.")
    return {name: tensor.to(device) for name, tensor in state.items()}, checksum


def image_to_codes(vae, pixels: torch.Tensor) -> torch.LongTensor:
    """Exact float32 VAE -> pooled channel-wise FSQ IDs, Bx256x4."""
    with torch.inference_mode():
        latents = vae.encode(pixels.float()).latent_dist.mode() * 0.18215
        pooled = F.avg_pool2d(latents, kernel_size=2, stride=2)
        bounded = torch.sigmoid(pooled)
        channels = [
            torch.round(bounded[:, channel] * (levels - 1)).long()
            for channel, levels in enumerate(CHANNEL_LEVELS)
        ]
        return torch.stack(channels, dim=-1).flatten(1, 2)


def build_prefix(codes: torch.LongTensor, bridge: dict[str, torch.Tensor]) -> torch.Tensor:
    if tuple(codes.shape[1:]) != (VISUAL_TOKEN_COUNT, len(CHANNEL_LEVELS)):
        raise ValueError(f"Expected codes shaped Bx256x4, got {tuple(codes.shape)}")
    visual = sum(
        F.embedding(codes[..., channel], bridge[f"codebooks.{channel}.weight"])
        for channel in range(len(CHANNEL_LEVELS))
    )
    return visual + bridge["position_embeddings.weight"].unsqueeze(0)
