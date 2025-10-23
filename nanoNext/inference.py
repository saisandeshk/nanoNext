"""Inference utilities for nanoNext."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch

from .cache import DynamicCache
from .config import GenerationConfig, NanoNextConfig
from .model import NanoNextForCausalLM


@dataclass
class GenerationStats:
    """Simple timing metrics captured during generation."""

    prefill_time_s: float
    prefill_tokens_per_sec: float
    decode_time_s: float
    decode_tokens_per_sec: float


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[NanoNextForCausalLM, dict]:
    """Load a model checkpoint and move it to the requested device."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_obj = checkpoint["config"]
    if isinstance(config_obj, NanoNextConfig):
        config = config_obj
    elif isinstance(config_obj, dict):
        config = NanoNextConfig(**config_obj)
    else:
        raise TypeError("Unsupported config format in checkpoint")

    model = NanoNextForCausalLM(config).to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


@torch.no_grad()
def generate(
    model: NanoNextForCausalLM,
    prompt: torch.LongTensor,
    config: GenerationConfig,
    *,
    device: Optional[torch.device] = None,
    log: Callable[[str], None] = print,
) -> tuple[torch.LongTensor, GenerationStats]:
    """Generate tokens with a prefill + decode schedule.

    Returns the generated tensor alongside basic performance statistics.
    """
    device = device or next(model.parameters()).device
    prompt = prompt.to(device)
    batch_size, prompt_len = prompt.shape

    cache = DynamicCache(model.config)
    autocast_enabled = device.type == "cuda"

    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        _ = model.model(
            input_ids=prompt,
            cache_params=cache,
            cache_position=torch.arange(prompt_len, device=device),
        )

    cache = DynamicCache(model.config)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        hidden_states = model.model(
            input_ids=prompt,
            cache_params=cache,
            cache_position=torch.arange(prompt_len, device=device),
        )
        logits = model.lm_head(hidden_states[:, -1:, :])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    prefill_time = time.perf_counter() - t0
    prefill_toks_per_sec = prompt_len / prefill_time if prefill_time > 0 else float("inf")
    log(f"prefill | {prompt_len} tokens in {prefill_time*1000:.1f}ms ({prefill_toks_per_sec:.0f} tok/s)")

    if config.temperature == 0:
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    else:
        logits = logits[:, -1, :] / config.temperature
        if config.top_k > 0:
            values, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    generated = torch.cat([prompt, next_token], dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for i in range(config.max_tokens - 1):
        cache_position = torch.tensor([prompt_len + i], device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            hidden_states = model.model(
                input_ids=next_token,
                cache_params=cache,
                cache_position=cache_position,
            )
            logits = model.lm_head(hidden_states[:, -1:, :])

        if config.temperature == 0:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        else:
            logits = logits[:, -1, :] / config.temperature
            if config.top_k > 0:
                values, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
                logits[logits < values[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    decode_time = time.perf_counter() - t0
    decode_tokens = max(config.max_tokens - 1, 1)
    decode_toks_per_sec = decode_tokens / decode_time if decode_time > 0 else float("inf")
    log(
        f"decode | {decode_tokens} tokens in {decode_time*1000:.1f}ms ({decode_toks_per_sec:.0f} tok/s)"
    )

    stats = GenerationStats(
        prefill_time_s=prefill_time,
        prefill_tokens_per_sec=prefill_toks_per_sec,
        decode_time_s=decode_time,
        decode_tokens_per_sec=decode_toks_per_sec,
    )
    return generated, stats


__all__ = [
    "GenerationStats",
    "generate",
    "load_checkpoint",
]
