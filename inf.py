"""Inference script for nanoNext with YAML-driven configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from nanoNext.config import GenerationConfig, load_generation_config
from nanoNext.inference import generate, load_checkpoint

DEFAULT_GENERATION_CONFIG = Path("config/inference/default.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with nanoNext")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--inference-config", type=Path, default=DEFAULT_GENERATION_CONFIG, help="Path to inference config YAML")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Override top-k sampling")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    return parser.parse_args()


def apply_overrides(config: GenerationConfig, args: argparse.Namespace) -> GenerationConfig:
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.device is not None:
        config.device = args.device
    return config


def main():
    args = parse_args()
    gen_config: GenerationConfig = load_generation_config(args.inference_config)
    gen_config = apply_overrides(gen_config, args)

    device = torch.device(gen_config.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and gen_config.device != "cpu":
        print("CUDA not available, falling back to CPU")

    model, checkpoint = load_checkpoint(args.checkpoint, device)
    print(f"Loaded checkpoint from step {checkpoint['step']}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    prompt_tokens = torch.tensor(
        [[ord(c) % model.config.vocab_size for c in args.prompt if 0 <= ord(c) < 256]],
        dtype=torch.long,
        device=device,
    )

    print(f"Prompt: '{args.prompt}'")
    print(
        f"Generating {gen_config.max_tokens} tokens (temp={gen_config.temperature}, top_k={gen_config.top_k})..."
    )

    generated, stats = generate(model, prompt_tokens, gen_config, device=device)

    generated_tokens = generated[0, prompt_tokens.shape[1] :].tolist()
    generated_text = "".join(chr(t) if t < 256 else "?" for t in generated_tokens)

    print(f"\nPrefill time: {stats.prefill_time_s*1000:.1f} ms | speed: {stats.prefill_tokens_per_sec:.0f} tok/s")
    print(f"Decode time: {stats.decode_time_s*1000:.1f} ms | speed: {stats.decode_tokens_per_sec:.0f} tok/s")

    print(f"\nGenerated text:\n'{generated_text}'")
    print(f"\nComplete output: '{args.prompt}{generated_text}'")


if __name__ == "__main__":
    main()
