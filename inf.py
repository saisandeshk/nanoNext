"""
Inference script for nanoNext with proper prefill and decode phases.

Two-phase generation:
1. Prefill: Process entire prompt at once, populate cache
2. Decode: Generate tokens one-by-one using cache

Clean and minimal, inspired by nanoGPT.
"""

from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import torch

from nanoNext.cache import DynamicCache
from nanoNext.config import NanoNextConfig
from nanoNext.model import NanoNextForCausalLM

# Suppress FLA format warnings (false positive for short sequences)
warnings.filterwarnings('ignore', message='.*Input tensor shape suggests potential format mismatch.*')


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[NanoNextForCausalLM, dict]:
    """Load model from checkpoint."""
    # weights_only=False is safe here since we're loading our own checkpoints
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = NanoNextForCausalLM(config).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


@torch.no_grad()
def generate(
    model: NanoNextForCausalLM,
    prompt: torch.LongTensor,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: int = 200,
    device: torch.device = None,
) -> torch.LongTensor:
    """
    Generate tokens using two-phase approach: prefill + decode.
    
    Args:
        model: The language model
        prompt: Input token IDs (batch, seq_len)
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (1.0 = normal, <1 = focused, >1 = diverse)
        top_k: Sample from top-k tokens only
        device: Device to run on
    
    Returns:
        Generated token IDs (batch, seq_len + max_new_tokens)
    """
    device = device or next(model.parameters()).device
    prompt = prompt.to(device)
    batch_size, prompt_len = prompt.shape
    
    # Initialize cache for efficient generation
    cache = DynamicCache(model.config)
    
    # Warmup: Compile CUDA kernels (first call is slow due to JIT)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        _ = model.model(
            input_ids=prompt,
            cache_params=cache,
            cache_position=torch.arange(prompt_len, device=device),
        )
    
    # Reset cache after warmup
    cache = DynamicCache(model.config)
    
    # Phase 1: PREFILL - Process entire prompt at once
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        hidden_states = model.model(
            input_ids=prompt,
            cache_params=cache,
            cache_position=torch.arange(prompt_len, device=device),
        )
        logits = model.lm_head(hidden_states[:, -1:, :])
    
    torch.cuda.synchronize()
    prefill_time = time.perf_counter() - t0
    prefill_toks_per_sec = prompt_len / prefill_time
    print(f"prefill | {prompt_len} tokens in {prefill_time*1000:.1f}ms ({prefill_toks_per_sec:.0f} tok/s)")  # Only last token
    
    # Sample first token
    if temperature == 0:
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    else:
        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    
    generated = torch.cat([prompt, next_token], dim=1)
    
    # Phase 2: DECODE - Generate one token at a time with cache
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    for i in range(max_new_tokens - 1):
        cache_position = torch.tensor([prompt_len + i], device=device)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            hidden_states = model.model(
                input_ids=next_token,
                cache_params=cache,
                cache_position=cache_position,
            )
            logits = model.lm_head(hidden_states[:, -1:, :])
        
        # Sample next token
        if temperature == 0:
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        else:
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        generated = torch.cat([generated, next_token], dim=1)
    
    torch.cuda.synchronize()
    decode_time = time.perf_counter() - t0
    decode_toks_per_sec = (max_new_tokens - 1) / decode_time if decode_time > 0 else 0
    print(f"decode | {max_new_tokens - 1} tokens in {decode_time*1000:.1f}ms ({decode_toks_per_sec:.0f} tok/s)")
    
    return generated


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text with nanoNext")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=200, help="Top-k sampling")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint: {args.checkpoint}")
    
    # Load model
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    print(f"Loaded model from step {checkpoint['step']}")
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Simple character-level tokenization (matches training)
    prompt_tokens = torch.tensor(
        [[ord(c) % model.config.vocab_size for c in args.prompt if 0 <= ord(c) < 256]],
        dtype=torch.long,
        device=device
    )
    
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Generating {args.max_tokens} tokens (temp={args.temperature}, top_k={args.top_k})...")
    
    # Generate
    output = generate(
        model,
        prompt_tokens,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device,
    )
    
    # Decode output
    generated_tokens = output[0, prompt_tokens.shape[1]:].tolist()
    generated_text = ''.join(chr(t) if t < 256 else '?' for t in generated_tokens)
    
    print(f"\nGenerated text:")
    print(f"'{generated_text}'")
    print(f"\nComplete output: '{args.prompt}{generated_text}'")


if __name__ == "__main__":
    main()

