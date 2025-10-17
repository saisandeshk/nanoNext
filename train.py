"""Minimal training loop for nanoNext with Hugging Face datasets."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.optim as optim
from datasets import load_dataset

from nanoNext.config import NanoNextConfig
from nanoNext.model import NanoNextForCausalLM
from nanoNext.utils import is_causal_conv1d_available, is_flash_linear_attention_available


def build_dataset(dataset_name: str, dataset_config: str, seq_len: int, batch: int, vocab_size: int, split: str = "train") -> Iterable[torch.Tensor]:
    stream = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    buffer = torch.empty(0, dtype=torch.long)

    for item in stream:
        text = item.get("text")
        if not text:
            continue
        tokens = torch.tensor([ord(c) % vocab_size for c in text if 0 <= ord(c) < 256], dtype=torch.long)
        if tokens.numel() == 0:
            continue
        buffer = torch.cat([buffer, tokens])
        while buffer.numel() >= seq_len * batch:
            batch_tokens = buffer[: seq_len * batch]
            buffer = buffer[seq_len * batch :]
            yield batch_tokens.view(batch, seq_len)


def evaluate_model(
    model: NanoNextForCausalLM,
    dataset_name: str,
    dataset_config: str,
    seq_len: int,
    batch: int,
    steps: int,
    device: torch.device | None = None,
) -> float:
    """Evaluate model and return average loss over eval steps."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    data_iter = build_dataset(dataset_name, dataset_config, seq_len, batch, model.config.vocab_size, split="validation")
    total_loss = 0.0
    eval_steps = 0

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for step, batch_tokens in enumerate(data_iter):
                inputs = batch_tokens.to(device)
                logits, loss = model(inputs, inputs)
                total_loss += loss.item()
                eval_steps += 1
                if step + 1 >= steps:
                    break

    avg_loss = total_loss / eval_steps if eval_steps > 0 else float("inf")
    model.train()
    return avg_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train nanoNext on a Hugging Face dataset")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Hugging Face dataset name")
    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
    parser.add_argument("--steps", type=int, default=100, help="Training steps")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--eval-steps", type=int, default=5, help="Evaluation steps per eval")
    parser.add_argument("--eval-freq", type=int, default=50, help="Evaluate every N training steps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--benchmark", action="store_true", help="Run kernel benchmarks before training")
    parser.add_argument("--benchmark-only", action="store_true", help="Run benchmarks and exit without training")
    parser.add_argument("--visualize", action="store_true", help="Run benchmarks with matplotlib visualization (10 scales)")
    parser.add_argument("--full-model", action="store_true", help="Use full 96-layer config (default: 2-layer demo)")
    return parser.parse_args()


def save_checkpoint(model: NanoNextForCausalLM, optimizer: optim.Optimizer, step: int, checkpoint_dir: str):
    """Save model checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_dir) / f"step_{step}.pt"
    
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config,
    }, checkpoint_path)
    
    print(f"  checkpoint saved: {checkpoint_path}")


def train_model(
    config: NanoNextConfig,
    dataset_name: str,
    dataset_config: str,
    seq_len: int,
    batch: int,
    steps: int,
    eval_steps: int = 10,
    eval_freq: int = 10,
    checkpoint_dir: str = "checkpoints",
    checkpoint_interval: int = 100,
    device: torch.device | None = None,
    model: NanoNextForCausalLM | None = None,
) -> NanoNextForCausalLM:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model with bfloat16 dtype for FLA compatibility
    if model is None:
        model = NanoNextForCausalLM(config)
    model = model.to(device=device, dtype=torch.bfloat16)

    # Count and display model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
    print(f"Using dtype: {next(model.parameters()).dtype}")

    data_iter = build_dataset(dataset_name, dataset_config, seq_len, batch, config.vocab_size)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    # Use GradScaler for mixed precision (though bfloat16 doesn't need it typically)
    scaler = torch.amp.GradScaler('cuda', enabled=False)  # bfloat16 doesn't need gradient scaling

    model.train()
    for step, batch_tokens in enumerate(data_iter):
        inputs = batch_tokens.to(device)
        
        # Use autocast for mixed precision
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, loss = model(inputs, inputs)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")

        # Evaluate periodically
        eval_loss_str = ""
        if (step + 1) % eval_freq == 0:
            eval_loss = evaluate_model(model, dataset_name, dataset_config, seq_len, batch, eval_steps, device)
            eval_ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
            eval_loss_str = f" eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"

        # Save checkpoint periodically
        if (step + 1) % checkpoint_interval == 0:
            save_checkpoint(model, optimizer, step + 1, checkpoint_dir)

        print(f"step {step}: train_loss={loss.item():.4f} train_ppl={train_ppl:.2f}{eval_loss_str}")
        if step + 1 >= steps:
            break
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, steps, checkpoint_dir)
    return model


def train():
    args = parse_args()

    # Check for required dependencies
    if not is_flash_linear_attention_available():
        raise ImportError(
            "Flash Linear Attention (flash-linear-attention >= 0.2.2) is required for training. "
            "Install with: pip install flash-linear-attention>=0.2.2"
        )

    if not is_causal_conv1d_available():
        raise ImportError(
            "Causal Conv1D is required for training. "
            "Install with: pip install causal-conv1d>=1.4.0"
        )

    # Run benchmarks if requested
    if args.visualize:
        from nanoNext.benchmark import visualize_benchmarks
        visualize_benchmarks(device="cuda", dtype=torch.bfloat16, save_path="benchmark_results.png")
        return
    elif args.benchmark or args.benchmark_only:
        from nanoNext.benchmark import run_all_benchmarks
        run_all_benchmarks(device="cuda", dtype=torch.bfloat16, num_precision_runs=3, num_benchmark_runs=10)
        if args.benchmark_only:
            return

    # Config selection
    if args.full_model:
        # Full 96-layer model (2x Qwen3-Next 80B which had 48 layers)
        config = NanoNextConfig()
        print(f"\nUsing FULL model: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
    else:
        # Tiny demo model for quick testing
        config = NanoNextConfig(
            num_hidden_layers=2,
            num_experts=4,
            num_experts_per_tok=2,
            hidden_size=256,
            intermediate_size=512,
        )
        print(f"\nUsing demo model: {config.num_hidden_layers} layers (use --full-model for 96 layers)")
    
    train_model(
        config,
        args.dataset,
        args.dataset_config,
        args.seq_len,
        args.batch,
        args.steps,
        args.eval_steps,
        args.eval_freq,
        args.checkpoint_dir,
        args.checkpoint_interval,
    )


if __name__ == "__main__":
    train()


