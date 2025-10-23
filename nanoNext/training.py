"""Training utilities for nanoNext."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterator, Optional

import torch
import torch.optim as optim
from datasets import load_dataset

from .config import NanoNextConfig, TrainingConfig
from .model import NanoNextForCausalLM


def build_dataset(
    dataset_name: str,
    dataset_config: str,
    seq_len: int,
    batch: int,
    vocab_size: int,
    split: str = "train",
) -> Iterator[torch.Tensor]:
    """Stream a Hugging Face dataset and yield batches of token IDs."""
    stream = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    buffer = torch.empty(0, dtype=torch.long)

    for item in stream:
        text = item.get("text")
        if not text:
            continue
        tokens = torch.tensor(
            [ord(c) % vocab_size for c in text if 0 <= ord(c) < 256],
            dtype=torch.long,
        )
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
    *,
    device: Optional[torch.device] = None,
) -> float:
    """Evaluate model and return the average loss over the requested number of steps."""
    device = device or next(model.parameters()).device
    model.eval()

    data_iter = build_dataset(
        dataset_name,
        dataset_config,
        seq_len,
        batch,
        model.config.vocab_size,
        split="validation",
    )
    total_loss = 0.0
    eval_steps = 0

    autocast_enabled = device.type == "cuda"
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            for step, batch_tokens in enumerate(data_iter, start=1):
                inputs = batch_tokens.to(device)
                _, loss = model(inputs, inputs)
                total_loss += loss.item()
                eval_steps += 1
                if step >= steps:
                    break

    avg_loss = total_loss / eval_steps if eval_steps > 0 else float("inf")
    model.train()
    return avg_loss


def save_checkpoint(
    model: NanoNextForCausalLM,
    optimizer: optim.Optimizer,
    step: int,
    checkpoint_dir: str,
) -> Path:
    """Persist model, optimizer, and config state for later reuse."""
    checkpoint_path = Path(checkpoint_dir) / f"step_{step}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model.config,
        },
        checkpoint_path,
    )
    return checkpoint_path


def train_model(
    model_config: NanoNextConfig,
    train_config: TrainingConfig,
    *,
    device: Optional[torch.device] = None,
    model: Optional[NanoNextForCausalLM] = None,
    log: Callable[[str], None] = print,
) -> NanoNextForCausalLM:
    """Run a simple nanoNext training loop suitable for educational experiments."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = NanoNextForCausalLM(model_config)
    target_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=target_dtype)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
    log(f"Using dtype: {next(model.parameters()).dtype}")

    data_iter = build_dataset(
        train_config.dataset,
        train_config.dataset_config,
        train_config.seq_len,
        train_config.batch,
        model_config.vocab_size,
    )
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    model.train()
    steps = train_config.steps
    autocast_enabled = device.type == "cuda"

    for step, batch_tokens in enumerate(data_iter, start=1):
        inputs = batch_tokens.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            _, loss = model(inputs, inputs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        train_ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")
        message = f"step {step}: train_loss={loss.item():.4f} train_ppl={train_ppl:.2f}"

        if train_config.eval_freq and step % train_config.eval_freq == 0:
            eval_loss = evaluate_model(
                model,
                train_config.dataset,
                train_config.dataset_config,
                train_config.seq_len,
                train_config.batch,
                train_config.eval_steps,
                device=device,
            )
            eval_ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
            message += f" eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"

        if train_config.checkpoint_interval and step % train_config.checkpoint_interval == 0:
            checkpoint_path = save_checkpoint(model, optimizer, step, train_config.checkpoint_dir)
            message += f" | checkpoint saved to {checkpoint_path}"

        log(message)
        if step >= steps:
            break

    save_checkpoint(model, optimizer, steps, train_config.checkpoint_dir)
    return model


__all__ = [
    "build_dataset",
    "evaluate_model",
    "save_checkpoint",
    "train_model",
]
