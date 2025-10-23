"""Minimal training loop for nanoNext with YAML-driven configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch

from nanoNext.config import (
    NanoNextConfig,
    TrainingConfig,
    load_model_config,
    load_training_config,
)
from nanoNext.training import train_model
from nanoNext.utils import is_causal_conv1d_available, is_flash_linear_attention_available

DEFAULT_MODEL_CONFIG = Path("config/models/demo.yaml")
FULL_MODEL_CONFIG = Path("config/models/full.yaml")
DEFAULT_TRAIN_CONFIG = Path("config/train/demo.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train nanoNext on a Hugging Face dataset")
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="Path to model config YAML")
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG, help="Path to training config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--dataset-config", type=str, default=None, help="Override dataset configuration")
    parser.add_argument("--steps", type=int, default=None, help="Override number of training steps")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override evaluation steps")
    parser.add_argument("--eval-freq", type=int, default=None, help="Override evaluation frequency")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Override checkpoint directory")
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=None,
        help="Override checkpoint interval",
    )
    parser.add_argument("--benchmark", action="store_true", help="Run kernel benchmarks before training")
    parser.add_argument("--benchmark-only", action="store_true", help="Run benchmarks and exit without training")
    parser.add_argument("--visualize", action="store_true", help="Run benchmarks with matplotlib visualization")
    parser.add_argument("--full-model", action="store_true", help="Shortcut to load the full 96-layer config")
    return parser.parse_args()


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
    if args.dataset:
        config.dataset = args.dataset
    if args.dataset_config:
        config.dataset_config = args.dataset_config
    if args.seq_len is not None:
        config.seq_len = args.seq_len
    if args.batch is not None:
        config.batch = args.batch
    if args.steps is not None:
        config.steps = args.steps
    if args.eval_steps is not None:
        config.eval_steps = args.eval_steps
    if args.eval_freq is not None:
        config.eval_freq = args.eval_freq
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = str(args.checkpoint_dir)
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = args.checkpoint_interval
    return config


def ensure_required_kernels():
    if not is_flash_linear_attention_available():
        raise ImportError(
            "Flash Linear Attention (flash-linear-attention >= 0.2.2) is required. "
            "Install with: pip install flash-linear-attention>=0.2.2"
        )
    if not is_causal_conv1d_available():
        raise ImportError(
            "Causal Conv1D (causal-conv1d >= 1.4.0) is required. "
            "Install with: pip install causal-conv1d>=1.4.0"
        )


def maybe_run_benchmarks(args: argparse.Namespace):
    if args.visualize:
        from nanoNext.benchmark import visualize_benchmarks

        visualize_benchmarks(device="cuda", dtype=torch.bfloat16, save_path="benchmark_results.png")
        return True
    if args.benchmark or args.benchmark_only:
        from nanoNext.benchmark import run_all_benchmarks

        run_all_benchmarks(device="cuda", dtype=torch.bfloat16, num_precision_runs=3, num_benchmark_runs=10)
        return args.benchmark_only
    return False


def main():
    args = parse_args()

    if maybe_run_benchmarks(args):
        return

    ensure_required_kernels()

    model_config_path = FULL_MODEL_CONFIG if args.full_model else args.model_config
    model_config: NanoNextConfig = load_model_config(model_config_path)
    train_config: TrainingConfig = load_training_config(args.train_config)
    train_config = apply_overrides(train_config, args)

    print(f"Using model config: {model_config_path}")
    print(f"Using train config: {args.train_config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(model_config, train_config, device=device)


if __name__ == "__main__":
    main()
