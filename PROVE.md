# PROVE

## Rationale
- Introduced YAML-driven configuration files and library adapters so the repo mirrors nanoGPT-style educational layering without losing model functionality.
- Split training and inference logic into reusable modules and refreshed CLIs/README for the new layout.

## Patch
```diff
--- a/inf.py
+++ b/inf.py
@@ -1,201 +1,77 @@
-"""
-Inference script for nanoNext with proper prefill and decode phases.
-
-Two-phase generation:
-1. Prefill: Process entire prompt at once, populate cache
-2. Decode: Generate tokens one-by-one using cache
-
-Clean and minimal, inspired by nanoGPT.
-"""
+"""Inference script for nanoNext with YAML-driven configuration."""
 
 from __future__ import annotations
 
 import argparse
-import time
-import warnings
 from pathlib import Path
 
 import torch
 
-from nanoNext.cache import DynamicCache
-from nanoNext.config import NanoNextConfig
-from nanoNext.model import NanoNextForCausalLM
+from nanoNext.config import GenerationConfig, load_generation_config
+from nanoNext.inference import generate, load_checkpoint
 
-# Suppress FLA format warnings (false positive for short sequences)
-warnings.filterwarnings('ignore', message='.*Input tensor shape suggests potential format mismatch.*')
+DEFAULT_GENERATION_CONFIG = Path("config/inference/default.yaml")
 
 
-def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[NanoNextForCausalLM, dict]:
-    """Load model from checkpoint."""
-    # weights_only=False is safe here since we're loading our own checkpoints
-    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
-    config = checkpoint['config']
-    
-    model = NanoNextForCausalLM(config).to(device=device, dtype=torch.bfloat16)
-    model.load_state_dict(checkpoint['model_state_dict'])
-    model.eval()
-    
-    return model, checkpoint
+def parse_args() -> argparse.Namespace:
+    parser = argparse.ArgumentParser(description="Generate text with nanoNext")
+    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint file")
+    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
+    parser.add_argument("--inference-config", type=Path, default=DEFAULT_GENERATION_CONFIG, help="Path to inference config YAML")
+    parser.add_argument("--max-tokens", type=int, default=None, help="Override number of tokens to generate")
+    parser.add_argument("--temperature", type=float, default=None, help="Override sampling temperature")
+    parser.add_argument("--top-k", type=int, default=None, help="Override top-k sampling")
+    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
+    return parser.parse_args()
 
 
-@torch.no_grad()
-def generate(
-    model: NanoNextForCausalLM,
-    prompt: torch.LongTensor,
-    max_new_tokens: int = 50,
-    temperature: float = 1.0,
-    top_k: int = 200,
-    device: torch.device = None,
-) -> torch.LongTensor:
-    """
-    Generate tokens using two-phase approach: prefill + decode.
-    
-    Args:
-        model: The language model
-        prompt: Input token IDs (batch, seq_len)
-        max_new_tokens: Number of tokens to generate
-        temperature: Sampling temperature (1.0 = normal, <1 = focused, >1 = diverse)
-        top_k: Sample from top-k tokens only
-        device: Device to run on
-    
-    Returns:
-        Generated token IDs (batch, seq_len + max_new_tokens)
-    """
-    device = device or next(model.parameters()).device
-    prompt = prompt.to(device)
-    batch_size, prompt_len = prompt.shape
-    
-    # Initialize cache for efficient generation
-    cache = DynamicCache(model.config)
-    
-    # Warmup: Compile CUDA kernels (first call is slow due to JIT)
-    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
-        _ = model.model(
-            input_ids=prompt,
-            cache_params=cache,
-            cache_position=torch.arange(prompt_len, device=device),
-        )
-    
-    # Reset cache after warmup
-    cache = DynamicCache(model.config)
-    
-    # Phase 1: PREFILL - Process entire prompt at once
-    torch.cuda.synchronize()
-    t0 = time.perf_counter()
-    
-    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
-        hidden_states = model.model(
-            input_ids=prompt,
-            cache_params=cache,
-            cache_position=torch.arange(prompt_len, device=device),
-        )
-        logits = model.lm_head(hidden_states[:, -1:, :])
-    
-    torch.cuda.synchronize()
-    prefill_time = time.perf_counter() - t0
-    prefill_toks_per_sec = prompt_len / prefill_time
-    print(f"prefill | {prompt_len} tokens in {prefill_time*1000:.1f}ms ({prefill_toks_per_sec:.0f} tok/s)")  # Only last token
-    
-    # Sample first token
-    if temperature == 0:
-        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
-    else:
-        logits = logits[:, -1, :] / temperature
-        if top_k > 0:
-            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
-            logits[logits < v[:, [-1]]] = -float('inf')
-        probs = torch.softmax(logits, dim=-1)
-        next_token = torch.multinomial(probs, num_samples=1)
-    
-    generated = torch.cat([prompt, next_token], dim=1)
-    
-    # Phase 2: DECODE - Generate one token at a time with cache
-    torch.cuda.synchronize()
-    t0 = time.perf_counter()
-    
-    for i in range(max_new_tokens - 1):
-        cache_position = torch.tensor([prompt_len + i], device=device)
-        
-        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
-            hidden_states = model.model(
-                input_ids=next_token,
-                cache_params=cache,
-                cache_position=cache_position,
-            )
-            logits = model.lm_head(hidden_states[:, -1:, :])
-        
-        # Sample next token
-        if temperature == 0:
-            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
-        else:
-            logits = logits[:, -1, :] / temperature
-            if top_k > 0:
-                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
-                logits[logits < v[:, [-1]]] = -float('inf')
-            probs = torch.softmax(logits, dim=-1)
-            next_token = torch.multinomial(probs, num_samples=1)
-        
-        generated = torch.cat([generated, next_token], dim=1)
-    
-    torch.cuda.synchronize()
-    decode_time = time.perf_counter() - t0
-    decode_toks_per_sec = (max_new_tokens - 1) / decode_time if decode_time > 0 else 0
-    print(f"decode | {max_new_tokens - 1} tokens in {decode_time*1000:.1f}ms ({decode_toks_per_sec:.0f} tok/s)")
-    
-    return generated
-
-
-def parse_args():
-    parser = argparse.ArgumentParser(description="Generate text with nanoNext")
-    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
-    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
-    parser.add_argument("--max-tokens", type=int, default=50, help="Maximum tokens to generate")
-    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
-    parser.add_argument("--top-k", type=int, default=200, help="Top-k sampling")
-    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
-    return parser.parse_args()
+def apply_overrides(config: GenerationConfig, args: argparse.Namespace) -> GenerationConfig:
+    if args.max_tokens is not None:
+        config.max_tokens = args.max_tokens
+    if args.temperature is not None:
+        config.temperature = args.temperature
+    if args.top_k is not None:
+        config.top_k = args.top_k
+    if args.device is not None:
+        config.device = args.device
+    return config
 
 
 def main():
     args = parse_args()
-    
-    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
-    print(f"Loading checkpoint: {args.checkpoint}")
-    
-    # Load model
+    gen_config: GenerationConfig = load_generation_config(args.inference_config)
+    gen_config = apply_overrides(gen_config, args)
+
+    device = torch.device(gen_config.device if torch.cuda.is_available() else "cpu")
+    if device.type == "cpu" and gen_config.device != "cpu":
+        print("CUDA not available, falling back to CPU")
+
     model, checkpoint = load_checkpoint(args.checkpoint, device)
-    print(f"Loaded model from step {checkpoint['step']}")
-    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
-    
-    # Simple character-level tokenization (matches training)
+    print(f"Loaded checkpoint from step {checkpoint['step']}")
+    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
+
     prompt_tokens = torch.tensor(
         [[ord(c) % model.config.vocab_size for c in args.prompt if 0 <= ord(c) < 256]],
         dtype=torch.long,
-        device=device
-    )
-    
-    print(f"\nPrompt: '{args.prompt}'")
-    print(f"Generating {args.max_tokens} tokens (temp={args.temperature}, top_k={args.top_k})...")
-    
-    # Generate
-    output = generate(
-        model,
-        prompt_tokens,
-        max_new_tokens=args.max_tokens,
-        temperature=args.temperature,
-        top_k=args.top_k,
         device=device,
     )
-    
-    # Decode output
-    generated_tokens = output[0, prompt_tokens.shape[1]:].tolist()
-    generated_text = ''.join(chr(t) if t < 256 else '?' for t in generated_tokens)
-    
-    print(f"\nGenerated text:")
-    print(f"'{generated_text}'")
+
+    print(f"Prompt: '{args.prompt}'")
+    print(
+        f"Generating {gen_config.max_tokens} tokens (temp={gen_config.temperature}, top_k={gen_config.top_k})..."
+    )
+
+    generated, stats = generate(model, prompt_tokens, gen_config, device=device)
+
+    generated_tokens = generated[0, prompt_tokens.shape[1] :].tolist()
+    generated_text = "".join(chr(t) if t < 256 else "?" for t in generated_tokens)
+
+    print(f"\nPrefill time: {stats.prefill_time_s*1000:.1f} ms | speed: {stats.prefill_tokens_per_sec:.0f} tok/s")
+    print(f"Decode time: {stats.decode_time_s*1000:.1f} ms | speed: {stats.decode_tokens_per_sec:.0f} tok/s")
+
+    print(f"\nGenerated text:\n'{generated_text}'")
     print(f"\nComplete output: '{args.prompt}{generated_text}'")
 
 
 if __name__ == "__main__":
     main()
-
--- a/nanoNext/__init__.py
+++ b/nanoNext/__init__.py
@@ -1,13 +1,29 @@
-"""nanoNext - Minimal Qwen3-Next architecture (2x scale, 96 layers)."""
+"""nanoNext - Minimal Qwen3-Next architecture (2x scale)."""
 
 from .cache import DynamicCache
-from .config import NanoNextConfig
-from .model import NanoNextModel, NanoNextForCausalLM
+from .config import (
+    GenerationConfig,
+    NanoNextConfig,
+    TrainingConfig,
+    load_generation_config,
+    load_model_config,
+    load_training_config,
+)
+from .model import NanoNextForCausalLM, NanoNextModel
+from .training import train_model
+from .inference import generate, load_checkpoint
 
 __all__ = [
     "DynamicCache",
+    "GenerationConfig",
     "NanoNextConfig",
     "NanoNextModel",
     "NanoNextForCausalLM",
+    "TrainingConfig",
+    "generate",
+    "load_checkpoint",
+    "load_generation_config",
+    "load_model_config",
+    "load_training_config",
+    "train_model",
 ]
-
--- a/nanoNext/config.py
+++ b/nanoNext/config.py
@@ -2,22 +2,98 @@
 
 from __future__ import annotations
 
+import json
 from dataclasses import dataclass, field
-from typing import List, Optional
+from pathlib import Path
+from typing import Any, Dict, List, Optional, Type, TypeVar
+
+T = TypeVar("T")
+
+
+def _parse_scalar(value: str) -> Any:
+    value = value.strip()
+    if not value:
+        return ""
+    if value.lower() in {"true", "false"}:
+        return value.lower() == "true"
+    try:
+        if value.startswith("0") and value != "0" and not value.startswith("0."):
+            return value  # preserve strings like 001
+        return int(value)
+    except ValueError:
+        try:
+            return float(value)
+        except ValueError:
+            if (value.startswith("\"") and value.endswith("\"")) or (
+                value.startswith("'") and value.endswith("'")
+            ):
+                return value[1:-1]
+            return value
+
+
+def _parse_simple_yaml(text: str) -> Dict[str, Any]:
+    data: Dict[str, Any] = {}
+    current_list: Optional[List[Any]] = None
+    current_key: Optional[str] = None
+
+    for raw_line in text.splitlines():
+        line = raw_line.split("#", 1)[0].rstrip()
+        if not line.strip():
+            continue
+        if line.lstrip().startswith("- "):
+            if current_list is None:
+                raise ValueError("List item found without preceding key")
+            current_list.append(_parse_scalar(line.lstrip()[2:]))
+            continue
+        if ":" in line:
+            key, value = line.split(":", 1)
+            key = key.strip()
+            value = value.strip()
+            if value == "":
+                current_list = []
+                data[key] = current_list
+                current_key = key
+            else:
+                data[key] = _parse_scalar(value)
+                current_list = None
+                current_key = None
+        else:
+            raise ValueError(f"Unable to parse line: {raw_line}")
+
+    return data
+
+
+def _load_yaml_like(path: Path) -> Dict[str, Any]:
+    text = path.read_text()
+    try:
+        import yaml  # type: ignore
+    except ModuleNotFoundError:
+        try:
+            return json.loads(text)
+        except json.JSONDecodeError:
+            return _parse_simple_yaml(text)
+    else:
+        loaded = yaml.safe_load(text)
+        if loaded is None:
+            return {}
+        if not isinstance(loaded, dict):
+            raise TypeError("Expected mapping at top level of YAML file")
+        return loaded
+
+
+def _load_dataclass(path: str | Path, cls: Type[T]) -> T:
+    data = _load_yaml_like(Path(path))
+    return cls(**data)
 
 
 @dataclass
 class NanoNextConfig:
-    """Configuration for Qwen3-Next architecture (2x scale).
-
-    Base Qwen3-Next 80B has 48 layers. This config doubles it to 96 layers
-    for a larger model while maintaining the same architecture.
-    """
+    """Configuration for Qwen3-Next architecture (2x scale)."""
 
     vocab_size: int = 151_936
-    hidden_size: int = 3072
-    intermediate_size: int = 5632
-    num_hidden_layers: int = 96  # 2x the base Qwen3-Next (48 → 96)
+    hidden_size: int = 3_072
+    intermediate_size: int = 5_632
+    num_hidden_layers: int = 96
     num_attention_heads: int = 16
     num_key_value_heads: int = 2
     head_dim: int = 256
@@ -42,13 +118,10 @@
     router_aux_loss_coef: float = 0.001
     mlp_only_layers: List[int] = field(default_factory=list)
     layer_types: Optional[List[str]] = None
-    
-    # Training
-    checkpoint_interval: int = 100  # Save checkpoint every N steps
+    checkpoint_interval: int = 100
 
     def __post_init__(self):
         if self.layer_types is None:
-            # Every 4th layer is a full attention layer; others use linear attention.
             interval_pattern = 4
             self.layer_types = [
                 "linear_attention" if (i + 1) % interval_pattern else "full_attention"
@@ -58,3 +131,51 @@
             raise ValueError("layer_types length must equal num_hidden_layers")
 
 
+@dataclass
+class TrainingConfig:
+    """Top-level controls for the educational training loop."""
+
+    dataset: str = "wikitext"
+    dataset_config: str = "wikitext-2-raw-v1"
+    seq_len: int = 256
+    batch: int = 8
+    steps: int = 100
+    eval_steps: int = 5
+    eval_freq: int = 50
+    checkpoint_dir: str = "checkpoints"
+    checkpoint_interval: int = 100
+
+
+@dataclass
+class GenerationConfig:
+    """Sampling controls for inference demos."""
+
+    max_tokens: int = 50
+    temperature: float = 1.0
+    top_k: int = 200
+    device: str = "cuda"
+
+
+def load_model_config(path: str | Path) -> NanoNextConfig:
+    """Load a model configuration from a YAML file."""
+    return _load_dataclass(path, NanoNextConfig)
+
+
+def load_training_config(path: str | Path) -> TrainingConfig:
+    """Load a training configuration from a YAML file."""
+    return _load_dataclass(path, TrainingConfig)
+
+
+def load_generation_config(path: str | Path) -> GenerationConfig:
+    """Load an inference configuration from a YAML file."""
+    return _load_dataclass(path, GenerationConfig)
+
+
+__all__ = [
+    "GenerationConfig",
+    "NanoNextConfig",
+    "TrainingConfig",
+    "load_generation_config",
+    "load_model_config",
+    "load_training_config",
+]
--- a/train.py
+++ b/train.py
@@ -1,231 +1,120 @@
-"""Minimal training loop for nanoNext with Hugging Face datasets."""
+"""Minimal training loop for nanoNext with YAML-driven configs."""
 
 from __future__ import annotations
 
 import argparse
-import math
-import os
 from pathlib import Path
-from typing import Iterable
+from typing import Optional
 
 import torch
-import torch.optim as optim
-from datasets import load_dataset
 
-from nanoNext.config import NanoNextConfig
-from nanoNext.model import NanoNextForCausalLM
+from nanoNext.config import (
+    NanoNextConfig,
+    TrainingConfig,
+    load_model_config,
+    load_training_config,
+)
+from nanoNext.training import train_model
 from nanoNext.utils import is_causal_conv1d_available, is_flash_linear_attention_available
 
-
-def build_dataset(dataset_name: str, dataset_config: str, seq_len: int, batch: int, vocab_size: int, split: str = "train") -> Iterable[torch.Tensor]:
-    stream = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
-    buffer = torch.empty(0, dtype=torch.long)
-
-    for item in stream:
-        text = item.get("text")
-        if not text:
-            continue
-        tokens = torch.tensor([ord(c) % vocab_size for c in text if 0 <= ord(c) < 256], dtype=torch.long)
-        if tokens.numel() == 0:
-            continue
-        buffer = torch.cat([buffer, tokens])
-        while buffer.numel() >= seq_len * batch:
-            batch_tokens = buffer[: seq_len * batch]
-            buffer = buffer[seq_len * batch :]
-            yield batch_tokens.view(batch, seq_len)
+DEFAULT_MODEL_CONFIG = Path("config/models/demo.yaml")
+FULL_MODEL_CONFIG = Path("config/models/full.yaml")
+DEFAULT_TRAIN_CONFIG = Path("config/train/demo.yaml")
 
 
-def evaluate_model(
-    model: NanoNextForCausalLM,
-    dataset_name: str,
-    dataset_config: str,
-    seq_len: int,
-    batch: int,
-    steps: int,
-    device: torch.device | None = None,
-) -> float:
-    """Evaluate model and return average loss over eval steps."""
-    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
-    model.eval()
-
-    data_iter = build_dataset(dataset_name, dataset_config, seq_len, batch, model.config.vocab_size, split="validation")
-    total_loss = 0.0
-    eval_steps = 0
-
-    with torch.no_grad():
-        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
-            for step, batch_tokens in enumerate(data_iter):
-                inputs = batch_tokens.to(device)
-                logits, loss = model(inputs, inputs)
-                total_loss += loss.item()
-                eval_steps += 1
-                if step + 1 >= steps:
-                    break
-
-    avg_loss = total_loss / eval_steps if eval_steps > 0 else float("inf")
-    model.train()
-    return avg_loss
-
-
-def parse_args():
+def parse_args() -> argparse.Namespace:
     parser = argparse.ArgumentParser(description="Train nanoNext on a Hugging Face dataset")
-    parser.add_argument("--dataset", type=str, default="wikitext", help="Hugging Face dataset name")
-    parser.add_argument("--dataset-config", type=str, default="wikitext-2-raw-v1", help="Dataset configuration")
-    parser.add_argument("--steps", type=int, default=100, help="Training steps")
-    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length")
-    parser.add_argument("--batch", type=int, default=8, help="Batch size")
-    parser.add_argument("--eval-steps", type=int, default=5, help="Evaluation steps per eval")
-    parser.add_argument("--eval-freq", type=int, default=50, help="Evaluate every N training steps")
-    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
-    parser.add_argument("--checkpoint-interval", type=int, default=100, help="Save checkpoint every N steps")
+    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG, help="Path to model config YAML")
+    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG, help="Path to training config YAML")
+    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
+    parser.add_argument("--dataset-config", type=str, default=None, help="Override dataset configuration")
+    parser.add_argument("--steps", type=int, default=None, help="Override number of training steps")
+    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")
+    parser.add_argument("--batch", type=int, default=None, help="Override batch size")
+    parser.add_argument("--eval-steps", type=int, default=None, help="Override evaluation steps")
+    parser.add_argument("--eval-freq", type=int, default=None, help="Override evaluation frequency")
+    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Override checkpoint directory")
+    parser.add_argument(
+        "--checkpoint-interval",
+        type=int,
+        default=None,
+        help="Override checkpoint interval",
+    )
     parser.add_argument("--benchmark", action="store_true", help="Run kernel benchmarks before training")
     parser.add_argument("--benchmark-only", action="store_true", help="Run benchmarks and exit without training")
-    parser.add_argument("--visualize", action="store_true", help="Run benchmarks with matplotlib visualization (10 scales)")
-    parser.add_argument("--full-model", action="store_true", help="Use full 96-layer config (default: 2-layer demo)")
+    parser.add_argument("--visualize", action="store_true", help="Run benchmarks with matplotlib visualization")
+    parser.add_argument("--full-model", action="store_true", help="Shortcut to load the full 96-layer config")
     return parser.parse_args()
 
 
-def save_checkpoint(model: NanoNextForCausalLM, optimizer: optim.Optimizer, step: int, checkpoint_dir: str):
-    """Save model checkpoint."""
-    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
-    checkpoint_path = Path(checkpoint_dir) / f"step_{step}.pt"
-    
-    torch.save({
-        'step': step,
-        'model_state_dict': model.state_dict(),
-        'optimizer_state_dict': optimizer.state_dict(),
-        'config': model.config,
-    }, checkpoint_path)
-    
-    print(f"  checkpoint saved: {checkpoint_path}")
+def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> TrainingConfig:
+    if args.dataset:
+        config.dataset = args.dataset
+    if args.dataset_config:
+        config.dataset_config = args.dataset_config
+    if args.seq_len is not None:
+        config.seq_len = args.seq_len
+    if args.batch is not None:
+        config.batch = args.batch
+    if args.steps is not None:
+        config.steps = args.steps
+    if args.eval_steps is not None:
+        config.eval_steps = args.eval_steps
+    if args.eval_freq is not None:
+        config.eval_freq = args.eval_freq
+    if args.checkpoint_dir is not None:
+        config.checkpoint_dir = str(args.checkpoint_dir)
+    if args.checkpoint_interval is not None:
+        config.checkpoint_interval = args.checkpoint_interval
+    return config
 
 
-def train_model(
-    config: NanoNextConfig,
-    dataset_name: str,
-    dataset_config: str,
-    seq_len: int,
-    batch: int,
-    steps: int,
-    eval_steps: int = 10,
-    eval_freq: int = 10,
-    checkpoint_dir: str = "checkpoints",
-    checkpoint_interval: int = 100,
-    device: torch.device | None = None,
-    model: NanoNextForCausalLM | None = None,
-) -> NanoNextForCausalLM:
-    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
-    
-    # Create model with bfloat16 dtype for FLA compatibility
-    if model is None:
-        model = NanoNextForCausalLM(config)
-    model = model.to(device=device, dtype=torch.bfloat16)
-
-    # Count and display model parameters
-    total_params = sum(p.numel() for p in model.parameters())
-    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
-    print(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
-    print(f"Using dtype: {next(model.parameters()).dtype}")
-
-    data_iter = build_dataset(dataset_name, dataset_config, seq_len, batch, config.vocab_size)
-    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
-
-    # Use GradScaler for mixed precision (though bfloat16 doesn't need it typically)
-    scaler = torch.amp.GradScaler('cuda', enabled=False)  # bfloat16 doesn't need gradient scaling
-
-    model.train()
-    for step, batch_tokens in enumerate(data_iter):
-        inputs = batch_tokens.to(device)
-        
-        # Use autocast for mixed precision
-        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
-            logits, loss = model(inputs, inputs)
-        
-        scaler.scale(loss).backward()
-        scaler.step(optimizer)
-        scaler.update()
-        optimizer.zero_grad()
-
-        train_ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")
-
-        # Evaluate periodically
-        eval_loss_str = ""
-        if (step + 1) % eval_freq == 0:
-            eval_loss = evaluate_model(model, dataset_name, dataset_config, seq_len, batch, eval_steps, device)
-            eval_ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
-            eval_loss_str = f" eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"
-
-        # Save checkpoint periodically
-        if (step + 1) % checkpoint_interval == 0:
-            save_checkpoint(model, optimizer, step + 1, checkpoint_dir)
-
-        print(f"step {step}: train_loss={loss.item():.4f} train_ppl={train_ppl:.2f}{eval_loss_str}")
-        if step + 1 >= steps:
-            break
-    
-    # Save final checkpoint
-    save_checkpoint(model, optimizer, steps, checkpoint_dir)
-    return model
-
-
-def train():
-    args = parse_args()
-
-    # Check for required dependencies
+def ensure_required_kernels():
     if not is_flash_linear_attention_available():
         raise ImportError(
-            "Flash Linear Attention (flash-linear-attention >= 0.2.2) is required for training. "
+            "Flash Linear Attention (flash-linear-attention >= 0.2.2) is required. "
             "Install with: pip install flash-linear-attention>=0.2.2"
         )
-
     if not is_causal_conv1d_available():
         raise ImportError(
-            "Causal Conv1D is required for training. "
+            "Causal Conv1D (causal-conv1d >= 1.4.0) is required. "
             "Install with: pip install causal-conv1d>=1.4.0"
         )
 
-    # Run benchmarks if requested
+
+def maybe_run_benchmarks(args: argparse.Namespace):
     if args.visualize:
         from nanoNext.benchmark import visualize_benchmarks
+
         visualize_benchmarks(device="cuda", dtype=torch.bfloat16, save_path="benchmark_results.png")
+        return True
+    if args.benchmark or args.benchmark_only:
+        from nanoNext.benchmark import run_all_benchmarks
+
+        run_all_benchmarks(device="cuda", dtype=torch.bfloat16, num_precision_runs=3, num_benchmark_runs=10)
+        return args.benchmark_only
+    return False
+
+
+def main():
+    args = parse_args()
+
+    if maybe_run_benchmarks(args):
         return
-    elif args.benchmark or args.benchmark_only:
-        from nanoNext.benchmark import run_all_benchmarks
-        run_all_benchmarks(device="cuda", dtype=torch.bfloat16, num_precision_runs=3, num_benchmark_runs=10)
-        if args.benchmark_only:
-            return
 
-    # Config selection
-    if args.full_model:
-        # Full 96-layer model (2x Qwen3-Next 80B which had 48 layers)
-        config = NanoNextConfig()
-        print(f"\nUsing FULL model: {config.num_hidden_layers} layers, {config.hidden_size} hidden size")
-    else:
-        # Tiny demo model for quick testing
-        config = NanoNextConfig(
-            num_hidden_layers=2,
-            num_experts=4,
-            num_experts_per_tok=2,
-            hidden_size=256,
-            intermediate_size=512,
-        )
-        print(f"\nUsing demo model: {config.num_hidden_layers} layers (use --full-model for 96 layers)")
-    
-    train_model(
-        config,
-        args.dataset,
-        args.dataset_config,
-        args.seq_len,
-        args.batch,
-        args.steps,
-        args.eval_steps,
-        args.eval_freq,
-        args.checkpoint_dir,
-        args.checkpoint_interval,
-    )
+    ensure_required_kernels()
+
+    model_config_path = FULL_MODEL_CONFIG if args.full_model else args.model_config
+    model_config: NanoNextConfig = load_model_config(model_config_path)
+    train_config: TrainingConfig = load_training_config(args.train_config)
+    train_config = apply_overrides(train_config, args)
+
+    print(f"Using model config: {model_config_path}")
+    print(f"Using train config: {args.train_config}")
+
+    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
+    train_model(model_config, train_config, device=device)
 
 
 if __name__ == "__main__":
-    train()
-
-
+    main()
--- a/nanoNext/inference.py
+++ b/nanoNext/inference.py
@@ -0,0 +1,149 @@
+"""Inference utilities for nanoNext."""
+
+from __future__ import annotations
+
+import time
+from dataclasses import dataclass
+from pathlib import Path
+from typing import Callable, Optional
+
+import torch
+
+from .cache import DynamicCache
+from .config import GenerationConfig, NanoNextConfig
+from .model import NanoNextForCausalLM
+
+
+@dataclass
+class GenerationStats:
+    """Simple timing metrics captured during generation."""
+
+    prefill_time_s: float
+    prefill_tokens_per_sec: float
+    decode_time_s: float
+    decode_tokens_per_sec: float
+
+
+def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[NanoNextForCausalLM, dict]:
+    """Load a model checkpoint and move it to the requested device."""
+    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
+    config_obj = checkpoint["config"]
+    if isinstance(config_obj, NanoNextConfig):
+        config = config_obj
+    elif isinstance(config_obj, dict):
+        config = NanoNextConfig(**config_obj)
+    else:
+        raise TypeError("Unsupported config format in checkpoint")
+
+    model = NanoNextForCausalLM(config).to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
+    model.load_state_dict(checkpoint["model_state_dict"])
+    model.eval()
+    return model, checkpoint
+
+
+@torch.no_grad()
+def generate(
+    model: NanoNextForCausalLM,
+    prompt: torch.LongTensor,
+    config: GenerationConfig,
+    *,
+    device: Optional[torch.device] = None,
+    log: Callable[[str], None] = print,
+) -> tuple[torch.LongTensor, GenerationStats]:
+    """Generate tokens with a prefill + decode schedule.
+
+    Returns the generated tensor alongside basic performance statistics.
+    """
+    device = device or next(model.parameters()).device
+    prompt = prompt.to(device)
+    batch_size, prompt_len = prompt.shape
+
+    cache = DynamicCache(model.config)
+    autocast_enabled = device.type == "cuda"
+
+    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
+        _ = model.model(
+            input_ids=prompt,
+            cache_params=cache,
+            cache_position=torch.arange(prompt_len, device=device),
+        )
+
+    cache = DynamicCache(model.config)
+
+    if device.type == "cuda":
+        torch.cuda.synchronize(device)
+    t0 = time.perf_counter()
+    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
+        hidden_states = model.model(
+            input_ids=prompt,
+            cache_params=cache,
+            cache_position=torch.arange(prompt_len, device=device),
+        )
+        logits = model.lm_head(hidden_states[:, -1:, :])
+    if device.type == "cuda":
+        torch.cuda.synchronize(device)
+    prefill_time = time.perf_counter() - t0
+    prefill_toks_per_sec = prompt_len / prefill_time if prefill_time > 0 else float("inf")
+    log(f"prefill | {prompt_len} tokens in {prefill_time*1000:.1f}ms ({prefill_toks_per_sec:.0f} tok/s)")
+
+    if config.temperature == 0:
+        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
+    else:
+        logits = logits[:, -1, :] / config.temperature
+        if config.top_k > 0:
+            values, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
+            logits[logits < values[:, [-1]]] = -float("inf")
+        probs = torch.softmax(logits, dim=-1)
+        next_token = torch.multinomial(probs, num_samples=1)
+
+    generated = torch.cat([prompt, next_token], dim=1)
+
+    if device.type == "cuda":
+        torch.cuda.synchronize(device)
+    t0 = time.perf_counter()
+
+    for i in range(config.max_tokens - 1):
+        cache_position = torch.tensor([prompt_len + i], device=device)
+        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
+            hidden_states = model.model(
+                input_ids=next_token,
+                cache_params=cache,
+                cache_position=cache_position,
+            )
+            logits = model.lm_head(hidden_states[:, -1:, :])
+
+        if config.temperature == 0:
+            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
+        else:
+            logits = logits[:, -1, :] / config.temperature
+            if config.top_k > 0:
+                values, _ = torch.topk(logits, min(config.top_k, logits.size(-1)))
+                logits[logits < values[:, [-1]]] = -float("inf")
+            probs = torch.softmax(logits, dim=-1)
+            next_token = torch.multinomial(probs, num_samples=1)
+
+        generated = torch.cat([generated, next_token], dim=1)
+
+    if device.type == "cuda":
+        torch.cuda.synchronize(device)
+    decode_time = time.perf_counter() - t0
+    decode_tokens = max(config.max_tokens - 1, 1)
+    decode_toks_per_sec = decode_tokens / decode_time if decode_time > 0 else float("inf")
+    log(
+        f"decode | {decode_tokens} tokens in {decode_time*1000:.1f}ms ({decode_toks_per_sec:.0f} tok/s)"
+    )
+
+    stats = GenerationStats(
+        prefill_time_s=prefill_time,
+        prefill_tokens_per_sec=prefill_toks_per_sec,
+        decode_time_s=decode_time,
+        decode_tokens_per_sec=decode_toks_per_sec,
+    )
+    return generated, stats
+
+
+__all__ = [
+    "GenerationStats",
+    "generate",
+    "load_checkpoint",
+]
--- a/nanoNext/training.py
+++ b/nanoNext/training.py
@@ -0,0 +1,186 @@
+"""Training utilities for nanoNext."""
+
+from __future__ import annotations
+
+import math
+from pathlib import Path
+from typing import Callable, Iterator, Optional
+
+import torch
+import torch.optim as optim
+from datasets import load_dataset
+
+from .config import NanoNextConfig, TrainingConfig
+from .model import NanoNextForCausalLM
+
+
+def build_dataset(
+    dataset_name: str,
+    dataset_config: str,
+    seq_len: int,
+    batch: int,
+    vocab_size: int,
+    split: str = "train",
+) -> Iterator[torch.Tensor]:
+    """Stream a Hugging Face dataset and yield batches of token IDs."""
+    stream = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
+    buffer = torch.empty(0, dtype=torch.long)
+
+    for item in stream:
+        text = item.get("text")
+        if not text:
+            continue
+        tokens = torch.tensor(
+            [ord(c) % vocab_size for c in text if 0 <= ord(c) < 256],
+            dtype=torch.long,
+        )
+        if tokens.numel() == 0:
+            continue
+        buffer = torch.cat([buffer, tokens])
+        while buffer.numel() >= seq_len * batch:
+            batch_tokens = buffer[: seq_len * batch]
+            buffer = buffer[seq_len * batch :]
+            yield batch_tokens.view(batch, seq_len)
+
+
+def evaluate_model(
+    model: NanoNextForCausalLM,
+    dataset_name: str,
+    dataset_config: str,
+    seq_len: int,
+    batch: int,
+    steps: int,
+    *,
+    device: Optional[torch.device] = None,
+) -> float:
+    """Evaluate model and return the average loss over the requested number of steps."""
+    device = device or next(model.parameters()).device
+    model.eval()
+
+    data_iter = build_dataset(
+        dataset_name,
+        dataset_config,
+        seq_len,
+        batch,
+        model.config.vocab_size,
+        split="validation",
+    )
+    total_loss = 0.0
+    eval_steps = 0
+
+    autocast_enabled = device.type == "cuda"
+    with torch.no_grad():
+        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
+            for step, batch_tokens in enumerate(data_iter, start=1):
+                inputs = batch_tokens.to(device)
+                _, loss = model(inputs, inputs)
+                total_loss += loss.item()
+                eval_steps += 1
+                if step >= steps:
+                    break
+
+    avg_loss = total_loss / eval_steps if eval_steps > 0 else float("inf")
+    model.train()
+    return avg_loss
+
+
+def save_checkpoint(
+    model: NanoNextForCausalLM,
+    optimizer: optim.Optimizer,
+    step: int,
+    checkpoint_dir: str,
+) -> Path:
+    """Persist model, optimizer, and config state for later reuse."""
+    checkpoint_path = Path(checkpoint_dir) / f"step_{step}.pt"
+    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
+    torch.save(
+        {
+            "step": step,
+            "model_state_dict": model.state_dict(),
+            "optimizer_state_dict": optimizer.state_dict(),
+            "config": model.config,
+        },
+        checkpoint_path,
+    )
+    return checkpoint_path
+
+
+def train_model(
+    model_config: NanoNextConfig,
+    train_config: TrainingConfig,
+    *,
+    device: Optional[torch.device] = None,
+    model: Optional[NanoNextForCausalLM] = None,
+    log: Callable[[str], None] = print,
+) -> NanoNextForCausalLM:
+    """Run a simple nanoNext training loop suitable for educational experiments."""
+    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
+
+    if model is None:
+        model = NanoNextForCausalLM(model_config)
+    target_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
+    model = model.to(device=device, dtype=target_dtype)
+
+    total_params = sum(p.numel() for p in model.parameters())
+    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
+    log(f"Model initialized with {total_params:,} total parameters ({trainable_params:,} trainable)")
+    log(f"Using dtype: {next(model.parameters()).dtype}")
+
+    data_iter = build_dataset(
+        train_config.dataset,
+        train_config.dataset_config,
+        train_config.seq_len,
+        train_config.batch,
+        model_config.vocab_size,
+    )
+    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
+    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
+
+    model.train()
+    steps = train_config.steps
+    autocast_enabled = device.type == "cuda"
+
+    for step, batch_tokens in enumerate(data_iter, start=1):
+        inputs = batch_tokens.to(device)
+        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
+            _, loss = model(inputs, inputs)
+
+        scaler.scale(loss).backward()
+        scaler.step(optimizer)
+        scaler.update()
+        optimizer.zero_grad(set_to_none=True)
+
+        train_ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")
+        message = f"step {step}: train_loss={loss.item():.4f} train_ppl={train_ppl:.2f}"
+
+        if train_config.eval_freq and step % train_config.eval_freq == 0:
+            eval_loss = evaluate_model(
+                model,
+                train_config.dataset,
+                train_config.dataset_config,
+                train_config.seq_len,
+                train_config.batch,
+                train_config.eval_steps,
+                device=device,
+            )
+            eval_ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
+            message += f" eval_loss={eval_loss:.4f} eval_ppl={eval_ppl:.2f}"
+
+        if train_config.checkpoint_interval and step % train_config.checkpoint_interval == 0:
+            checkpoint_path = save_checkpoint(model, optimizer, step, train_config.checkpoint_dir)
+            message += f" | checkpoint saved to {checkpoint_path}"
+
+        log(message)
+        if step >= steps:
+            break
+
+    save_checkpoint(model, optimizer, steps, train_config.checkpoint_dir)
+    return model
+
+
+__all__ = [
+    "build_dataset",
+    "evaluate_model",
+    "save_checkpoint",
+    "train_model",
+]
```

## Validate
- `python3 -m compileall nanoNext train.py inf.py`
