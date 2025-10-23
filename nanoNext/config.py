"""nanoNext configuration module."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

T = TypeVar("T")


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if value.startswith("0") and value != "0" and not value.startswith("0."):
            return value  # preserve strings like 001
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            if (value.startswith("\"") and value.endswith("\"")) or (
                value.startswith("'") and value.endswith("'")
            ):
                return value[1:-1]
            return value


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current_list: Optional[List[Any]] = None
    current_key: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("- "):
            if current_list is None:
                raise ValueError("List item found without preceding key")
            current_list.append(_parse_scalar(line.lstrip()[2:]))
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value == "":
                current_list = []
                data[key] = current_list
                current_key = key
            else:
                data[key] = _parse_scalar(value)
                current_list = None
                current_key = None
        else:
            raise ValueError(f"Unable to parse line: {raw_line}")

    return data


def _load_yaml_like(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return _parse_simple_yaml(text)
    else:
        loaded = yaml.safe_load(text)
        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise TypeError("Expected mapping at top level of YAML file")
        return loaded


def _load_dataclass(path: str | Path, cls: Type[T]) -> T:
    data = _load_yaml_like(Path(path))
    return cls(**data)


@dataclass
class NanoNextConfig:
    """Configuration for Qwen3-Next architecture (2x scale)."""

    vocab_size: int = 151_936
    hidden_size: int = 3_072
    intermediate_size: int = 5_632
    num_hidden_layers: int = 96
    num_attention_heads: int = 16
    num_key_value_heads: int = 2
    head_dim: int = 256
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32
    linear_conv_kernel_dim: int = 4
    max_position_embeddings: int = 262_144
    partial_rotary_factor: float = 0.25
    rope_theta: float = 10_000.0
    rope_scaling: Optional[dict] = None
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    attention_dropout: float = 0.0
    decoder_sparse_step: int = 1
    moe_intermediate_size: int = 512
    shared_expert_intermediate_size: int = 512
    num_experts_per_tok: int = 10
    num_experts: int = 512
    norm_topk_prob: bool = True
    router_aux_loss_coef: float = 0.001
    mlp_only_layers: List[int] = field(default_factory=list)
    layer_types: Optional[List[str]] = None
    checkpoint_interval: int = 100

    def __post_init__(self):
        if self.layer_types is None:
            interval_pattern = 4
            self.layer_types = [
                "linear_attention" if (i + 1) % interval_pattern else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")


@dataclass
class TrainingConfig:
    """Top-level controls for the educational training loop."""

    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    seq_len: int = 256
    batch: int = 8
    steps: int = 100
    eval_steps: int = 5
    eval_freq: int = 50
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 100


@dataclass
class GenerationConfig:
    """Sampling controls for inference demos."""

    max_tokens: int = 50
    temperature: float = 1.0
    top_k: int = 200
    device: str = "cuda"


def load_model_config(path: str | Path) -> NanoNextConfig:
    """Load a model configuration from a YAML file."""
    return _load_dataclass(path, NanoNextConfig)


def load_training_config(path: str | Path) -> TrainingConfig:
    """Load a training configuration from a YAML file."""
    return _load_dataclass(path, TrainingConfig)


def load_generation_config(path: str | Path) -> GenerationConfig:
    """Load an inference configuration from a YAML file."""
    return _load_dataclass(path, GenerationConfig)


__all__ = [
    "GenerationConfig",
    "NanoNextConfig",
    "TrainingConfig",
    "load_generation_config",
    "load_model_config",
    "load_training_config",
]
