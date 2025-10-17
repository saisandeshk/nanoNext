"""nanoNext configuration module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class NanoNextConfig:
    """Configuration for Qwen3-Next architecture (2x scale).

    Base Qwen3-Next 80B has 48 layers. This config doubles it to 96 layers
    for a larger model while maintaining the same architecture.
    """

    vocab_size: int = 151_936
    hidden_size: int = 3072
    intermediate_size: int = 5632
    num_hidden_layers: int = 96  # 2x the base Qwen3-Next (48 → 96)
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
    
    # Training
    checkpoint_interval: int = 100  # Save checkpoint every N steps

    def __post_init__(self):
        if self.layer_types is None:
            # Every 4th layer is a full attention layer; others use linear attention.
            interval_pattern = 4
            self.layer_types = [
                "linear_attention" if (i + 1) % interval_pattern else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types length must equal num_hidden_layers")


