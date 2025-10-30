import os
import math
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional
from dataclasses import field

# Ensure repo root is on sys.path when running pytest from parent dirs
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root containing pyproject.toml
ROOT_STR = str(ROOT)
if ROOT_STR not in sys.path:
    sys.path.insert(0, ROOT_STR)

from nanoNext.modules.attention import (
    MultiHeadAttention,
    create_attention_cache,
)

# define config 
@dataclass
class NanoNextConfig:
    """Configuration for Qwen3-Next architecture (2x scale)."""

    vocab_size: int = 151_936 # total vocabulary size 
    hidden_size: int = 3_072 # d_model 
    intermediate_size: int = 5_632 # d_ff
    num_hidden_layers: int = 96 # n_layers 
    num_attention_heads: int = 16 # n_heads
    num_key_value_heads: int = 2 # n_kv_heads - for GQA or MLA 
    head_dim: int = 256 # head_dim 
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

def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

@torch.no_grad()
def test_attention_shapes_and_types():
    device = get_device()
    cfg = NanoNextConfig(
        attention_dropout=0.0,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=64,           # smaller for test speed
        hidden_size=16*64,     # Hq * D
        partial_rotary_factor=0.5,
    )
    attn = MultiHeadAttention(cfg, layer_idx=0).to(device=device, dtype=torch.float32).eval()
    B, T = 2, 7
    x = torch.randn(B, T, cfg.hidden_size, device=device, dtype=torch.float32)

    y, cache = attn(x)  # no mask -> causal
    assert y.shape == (B, T, cfg.hidden_size)
    assert cache.key.shape == (B, cfg.num_key_value_heads, T, cfg.head_dim)
    assert cache.value.shape == cache.key.shape
    assert y.dtype == torch.float32
    assert y.device.type == device.type

@torch.no_grad()
def test_masking_consistency_sdpa_vs_manual():
    # Compare SDPA vs manual path with fixed seed
    device = get_device()
    torch.manual_seed(0)
    cfg = NanoNextConfig(
        attention_dropout=0.0,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        hidden_size=8*32,
        partial_rotary_factor=0.5,
    )
    attn_sdpa = MultiHeadAttention(cfg, layer_idx=0).to(device, dtype=torch.float32).eval()
    attn_eager = MultiHeadAttention(cfg, layer_idx=0).to(device, dtype=torch.float32).eval()
    # copy weights to ensure identical params
    attn_eager.load_state_dict(attn_sdpa.state_dict())

    B, T = 2, 16
    x = torch.randn(B, T, cfg.hidden_size, device=device, dtype=torch.float32)

    y1, _ = attn_sdpa(x, use_sdpa=True)
    y2, _ = attn_eager(x, use_sdpa=False)
    # Should be very close numerically
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)

@torch.no_grad()
def test_kv_cache_prefill_and_decode_parity():
    device = get_device()
    cfg = NanoNextConfig(
        attention_dropout=0.0,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        hidden_size=8*32,
        partial_rotary_factor=0.5,
    )
    attn = MultiHeadAttention(cfg, layer_idx=0).to(device, dtype=torch.float32).eval()
    B, T_total = 2, 18
    T_prefill = 13

    x_full = torch.randn(B, T_total, cfg.hidden_size, device=device, dtype=torch.float32)
    # Reference: single forward over full context
    y_full, _ = attn(x_full, use_sdpa=False)  # manual to avoid kernel nondeterminism issues
    last_token_ref = y_full[:, -1, :]  # [B, hidden]

    # Prefill + decode one token
    kv = create_attention_cache(
        batch_size=B,
        num_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        max_seq_len=T_total,
        device=device,
        dtype=torch.float32,
    )
    # prefill first T_prefill
    _ = attn(x_full[:, :T_prefill, :], past_kv=kv, cache_position=0, use_sdpa=False)
    # decode last T_total - T_prefill tokens one by one (here 5 tokens)
    pos = T_prefill
    y_inc = None
    while pos < T_total:
        y_inc, _ = attn(x_full[:, pos:pos+1, :], past_kv=kv, cache_position=pos, use_sdpa=False)
        pos += 1

    last_token_inc = y_inc[:, 0, :] #type: ignore [B, hidden]
    assert torch.allclose(last_token_inc, last_token_ref, atol=1e-5, rtol=1e-5)