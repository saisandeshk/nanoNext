"""Attention modules used by nanoNext."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NanoNextConfig
from ..utils import is_causal_conv1d_available, is_flash_linear_attention_available
from .norm import RMSNorm


# Import required dependencies
if not is_causal_conv1d_available():
    raise ImportError(
        "causal_conv1d is required. Install with: pip install causal-conv1d>=1.4.0"
    )

if not is_flash_linear_attention_available():
    raise ImportError(
        "flash-linear-attention is required. Install with: pip install flash-linear-attention>=0.2.2"
    )

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


@dataclass
class AttentionCache:
    key: torch.Tensor
    value: torch.Tensor


def create_attention_cache(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AttentionCache:
    key = torch.zeros(batch_size, num_heads, max_seq_len, head_dim, device=device, dtype=dtype)
    value = torch.zeros_like(key)
    return AttentionCache(key=key, value=value)


class RotaryEmbedding(nn.Module):
    """Minimal rotary embedding implementation with optional partial rotary factor."""

    def __init__(self, config: NanoNextConfig):
        super().__init__()
        self.dim = int(config.head_dim * config.partial_rotary_factor)
        if self.dim % 2 != 0:
            self.dim -= self.dim % 2
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        positions = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot, q_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    k_rot, k_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class MultiHeadAttention(nn.Module):
    """Grouped-query gated attention used in Qwen3-Next."""

    def __init__(self, config: NanoNextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        rotary_factor = config.partial_rotary_factor if config.partial_rotary_factor > 0 else 1.0
        self.rotary_dim = max(2, int(self.head_dim * rotary_factor))
        if self.rotary_dim % 2:
            self.rotary_dim -= 1
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.rotary_emb = RotaryEmbedding(config)

    def _shape(self, tensor: torch.Tensor, seq_len: int, num_heads: int) -> torch.Tensor:
        return tensor.view(*tensor.shape[:-1], num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[AttentionCache] = None,
        cache_position: Optional[int] = None,
    ) -> Tuple[torch.Tensor, AttentionCache]:
        batch_size, seq_len, _ = hidden_states.size()
        device = hidden_states.device
        dtype = hidden_states.dtype

        q, gate = torch.chunk(self.q_proj(hidden_states), 2, dim=-1)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = self._shape(q, seq_len, self.num_heads)
        k = self._shape(k, seq_len, self.num_kv_heads)
        v = self._shape(v, seq_len, self.num_kv_heads)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self.rotary_emb(seq_len, device, dtype)
        cos = cos[..., : self.rotary_dim]
        sin = sin[..., : self.rotary_dim]
        cos = cos.view(1, 1, seq_len, -1)
        sin = sin.view(1, 1, seq_len, -1)
        q, k = apply_rotary(q, k, cos, sin, self.rotary_dim)

        if past_kv is not None and cache_position is not None:
            past_kv.key[:, :, cache_position : cache_position + seq_len] = k
            past_kv.value[:, :, cache_position : cache_position + seq_len] = v
            k = past_kv.key[:, :, : cache_position + seq_len]
            v = past_kv.value[:, :, : cache_position + seq_len]
        else:
            past_kv = create_attention_cache(
                batch_size,
                self.num_kv_heads,
                self.head_dim,
                seq_len,
                device,
                dtype,
            )
            past_kv.key.copy_(k)
            past_kv.value.copy_(v)

        repeat_shape = (1, self.num_groups, 1, 1)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)

        gate = torch.sigmoid(gate)
        attn_output = attn_output * gate

        attn_output = self.o_proj(attn_output)
        return attn_output, past_kv


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet matching Qwen3NextGatedDeltaNet implementation."""

    def __init__(self, config: NanoNextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = "silu"
        self.layer_norm_epsilon = config.rms_norm_eps

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # projection of the input hidden states
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        self.in_proj_qkvz = nn.Linear(self.hidden_size, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(self.hidden_size, projection_size_ba, bias=False)

        # time step projection (discretization)
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = FusedRMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            activation=self.activation,
        )

        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def fix_query_key_value_ordering(self, mixed_qkvz: torch.Tensor, mixed_ba: torch.Tensor):
        """Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`."""
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads,
            2 * self.head_k_dim + 2 * self.head_v_dim * self.num_v_heads // self.num_k_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (self.num_k_heads, 2 * self.num_v_heads // self.num_k_heads)

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [self.num_v_heads // self.num_k_heads, self.num_v_heads // self.num_k_heads]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), value.size(1), -1, self.head_v_dim)
        z = z.reshape(z.size(0), z.size(1), -1, self.head_v_dim)
        b = b.reshape(b.size(0), b.size(1), self.num_v_heads)
        a = a.reshape(a.size(0), a.size(1), self.num_v_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_position: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        conv_state: Optional[torch.Tensor] = None,
        recurrent_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = (
            conv_state is not None
            and recurrent_state is not None
            and seq_len == 1
            and cache_position is not None
        )

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states:
            # Convolution sequence transformation
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            mixed_qkv = causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                head_first=False,  # Inputs are [B, T, H, D] not [B, H, T, D]
            )
        else:
            core_attn_out, last_recurrent_state = fused_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                head_first=False,  # Inputs are [B, T, H, D] not [B, H, T, D]
            )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)
        return output


