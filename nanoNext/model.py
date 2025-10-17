"""nanoNext model definition."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import DynamicCache
from .config import NanoNextConfig
from .modules import GatedDeltaNet, MultiHeadAttention, RMSNorm, SparseMoe


class NanoNextDecoderLayer(nn.Module):
    def __init__(self, config: NanoNextConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "full_attention":
            self.mixer = MultiHeadAttention(config, layer_idx)
        else:
            self.mixer = GatedDeltaNet(config, layer_idx)

        use_moe = (layer_idx + 1) % config.decoder_sparse_step == 0 and layer_idx not in config.mlp_only_layers
        self.mlp = SparseMoe(config) if use_moe else nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
            nn.SiLU(),
            nn.Linear(config.intermediate_size, config.hidden_size, bias=False),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache_params: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Token Mixer
        if self.layer_type == "linear_attention":
            # For linear attention, get conv and recurrent states if they exist
            conv_state = cache_params.conv_states[self.layer_idx] if cache_params is not None else None
            recurrent_state = cache_params.recurrent_states[self.layer_idx] if cache_params is not None else None
            hidden_states = self.mixer(
                hidden_states=hidden_states,
                cache_position=cache_position,
                attention_mask=attention_mask,
                conv_state=conv_state,
                recurrent_state=recurrent_state,
            )
        elif self.layer_type == "full_attention":
            # Self Attention
            hidden_states, _ = self.mixer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class NanoNextModel(nn.Module):
    def __init__(self, config: NanoNextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NanoNextDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _update_linear_attn_mask(self, attention_mask: Optional[torch.Tensor], cache_position: Optional[torch.LongTensor]) -> Optional[torch.Tensor]:
        """
        NOTE: Left-padding is used for linear attention mask.
        No need for zeroing states when
            1. Cached forward
            2. Attending to all inputs
        """
        linear_attn_mask = attention_mask
        if cache_position is not None and (cache_position[0] > 0 or (attention_mask is not None and torch.all(attention_mask == 1))):
            linear_attn_mask = None
        return linear_attn_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_params: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds

        batch_size, seq_len, _ = hidden_states.shape

        if cache_position is None:
            past_seen_tokens = cache_params.get_seq_length() if cache_params is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, device=hidden_states.device
            )

        # Create causal mask for full attention layers
        if attention_mask is not None and attention_mask.dim() == 2:
            # Convert 2D attention mask to 4D causal mask
            causal_mask = self._prepare_causal_mask(attention_mask, seq_len, hidden_states.device, hidden_states.dtype)
        else:
            causal_mask = self._prepare_causal_mask(None, seq_len, hidden_states.device, hidden_states.dtype)

        # Linear attention mask (for masking padding tokens)
        linear_attn_mask = self._update_linear_attn_mask(attention_mask, cache_position)

        for layer in self.layers:
            # Select appropriate mask based on layer type
            layer_mask = linear_attn_mask if layer.layer_type == "linear_attention" else causal_mask
            hidden_states = layer(
                hidden_states,
                attention_mask=layer_mask,
                cache_params=cache_params,
                cache_position=cache_position,
            )

        return self.norm(hidden_states)

    @staticmethod
    def _prepare_causal_mask(
        attention_mask: Optional[torch.Tensor],
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare a 4D causal attention mask."""
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )
        causal_mask = torch.where(causal_mask, torch.finfo(dtype).min, 0.0).to(dtype)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

        if attention_mask is not None:
            # Expand attention_mask from [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype)
            # Create padding mask
            padding_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
            causal_mask = causal_mask + padding_mask

        return causal_mask


class NanoNextForCausalLM(nn.Module):
    def __init__(self, config: NanoNextConfig):
        super().__init__()
        self.config = config
        self.model = NanoNextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.LongTensor, labels: Optional[torch.LongTensor] = None):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100
            )

        return logits, loss


