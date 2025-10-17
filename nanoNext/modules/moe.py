"""Mixture-of-experts utilities for nanoNext."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import NanoNextConfig


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = getattr(F, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class SparseMoe(nn.Module):
    def __init__(self, config: NanoNextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk = config.norm_topk_prob

        self.gate = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(self.hidden_size, config.moe_intermediate_size) for _ in range(self.num_experts)]
        )
        self.shared_expert = FeedForward(self.hidden_size, config.shared_expert_intermediate_size)
        self.shared_gate = nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        flat_states = hidden_states.view(-1, hidden_size)

        router_logits = self.gate(flat_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        expert_outputs = torch.zeros_like(flat_states)

        for slot in range(self.top_k):
            expert_indices = selected_experts[:, slot]
            weights = routing_weights[:, slot].unsqueeze(-1)

            for expert_idx in expert_indices.unique():
                mask = expert_indices == expert_idx
                if mask.any():
                    expert_input = flat_states[mask]
                    expert_weight = weights[mask]
                    expert_outputs[mask] += self.experts[int(expert_idx)](expert_input) * expert_weight

        shared_out = self.shared_expert(flat_states)
        shared_gate = torch.sigmoid(self.shared_gate(flat_states))
        expert_outputs += shared_gate * shared_out

        return expert_outputs.view(batch_size, seq_len, hidden_size)


