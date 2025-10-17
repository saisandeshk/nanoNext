"""
Benchmark optimized CUDA kernels vs PyTorch reference implementations.

Simple, clean benchmarks to verify numerical accuracy and measure speedups.
Perfect for understanding the performance gains from causal-conv1d and FLA.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule


def l2norm_torch(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """PyTorch L2 normalization."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: str = "silu",
) -> torch.Tensor:
    """PyTorch fallback for causal_conv1d_fn."""
    # x: (batch, dim, seqlen)
    # weight: (dim, 1, kernel_size)
    batch, dim, seqlen = x.shape
    kernel_size = weight.shape[-1]
    
    # Pad left side for causal convolution
    x_padded = F.pad(x, (kernel_size - 1, 0))
    out = F.conv1d(x_padded, weight.unsqueeze(1), bias, groups=dim)
    
    if activation == "silu":
        out = F.silu(out)
    
    return out


def torch_causal_conv1d_update_fn(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    activation: Optional[str] = None,
) -> torch.Tensor:
    """PyTorch fallback for causal_conv1d_update."""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """PyTorch fallback for chunk_gated_delta_rule."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm_torch(query, dim=-1, eps=1e-6)
        key = l2norm_torch(key, dim=-1, eps=1e-6)
    
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """PyTorch fallback for fused_recurrent_gated_delta_rule."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm_torch(query, dim=-1, eps=1e-6)
        key = l2norm_torch(key, dim=-1, eps=1e-6)
    
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class TorchRMSNormGated(torch.nn.Module):
    """PyTorch fallback for FusedRMSNormGated."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6, activation: str = "silu"):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def benchmark_causal_conv1d(
    batch_size: int = 4,
    dim: int = 512,
    seqlen: int = 256,
    kernel_size: int = 4,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_precision_runs: int = 3,
    num_benchmark_runs: int = 10,
    quiet: bool = False,
):
    """Benchmark causal_conv1d_fn vs PyTorch."""
    if not quiet:
        print(f"\ncausal_conv1d | shape ({batch_size},{dim},{seqlen}) kernel={kernel_size}")
    
    # Create random inputs
    x = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
    weight = torch.randn(dim, kernel_size, device=device, dtype=dtype)
    bias = torch.randn(dim, device=device, dtype=dtype)
    
    # Numerical precision check
    max_diff = 0.0
    for _ in range(num_precision_runs):
        x_test = torch.randn(batch_size, dim, seqlen, device=device, dtype=dtype)
        weight_test = torch.randn(dim, kernel_size, device=device, dtype=dtype)
        bias_test = torch.randn(dim, device=device, dtype=dtype)
        
        out_optimized = causal_conv1d_fn(x_test, weight_test, bias_test, activation="silu")
        out_torch = torch_causal_conv1d_fn(x_test, weight_test, bias_test, activation="silu")
        
        diff = (out_optimized - out_torch).abs().max().item()
        max_diff = max(max_diff, diff)
    
    if not quiet:
        print(f"  precision check: max_diff={max_diff:.2e}")
    
    # Performance benchmark
    for _ in range(3):  # warmup
        _ = causal_conv1d_fn(x, weight, bias, activation="silu")
        _ = torch_causal_conv1d_fn(x, weight, bias, activation="silu")
    
    torch.cuda.synchronize()
    
    times_opt = []
    times_torch = []
    for _ in range(num_benchmark_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = causal_conv1d_fn(x, weight, bias, activation="silu")
        torch.cuda.synchronize()
        times_opt.append(time.perf_counter() - t0)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = torch_causal_conv1d_fn(x, weight, bias, activation="silu")
        torch.cuda.synchronize()
        times_torch.append(time.perf_counter() - t0)
    
    avg_opt = sum(times_opt) / len(times_opt) * 1000
    avg_torch = sum(times_torch) / len(times_torch) * 1000
    speedup = avg_torch / avg_opt
    
    if not quiet:
        print(f"  optimized: {avg_opt:.3f}ms | pytorch: {avg_torch:.3f}ms | speedup: {speedup:.1f}x")
    return max_diff, speedup, avg_opt, avg_torch


def benchmark_chunk_gated_delta_rule(
    batch_size: int = 2,
    num_heads: int = 8,
    seqlen: int = 256,
    head_dim: int = 32,
    chunk_size: int = 64,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_precision_runs: int = 3,
    num_benchmark_runs: int = 10,
    quiet: bool = False,
):
    """Benchmark chunk_gated_delta_rule vs PyTorch."""
    num_chunks = (seqlen + chunk_size - 1) // chunk_size
    if not quiet:
        print(f"\nchunk_delta_rule | shape ({batch_size},{seqlen},{num_heads},{head_dim}) chunks={num_chunks}")
    
    # Create random inputs
    query = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    g = torch.randn(batch_size, seqlen, num_heads, device=device, dtype=dtype) - 3  # negative
    beta = torch.randn(batch_size, seqlen, num_heads, device=device, dtype=dtype).sigmoid()
    
    # Numerical precision check
    max_diff = 0.0
    for _ in range(num_precision_runs):
        q_test = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k_test = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v_test = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        g_test = torch.randn(batch_size, seqlen, num_heads, device=device, dtype=dtype) - 3
        beta_test = torch.randn(batch_size, seqlen, num_heads, device=device, dtype=dtype).sigmoid()
        
        out_optimized, _ = chunk_gated_delta_rule(
            q_test, k_test, v_test, g=g_test, beta=beta_test,
            initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
        )
        out_torch, _ = torch_chunk_gated_delta_rule(
            q_test, k_test, v_test, g=g_test, beta=beta_test,
            chunk_size=chunk_size, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
        )
        
        diff = (out_optimized - out_torch).abs().max().item()
        max_diff = max(max_diff, diff)
    
    if not quiet:
        print(f"  precision check: max_diff={max_diff:.2e}")
    
    # Performance benchmark
    for _ in range(3):  # warmup
        _ = chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        _ = torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, chunk_size=chunk_size, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
    
    torch.cuda.synchronize()
    
    times_opt = []
    times_torch = []
    for _ in range(num_benchmark_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, _ = chunk_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        times_opt.append(time.perf_counter() - t0)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, _ = torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, chunk_size=chunk_size, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        times_torch.append(time.perf_counter() - t0)
    
    avg_opt = sum(times_opt) / len(times_opt) * 1000
    avg_torch = sum(times_torch) / len(times_torch) * 1000
    speedup = avg_torch / avg_opt
    
    if not quiet:
        print(f"  optimized: {avg_opt:.3f}ms | pytorch: {avg_torch:.3f}ms | speedup: {speedup:.1f}x")
    return max_diff, speedup, avg_opt, avg_torch


def benchmark_recurrent_gated_delta_rule(
    batch_size: int = 2,
    num_heads: int = 8,
    seqlen: int = 16,  # Shorter for recurrent
    head_dim: int = 32,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_precision_runs: int = 3,
    num_benchmark_runs: int = 10,
    quiet: bool = False,
):
    """Benchmark fused_recurrent_gated_delta_rule vs PyTorch."""
    if not quiet:
        print(f"\nrecurrent_delta_rule | shape ({batch_size},{seqlen},{num_heads},{head_dim}) recurrent")
    
    # Create random inputs
    query = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
    g = torch.randn(batch_size, seqlen, num_heads, device=device, dtype=dtype) - 3
    beta = torch.randn(batch_size, seqlen, num_heads, device=device, dtype=dtype).sigmoid()
    
    # Numerical precision check
    max_diff = 0.0
    for _ in range(num_precision_runs):
        q_test = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        k_test = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        v_test = torch.randn(batch_size, seqlen, num_heads, head_dim, device=device, dtype=dtype)
        g_test = torch.randn(batch_size, seqlen, num_heads, device=device, dtype=dtype) - 3
        beta_test = torch.randn(batch_size, seqlen, num_heads, device=device, dtype=dtype).sigmoid()
        
        out_optimized, _ = fused_recurrent_gated_delta_rule(
            q_test, k_test, v_test, g=g_test, beta=beta_test,
            initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
        )
        out_torch, _ = torch_recurrent_gated_delta_rule(
            q_test, k_test, v_test, g=g_test, beta=beta_test,
            initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True,
        )
        
        diff = (out_optimized - out_torch).abs().max().item()
        max_diff = max(max_diff, diff)
    
    if not quiet:
        print(f"  precision check: max_diff={max_diff:.2e}")
    
    # Performance benchmark
    for _ in range(3):  # warmup
        _ = fused_recurrent_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        _ = torch_recurrent_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
    
    torch.cuda.synchronize()
    
    times_opt = []
    times_torch = []
    for _ in range(num_benchmark_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, _ = fused_recurrent_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        times_opt.append(time.perf_counter() - t0)
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, _ = torch_recurrent_gated_delta_rule(query, key, value, g=g, beta=beta, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=True)
        torch.cuda.synchronize()
        times_torch.append(time.perf_counter() - t0)
    
    avg_opt = sum(times_opt) / len(times_opt) * 1000
    avg_torch = sum(times_torch) / len(times_torch) * 1000
    speedup = avg_torch / avg_opt
    
    if not quiet:
        print(f"  optimized: {avg_opt:.3f}ms | pytorch: {avg_torch:.3f}ms | speedup: {speedup:.1f}x")
    return max_diff, speedup, avg_opt, avg_torch


def run_all_benchmarks(
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_precision_runs: int = 3,
    num_benchmark_runs: int = 10,
):
    """Run all kernel benchmarks."""
    print(f"\nBenchmarking kernels on {device} ({dtype})")
    
    if not torch.cuda.is_available():
        print("Error: CUDA not available")
        return
    
    results = {}
    
    # Benchmark 1: causal_conv1d
    diff1, speedup1, time_opt1, time_torch1 = benchmark_causal_conv1d(
        device=device, dtype=dtype,
        num_precision_runs=num_precision_runs,
        num_benchmark_runs=num_benchmark_runs,
    )
    results['causal_conv1d'] = {
        'max_diff': diff1, 
        'speedup': speedup1,
        'time_optimized': time_opt1,
        'time_torch': time_torch1
    }
    
    # Benchmark 2: chunk_gated_delta_rule
    diff2, speedup2, time_opt2, time_torch2 = benchmark_chunk_gated_delta_rule(
        device=device, dtype=dtype,
        num_precision_runs=num_precision_runs,
        num_benchmark_runs=num_benchmark_runs,
    )
    results['chunk_gated_delta_rule'] = {
        'max_diff': diff2, 
        'speedup': speedup2,
        'time_optimized': time_opt2,
        'time_torch': time_torch2
    }
    
    # Benchmark 3: recurrent_gated_delta_rule
    diff3, speedup3, time_opt3, time_torch3 = benchmark_recurrent_gated_delta_rule(
        device=device, dtype=dtype,
        num_precision_runs=num_precision_runs,
        num_benchmark_runs=num_benchmark_runs,
    )
    results['recurrent_gated_delta_rule'] = {
        'max_diff': diff3, 
        'speedup': speedup3,
        'time_optimized': time_opt3,
        'time_torch': time_torch3
    }
    
    # Summary
    print("\nSummary:")
    avg_speedup = sum(r['speedup'] for r in results.values()) / len(results)
    print(f"  average speedup: {avg_speedup:.1f}x")
    
    total_opt = sum(r['time_optimized'] for r in results.values())
    total_torch = sum(r['time_torch'] for r in results.values())
    if total_torch > 0:
        print(f"  throughput: {1000/total_torch:.0f} iter/s (pytorch) → {1000/total_opt:.0f} iter/s (optimized)")
    
    return results


# -----------------------------------------------------------------------------
# Visualization functions (require matplotlib)

def generate_scale_configs() -> List[Dict[str, Dict]]:
    """Generate 10 configurations with increasing scales."""
    return [
        # Scale 1: Small
        {
            'causal_conv1d': {'batch_size': 2, 'dim': 256, 'seqlen': 128, 'kernel_size': 4},
            'chunk_gated_delta': {'batch_size': 1, 'num_heads': 4, 'seqlen': 128, 'head_dim': 32, 'chunk_size': 64},
            'recurrent_gated_delta': {'batch_size': 1, 'num_heads': 4, 'seqlen': 8, 'head_dim': 32},
        },
        # Scale 2-10: Progressively larger
        {'causal_conv1d': {'batch_size': 4, 'dim': 384, 'seqlen': 256, 'kernel_size': 4},
         'chunk_gated_delta': {'batch_size': 2, 'num_heads': 6, 'seqlen': 256, 'head_dim': 32, 'chunk_size': 64},
         'recurrent_gated_delta': {'batch_size': 2, 'num_heads': 6, 'seqlen': 12, 'head_dim': 32}},
        {'causal_conv1d': {'batch_size': 4, 'dim': 512, 'seqlen': 256, 'kernel_size': 4},
         'chunk_gated_delta': {'batch_size': 2, 'num_heads': 8, 'seqlen': 256, 'head_dim': 32, 'chunk_size': 64},
         'recurrent_gated_delta': {'batch_size': 2, 'num_heads': 8, 'seqlen': 16, 'head_dim': 32}},
        {'causal_conv1d': {'batch_size': 8, 'dim': 512, 'seqlen': 512, 'kernel_size': 4},
         'chunk_gated_delta': {'batch_size': 4, 'num_heads': 8, 'seqlen': 512, 'head_dim': 32, 'chunk_size': 64},
         'recurrent_gated_delta': {'batch_size': 4, 'num_heads': 8, 'seqlen': 16, 'head_dim': 32}},
        {'causal_conv1d': {'batch_size': 8, 'dim': 768, 'seqlen': 512, 'kernel_size': 4},
         'chunk_gated_delta': {'batch_size': 4, 'num_heads': 12, 'seqlen': 512, 'head_dim': 64, 'chunk_size': 64},
         'recurrent_gated_delta': {'batch_size': 4, 'num_heads': 12, 'seqlen': 16, 'head_dim': 64}},
        {'causal_conv1d': {'batch_size': 16, 'dim': 768, 'seqlen': 1024, 'kernel_size': 4},
         'chunk_gated_delta': {'batch_size': 8, 'num_heads': 12, 'seqlen': 1024, 'head_dim': 64, 'chunk_size': 64},
         'recurrent_gated_delta': {'batch_size': 8, 'num_heads': 12, 'seqlen': 16, 'head_dim': 64}},
        {'causal_conv1d': {'batch_size': 16, 'dim': 1024, 'seqlen': 1024, 'kernel_size': 4},
         'chunk_gated_delta': {'batch_size': 8, 'num_heads': 16, 'seqlen': 1024, 'head_dim': 64, 'chunk_size': 64},
         'recurrent_gated_delta': {'batch_size': 8, 'num_heads': 16, 'seqlen': 20, 'head_dim': 64}},
        {'causal_conv1d': {'batch_size': 32, 'dim': 1024, 'seqlen': 2048, 'kernel_size': 4},
         'chunk_gated_delta': {'batch_size': 16, 'num_heads': 16, 'seqlen': 2048, 'head_dim': 64, 'chunk_size': 64},
         'recurrent_gated_delta': {'batch_size': 16, 'num_heads': 16, 'seqlen': 24, 'head_dim': 64}},
        {'causal_conv1d': {'batch_size': 32, 'dim': 1536, 'seqlen': 2048, 'kernel_size': 4},
         'chunk_gated_delta': {'batch_size': 16, 'num_heads': 24, 'seqlen': 2048, 'head_dim': 64, 'chunk_size': 64},
         'recurrent_gated_delta': {'batch_size': 16, 'num_heads': 24, 'seqlen': 32, 'head_dim': 64}},
        # Scale 10: Large
        {'causal_conv1d': {'batch_size': 64, 'dim': 2048, 'seqlen': 4096, 'kernel_size': 4},
         'chunk_gated_delta': {'batch_size': 32, 'num_heads': 32, 'seqlen': 4096, 'head_dim': 64, 'chunk_size': 64},
         'recurrent_gated_delta': {'batch_size': 32, 'num_heads': 32, 'seqlen': 32, 'head_dim': 64}},
    ]


def visualize_benchmarks(device: str = "cuda", dtype: torch.dtype = torch.bfloat16, save_path: str = "benchmark_results.png"):
    """Run benchmarks across 10 scales and create matplotlib visualizations."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"\nVisualizing benchmarks on {device} ({dtype})")
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    configs = generate_scale_configs()
    results = {'scales': [], 'causal_conv1d': {'optimized': [], 'pytorch': [], 'flops': []},
               'chunk_gated_delta': {'optimized': [], 'pytorch': [], 'flops': []},
               'recurrent_gated_delta': {'optimized': [], 'pytorch': [], 'flops': []}}
    
    print("Benchmarking 10 scales...")
    for i, config in enumerate(configs, 1):
        print(f"scale {i}/10... ", end="", flush=True)
        results['scales'].append(f"Scale {i}")
        
        # Run benchmarks quietly
        _, _, t_opt, t_torch = benchmark_causal_conv1d(**config['causal_conv1d'], device=device, dtype=dtype, num_precision_runs=1, num_benchmark_runs=5, quiet=True)
        cfg = config['causal_conv1d']
        flops = cfg['batch_size'] * cfg['dim'] * cfg['seqlen'] * cfg['kernel_size'] * 2
        results['causal_conv1d']['optimized'].append(t_opt)
        results['causal_conv1d']['pytorch'].append(t_torch)
        results['causal_conv1d']['flops'].append(flops / 1e9)
        
        _, _, t_opt, t_torch = benchmark_chunk_gated_delta_rule(**config['chunk_gated_delta'], device=device, dtype=dtype, num_precision_runs=1, num_benchmark_runs=5, quiet=True)
        cfg = config['chunk_gated_delta']
        num_chunks = (cfg['seqlen'] + cfg['chunk_size'] - 1) // cfg['chunk_size']
        flops = cfg['batch_size'] * cfg['num_heads'] * num_chunks * cfg['chunk_size'] ** 2 * cfg['head_dim'] * 2
        results['chunk_gated_delta']['optimized'].append(t_opt)
        results['chunk_gated_delta']['pytorch'].append(t_torch)
        results['chunk_gated_delta']['flops'].append(flops / 1e9)
        
        _, _, t_opt, t_torch = benchmark_recurrent_gated_delta_rule(**config['recurrent_gated_delta'], device=device, dtype=dtype, num_precision_runs=1, num_benchmark_runs=5, quiet=True)
        cfg = config['recurrent_gated_delta']
        flops = cfg['batch_size'] * cfg['num_heads'] * cfg['seqlen'] * cfg['head_dim'] ** 2 * 4
        results['recurrent_gated_delta']['optimized'].append(t_opt)
        results['recurrent_gated_delta']['pytorch'].append(t_torch)
        results['recurrent_gated_delta']['flops'].append(flops / 1e6)
        print("done")
    
    print("Generating visualizations...")
    
    # Create figures
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    scales, x = results['scales'], np.arange(len(results['scales']))
    width, colors_opt, colors_torch = 0.35, '#2ecc71', '#e74c3c'
    
    for idx, (kernel_key, kernel_name, flops_unit) in enumerate([('causal_conv1d', 'Causal Conv1D', 'GFLOPs'),
                                                                   ('chunk_gated_delta', 'Chunk Gated Delta Rule', 'GFLOPs'),
                                                                   ('recurrent_gated_delta', 'Recurrent Gated Delta Rule', 'MFLOPs')]):
        kernel_data = results[kernel_key]
        
        # Time comparison
        ax_time = fig.add_subplot(gs[idx, 0])
        ax_time.bar(x - width/2, kernel_data['optimized'], width, label='Optimized', color=colors_opt, alpha=0.8)
        ax_time.bar(x + width/2, kernel_data['pytorch'], width, label='PyTorch', color=colors_torch, alpha=0.8)
        ax_time.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax_time.set_title(f'{kernel_name} - Time', fontsize=14, fontweight='bold')
        ax_time.set_xticks(x)
        ax_time.set_xticklabels(scales, rotation=45, ha='right')
        ax_time.legend(fontsize=10)
        ax_time.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Throughput
        ax_flops = fig.add_subplot(gs[idx, 1])
        throughput_opt = [f / t if t > 0 else 0 for f, t in zip(kernel_data['flops'], kernel_data['optimized'])]
        throughput_torch = [f / t if t > 0 else 0 for f, t in zip(kernel_data['flops'], kernel_data['pytorch'])]
        ax_flops.bar(x - width/2, throughput_opt, width, label='Optimized', color=colors_opt, alpha=0.8)
        ax_flops.bar(x + width/2, throughput_torch, width, label='PyTorch', color=colors_torch, alpha=0.8)
        ax_flops.set_ylabel(f'{flops_unit}/ms', fontsize=12, fontweight='bold')
        ax_flops.set_title(f'{kernel_name} - Throughput', fontsize=14, fontweight='bold')
        ax_flops.set_xticks(x)
        ax_flops.set_xticklabels(scales, rotation=45, ha='right')
        ax_flops.legend(fontsize=10)
        ax_flops.grid(axis='y', alpha=0.3, linestyle='--')
    
    fig.suptitle('Optimized vs PyTorch (Green = Faster)', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Speedup summary
    fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
    for idx, (kernel_key, kernel_name) in enumerate([('causal_conv1d', 'Causal Conv1D'),
                                                       ('chunk_gated_delta', 'Chunk Gated Delta Rule'),
                                                       ('recurrent_gated_delta', 'Recurrent Gated Delta Rule')]):
        ax = axes[idx]
        kernel_data = results[kernel_key]
        speedups = [t / o if o > 0 else 0 for t, o in zip(kernel_data['pytorch'], kernel_data['optimized'])]
        colors = plt.cm.RdYlGn(np.array(speedups) / max(speedups) if max(speedups) > 0 else [0.5] * len(speedups))
        bars = ax.bar(x, speedups, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='PyTorch (1x)')
        ax.set_ylabel('Speedup (x)', fontsize=12, fontweight='bold')
        ax.set_title(kernel_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scales, rotation=45, ha='right')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}x',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        avg_speedup = np.mean(speedups)
        ax.text(0.5, 0.95, f'Avg: {avg_speedup:.1f}x', transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=11, fontweight='bold')
    
    fig2.suptitle('Speedup Factors (Higher = Better)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_speedup.png'), dpi=300, bbox_inches='tight')
    
    print(f"\nSaved: {save_path}")
    print(f"       {save_path.replace('.png', '_speedup.png')}")


if __name__ == "__main__":
    run_all_benchmarks()

