#!/usr/bin/env python3
"""Benchmark script for nanoNext kernel comparisons.

Usage:
    python benchmark.py                          # Text output (default)
    python benchmark.py --visualize              # Matplotlib visualization
    python benchmark.py --precision-runs 5       # Custom precision runs
    python benchmark.py --benchmark-runs 20      # Custom benchmark runs
    python benchmark.py --visualize --output results.png  # Custom output path
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark optimized kernels vs PyTorch implementations"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate matplotlib visualization across 10 scales (default: text output)",
    )
    parser.add_argument(
        "--precision-runs",
        type=int,
        default=3,
        help="Number of numerical precision test runs (default: 3)",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=10,
        help="Number of performance benchmark runs (default: 10)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run benchmarks on (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Data type for benchmarks (default: bfloat16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.png",
        help="Output path for visualization (default: benchmark_results.png)",
    )
    
    args = parser.parse_args()
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    from nanoNext.benchmark import visualize_benchmarks, run_all_benchmarks
    
    if args.visualize:
        visualize_benchmarks(device=args.device, dtype=dtype_map[args.dtype], save_path=args.output)
    else:
        run_all_benchmarks(device=args.device, dtype=dtype_map[args.dtype],
                          num_precision_runs=args.precision_runs, num_benchmark_runs=args.benchmark_runs)


if __name__ == "__main__":
    main()
