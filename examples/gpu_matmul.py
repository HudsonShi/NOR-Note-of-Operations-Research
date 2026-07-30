# -*- coding: utf-8 -*-
"""
GPU vs CPU 矩阵乘法性能对比

展示 GPU 在密集矩阵运算上的加速效果。
分别用 NumPy (CPU) 和 PyTorch (GPU) 做 GEMM，比较吞吐量。

依赖：pip install torch numpy
"""

import time
import numpy as np
import torch


def benchmark_gemm(size, dtype=torch.float32, iters=10):
    """对比 CPU (NumPy) 和 GPU (PyTorch) 的矩阵乘法吞吐。"""

    # ----- CPU (NumPy) -----
    a_cpu = np.random.randn(size, size).astype(np.float32)
    b_cpu = np.random.randn(size, size).astype(np.float32)

    start = time.perf_counter()
    for _ in range(iters):
        c_cpu = a_cpu @ b_cpu
    cpu_time = time.perf_counter() - start

    # ----- GPU (PyTorch) -----
    if torch.cuda.is_available():
        a_gpu = torch.randn(size, size, dtype=dtype, device='cuda')
        b_gpu = torch.randn(size, size, dtype=dtype, device='cuda')

        # Warm-up
        for _ in range(3):
            _ = a_gpu @ b_gpu
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            c_gpu = a_gpu @ b_gpu
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - start
    else:
        gpu_time = None

    # ----- 输出 -----
    flops = 2.0 * size**3 * iters  # GEMM: 2*M*N*K

    print(f"\nMatrix size: {size}×{size}  ({iters} iters)")
    print(f"  CPU: {cpu_time:.4f}s  →  {flops / cpu_time / 1e9:.2f} GFLOPS")
    if gpu_time is not None:
        print(f"  GPU: {gpu_time:.4f}s  →  {flops / gpu_time / 1e9:.2f} GFLOPS")
        print(f"  Speedup: {cpu_time / gpu_time:.1f}×")
    else:
        print("  GPU: not available")


if __name__ == "__main__":
    for sz in [512, 1024, 2048, 4096]:
        benchmark_gemm(sz, iters=5)

    if torch.cuda.is_available():
        print(f"\nDevice: {torch.cuda.get_device_name(0)}")
        print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
