# -*- coding: utf-8 -*-
"""
CUDA 并行计算入门：SAXPY 向量操作

SAXPY = Single-precision A·X Plus Y:  y ← α·x + y
这是 CUDA 编程的 "Hello World"，展示 GPU 线程网格模型。

依赖：pip install numba (含 numba.cuda)
"""

import numpy as np
from numba import cuda
import math


@cuda.jit
def saxpy_kernel(a, x, y, n):
    """每个线程处理一个元素：y[i] += a * x[i]。

    CUDA 内建变量：
      cuda.grid(1) → 线程在 1D 网格中的全局索引
      cuda.blockIdx.x → 当前 Block 索引
      cuda.blockDim.x → 每个 Block 的线程数
      cuda.threadIdx.x → 线程在 Block 内的索引
    """
    i = cuda.grid(1)
    if i < n:
        y[i] += a * x[i]


def run_saxpy(n=2**24, a=2.0):
    """在 GPU 上执行 SAXPY 并与 CPU 比较。"""

    # 主机端数据
    x_host = np.ones(n, dtype=np.float32)
    y_host = np.full(n, 3.0, dtype=np.float32)

    # CPU 参考结果
    y_cpu = y_host.copy()
    y_cpu += a * x_host

    # -------- GPU 执行 --------
    x_dev = cuda.to_device(x_host)
    y_dev = cuda.to_device(y_host)

    # 线程配置
    threads_per_block = 256
    blocks_per_grid = math.ceil(n / threads_per_block)

    print(f"N = {n:,}")
    print(f"Threads per block = {threads_per_block}")
    print(f"Blocks per grid   = {blocks_per_grid}")

    # 启动 kernel
    saxpy_kernel[blocks_per_grid, threads_per_block](a, x_dev, y_dev, n)
    cuda.synchronize()

    y_gpu = y_dev.copy_to_host()

    # 验证
    max_err = np.max(np.abs(y_gpu - y_cpu))
    print(f"Max error (GPU vs CPU) = {max_err:.2e}")
    print(f"First few GPU results: {y_gpu[:5]}")

    assert max_err < 1e-5, f"Error too large: {max_err}"
    print("\n✓ SAXPY kernel executed correctly!")


if __name__ == "__main__":
    if cuda.is_available():
        print(f"CUDA device: {cuda.get_current_device().name}")
        run_saxpy(n=2**24, a=2.0)
    else:
        print("CUDA device not available. Install numba and ensure GPU drivers.")
