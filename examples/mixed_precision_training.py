# -*- coding: utf-8 -*-
"""
混合精度训练 (Automatic Mixed Precision, AMP)

使用 PyTorch 的 torch.cuda.amp 实现 FP16 混合精度训练。
对比 FP32 vs AMP 的速度和内存差异。

关键组件：
  - autocast()：自动将计算转换为 FP16
  - GradScaler：防止小梯度 underflow，动态调整 loss scale
  - 权重主副本保持 FP32 精度

依赖：pip install torch torchvision tqdm
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler


def build_model():
    """构建一个适中的 MLP 用于 benchmark。"""
    return nn.Sequential(
        nn.Linear(1024, 4096), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(4096, 1024), nn.ReLU(),
        nn.Linear(1024, 10),
    )


def train_loop(model, optimizer, data, targets, criterion, scaler=None):
    """单步训练循环。返回耗时 + 是否发生 scale 调整。"""
    model.train()
    scale_adjusted = False

    start = time.perf_counter()
    optimizer.zero_grad()

    if scaler is not None:
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, targets)

        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scale_adjusted = scaler.get_scale() != old_scale
    else:
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    elapsed = time.perf_counter() - start
    return elapsed, scale_adjusted


def benchmark(batch_size=512, hidden=4096, steps=200):
    """对比 FP32 和 AMP 的性能。"""
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = 'cuda'
    data    = torch.randn(batch_size, hidden, device=device)
    targets = torch.randint(0, 10, (batch_size,), device=device)
    criterion = nn.CrossEntropyLoss()

    # ----- FP32 Baseline -----
    model_fp32 = build_model().to(device)
    opt_fp32 = optim.AdamW(model_fp32.parameters(), lr=1e-3)

    # Warm-up
    for _ in range(10):
        train_loop(model_fp32, opt_fp32, data, targets, criterion)
    torch.cuda.reset_peak_memory_stats()

    t_fp32 = 0.0
    for _ in range(steps):
        dt, _ = train_loop(model_fp32, opt_fp32, data, targets, criterion)
        t_fp32 += dt

    mem_fp32 = torch.cuda.max_memory_allocated() / 1e6  # MB

    # ----- AMP -----
    model_amp = build_model().to(device)
    opt_amp = optim.AdamW(model_amp.parameters(), lr=1e-3)
    scaler = GradScaler()

    for _ in range(10):
        train_loop(model_amp, opt_amp, data, targets, criterion, scaler)
    torch.cuda.reset_peak_memory_stats()

    t_amp = 0.0
    for _ in range(steps):
        dt, _ = train_loop(model_amp, opt_amp, data, targets, criterion, scaler)
        t_amp += dt

    mem_amp = torch.cuda.max_memory_allocated() / 1e6

    # ----- 结果 -----
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"{'Mode':<10} {'Time':>10} {'Memory':>10} {'Speedup':>10}")
    print("-" * 42)
    print(f"{'FP32':<10} {t_fp32:>8.4f}s {mem_fp32:>8.0f} MB {'1.0x':>10}")
    print(f"{'AMP':<10} {t_amp:>8.4f}s {mem_amp:>8.0f} MB "
          f"{t_fp32 / t_amp:>8.1f}x")
    print(f"\nPeak memory saved: {mem_fp32 - mem_amp:.0f} MB "
          f"({(1 - mem_amp/mem_fp32)*100:.1f}%)")


if __name__ == "__main__":
    benchmark(batch_size=512, steps=200)
