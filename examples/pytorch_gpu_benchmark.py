# -*- coding: utf-8 -*-
"""
PyTorch GPU 训练基准：CNN 在 CIFAR-10 上的吞吐测试

对比不同 batch size、混合精度、编译优化下的训练吞吐。
展示 GPU 利用率优化的实战技巧。

依赖：pip install torch torchvision tqdm
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler


def build_resnet():
    """构建一个简单的 ResNet-18 分类器。"""
    return torchvision.models.resnet18(num_classes=10)


def train_epoch(model, loader, optimizer, criterion, scaler=None, device='cuda'):
    """训练一个 epoch，返回吞吐 (samples/sec)。"""
    model.train()
    total_samples = 0
    start = time.perf_counter()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_samples += images.size(0)

    elapsed = time.perf_counter() - start
    return total_samples / elapsed   # samples/sec


def benchmark(batch_size=128, use_amp=False, use_compile=False):
    """单次 benchmark 运行。"""
    if not torch.cuda.is_available():
        print("CUDA not available.")
        return

    device = 'cuda'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='/tmp/cifar10', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)

    model = build_resnet().to(device)
    if use_compile:
        model = torch.compile(model, mode='reduce-overhead')

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if use_amp else None

    # Warm-up
    train_epoch(model, loader, optimizer, criterion, scaler, device)
    torch.cuda.reset_peak_memory_stats()

    # Measured run
    throughput = train_epoch(model, loader, optimizer, criterion, scaler, device)
    peak_mem = torch.cuda.max_memory_allocated() / 1e9  # GB

    return throughput, peak_mem


if __name__ == "__main__":
    configs = [
        ("FP32,   bs=64 ", 64, False, False),
        ("FP32,   bs=256", 256, False, False),
        ("AMP,    bs=64 ", 64, True, False),
        ("AMP,    bs=256", 256, True, False),
        ("AMP+compile, bs=256", 256, True, True),
    ]

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"{'Config':<22} {'Throughput':>14} {'Peak Mem':>10}")
    print("-" * 48)

    for name, bs, amp, comp in configs:
        try:
            tput, mem = benchmark(bs, amp, comp)
            print(f"{name:<22} {tput:>10.0f} img/s {mem:>8.2f} GB")
        except Exception as e:
            print(f"{name:<22} ERROR: {e}")
