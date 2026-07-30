# -*- coding: utf-8 -*-
"""
梯度累积 (Gradient Accumulation) 实现

在显存不足以支持大 batch 时，通过多次 micro-batch 前向+反向
累积梯度，等效实现大批量训练。

核心技巧：
  - loss 需要除以累积步数 (或对 micro-batch loss 取平均)
  - optimizer.step() 只在累积完成后调用一次
  - optimizer.zero_grad() 只在 step 前调用

依赖：pip install torch
"""

import torch
import torch.nn as nn


class GradientAccumulator:
    """梯度累积包装器：透明地将小 batch 模拟为大 batch 训练。"""

    def __init__(self, model, optimizer, accum_steps):
        """
        Args:
            model: PyTorch 模型
            optimizer: PyTorch 优化器
            accum_steps: 累积步数 (等效 batch_size × accum_steps)
        """
        self.model = model
        self.optimizer = optimizer
        self.accum_steps = accum_steps
        self.current_step = 0

    def zero_grad(self):
        """重置梯度（每个 accum cycle 开始时调用一次）。"""
        if self.current_step == 0:
            self.optimizer.zero_grad()

    def backward(self, loss):
        """
        反向传播并累积梯度。

        关键：loss 必须除以 accum_steps，因为 PyTorch 的 backward()
        默认累加梯度。除以 accum_steps 使得最终梯度等价于
        在 batch_size × accum_steps 数据上的平均梯度。
        """
        scaled_loss = loss / self.accum_steps
        scaled_loss.backward()
        self.current_step += 1

    def step(self):
        """累积完成后更新参数。"""
        if self.current_step == self.accum_steps:
            self.optimizer.step()
            self.current_step = 0


def demo():
    """演示梯度累积与直接大批量训练的等价性。"""
    torch.manual_seed(42)

    # 两个相同的简单模型
    model_direct = nn.Linear(10, 1)
    model_accum  = nn.Linear(10, 1)
    model_accum.load_state_dict(model_direct.state_dict())  # 相同初始化

    # 大批量训练的参考
    optim_direct = torch.optim.SGD(model_direct.parameters(), lr=0.01)
    optim_accum  = torch.optim.SGD(model_accum.parameters(), lr=0.01)

    # 生成数据
    x_big = torch.randn(256, 10)
    y_big = torch.randn(256, 1)

    # ----- 直接大批量 -----
    optim_direct.zero_grad()
    loss_d = torch.nn.functional.mse_loss(model_direct(x_big), y_big)
    loss_d.backward()
    optim_direct.step()

    # ----- 梯度累积：4 步 × batch_size=64 = 等效 256 -----
    accum = GradientAccumulator(model_accum, optim_accum, accum_steps=4)
    for i in range(4):
        x_micro = x_big[i*64:(i+1)*64]
        y_micro = y_big[i*64:(i+1)*64]

        accum.zero_grad()
        loss_a = torch.nn.functional.mse_loss(model_accum(x_micro), y_micro)
        accum.backward(loss_a)
        accum.step()

    # 验证等价性
    w_d = model_direct.weight.data
    w_a = model_accum.weight.data
    diff = (w_d - w_a).abs().max().item()

    print(f"Direct loss:     {loss_d.item():.6f}")
    print(f"Max param diff:  {diff:.2e}")
    print(f"Equivalent?      {diff < 1e-6}")


if __name__ == "__main__":
    demo()
