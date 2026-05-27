# -*- coding: utf-8 -*-
"""
RMSNorm 层
"""

import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 唯一的可学习参数：增益 γ，初始化为全 1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算 RMS，在最后一个维度（特征维）上
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms  # x̄ = x / RMS(x)

    def forward(self, x: torch.Tensor):
        # 在 float32 下计算以提高精度，再转回原类型
        output = self._norm(x.float()).type_as(x)
        return output * self.weight  # γ · x̄

# 使用示例
norm = RMSNorm(dim=4096)
x = torch.randn(2, 512, 4096)  # [batch, seq_len, dim]
out = norm(x)  # out.shape = (2, 512, 4096)