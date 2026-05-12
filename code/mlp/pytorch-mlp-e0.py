# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
MLP 极简代码示例
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    mlp = MLP(in_dim=768, hidden_dim=256, out_dim=10)
    x = torch.randn(32, 768)
    out = mlp(x)
    print(out.shape)
