from __future__ import annotations
import torch
from torch import nn

class DnCNN(nn.Module):
    def __init__(self, channels: int = 1, features: int = 64, depth: int = 12):
        super().__init__()
        if depth < 3: raise ValueError("depth must be at least 3")
        layers = [nn.Conv2d(channels, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2): layers += [nn.Conv2d(features, features, 3, padding=1, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(features, channels, 3, padding=1)); self.net = nn.Sequential(*layers)
    def forward(self, noisy: torch.Tensor) -> torch.Tensor: return noisy - self.net(noisy)
