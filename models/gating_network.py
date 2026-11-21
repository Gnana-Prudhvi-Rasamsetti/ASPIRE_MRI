from __future__ import annotations
import torch
from torch import nn
class AdaptiveGatingNetwork(nn.Module):
    def __init__(self, embedding_dim=128, classes=3, hidden=128):
        super().__init__(); self.net=nn.Sequential(nn.Linear(embedding_dim+classes+1,hidden),nn.ReLU(),nn.Linear(hidden,hidden//2),nn.ReLU(),nn.Linear(hidden//2,3))
    def forward(self, embedding, probabilities, severity):
        if severity.ndim == 1: severity=severity[:,None]
        return torch.softmax(self.net(torch.cat([embedding,probabilities,severity],1)),1)
