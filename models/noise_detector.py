from __future__ import annotations
import torch
from torch import nn
class NoiseCharacterizationNet(nn.Module):
    def __init__(self, channels=1, embedding_dim=128, classes=3):
        super().__init__(); self.encoder=nn.Sequential(nn.Conv2d(channels,32,3,2,1),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,embedding_dim,3,2,1),nn.BatchNorm2d(embedding_dim),nn.ReLU(),nn.AdaptiveAvgPool2d(1)); self.classifier=nn.Linear(embedding_dim,classes); self.regressor=nn.Linear(embedding_dim,1)
    def forward(self,x):
        features=self.encoder(x).flatten(1)
        return {"embedding":features,"logits":self.classifier(features),"severity":torch.sigmoid(self.regressor(features).squeeze(1))}
