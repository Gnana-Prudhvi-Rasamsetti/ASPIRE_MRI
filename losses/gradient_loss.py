from __future__ import annotations
import torch
from torch import nn
class GradientLoss(nn.Module):
    def forward(self,prediction,target):
        dx=lambda x:x[:,:,:,1:]-x[:,:,:,:-1]; dy=lambda x:x[:,:,1:,:]-x[:,:,:-1,:]
        return torch.nn.functional.l1_loss(dx(prediction),dx(target))+torch.nn.functional.l1_loss(dy(prediction),dy(target))
