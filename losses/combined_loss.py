from __future__ import annotations
from torch import nn
from .ssim_loss import SSIMLoss
from .gradient_loss import GradientLoss
class StructurePreservingLoss(nn.Module):
    def __init__(self,mse_weight=1.,ssim_weight=.2,gradient_weight=.1): super().__init__(); self.weights=(mse_weight,ssim_weight,gradient_weight); self.mse=nn.MSELoss(); self.ssim=SSIMLoss(); self.gradient=GradientLoss()
    def forward(self,prediction,target):
        a,b,c=self.weights; return a*self.mse(prediction,target)+b*self.ssim(prediction,target)+c*self.gradient(prediction,target)
