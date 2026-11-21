from __future__ import annotations
import torch
from torch import nn
class SSIMLoss(nn.Module):
    def __init__(self, window=7, data_range=1.0): super().__init__(); self.window=window; self.data_range=data_range
    def forward(self, prediction, target):
        c1=(.01*self.data_range)**2; c2=(.03*self.data_range)**2; pad=self.window//2
        mu_x=torch.nn.functional.avg_pool2d(prediction,self.window,1,pad); mu_y=torch.nn.functional.avg_pool2d(target,self.window,1,pad)
        sx=torch.nn.functional.avg_pool2d(prediction**2,self.window,1,pad)-mu_x**2; sy=torch.nn.functional.avg_pool2d(target**2,self.window,1,pad)-mu_y**2; sxy=torch.nn.functional.avg_pool2d(prediction*target,self.window,1,pad)-mu_x*mu_y
        ssim=((2*mu_x*mu_y+c1)*(2*sxy+c2))/((mu_x**2+mu_y**2+c1)*(sx+sy+c2)+1e-8)
        return 1-ssim.mean()
