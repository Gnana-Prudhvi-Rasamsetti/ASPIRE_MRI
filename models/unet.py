from __future__ import annotations
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self, i, o): super().__init__(); self.net = nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(o,o,3,padding=1),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)
class UNet(nn.Module):
    def __init__(self, channels=1, base=32):
        super().__init__(); self.e1=Block(channels,base); self.e2=Block(base,base*2); self.e3=Block(base*2,base*4); self.pool=nn.MaxPool2d(2); self.b=Block(base*4,base*8); self.u3=nn.ConvTranspose2d(base*8,base*4,2,2); self.d3=Block(base*8,base*4); self.u2=nn.ConvTranspose2d(base*4,base*2,2,2); self.d2=Block(base*4,base*2); self.u1=nn.ConvTranspose2d(base*2,base,2,2); self.d1=Block(base*2,base); self.out=nn.Conv2d(base,channels,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); b=self.b(self.pool(e3)); d3=self.d3(torch.cat([self.u3(b),e3],1)); d2=self.d2(torch.cat([self.u2(d3),e2],1)); return self.out(self.d1(torch.cat([self.u1(d2),e1],1)))
