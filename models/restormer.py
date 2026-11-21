from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GDFN(nn.Module):
    def __init__(self, dim, expansion=2):
        super().__init__()
        hidden=dim*expansion
        self.project_in=nn.Conv2d(dim,hidden*2,1)
        self.depthwise=nn.Conv2d(hidden*2,hidden*2,3,padding=1,groups=hidden*2)
        self.project_out=nn.Conv2d(hidden,dim,1)

    def forward(self,x):
        a,b=self.depthwise(self.project_in(x)).chunk(2,dim=1)
        return self.project_out(F.gelu(a)*b)


class MDTA(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        if dim%heads: raise ValueError(f'Feature dimension {dim} must be divisible by attention heads {heads}.')
        self.heads=heads
        self.qkv=nn.Conv2d(dim,dim*3,1)
        self.qkv_depthwise=nn.Conv2d(dim*3,dim*3,3,padding=1,groups=dim*3)
        self.project=nn.Conv2d(dim,dim,1)
        self.temperature=nn.Parameter(torch.ones(heads,1,1))

    def forward(self,x):
        batch,channels,height,width=x.shape
        q,k,v=self.qkv_depthwise(self.qkv(x)).chunk(3,dim=1)
        head_dim=channels//self.heads
        q=q.reshape(batch,self.heads,head_dim,height*width)
        k=k.reshape(batch,self.heads,head_dim,height*width)
        v=v.reshape(batch,self.heads,head_dim,height*width)
        q=F.normalize(q,dim=-1); k=F.normalize(k,dim=-1)
        attention=((q@k.transpose(-2,-1))*self.temperature).softmax(dim=-1)
        return self.project((attention@v).reshape(batch,channels,height,width))


class TransformerBlock(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        self.norm1=nn.GroupNorm(1,dim); self.attention=MDTA(dim,heads)
        self.norm2=nn.GroupNorm(1,dim); self.feed_forward=GDFN(dim)

    def forward(self,x):
        x=x+self.attention(self.norm1(x))
        return x+self.feed_forward(self.norm2(x))


def _blocks(dim,count,heads):
    return nn.Sequential(*(TransformerBlock(dim,heads) for _ in range(count)))


class Restormer(nn.Module):
    """Compact multi-scale Restormer-style encoder-decoder for single-channel MRI slices."""
    def __init__(self,channels=1,dim=16,blocks=2,heads=4):
        super().__init__()
        self.embed=nn.Conv2d(channels,dim,3,padding=1)
        self.encoder1=_blocks(dim,blocks,heads)
        self.down1=nn.Conv2d(dim,dim*2,3,stride=2,padding=1)
        self.encoder2=_blocks(dim*2,blocks,heads)
        self.down2=nn.Conv2d(dim*2,dim*4,3,stride=2,padding=1)
        self.latent=_blocks(dim*4,blocks,heads)
        self.up2=nn.Conv2d(dim*4,dim*2,1)
        self.merge2=nn.Conv2d(dim*4,dim*2,1)
        self.decoder2=_blocks(dim*2,blocks,heads)
        self.up1=nn.Conv2d(dim*2,dim,1)
        self.merge1=nn.Conv2d(dim*2,dim,1)
        self.decoder1=_blocks(dim,blocks,heads)
        self.out=nn.Conv2d(dim,channels,3,padding=1)

    def forward(self,x):
        original=x
        e1=self.encoder1(self.embed(x))
        e2=self.encoder2(self.down1(e1))
        z=self.latent(self.down2(e2))
        d2=F.interpolate(self.up2(z),size=e2.shape[-2:],mode='bilinear',align_corners=False)
        d2=self.decoder2(self.merge2(torch.cat((d2,e2),dim=1)))
        d1=F.interpolate(self.up1(d2),size=e1.shape[-2:],mode='bilinear',align_corners=False)
        d1=self.decoder1(self.merge1(torch.cat((d1,e1),dim=1)))
        return original+self.out(d1)
