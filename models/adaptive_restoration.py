from __future__ import annotations
import torch
from torch import nn
class AdaptiveRestoration(nn.Module):
    def __init__(self, dncnn, unet, restormer, detector, gate): super().__init__(); self.dncnn,self.unet,self.restormer,self.detector,self.gate=dncnn,unet,restormer,detector,gate
    def freeze_experts(self):
        for module in (self.dncnn,self.unet,self.restormer,self.detector):
            module.eval()
            for parameter in module.parameters(): parameter.requires_grad_(False)
    def forward(self,noisy):
        characterization=self.detector(noisy); probabilities=characterization["logits"].softmax(1); weights=self.gate(characterization["embedding"],probabilities,characterization["severity"])
        outputs=torch.stack([self.dncnn(noisy),self.unet(noisy),self.restormer(noisy)],1); fused=(outputs*weights[:,:,None,None,None]).sum(1)
        return {"restored":fused,"weights":weights,"experts":outputs,"noise_logits":characterization["logits"],"severity":characterization["severity"]}
