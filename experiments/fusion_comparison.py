from __future__ import annotations
import torch
from evaluation.metrics import restoration_metrics
@torch.inference_mode()
def compare(model,loader,device):
    """Compare individual experts, equal weights, and learned fusion on identical samples."""
    scores={name:[] for name in ('dncnn','unet','restormer','equal_weight','adaptive')}; model.eval()
    for batch in loader:
        result=model(batch['noisy'].to(device)); target=batch['clean'].to(device); experts=result['experts']; predictions=[experts[:,0],experts[:,1],experts[:,2],experts.mean(1),result['restored']]
        for name,prediction in zip(scores,predictions): scores[name].append(restoration_metrics(prediction,target))
    return {name:{metric:sum(x[metric] for x in values)/len(values) for metric in values[0]} for name,values in scores.items()}
