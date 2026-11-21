from __future__ import annotations
import torch
from datasets.noise_generator import corrupt
from .metrics import restoration_metrics
@torch.inference_mode()
def evaluate_noise_levels(model, clean_images, levels, kinds, device):
    """Controlled seen/unseen corruption evaluation; no result is inferred from expectations."""
    model.eval(); output=[]
    for kind in kinds:
        for severity in levels:
            records=[]
            for index, clean in enumerate(clean_images):
                noisy,_=corrupt(clean.numpy(),kind,severity,index); result=model(torch.from_numpy(noisy)[None,None].to(device)); records.append(restoration_metrics(result['restored'],clean[None,None].to(device)))
            output.append({'kind':kind,'severity':severity,**{key:sum(x[key] for x in records)/len(records) for key in records[0]}})
    return output
