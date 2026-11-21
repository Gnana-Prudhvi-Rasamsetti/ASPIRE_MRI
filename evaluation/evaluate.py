from __future__ import annotations
import torch
from .metrics import restoration_metrics
def evaluate_restorer(model, loader, device):
    results=[]; model.eval()
    with torch.inference_mode():
        for batch in loader:
            output=model(batch['noisy'].to(device)); prediction=output['restored'] if isinstance(output,dict) else output
            results.append(restoration_metrics(prediction,batch['clean'].to(device)))
    return {key:sum(r[key] for r in results)/len(results) for key in results[0]}
