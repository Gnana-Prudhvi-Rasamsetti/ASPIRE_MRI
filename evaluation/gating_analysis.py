from __future__ import annotations
import torch
@torch.inference_mode()
def summarize_gating(model, loader, device):
    """Report measured (never hand-assigned) average expert weights per noise label."""
    model.eval(); groups = {}
    for batch in loader:
        result=model(batch['noisy'].to(device)); labels=batch['noise_class'].tolist()
        for label, weights in zip(labels, result['weights'].cpu().tolist()): groups.setdefault(int(label),[]).append(weights)
    return {label: torch.tensor(weights).mean(0).tolist() for label,weights in groups.items()}
