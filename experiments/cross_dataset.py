"""Optional cross-dataset protocol. Keep results separate from IXI primary evaluation."""
from evaluation.evaluate import evaluate_restorer
def run(model, external_loader, device): return evaluate_restorer(model,external_loader,device)
