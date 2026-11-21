from evaluation.evaluate import evaluate_restorer
from evaluation.benchmark import benchmark
def run(models,loader,sample,device): return {name:{'restoration':evaluate_restorer(model,loader,device),'efficiency':benchmark(model,sample.to(device))} for name,model in models.items()}
