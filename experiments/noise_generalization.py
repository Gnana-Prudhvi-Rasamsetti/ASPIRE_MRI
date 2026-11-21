from evaluation.robustness import evaluate_noise_levels
def run(model,clean_images,device): return evaluate_noise_levels(model,clean_images,[.02,.03,.05,.075,.10,.15],['gaussian','rician','mixed'],device)
