from __future__ import annotations
import time, torch
from .metrics import parameter_count
def benchmark(model, sample, warmup=3, repetitions=10):
    device=sample.device; model.eval()
    with torch.inference_mode():
        for _ in range(warmup): model(sample)
        if device.type=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter()
        for _ in range(repetitions): model(sample)
        if device.type=='cuda': torch.cuda.synchronize()
    seconds=(time.perf_counter()-start)/repetitions
    batch_size = int(sample.shape[0])
    return {"parameters":parameter_count(model),"latency_seconds":seconds,"images_per_second":batch_size/seconds,"peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0}
