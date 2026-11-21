"""Benchmark the trained ASPIRE-MRI experts and frozen adaptive system."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets.dataloader import MRINoiseDataset
from evaluation.benchmark import benchmark
from models import AdaptiveGatingNetwork, AdaptiveRestoration, DnCNN, NoiseCharacterizationNet, Restormer, UNet
from training.common import load_checkpoint, seed_everything
from training.run_stage import checkpoint_path, read_subjects


def load_experts(device: torch.device):
    models = {
        "dncnn": DnCNN(),
        "unet": UNet(),
        "restormer": Restormer(),
    }
    for name, model in models.items():
        load_checkpoint(checkpoint_path(name), model, device)
    detector = NoiseCharacterizationNet()
    load_checkpoint(checkpoint_path("noise_detector"), detector, device)
    return models, detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure inference latency, throughput, parameters, and peak memory.")
    parser.add_argument("--clean-root", type=Path, default=Path("data/processed/clean"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=(1, 8))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--resume", action="store_true", help="Skip model and batch-size measurements already saved in the output JSON.")
    parser.add_argument("--output", type=Path, default=Path("results/efficiency.json"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Efficiency benchmark requires the CUDA GPU used for the reported training runs.")
    validation_subjects = read_subjects(args.splits_dir, "validation")
    example = MRINoiseDataset(args.clean_root, validation_subjects, seed=args.seed)[0]["noisy"]
    expert_types = {"dncnn": DnCNN, "unet": UNet, "restormer": Restormer}
    device_name = torch.cuda.get_device_name(device)
    batch_keys = [str(size) for size in args.batch_sizes]
    throughput_formula = "batch_size / latency_seconds"
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8-sig"))
        expected = (report.get("device"), report.get("seed"), report.get("warmup_iterations"),
                    report.get("timed_iterations"), report.get("batch_sizes"))
        requested = (device_name, args.seed, args.warmup, args.repetitions, batch_keys)
        if expected != requested:
            raise ValueError("Existing efficiency results use different settings; choose a new --output path.")
        if report.get("throughput_formula") == throughput_formula:
            results = report.get("results", {})
            print("Resuming efficiency benchmark from saved model/batch measurements.", flush=True)
        else:
            results = {}
            print("Efficiency report uses an older throughput formula; refreshing measurements.", flush=True)
    else:
        results = {}
    report = {
        "device": device_name,
        "seed": args.seed,
        "batch_sizes": batch_keys,
        "warmup_iterations": args.warmup,
        "timed_iterations": args.repetitions,
        "throughput_formula": throughput_formula,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save_progress() -> None:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, model_type in expert_types.items():
        existing = results.setdefault(name, {})
        pending_sizes = [size for size in args.batch_sizes if str(size) not in existing]
        if not pending_sizes:
            print(f"Skipping completed benchmark: {name}", flush=True)
            continue
        model = model_type().to(device)
        load_checkpoint(checkpoint_path(name), model, device)
        model.eval()
        for batch_size in pending_sizes:
            sample = example.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            existing[str(batch_size)] = benchmark(
                model, sample, warmup=args.warmup, repetitions=args.repetitions,
            )
            save_progress()
            del sample
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    adaptive_results = results.setdefault("adaptive_fusion", {})
    pending_sizes = [size for size in args.batch_sizes if str(size) not in adaptive_results]
    if pending_sizes:
        experts, detector = load_experts(device)
        gate = AdaptiveGatingNetwork()
        adaptive = AdaptiveRestoration(
            experts["dncnn"], experts["unet"], experts["restormer"], detector, gate,
        ).to(device)
        load_checkpoint(checkpoint_path("adaptive"), adaptive, device)
        adaptive.eval()
        for batch_size in pending_sizes:
            sample = example.unsqueeze(0).repeat(batch_size, 1, 1, 1).to(device)
            adaptive_results[str(batch_size)] = benchmark(
                adaptive, sample, warmup=args.warmup, repetitions=args.repetitions,
            )
            save_progress()
            del sample
            torch.cuda.empty_cache()
    else:
        print("Skipping completed benchmark: adaptive_fusion", flush=True)

    print(f"Efficiency results saved to {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
