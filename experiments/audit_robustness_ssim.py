"""Audit the low SSIM scores at intermediate synthetic noise severities."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.noise_generator import NOISE_TO_ID, corrupt
from evaluation.metrics import parameter_count
from experiments.run_loss_ablation import per_image_ssim
from experiments.run_robustness import CleanTestSlices, build_model, noise_seed
from training.common import BatchProgress, seed_everything
from training.run_stage import read_subjects


def paired_seed(seed: int, source: str, kind: str) -> int:
    digest = hashlib.sha256(f"{seed}:{source}:{kind}:paired-severity".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def score_batch(prediction: torch.Tensor, clean: torch.Tensor) -> dict[str, list[float]]:
    prediction = prediction.float().clamp(0, 1)
    clean = clean.float().clamp(0, 1)
    mse = (prediction - clean).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return {
        "rmse": mse.sqrt().cpu().tolist(),
        "psnr": (-10 * torch.log10(mse)).cpu().tolist(),
        "ssim": per_image_ssim(prediction, clean).cpu().tolist(),
    }


@torch.inference_mode()
def audit_condition(model, loader, device, kind, severity, seed, seed_mode):
    metrics = ("rmse", "psnr", "ssim")
    slice_totals = {source: {metric: 0.0 for metric in metrics} for source in ("noisy", "restored")}
    subject_totals = {
        source: defaultdict(lambda: {metric: 0.0 for metric in metrics})
        for source in ("noisy", "restored")
    }
    subject_counts = defaultdict(int)
    predicted_severity = []
    detector_logits = []
    gate_sum = torch.zeros(3)
    slice_count = 0

    with BatchProgress(loader, desc=f"SSIM audit {seed_mode} {kind} {severity:.3f}") as progress:
        for clean_batch, subjects, sources in progress:
            noisy_items = []
            for clean, source in zip(clean_batch, sources):
                item_seed = (noise_seed(seed, source, kind, severity) if seed_mode == "independent"
                             else paired_seed(seed, source, kind))
                noisy_item, _ = corrupt(clean.squeeze(0).numpy(), kind, severity, item_seed)
                noisy_items.append(noisy_item)
            noisy = torch.from_numpy(np.stack(noisy_items)[:, None]).to(device)
            clean = clean_batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                result = model(noisy)

            predictions = {"noisy": noisy, "restored": result["restored"]}
            batch_scores = {name: score_batch(prediction, clean) for name, prediction in predictions.items()}
            for index, subject in enumerate(subjects):
                subject_counts[subject] += 1
                slice_count += 1
                for source_name in predictions:
                    for metric in metrics:
                        value = batch_scores[source_name][metric][index]
                        slice_totals[source_name][metric] += value
                        subject_totals[source_name][subject][metric] += value

            predicted_severity.extend(result["severity"].float().cpu().tolist())
            detector_logits.append(result["noise_logits"].float().detach().cpu())
            gate_sum += result["weights"].float().sum(0).cpu()
            current_count = slice_count
            progress.set_postfix(
                restored_ssim=slice_totals["restored"]["ssim"] / current_count,
                noisy_ssim=slice_totals["noisy"]["ssim"] / current_count,
            )

    subject_means = {
        source_name: {
            metric: sum(values[metric] / subject_counts[subject] for subject, values in by_subject.items()) / len(by_subject)
            for metric in metrics
        }
        for source_name, by_subject in subject_totals.items()
    }
    noisy_mean = {key: value / slice_count for key, value in slice_totals["noisy"].items()}
    restored_mean = {key: value / slice_count for key, value in slice_totals["restored"].items()}
    errors = torch.tensor(predicted_severity, dtype=torch.float32) - severity
    logits = torch.cat(detector_logits)
    labels = torch.full((slice_count,), NOISE_TO_ID[kind], dtype=torch.long)
    predicted = logits.argmax(1)
    return {
        "subject_count": len(subject_counts),
        "slice_count": slice_count,
        "slice_mean": {"noisy_input": noisy_mean, "restored": restored_mean},
        "subject_mean": subject_means,
        "restoration_change": {
            metric: restored_mean[metric] - noisy_mean[metric] for metric in metrics
        },
        "noise_detector": {
            "accuracy": float((predicted == labels).float().mean()),
            "severity_mae": float(errors.abs().mean()),
            "severity_rmse": float(errors.square().mean().sqrt()),
        },
        "mean_gate_weights": (gate_sum / slice_count).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recheck intermediate-severity SSIM using noisy-input controls and paired seeds.")
    parser.add_argument("--clean-root", type=Path, default=Path("data/processed/clean"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/robustness_ssim_audit.json"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SSIM audit requires the CUDA environment used for the robustness run.")
    subjects = read_subjects(args.splits_dir, "test")
    dataset = CleanTestSlices(args.clean_root, subjects)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    model = build_model(device)
    robustness_path = Path("results/robustness_test.json")
    robustness = json.loads(robustness_path.read_text(encoding="utf-8-sig"))
    config = {"split": "test", "seed": args.seed, "subject_count": len(subjects), "checkpoint": str(Path("checkpoints/adaptive/best.pt"))}
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8-sig"))
        if report.get("config") != config:
            raise ValueError("Existing audit results use different settings; choose another --output path.")
        results = report.get("conditions", {})
        print(f"Resuming SSIM audit with {len(results)} saved condition results.", flush=True)
    else:
        results = {}
    report = {
        "config": config,
        "purpose": "Verify intermediate-severity SSIM, compare noisy-input controls, and test whether independent noise seeds caused the dip.",
        "seed_modes": ["independent", "paired"],
        "noise_types": list(NOISE_TO_ID),
        "severity_levels": [.02, .03, .05],
        "conditions": results,
    }
    conditions = [(mode, kind, level) for mode in ("independent", "paired")
                  for kind in NOISE_TO_ID for level in (.02, .03, .05)]
    saved_main = robustness.get("conditions", {})
    for index, (mode, kind, level) in enumerate(conditions, 1):
        key = f"{mode}_{kind}_{level:.3f}"
        if key in results:
            print(f"Skipping saved audit condition {index}/{len(conditions)}: {key}", flush=True)
            continue
        scores = audit_condition(model, loader, device, kind, level, args.seed, mode)
        if mode == "independent":
            original = saved_main.get(f"{kind}_{level:.3f}", {}).get("subject_mean", {}).get("ssim")
            scores["original_report_subject_ssim"] = original
            scores["difference_from_original_report"] = (
                scores["subject_mean"]["restored"]["ssim"] - original if original is not None else None
            )
        results[key] = scores
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Audit condition {index}/{len(conditions)} saved: {key}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SSIM audit saved to {args.output}; adaptive parameters={parameter_count(model):,}")


if __name__ == "__main__":
    main()
