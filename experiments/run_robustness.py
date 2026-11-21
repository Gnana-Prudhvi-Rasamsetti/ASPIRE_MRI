"""Evaluate frozen adaptive restoration across seen and unseen noise conditions."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.noise_generator import NOISE_TO_ID, corrupt
from experiments.run_loss_ablation import per_image_ssim
from models import AdaptiveGatingNetwork, AdaptiveRestoration, DnCNN, NoiseCharacterizationNet, Restormer, UNet
from training.common import BatchProgress, load_checkpoint, seed_everything
from training.run_stage import checkpoint_path, read_subjects


class CleanTestSlices(Dataset):
    def __init__(self, clean_root: Path, subjects: list[str]):
        self.root = clean_root
        self.files = [path for subject in subjects for path in sorted((clean_root / subject).glob("*.npy"))]
        if not self.files:
            raise ValueError("No clean test slices found for the supplied subjects.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        clean = torch.from_numpy(np.load(path).astype(np.float32))[None]
        return clean, path.parent.name, path.relative_to(self.root).as_posix()


def build_model(device: torch.device) -> AdaptiveRestoration:
    parts = (DnCNN(), UNet(), Restormer(), NoiseCharacterizationNet(), AdaptiveGatingNetwork())
    for model, name in zip(parts[:4], ("dncnn", "unet", "restormer", "noise_detector")):
        load_checkpoint(checkpoint_path(name), model, device)
    model = AdaptiveRestoration(*parts).to(device)
    load_checkpoint(checkpoint_path("adaptive"), model, device)
    model.eval()
    return model


def noise_seed(seed: int, source: str, kind: str, severity: float) -> int:
    data = f"{seed}:{source}:{kind}:{severity:.6f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**32)


@torch.inference_mode()
def evaluate_condition(model, loader, device, use_amp, kind, severity, seed):
    aggregate = {key: 0.0 for key in ("rmse", "psnr", "ssim")}
    by_subject = defaultdict(lambda: {key: 0.0 for key in aggregate})
    subject_counts = defaultdict(int)
    logits_all, predicted_severity_all = [], []
    label_all, severity_all = [], []
    gate_sum = torch.zeros(3)
    count = 0
    total_batches = (len(loader) if len(loader) else 0)
    with BatchProgress(loader, desc=f"{kind} sigma={severity:.3f}", total=total_batches) as progress:
        for clean_batch, subjects, sources in progress:
            noisy_np = []
            for clean, source in zip(clean_batch, sources):
                noisy, _ = corrupt(
                    clean.squeeze(0).numpy(), kind, severity,
                    noise_seed(seed, source, kind, severity),
                )
                noisy_np.append(noisy)
            noisy = torch.from_numpy(np.stack(noisy_np)[:, None]).to(device)
            clean = clean_batch.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                result = model(noisy)
            prediction = result["restored"].float().clamp(0, 1)
            target = clean.float().clamp(0, 1)
            mse = (prediction - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
            scores = {
                "rmse": mse.sqrt().cpu().tolist(),
                "psnr": (-10 * torch.log10(mse)).cpu().tolist(),
                "ssim": per_image_ssim(prediction, target).cpu().tolist(),
            }
            for index, subject in enumerate(subjects):
                for metric, values in scores.items():
                    value = values[index]
                    aggregate[metric] += value
                    by_subject[subject][metric] += value
                subject_counts[subject] += 1
                count += 1

            labels = torch.full((len(subjects),), NOISE_TO_ID[kind], dtype=torch.long, device=device)
            severity_targets = torch.full((len(subjects),), severity, dtype=torch.float32, device=device)
            logits_all.append(result["noise_logits"].detach().float())
            predicted_severity_all.append(result["severity"].detach().float())
            label_all.append(labels)
            severity_all.append(severity_targets)
            gate_sum += result["weights"].detach().float().sum(0).cpu()
            progress.set_postfix(psnr=aggregate["psnr"] / count, ssim=aggregate["ssim"] / count)

    subject_mean = {
        metric: sum(values[metric] / subject_counts[subject] for subject, values in by_subject.items()) / len(by_subject)
        for metric in aggregate
    }
    logits = torch.cat(logits_all)
    predicted = logits.argmax(1).detach().cpu()
    actual = torch.cat(label_all).detach().cpu()
    matrix = torch.zeros(3, 3, dtype=torch.long)
    for truth, guess in zip(actual.tolist(), predicted.tolist()):
        matrix[truth, guess] += 1
    class_id = NOISE_TO_ID[kind]
    true_positive = matrix[class_id, class_id].float()
    precision = true_positive / (matrix[:, class_id].sum().float() + 1e-8)
    recall = true_positive / (matrix[class_id, :].sum().float() + 1e-8)
    class_f1 = 2 * precision * recall / (precision + recall + 1e-8)
    severity_error = (torch.cat(predicted_severity_all).detach().cpu() - torch.cat(severity_all).detach().cpu()).float()
    detector = {
        "accuracy": float((predicted == actual).float().mean()),
        "target_noise_class": kind,
        "target_class_precision": float(precision),
        "target_class_recall": float(recall),
        "target_class_f1": float(class_f1),
        "confusion_matrix": matrix.tolist(),
        "severity_mae": float(severity_error.abs().mean()),
        "severity_rmse": float(severity_error.square().mean().sqrt()),
    }
    return {
        "slice_count": count,
        "subject_count": len(by_subject),
        "slice_mean": {key: value / count for key, value in aggregate.items()},
        "subject_mean": subject_mean,
        "noise_characterization": detector,
        "mean_gate_weights": (gate_sum / count).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness evaluation on untouched test subjects.")
    parser.add_argument("--clean-root", type=Path, default=Path("data/processed/clean"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Skip conditions already saved in the output JSON.")
    parser.add_argument("--output", type=Path, default=Path("results/robustness_test.json"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Robustness evaluation requires the CUDA environment used for training.")
    subjects = read_subjects(args.splits_dir, "test")
    dataset = CleanTestSlices(args.clean_root, subjects)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    model = build_model(device)
    conditions = [(kind, level) for kind in NOISE_TO_ID for level in (.02, .03, .05, .075, .10, .15)]
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8-sig"))
        expected = (report.get("split"), report.get("seed"), report.get("checkpoint"))
        requested = ("test", args.seed, str(checkpoint_path("adaptive")))
        if expected != requested:
            raise ValueError("Existing robustness results use different settings; choose a new --output path.")
        results = report.get("conditions", {})
        print(f"Resuming robustness run; {len(results)}/{len(conditions)} conditions already saved.", flush=True)
    else:
        results = {}
    report = {
        "split": "test",
        "noise_types": list(NOISE_TO_ID),
        "severity_levels": [.02, .03, .05, .075, .10, .15],
        "seen_training_levels": [.02, .05, .10],
        "unseen_or_intermediate_levels": [.03, .075, .15],
        "subject_count": len(subjects),
        "seed": args.seed,
        "device": torch.cuda.get_device_name(device),
        "checkpoint": str(checkpoint_path("adaptive")),
        "selection_note": "This fixed robustness grid is for reporting; do not retune from test results.",
        "conditions": results,
    }
    for index, (kind, level) in enumerate(conditions, 1):
        key = f"{kind}_{level:.3f}"
        if key in results:
            print(f"Skipping completed condition {index}/{len(conditions)}: {key}", flush=True)
            continue
        print(f"Robustness condition {index}/{len(conditions)}", flush=True)
        results[key] = evaluate_condition(
            model, loader, device, True, kind, level, args.seed,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Robustness results saved to {args.output}")


if __name__ == "__main__":
    main()
