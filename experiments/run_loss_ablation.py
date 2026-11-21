"""Train and evaluate the structure-preserving loss ablation."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.loss_ablation import configurations
from models import DnCNN, Restormer, UNet
from training.common import BatchProgress, load_checkpoint, seed_everything, write_log
from training.run_stage import make_loader, read_subjects
from training.train_expert import train_expert


MODEL_TYPES = {"dncnn": DnCNN, "unet": UNet, "restormer": Restormer}
BASELINE_CHECKPOINTS = {
    name: Path("checkpoints") / name / "best.pt" for name in MODEL_TYPES
}


def per_image_ssim(prediction: torch.Tensor, target: torch.Tensor, window: int = 7) -> torch.Tensor:
    """Match SSIMLoss's formula while returning one score per image in the batch."""
    c1, c2, pad = 0.01 ** 2, 0.03 ** 2, window // 2
    mu_x = F.avg_pool2d(prediction, window, 1, pad)
    mu_y = F.avg_pool2d(target, window, 1, pad)
    sigma_x = F.avg_pool2d(prediction.square(), window, 1, pad) - mu_x.square()
    sigma_y = F.avg_pool2d(target.square(), window, 1, pad) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window, 1, pad) - mu_x * mu_y
    scores = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2) + 1e-8
    )
    return scores.mean(dim=(1, 2, 3))


@torch.inference_mode()
def evaluate(model, loader, device: torch.device, use_amp: bool, description: str) -> dict:
    slice_totals = {key: 0.0 for key in ("rmse", "psnr", "ssim")}
    subject_totals: dict[str, dict[str, float]] = {}
    subject_counts: dict[str, int] = {}
    sample_count = 0
    with BatchProgress(loader, desc=description) as progress:
        for batch in progress:
            noisy, target = batch["noisy"].to(device), batch["clean"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(noisy)
            prediction = prediction.float().clamp(0, 1)
            target = target.float().clamp(0, 1)
            mse = (prediction - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
            rows = {
                "rmse": mse.sqrt().cpu().tolist(),
                "psnr": (-10 * torch.log10(mse)).cpu().tolist(),
                "ssim": per_image_ssim(prediction, target).cpu().tolist(),
            }
            subjects = batch["subject"]
            for index, subject in enumerate(subjects):
                subject = str(subject)
                if subject not in subject_totals:
                    subject_totals[subject] = {key: 0.0 for key in slice_totals}
                    subject_counts[subject] = 0
                subject_counts[subject] += 1
                for key in slice_totals:
                    value = rows[key][index]
                    slice_totals[key] += value
                    subject_totals[subject][key] += value
                sample_count += 1
            if sample_count:
                progress.set_postfix(
                    psnr=slice_totals["psnr"] / sample_count,
                    ssim=slice_totals["ssim"] / sample_count,
                )

    if sample_count == 0:
        raise ValueError("Evaluation loader returned no slices.")
    subject_means = {
        subject: {key: value / subject_counts[subject] for key, value in metrics.items()}
        for subject, metrics in subject_totals.items()
    }
    return {
        "subject_count": len(subject_means),
        "slice_count": sample_count,
        "slice_mean": {key: value / sample_count for key, value in slice_totals.items()},
        "subject_mean": {
            key: sum(metrics[key] for metrics in subject_means.values()) / len(subject_means)
            for key in slice_totals
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MSE/SSIM/gradient loss ablation on one restoration expert.")
    parser.add_argument("--model", choices=tuple(MODEL_TYPES), default="unet")
    parser.add_argument("--clean-root", type=Path, default=Path("data/processed/clean"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume each variant from its latest checkpoint if present.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Loss ablation is intended for the CUDA training environment; no CUDA device was detected.")
    batch_size = args.batch_size or (8 if args.model == "restormer" else 16)
    seed_everything(args.seed)
    train_ids = read_subjects(args.splits_dir, "expert_train")
    validation_ids = read_subjects(args.splits_dir, "validation")
    test_ids = read_subjects(args.splits_dir, "test")
    train_loader = make_loader(args.clean_root, train_ids, batch_size, args.seed, True, args.num_workers)
    validation_loader = make_loader(args.clean_root, validation_ids, batch_size, args.seed, False, args.num_workers)
    test_loader = make_loader(args.clean_root, test_ids, batch_size, args.seed, False, args.num_workers)
    use_amp = True
    losses = configurations()
    output = args.output or Path(f"results/loss_ablation_{args.model}.json")
    log_path = Path("logs") / f"loss_ablation_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ["PIPELINE_LOG_FILE"] = str(log_path)
    variant_results = {}

    for variant in ("mse", "mse_ssim", "mse_ssim_gradient"):
        checkpoint = (BASELINE_CHECKPOINTS[args.model] if variant == "mse"
                      else Path("checkpoints/ablations") / f"{args.model}_{variant}.pt")
        latest = checkpoint.with_name(checkpoint.stem + "_latest.pt")
        if variant == "mse":
            if not checkpoint.exists():
                raise FileNotFoundError(f"MSE baseline checkpoint not found: {checkpoint}")
            model = MODEL_TYPES[args.model]().to(device)
            info = load_checkpoint(checkpoint, model, device)
            write_log(f"Using existing MSE baseline {checkpoint} (epoch {info.get('epoch')}).")
        else:
            seed_everything(args.seed)
            model = MODEL_TYPES[args.model]().to(device)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            resume_variant = args.resume and latest.exists()
            if args.resume and not latest.exists():
                write_log(f"No resumable checkpoint for {variant}; starting this variant from epoch 1.")
            best = train_expert(
                model, train_loader, validation_loader, device, args.epochs,
                checkpoint_path=str(checkpoint), loss=losses[variant],
                latest_checkpoint_path=str(latest), resume=resume_variant,
            )
            info = load_checkpoint(checkpoint, model, device)
            write_log(f"Trained {variant}; best validation loss={best:.6f}; epoch={info.get('epoch')}.")

        model.eval()
        variant_results[variant] = {
            "checkpoint": str(checkpoint),
            "best_epoch": info.get("epoch"),
            "validation": evaluate(model, validation_loader, device, use_amp, f"{variant} validation"),
            "test": evaluate(model, test_loader, device, use_amp, f"{variant} test"),
        }
        del model
        torch.cuda.empty_cache()

    report = {
        "model": args.model,
        "seed": args.seed,
        "training_subjects": len(train_ids),
        "validation_subjects": len(validation_ids),
        "test_subjects": len(test_ids),
        "device": str(device),
        "mixed_precision": use_amp,
        "selection_rule": "Select the loss configuration using validation subject_mean metrics only; use test metrics for final reporting, not tuning.",
        "variants": variant_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_log(f"Loss ablation saved to {output}")
    print(json.dumps({name: value["validation"]["subject_mean"] for name, value in variant_results.items()}, indent=2))


if __name__ == "__main__":
    main()
