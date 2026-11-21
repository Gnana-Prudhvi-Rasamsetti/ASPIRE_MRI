"""Compare trained ASPIRE-MRI checkpoints on one subject split."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from evaluation.metrics import characterization_metrics
from losses.ssim_loss import SSIMLoss
from models import AdaptiveGatingNetwork, AdaptiveRestoration, DnCNN, NoiseCharacterizationNet, Restormer, UNet
from training.common import BatchProgress, load_checkpoint, seed_everything
from training.run_stage import checkpoint_path, make_loader, read_subjects


def build_model(device: torch.device, checkpoint: Path) -> tuple[AdaptiveRestoration, dict]:
    """Load all expert weights, then load one fusion checkpoint."""
    parts = (DnCNN(), UNet(), Restormer(), NoiseCharacterizationNet(), AdaptiveGatingNetwork())
    for model, name in zip(parts[:4], ("dncnn", "unet", "restormer", "noise_detector")):
        path = checkpoint_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Missing required checkpoint: {path}")
        load_checkpoint(path, model, device)
    model = AdaptiveRestoration(*parts).to(device)
    info = load_checkpoint(checkpoint, model, device)
    model.eval()
    return model, info


@torch.inference_mode()
def evaluate(model: AdaptiveRestoration, loader, device: torch.device, use_amp: bool,
             description: str) -> dict:
    names = ("dncnn", "unet", "restormer", "equal_weight", "adaptive")
    totals = {name: defaultdict(float) for name in names}
    count = 0
    all_logits, all_predicted_severity, all_classes, all_target_severity = [], [], [], []
    gate_by_class = defaultdict(list)
    ssim_loss = SSIMLoss()
    noise_correct = 0

    with BatchProgress(loader, desc=description) as progress:
        for batch_index, batch in enumerate(progress, start=1):
            noisy = batch["noisy"].to(device)
            clean = batch["clean"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                result = model(noisy)
            experts = result["experts"]
            predictions = (experts[:, 0], experts[:, 1], experts[:, 2], experts.mean(1), result["restored"])
            clean_float = clean.float().clamp(0, 1)
            batch_count = clean.shape[0]
            for name, prediction in zip(names, predictions):
                prediction = prediction.float().clamp(0, 1)
                per_image_mse = (prediction - clean_float).square().mean(dim=(1, 2, 3))
                totals[name]["rmse"] += per_image_mse.sqrt().sum().item()
                totals[name]["psnr"] += (-10 * torch.log10(per_image_mse.clamp_min(1e-12))).sum().item()
                totals[name]["ssim"] += (1 - ssim_loss(prediction, clean_float).item()) * batch_count
            labels = batch["noise_class"].to(device)
            all_logits.append(result["noise_logits"].detach().float())
            all_predicted_severity.append(result["severity"].detach().float().reshape(-1))
            all_classes.append(labels.detach().reshape(-1))
            all_target_severity.append(batch["severity"].to(device).detach().reshape(-1))
            noise_correct += (result["noise_logits"].argmax(1) == labels).sum().item()
            for label, weights in zip(labels.cpu().tolist(), result["weights"].cpu().tolist()):
                gate_by_class[int(label)].append(weights)
            count += clean.shape[0]
            progress.set_postfix(
                psnr=totals["adaptive"]["psnr"] / count,
                ssim=totals["adaptive"]["ssim"] / count,
                noise_acc=noise_correct / count,
            )

    if not count:
        raise ValueError("Validation loader produced no samples.")
    class_metrics = characterization_metrics(
        torch.cat(all_logits), torch.cat(all_predicted_severity), torch.cat(all_classes),
        torch.cat(all_target_severity),
    )
    return {
        "sample_count": count,
        "restoration": {
            name: {metric: value / count for metric, value in metrics.items()}
            for name, metrics in totals.items()
        },
        "noise_characterization": class_metrics,
        "mean_gate_weights_by_noise_class": {
            str(label): [sum(row[i] for row in rows) / len(rows) for i in range(3)]
            for label, rows in sorted(gate_by_class.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained checkpoints on validation or held-out test subjects.")
    parser.add_argument("--clean-root", type=Path, default=Path("data/processed/clean"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed-precision inference.")
    parser.add_argument(
        "--include-fine-tuned",
        action="store_true",
        help="Also evaluate the optional end-to-end fine-tuned checkpoint.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subjects = read_subjects(args.splits_dir, args.split)
    loader = make_loader(args.clean_root, subjects, args.batch_size, args.seed, False, args.num_workers)
    use_amp = device.type == "cuda" and not args.no_amp

    checkpoints = {"frozen_adaptive": checkpoint_path("adaptive")}
    if args.include_fine_tuned:
        checkpoints["fine_tuned"] = checkpoint_path("fine_tune")
    report = {
        "split": args.split,
        "subject_count": len(subjects),
        "seed": args.seed,
        "device": str(device),
        "mixed_precision": use_amp,
        "checkpoints": {},
        "note": ("Use validation results for model selection." if args.split == "validation"
                 else "Held-out test results. Do not use these scores to tune or select models."),
    }
    for name, path in checkpoints.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing fusion checkpoint: {path}")
        print(f"Loading {name} checkpoint: {path}", flush=True)
        model, info = build_model(device, path)
        print(f"Evaluating {name} on {args.split} split ({len(loader)} batches)", flush=True)
        report["checkpoints"][name] = {
            "path": str(path),
            "epoch": info.get("epoch"),
            **evaluate(model, loader, device, use_amp, f"{name} / {args.split}"),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = args.output or Path(f"results/{args.split}_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"{args.split.capitalize()} comparison saved to {output}")
    print(json.dumps({key: value["restoration"] for key, value in report["checkpoints"].items()}, indent=2))


if __name__ == "__main__":
    main()
