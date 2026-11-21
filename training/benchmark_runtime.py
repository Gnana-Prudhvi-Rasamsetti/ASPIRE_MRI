"""Measure training and validation throughput on the prepared IXI slices.

This benchmark performs real forward/backward optimizer steps, but intentionally
does not save checkpoints or modify project artifacts. Example:
  python -m training.benchmark_runtime --train-batches 100 --validation-batches 100
"""
from __future__ import annotations

import argparse
import csv
import copy
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.dataloader import MRINoiseDataset
from losses import StructurePreservingLoss
from models import (
    AdaptiveGatingNetwork,
    AdaptiveRestoration,
    DnCNN,
    NoiseCharacterizationNet,
    Restormer,
    UNet,
)


def subjects(split_dir: Path, split: str) -> list[str]:
    with (split_dir / f"{split}_subjects.csv").open(newline="", encoding="utf-8") as f:
        return [row["subject_id"] for row in csv.DictReader(f)]


def loader(root: Path, ids: list[str], batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        MRINoiseDataset(root, ids, seed=42), batch_size=batch_size,
        shuffle=shuffle, num_workers=workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0, prefetch_factor=2 if workers > 0 else None,
    )


def move_tensor(value, device, channels_last=False):
    value = value.to(device, non_blocking=device.type == "cuda")
    if channels_last:
        value = value.contiguous(memory_format=torch.channels_last)
    return value


def run_batches(model, data, optimizer, device, stage, count, warmup, criterion, amp=False, channels_last=False):
    model.train()
    if stage == "adaptive":
        model.freeze_experts()
    iterator = iter(data)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp)
    elapsed = 0.0
    measured = 0
    for index in range(count + warmup):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(data)
            batch = next(iterator)
        noisy = move_tensor(batch["noisy"], device, channels_last)
        clean = move_tensor(batch["clean"], device, channels_last)
        optimizer.zero_grad(set_to_none=True)
        if stage in ("dncnn", "unet", "restormer"):
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                output = model(noisy)
                loss = criterion(output, clean)
        elif stage == "noise_detector":
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                result = model(noisy)
                loss = torch.nn.functional.cross_entropy(result["logits"].float(), batch["noise_class"].to(device))
                loss = loss + torch.nn.functional.mse_loss(result["severity"].float(), batch["severity"].to(device))
        else:
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                result = model(noisy)
            loss = criterion(result["restored"].float(), clean)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - start
        if index >= warmup:
            elapsed += duration
            measured += 1
    return elapsed, measured


def validation_batches(model, data, device, stage, count, criterion, amp=False, channels_last=False):
    model.eval()
    iterator = iter(data)
    elapsed = 0.0
    measured = 0
    with torch.inference_mode():
        for _ in range(count):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                break
            noisy = move_tensor(batch["noisy"], device, channels_last)
            clean = move_tensor(batch["clean"], device, channels_last)
            if stage in ("dncnn", "unet", "restormer"):
                with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    model(noisy)
            elif stage == "noise_detector":
                with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    model(noisy)
            else:
                with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    result = model(noisy)
                criterion(result["restored"].float(), clean)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            measured += 1
    return elapsed, measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, default=Path("data/processed/clean"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--train-batches", type=int, default=100)
    parser.add_argument("--validation-batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--stage", choices=("dncnn", "unet", "restormer", "noise_detector", "adaptive", "fine_tune"))
    parser.add_argument("--amp", action="store_true", help="Use autocast and gradient scaling for stages that currently train in FP32.")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--cudnn-benchmark", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; run this benchmark on the project GPU environment.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    print(f"GPU: {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}", flush=True)
    expert_ids = subjects(args.splits_dir, "expert_train")
    fusion_ids = subjects(args.splits_dir, "fusion_train")
    validation_ids = subjects(args.splits_dir, "validation")
    stage_specs = [
        ("dncnn", DnCNN(), expert_ids, 8, "expert"),
        ("unet", UNet(), expert_ids, 8, "expert"),
        ("restormer", Restormer(), expert_ids, 4, "expert"),
        ("noise_detector", NoiseCharacterizationNet(), expert_ids, 8, "detector"),
    ]
    trained = {name: model for name, model, *_ in stage_specs}
    adaptive = AdaptiveRestoration(
        copy.deepcopy(trained["dncnn"]), copy.deepcopy(trained["unet"]),
        copy.deepcopy(trained["restormer"]), copy.deepcopy(trained["noise_detector"]),
        AdaptiveGatingNetwork(),
    )
    adaptive.freeze_experts()
    stage_specs.append(("adaptive", adaptive, fusion_ids, 8, "adaptive"))
    fine_tune_model = copy.deepcopy(adaptive)
    for parameter in fine_tune_model.parameters():
        parameter.requires_grad_(True)
    stage_specs.append(("fine_tune", fine_tune_model, fusion_ids, 8, "fine_tune"))
    mse = torch.nn.MSELoss()
    structure_loss = StructurePreservingLoss()

    if args.stage:
        stage_specs = [spec for spec in stage_specs if spec[0] == args.stage]
    for name, model, train_ids, default_batch_size, _ in stage_specs:
        batch_size = args.batch_size or default_batch_size
        training = loader(args.clean_root, train_ids, batch_size, True, args.workers)
        validation = loader(args.clean_root, validation_ids, batch_size, False, args.workers)
        train_steps = math.ceil(len(training.dataset) / batch_size)
        validation_steps = math.ceil(len(validation.dataset) / batch_size)
        if name == "adaptive":
            trainable = list(model.gate.parameters())
        else:
            trainable = list(model.parameters())
        model.to(device, memory_format=torch.channels_last if args.channels_last else torch.contiguous_format)
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
        criterion = structure_loss if name in ("adaptive", "fine_tune") else mse
        train_seconds, ntrain = run_batches(model, training, optimizer, device, name,
                                            args.train_batches, args.warmup_batches, criterion, args.amp, args.channels_last)
        val_seconds, nval = validation_batches(model, validation, device, name,
                                               args.validation_batches, criterion, args.amp, args.channels_last)
        train_rate = train_seconds / ntrain
        val_rate = val_seconds / max(1, nval)
        epoch_seconds = train_rate * train_steps + val_rate * validation_steps
        total_hours = epoch_seconds * args.epochs / 3600
        print(
            f"{name}: train={train_rate:.3f}s/batch ({ntrain} measured), "
            f"validation={val_rate:.3f}s/batch ({nval} measured), "
            f"steps/epoch={train_steps}+{validation_steps}, "
            f"estimated={epoch_seconds/60:.1f} min/epoch, "
            f"{args.epochs} epochs={total_hours:.1f} h",
            flush=True,
        )


if __name__ == "__main__":
    main()
