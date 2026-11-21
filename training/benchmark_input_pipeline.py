"""Compare on-the-fly noise generation with prepared noisy-pair reads."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.dataloader import MRINoiseDataset
from datasets.noise_generator import NOISE_TO_ID, deterministic_corruption


class PreparedPairs(Dataset):
    """Read the exact deterministic pair written by datasets.generate_pairs."""
    def __init__(self, clean_root: Path, pair_root: Path, subject_ids: list[str], seed: int):
        self.clean_root, self.pair_root, self.seed = clean_root, pair_root, seed
        self.files = [p for subject in subject_ids for p in sorted((clean_root / subject).glob("*.npy"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        relative = path.relative_to(self.clean_root).as_posix()
        kind, severity, _ = deterministic_corruption(relative, self.seed)
        noisy_path = self.pair_root / kind / path.parent.name / f"{path.stem}_noisy.npy"
        clean = np.load(path).astype(np.float32)
        noisy = np.load(noisy_path).astype(np.float32)
        return {
            "noisy": torch.from_numpy(noisy)[None],
            "clean": torch.from_numpy(clean)[None],
            "noise_class": torch.tensor(NOISE_TO_ID[kind]),
            "severity": torch.tensor(severity),
        }


def read_subjects(path: Path, split: str) -> list[str]:
    with (path / f"{split}_subjects.csv").open(newline="", encoding="utf-8") as f:
        return [row["subject_id"] for row in csv.DictReader(f)]


def measure(dataset, batch_size: int, workers: int, batches: int, warmup: int):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
    iterator = iter(loader)
    durations = []
    for i in range(batches + warmup):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        duration = time.perf_counter() - start
        if i >= warmup:
            durations.append(duration)
        del batch
    return sum(durations) / len(durations), len(dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path, default=Path("data/processed/clean"))
    parser.add_argument("--pairs-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--warmup-batches", type=int, default=10)
    args = parser.parse_args()
    ids = read_subjects(args.splits_dir, "expert_train")
    cases = {
        "dynamic noise, workers=0": (MRINoiseDataset(args.clean_root, ids, seed=42), 0),
        "dynamic noise, workers=2": (MRINoiseDataset(args.clean_root, ids, seed=42), 2),
        "saved pairs, workers=0": (PreparedPairs(args.clean_root, args.pairs_root, ids, 42), 0),
        "saved pairs, workers=2": (PreparedPairs(args.clean_root, args.pairs_root, ids, 42), 2),
    }
    print(f"GPU available: {torch.cuda.is_available()} | samples: {len(cases[next(iter(cases))][0])}", flush=True)
    for label, (dataset, workers) in cases.items():
        mean, sample_count = measure(dataset, args.batch_size, workers, args.batches, args.warmup_batches)
        print(
            f"{label}: {mean:.3f}s/batch, {args.batch_size / mean:.1f} samples/s "
            f"({args.batches} measured batches; {sample_count} samples total)", flush=True,
        )


if __name__ == "__main__":
    main()
