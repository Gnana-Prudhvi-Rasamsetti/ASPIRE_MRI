"""Subject-aware datasets. Splits are made before slice paths are enumerated."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .noise_generator import corrupt, NOISE_TO_ID, deterministic_corruption
from .splitting import make_subject_splits, write_splits

class MRINoiseDataset(Dataset):
    def __init__(self, clean_root: str | Path, subjects: list[str], noise_types=("gaussian", "rician", "mixed"), levels=(.02, .05, .10), seed: int = 42):
        self.seed, self.noise_types, self.levels = seed, tuple(noise_types), tuple(levels)
        root = Path(clean_root); self.files = [p for s in subjects for p in sorted((root / s).glob("*.npy"))]
        self.root = root
        if not self.files: raise ValueError("No preprocessed slices found for supplied subject split.")
    def __len__(self): return len(self.files)
    def __getitem__(self, index):
        path = self.files[index]
        clean = np.load(path).astype(np.float32)
        kind, severity, sample_seed = deterministic_corruption(path.relative_to(self.root).as_posix(), self.seed, self.noise_types, self.levels)
        noisy, meta = corrupt(clean, kind, severity, sample_seed)
        return {"noisy": torch.from_numpy(noisy)[None], "clean": torch.from_numpy(clean)[None], "noise_class": torch.tensor(NOISE_TO_ID[kind]), "severity": torch.tensor(meta.severity), "subject": self.files[index].parent.name}
