"""Deterministic subject-level partitions, independent of the training framework."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def make_subject_splits(subjects: list[str], seed: int = 42) -> dict[str, list[str]]:
    unique=sorted(set(subjects)); rng=np.random.default_rng(seed); rng.shuffle(unique)
    n=len(unique)
    if n<4: raise ValueError('At least four unique subjects are required for non-empty expert, fusion, validation, and test splits.')
    train_end=max(2,min(n-2,round(.70*n)))
    val_end=max(train_end+1,min(n-1,round(.85*n)))
    train=unique[:train_end]; expert_end=max(1,min(len(train)-1,round(.80*len(train))))
    return {'expert_train':train[:expert_end],'fusion_train':train[expert_end:],'validation':unique[train_end:val_end],'test':unique[val_end:]}


def write_splits(splits: dict[str, list[str]], directory: str | Path) -> None:
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    for name,subjects in splits.items():
        with (directory/f'{name}_subjects.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream); writer.writerow(['subject_id']); writer.writerows([[subject] for subject in subjects])
