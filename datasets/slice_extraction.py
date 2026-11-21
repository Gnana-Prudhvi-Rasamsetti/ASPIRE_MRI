"""Command-line preprocessing of IXI NIfTI volumes into .npy slices."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .ixi_loader import load_ixi_volume, list_t1_volumes
from .preprocessing import robust_normalize, axial_slices


def extract_volume(source: Path, destination: Path, size: int = 128) -> int:
    volume, metadata = load_ixi_volume(source)
    volume = robust_normalize(volume)
    subject = source.name.replace(".nii.gz", "").replace(".nii", "")
    subject_dir = destination / subject
    subject_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for index, image in axial_slices(volume, size):
        np.save(subject_dir / f"axial_{index:03d}.npy", image)
        count += 1
    (subject_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path); parser.add_argument("output_dir", type=Path)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()
    files = list_t1_volumes(args.input_dir)
    if not files:
        raise ValueError(f"No T1-weighted NIfTI volumes found in {args.input_dir}.")
    for file in files: print(file.name, extract_volume(file, args.output_dir, args.size))
