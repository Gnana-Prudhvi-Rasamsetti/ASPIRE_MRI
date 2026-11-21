"""NIfTI loading with canonical orientation and explicit metadata."""
from __future__ import annotations
from pathlib import Path
import numpy as np


def nifti_stem(path: str | Path) -> str:
    name=Path(path).name; lowered=name.lower()
    if lowered.endswith('.nii.gz'): return name[:-7]
    if lowered.endswith('.nii'): return name[:-4]
    return Path(path).stem


def list_t1_volumes(directory: str | Path) -> list[Path]:
    """Return only T1-weighted NIfTI volumes from a flat IXI directory."""
    root = Path(directory)
    files = sorted(set(root.glob("*.nii")) | set(root.glob("*.nii.gz")))
    return [path for path in files if nifti_stem(path).lower().endswith("-t1")]


def load_ixi_volume(path: str | Path) -> tuple[np.ndarray, dict]:
    try:
        import nibabel as nib
    except ImportError as exc:
        raise ImportError("Install nibabel to load NIfTI volumes.") from exc
    image = nib.as_closest_canonical(nib.load(str(path)))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D T1 volume at {path}, got shape {data.shape}")
    # Convert NumPy scalar metadata to JSON-native Python values.
    metadata = {
        "path": str(path),
        "shape": [int(dimension) for dimension in data.shape],
        "voxel_spacing": [float(spacing) for spacing in image.header.get_zooms()[:3]],
        "affine": [[float(value) for value in row] for row in image.affine],
    }
    return data, metadata
