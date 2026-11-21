"""Volume-level, leakage-safe MRI preprocessing primitives."""
from __future__ import annotations
import numpy as np


def robust_normalize(volume: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Normalize one volume using non-background percentiles, preserving slice relations."""
    x = np.asarray(volume, dtype=np.float32)
    foreground = x[np.isfinite(x) & (x > 0)]
    if foreground.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(foreground, [lower, upper])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def axial_slices(volume: np.ndarray, size: int = 128, min_foreground_fraction: float = 0.02):
    """Yield resized axial slices that contain sufficient anatomy."""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3-D volume, got {volume.shape}")
    for index in range(volume.shape[2]):
        image = volume[:, :, index]
        if (image > 0).mean() < min_foreground_fraction:
            continue
        height,width=image.shape
        rows=np.clip((np.arange(size,dtype=np.float32)+0.5)*(height/size)-0.5,0,height-1)
        columns=np.clip((np.arange(size,dtype=np.float32)+0.5)*(width/size)-0.5,0,width-1)
        y0=rows.astype(np.intp); x0=columns.astype(np.intp); y1=np.minimum(y0+1,height-1); x1=np.minimum(x0+1,width-1)
        wy=(rows-y0)[:,None]; wx=(columns-x0)[None,:]
        top=image[y0[:,None],x0[None,:]]*(1-wx)+image[y0[:,None],x1[None,:]]*wx
        bottom=image[y1[:,None],x0[None,:]]*(1-wx)+image[y1[:,None],x1[None,:]]*wx
        yield index,(top*(1-wy)+bottom*wy).astype(np.float32)
