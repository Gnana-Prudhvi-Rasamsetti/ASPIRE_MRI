"""Deterministic synthetic MRI corruptions with explicit, auditable labels."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

NOISE_TO_ID = {"gaussian": 0, "rician": 1, "mixed": 2}
DEFAULT_NOISE_TYPES = tuple(NOISE_TO_ID)
DEFAULT_SEVERITIES = (0.02, 0.05, 0.10)

@dataclass(frozen=True)
class NoiseMetadata:
    kind: str
    severity: float
    seed: int


def _rng(seed: int | None) -> np.random.Generator: return np.random.default_rng(seed)


def deterministic_corruption(source_id: str, seed: int = 42,
                             noise_types=DEFAULT_NOISE_TYPES,
                             severities=DEFAULT_SEVERITIES) -> tuple[str, float, int]:
    """Choose one reproducible corruption per slice, independent of split ordering."""
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
    choice = int.from_bytes(digest[:8], "little")
    kind = noise_types[choice % len(noise_types)]
    severity = float(severities[(choice // len(noise_types)) % len(severities)])
    noise_seed = int.from_bytes(digest[8:16], "little")
    return kind, severity, noise_seed

def add_gaussian(clean: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    return np.asarray(clean, dtype=np.float32) + _rng(seed).normal(0, sigma, clean.shape).astype(np.float32)

def add_rician(clean: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    rng = _rng(seed); n1 = rng.normal(0, sigma, clean.shape); n2 = rng.normal(0, sigma, clean.shape)
    return np.sqrt((clean + n1) ** 2 + n2 ** 2).astype(np.float32)

def add_mixed(clean: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    """Gaussian left / Rician right, with a smooth severity map; severity is max sigma."""
    rng = _rng(seed); h, w = clean.shape[-2:]
    gradient = np.linspace(0.5, 1.0, w, dtype=np.float32)[None, :]
    local = sigma * gradient
    gaussian = clean + rng.normal(0, local, clean.shape)
    r1, r2 = rng.normal(0, local, clean.shape), rng.normal(0, local, clean.shape)
    rician = np.sqrt((clean + r1) ** 2 + r2 ** 2)
    mask = np.linspace(0, 1, w, dtype=np.float32)[None, :] >= 0.5
    return np.where(mask, rician, gaussian).astype(np.float32)

def corrupt(clean: np.ndarray, kind: str, severity: float, seed: int) -> tuple[np.ndarray, NoiseMetadata]:
    if kind not in NOISE_TO_ID: raise ValueError(f"Unknown noise type: {kind}")
    fn = {"gaussian": add_gaussian, "rician": add_rician, "mixed": add_mixed}[kind]
    return fn(clean, severity, seed), NoiseMetadata(kind, float(severity), seed)
