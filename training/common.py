from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Ensure stdout supports UTF-8 on Windows consoles without charmap encode errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def format_duration(seconds: float) -> str:
    """Format elapsed seconds into a compact human-readable duration."""
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


class BatchProgress:
    """Safe batch iterator with live progress bar and rate metrics."""

    def __init__(self, loader, desc: str | None = None, total: int | None = None):
        self.loader = loader
        self.desc = desc
        self.total = total
        self.pbar = None
        self.is_tty = sys.stdout.isatty()

    def __enter__(self):
        if tqdm is not None:
            total_items = self.total or (len(self.loader) if hasattr(self.loader, "__len__") else None)
            self.pbar = tqdm(
                self.loader,
                desc=self.desc,
                total=total_items,
                file=sys.stdout,
                leave=False,
                ascii=" ▏▎▍▌▋▊▉█",
                mininterval=0.2 if self.is_tty else 3.0,
                dynamic_ncols=self.is_tty,
                ncols=None if self.is_tty else 80,
            )
        return self

    def __iter__(self):
        if self.pbar is not None:
            for item in self.pbar:
                yield item
        else:
            for item in self.loader:
                yield item

    def set_postfix(self, **kwargs):
        if self.pbar is not None:
            formatted = {}
            for key, val in kwargs.items():
                if isinstance(val, float):
                    formatted[key] = f"{val:.4f}"
                else:
                    formatted[key] = str(val)
            self.pbar.set_postfix(formatted, refresh=False)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pbar is not None:
            self.pbar.close()
            if not self.is_tty:
                sys.stdout.write("\n")
                sys.stdout.flush()
        return False  # Propagate exceptions without swallowing


def write_log(line: str) -> None:
    """Print a line to stdout and append it to PIPELINE_LOG_FILE if configured."""
    print(line, flush=True)
    log_file = os.environ.get("PIPELINE_LOG_FILE")
    if log_file:
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def log_epoch_summary(
    epoch: int,
    epochs: int,
    duration_seconds: float,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    saved: bool,
) -> None:
    """Print a clean, single-line epoch summary with timing and loss metrics."""
    duration_str = format_duration(duration_seconds)
    saved_tag = " (Saved checkpoint)" if saved else ""
    line = (
        f"Epoch {epoch:2d}/{epochs:<2d} [{duration_str}] | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"Best: {best_val_loss:.6f}{saved_tag}"
    )
    write_log(line)


def log_early_stopping(epoch: int, patience: int) -> None:
    """Log when training halts early due to plateaued validation loss."""
    write_log(
        f"Early stopping triggered at epoch {epoch} (no validation improvement for {patience} epochs)."
    )


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().cpu(),
        "cuda": [state.cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available():
        for device_index, device_state in enumerate(state.get("cuda", [])[:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(device_state.cpu(), device_index)


def save_checkpoint(
    path, model, optimizer, epoch, metrics, config,
    scheduler=None, scaler=None, training_state=None,
):
    """Atomically save a model checkpoint and optional full resume state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    if training_state is not None:
        checkpoint["training_state"] = training_state
    checkpoint["rng_state"] = capture_rng_state()

    temporary_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(path, model, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def resume_checkpoint(path, model, optimizer, scheduler, scaler, device="cpu"):
    """Restore a complete epoch-boundary training state from a latest checkpoint."""
    checkpoint = load_checkpoint(path, model, device)
    if "optimizer_state" not in checkpoint or "scheduler_state" not in checkpoint:
        raise ValueError(
            f"{path} is a model-only checkpoint and cannot resume training. "
            "Resume requires a latest.pt checkpoint created by the resumable trainer."
        )
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])
    restore_rng_state(checkpoint.get("rng_state"))
    state = checkpoint.get("training_state", {})
    metrics = checkpoint.get("metrics", {})
    return {
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "best_validation_loss": float(state.get("best_validation_loss", metrics.get("validation_loss", float("inf")))),
        "stale_epochs": int(state.get("stale_epochs", 0)),
    }
