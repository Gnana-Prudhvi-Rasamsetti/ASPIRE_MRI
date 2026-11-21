"""Unit tests for training progress reporting utilities."""
import io
import sys
import pytest
from training.common import BatchProgress, format_duration, log_early_stopping, log_epoch_summary


def test_format_duration():
    assert format_duration(0) == "0m 00s"
    assert format_duration(45.2) == "0m 45s"
    assert format_duration(65) == "1m 05s"
    assert format_duration(3665) == "1h 01m 05s"


def test_batch_progress_iteration():
    items = [10, 20, 30]
    seen = []
    with BatchProgress(items, desc="Testing") as pbar:
        for item in pbar:
            seen.append(item)
            pbar.set_postfix(val=item, rate=1.2345)
    assert seen == items


def test_batch_progress_exception_propagation():
    with pytest.raises(ZeroDivisionError):
        with BatchProgress(range(5), desc="Fail") as pbar:
            for item in pbar:
                if item == 2:
                    _ = 1 / 0


def test_log_epoch_summary_output(capsys):
    log_epoch_summary(
        epoch=1,
        epochs=50,
        duration_seconds=75.4,
        train_loss=0.004521,
        val_loss=0.003912,
        best_val_loss=0.003912,
        saved=True,
    )
    captured = capsys.readouterr()
    assert "Epoch  1/50" in captured.out
    assert "[1m 15s]" in captured.out
    assert "Train Loss: 0.004521" in captured.out
    assert "Val Loss: 0.003912" in captured.out
    assert "Best: 0.003912" in captured.out
    assert "(Saved checkpoint)" in captured.out


def test_log_early_stopping_output(capsys):
    log_early_stopping(epoch=12, patience=10)
    captured = capsys.readouterr()
    assert "Early stopping triggered at epoch 12" in captured.out
    assert "10 epochs" in captured.out
