# ASPIRE-MRI

ASPIRE-MRI is an academic, reproducible framework for restoring synthetically degraded T1-weighted brain MRI slices. It compares DnCNN, U-Net, and a compact multi-scale Restormer-style model, then learns a noise-aware adaptive fusion of their outputs.

> Research use only. This project is not a clinically validated enhancement, diagnostic, or treatment system. It evaluates controlled synthetic degradation rather than claiming to model every real scanner-noise process.

## Implemented architecture

### Data pipeline

- NiBabel loading of `.nii` and `.nii.gz` IXI volumes in canonical orientation.
- Robust per-volume percentile normalization over non-background voxels.
- Axial 2-D slice extraction, empty-slice rejection, and 128 x 128 resizing.
- Leakage-safe subject-level 70/15/15 train/validation/test partitioning, with training divided into `expert_train` and `fusion_train`.
- Deterministic Gaussian, Rician, and spatially varying mixed corruption, including type, severity, and seed metadata.
- `MRINoiseDataset` for on-the-fly training pairs and `datasets.generate_pairs` for persisted noisy-clean pairs.

### Models

| Component | Purpose |
| --- | --- |
| DnCNN | Residual-learning CNN denoising baseline. |
| U-Net | Encoder-decoder restoration with spatial skip connections. |
| Restormer | Compact single-channel multi-scale encoder-decoder with MDTA and gated feed-forward blocks. |
| Noise characterization network | CNN producing Gaussian/Rician/mixed logits, continuous severity, and learned embedding. |
| Adaptive gate | Softmax MLP using detector embedding, class probabilities, and severity to make three weights summing to one. |
| Adaptive restoration | Runs experts and produces expert outputs, fused restoration, detector predictions, and weights. |

### Losses, evaluation, and experiments

- MSE, SSIM, gradient, and combined structure-preserving objectives.
- PSNR, SSIM, RMSE, parameter count, inference latency, throughput, and GPU memory utilities.
- Accuracy, macro precision/recall/F1, confusion matrix, severity MAE/RMSE for noise characterization.
- Fair expert comparison, equal-weight versus adaptive fusion, loss ablation, unseen-noise robustness, gating analysis, and optional cross-dataset helpers.

## Repository map

```text
data/          Raw IXI volumes, processed slices/pairs, subject splits
datasets/      NIfTI loading, preprocessing, slicing, corruption, dataloader
models/        Three experts, noise detector, gate, adaptive fusion
losses/        MSE, SSIM, gradient, combined loss
training/      Trainers, fine-tuning, and `run_stage` training CLI
evaluation/    Metrics, evaluation, benchmark, robustness, plots, gate analysis
experiments/   Comparisons, ablations, and generalization experiments
checkpoints/   Best saved model states
results/       Metrics, graphs, restorations, and error maps
app/           Streamlit demonstration
configs/       Dataset, training, and experiment configuration defaults
```

## Setup and verification

Run every command from the project root:

```powershell
cd "C:\Deep Learning\DL_Project_Final"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --smoke-test
python -m pytest -q
```

The smoke test builds the complete model, runs a 128 x 128 forward pass, and checks that the gate weights sum to one. The test suite verifies deterministic noise, normalization, model/fusion shapes, differentiable loss, and characterization metrics. CPU is suitable for verification; full training is intended for a CUDA-capable GPU.

## Dataset preparation

The dataset is intentionally not bundled in this share-ready repository. Download the official **T1 images (all images)** archive from [the IXI dataset page](https://brain-development.org/ixi-dataset/), extract its T1 `.nii` or `.nii.gz` volumes into `data/raw/IXI/`, and keep other modalities out of that folder. The dataset page identifies the data as CC BY-SA 3.0 and requests source acknowledgment; preserve that attribution in reports and redistributed derivatives. Preparation and extraction accept filenames ending in `-T1`. Run the following in order afterward.

### 1. Make leakage-safe subject partitions

```powershell
python -m datasets.prepare data/raw/IXI data/splits
```

This writes `expert_train_subjects.csv`, `fusion_train_subjects.csv`, `validation_subjects.csv`, and `test_subjects.csv`. Never use test subjects for tuning.

### 2. Create normalized clean slices

```powershell
python -m datasets.slice_extraction data/raw/IXI data/processed/clean
```

Each subject receives suitable normalized 128 x 128 `.npy` slices and `metadata.json`. Processing the full IXI collection takes time.

### 3. Persist auditable noisy-clean pairs

```powershell
python -m datasets.generate_pairs data/processed/clean data/processed --progress-every 100
```

This command is resumable. It skips complete pairs and writes the noisy slice plus JSON metadata into `gaussian`, `rician`, and `mixed` folders. Metadata points back to the existing clean slice instead of duplicating it, reducing storage. Each slice receives one deterministic noise type, severity, and random seed; the training loader uses that same assignment so persisted pairs can be audited against training inputs. If you change `--seed`, pass the same seed to training and verification.

### 4. Verify the complete pre-training boundary

```powershell
python -m datasets.verify_preprocessing
```

This read-only validation confirms that T1 raw volumes match the four non-overlapping subject splits, clean slice folders/metadata exist, and every slice has a persisted pair with the same noise type, severity, and seed used by training.

## Complete training workflow

Training is staged so adaptive fusion learns from subjects that did not train the individual experts. The `training.run_stage` command uses subject CSVs, deterministic corruption assignments, validation-selected checkpoints, learning-rate reduction, and early stopping for trainable stages. CUDA mixed precision is used for the restoration experts, adaptive fusion, and optional fine-tuning when available. On the project RTX 4060, a short throughput benchmark favored two data workers, batch size 16 for most stages, and batch size 8 for Restormer. Override `--num-workers 0` if multiprocessing is unavailable in another environment.

### Automated sequential pipeline runner

To run the complete training pipeline unattended without manually triggering each stage, use `run_pipeline.ps1`. It activates `.venv-gpu`, trains each required stage sequentially (`dncnn` -> `unet` -> `restormer` -> `noise_detector` -> `adaptive`), aborts immediately if any stage fails, automatically resumes interrupted stages from `latest.pt`, streams console progress in real time, and logs to `logs/pipeline_<timestamp>.log`:

```powershell
# Run core stages 1 through 3 sequentially:
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1

# Or include optional Stage 4 (end-to-end fine-tuning):
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1 -IncludeFineTune
```

### Stage 1 - Independent restoration experts

```powershell
python -m training.run_stage dncnn --epochs 50 --batch-size 16
python -m training.run_stage unet --epochs 50 --batch-size 16
python -m training.run_stage restormer --epochs 50 --batch-size 8
```

These use `expert_train`, select using validation data, and save `checkpoints/dncnn/best.pt`, `checkpoints/unet/best.pt`, and `checkpoints/restormer/best.pt`.

### Stage 2 - Noise characterization

```powershell
python -m training.run_stage noise_detector --epochs 50 --batch-size 16
```

The CNN learns noise class and severity, selects its checkpoint using validation loss, and saves `checkpoints/noise_detector/best.pt`.

### Stage 3 - Adaptive fusion

```powershell
python -m training.run_stage adaptive --epochs 50 --batch-size 16
```

This requires the four prior checkpoints, freezes the experts and detector, trains the gate on `fusion_train`, selects the checkpoint using validation loss, and saves `checkpoints/adaptive/best.pt`.

### Stage 4 - Optional end-to-end fine-tuning

```powershell
python -m training.run_stage fine_tune --epochs 10 --batch-size 16 --learning-rate 0.00001
```

Evaluate frozen fusion before this optional step and retain only the approach with better validation results. Use `--clean-root`, `--splits-dir`, `--epochs`, `--batch-size`, `--learning-rate`, and `--seed` to customize a run. Lower batch size if GPU memory is exhausted.

Each stage keeps its best validation model (`best.pt`) and an atomic latest-epoch checkpoint (`latest.pt`; fine-tuning uses `fine_tuned_latest.pt`) with optimizer, scheduler, mixed-precision scaler, early-stopping, and random-number state. If a run is interrupted, rerun the same stage with `--resume`; it continues after the last fully completed epoch. An interruption during an epoch may require that epoch to run again. For example:

```powershell
python -m training.run_stage restormer --epochs 50 --resume
```

Keep the same stage and data split when resuming. You can raise `--epochs` to extend a run that has not already stopped early. Older `best.pt` files remain usable for inference, but they do not contain the full state needed to resume; the new `latest.pt` is created as the stage trains.

## Evaluation and reporting

Use untouched test subjects only for final reporting. Import helpers from `evaluation` and `experiments` in an analysis notebook/script:

```python
from evaluation.evaluate import evaluate_restorer
from evaluation.benchmark import benchmark
from experiments.model_comparison import run as compare_models
from experiments.fusion_comparison import compare as compare_fusion
from experiments.noise_generalization import run as noise_generalization
from evaluation.gating_analysis import summarize_gating
```

The project also includes command-line runners for the final comparison and experiments. Run them from the project root with the CUDA environment active. The validation comparison is used for model selection; after selecting and documenting the model, run the held-out test comparison once. Do not tune from test results.

```powershell
# Compare expert, equal-weight, frozen adaptive, and fine-tuned outputs on validation.
python -u -m evaluation.run_validation --split validation --batch-size 8 --num-workers 2

# After model selection is fixed, record the held-out test comparison.
python -u -m evaluation.run_validation --split test --batch-size 8 --num-workers 2

# Compare MSE, MSE+SSIM, and MSE+SSIM+gradient on the selected expert.
# Reuses its existing MSE checkpoint, trains the two additional variants,
# selects from validation metrics, and records test metrics for reporting.
python -u -m experiments.run_loss_ablation --model unet --epochs 50 --batch-size 16 --num-workers 2

# Evaluate the fixed noise and severity grid on the test subjects.
python -u -m experiments.run_robustness --batch-size 8 --num-workers 2

# Measure parameters, inference time, throughput, and peak GPU memory.
python -u -m evaluation.run_efficiency --batch-sizes 1 8
```

To run the remaining GPU experiments sequentially with one command, use the PowerShell runner. It checks CUDA, resumes saved ablation/robustness/efficiency progress, stops if a stage fails, and leaves progress output attached to the terminal:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_remaining_experiments.ps1
```

The runners save machine-readable JSON reports under `results/` and display progress while processing batches. Loss-ablation checkpoints and training logs are saved under `checkpoints/ablations/` and `logs/`.

Required experiments are: fair comparison of three experts; each expert versus equal and adaptive fusion; loss ablations; seen/intermediate/higher unseen noise severity; cross-noise robustness; detector accuracy/regression; measured gating weights; and computational efficiency. Do not make performance or clinical claims until held-out results are completed.

## Streamlit demo

After adaptive training:

```powershell
streamlit run app/streamlit_app.py
```

Upload a NIfTI volume, provide `checkpoints/adaptive/best.pt`, select a slice, and inspect all three expert outputs, the adaptive restoration, noise probabilities, predicted severity, and gating weights. Upload a clean reference with matching geometry to see PSNR and SSIM. It is a research visualization tool only.

## Troubleshooting

| Symptom | Resolution |
| --- | --- |
| `ModuleNotFoundError: nibabel` | Run `python -m pip install -r requirements.txt` in the same environment. |
| Pair generation appears silent | Add `--progress-every 100`; wait for `Completed:`. |
| CUDA out-of-memory | Reduce `--batch-size`; the RTX 4060 benchmark favored batch size 8 for Restormer and 16 for the other stages. |
| Adaptive stage says a checkpoint is missing | Train all three experts and the detector first. Fine-tuning also requires `checkpoints/adaptive/best.pt`. |
| No clean slices found | Complete slice extraction and verify `data/processed/clean/<subject>/*.npy`. |

## Reproducibility and data handling

Record splits, seed, package versions, model/loss settings, optimizer settings, hardware, and checkpoint versions alongside every result. Respect the IXI licence and source attribution; do not commit licensed source data or large checkpoints without an appropriate policy.

## Sharing this repository

This repository is deliberately lightweight for sharing. It contains the source code, configurations, tests, documentation, and deterministic subject split CSVs only. It does **not** include downloaded IXI volumes, generated slices/pairs, model checkpoints, experiment outputs, or Python caches. A teammate should install dependencies, download IXI from the official source, and follow the dataset-preparation sequence above before training.
