"""Non-diagnostic ASPIRE-MRI demo. Requires a trained adaptive checkpoint."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import streamlit as st
import torch

# Streamlit may put only ``app/`` on sys.path when launched with
# ``streamlit run app/streamlit_app.py``. Add the repository root explicitly so
# local packages such as ``datasets`` and ``models`` resolve in every launcher.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.ixi_loader import load_ixi_volume
from datasets.noise_generator import NOISE_TO_ID, corrupt
from datasets.preprocessing import axial_slices, robust_normalize
from evaluation.metrics import restoration_metrics
from main import build_model
from training.common import load_checkpoint


st.set_page_config(page_title="ASPIRE-MRI", layout="wide")
st.title("ASPIRE-MRI Restoration Demo")
st.caption("Academic research demonstration only — not a clinical or diagnostic tool.")


@st.cache_data(show_spinner="Reading and preparing NIfTI volume…")
def prepare_uploaded_volume(file_bytes: bytes, filename: str):
    """Read and preprocess an uploaded NIfTI once per file content."""
    suffix = ".nii.gz" if filename.lower().endswith(".nii.gz") else ".nii"
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
            handle.write(file_bytes)
            temporary_path = Path(handle.name)
        volume, metadata = load_ixi_volume(temporary_path)
        slices = list(axial_slices(robust_normalize(volume)))
        return slices, metadata
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@st.cache_resource(show_spinner=False)
def load_inference_model(checkpoint_path: str, checkpoint_mtime_ns: int, device_name: str):
    """Cache the loaded model across Streamlit reruns; reload if its file changes."""
    device = torch.device(device_name)
    model = build_model().to(device)
    load_checkpoint(checkpoint_path, model, device)
    model.eval()
    return model


uploaded = st.file_uploader("Upload a 3-D NIfTI MRI volume", type=["nii", "gz"])
checkpoint_input = st.text_input(
    "Adaptive checkpoint path", value="checkpoints/adaptive/best.pt"
)
input_mode = st.radio(
    "Input mode",
    ("Simulate synthetic noise from this clean volume", "Use this volume as-is"),
    horizontal=True,
)
simulate_noise = input_mode.startswith("Simulate")

reference = None
if simulate_noise:
    noise_columns = st.columns(3)
    noise_type = noise_columns[0].selectbox(
        "Synthetic noise type", tuple(NOISE_TO_ID), format_func=str.capitalize
    )
    severity = noise_columns[1].select_slider(
        "Noise severity (σ)",
        options=(0.02, 0.03, 0.05, 0.075, 0.10, 0.15),
        value=0.05,
        format_func=lambda value: f"{value:.3f}",
    )
    noise_seed = noise_columns[2].number_input(
        "Noise seed", min_value=0, max_value=2**31 - 1, value=42, step=1
    )
    st.caption(
        "Synthetic corruption is applied to the selected normalized slice for demonstration; "
        "it does not model every real scanner-noise process."
    )
else:
    reference = st.file_uploader(
        "Optional clean reference NIfTI volume", type=["nii", "gz"], key="reference"
    )

if uploaded is not None:
    try:
        slices, metadata = prepare_uploaded_volume(uploaded.getvalue(), uploaded.name)
    except Exception as exc:
        st.error(f"Could not read this NIfTI volume: {exc}")
        st.stop()

    if not slices:
        st.error("No usable axial slices were found in this volume.")
        st.stop()

    checkpoint = Path(checkpoint_input).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        st.error(
            f"Adaptive checkpoint not found: {checkpoint}. Train the model or enter a valid checkpoint path."
        )
        st.stop()
    checkpoint = checkpoint.resolve()

    index = st.slider("Axial slice", 0, len(slices) - 1, len(slices) // 2)
    source_slice_index, image = slices[index]
    if simulate_noise:
        noisy_image, _ = corrupt(
            image, noise_type, float(severity), int(noise_seed) + source_slice_index
        )
        input_caption = f"Synthetic {noise_type} input, σ={severity:.3f}"
    else:
        noisy_image = image
        input_caption = "Uploaded input (used as-is)"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = load_inference_model(
            str(checkpoint), checkpoint.stat().st_mtime_ns, str(device)
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            input_tensor = torch.from_numpy(noisy_image)[None, None].to(device)
            output = model(input_tensor)
        restored = output["restored"][0, 0].float().clamp(0, 1).cpu().numpy()
        experts = output["experts"][0, :, 0].float().clamp(0, 1).cpu().numpy()
        probabilities = output["noise_logits"].softmax(1)[0].float().cpu().tolist()
        severity = float(output["severity"][0].float().cpu())
        weights = output["weights"][0].float().cpu().tolist()
    except Exception as exc:
        st.error(f"Could not load the checkpoint or run inference: {exc}")
        st.stop()

    st.caption(f"Inference device: {device} · Original axial slice: {source_slice_index}")
    if simulate_noise:
        clean_column, noisy_column, restored_column = st.columns(3)
        clean_column.image(image, caption="Clean reference", clamp=True)
        noisy_column.image(noisy_image, caption=input_caption, clamp=True)
        restored_column.image(restored, caption="Adaptive restoration", clamp=True)
        target = torch.from_numpy(image)[None, None]
        noisy_tensor = torch.from_numpy(noisy_image)[None, None]
        st.subheader("Selected-slice restoration metrics")
        metric_columns = st.columns(2)
        metric_columns[0].json(restoration_metrics(noisy_tensor, target))
        metric_columns[1].json(restoration_metrics(torch.from_numpy(restored)[None, None], target))
        metric_columns[0].caption("Before restoration")
        metric_columns[1].caption("After restoration")
    else:
        input_column, restored_column = st.columns(2)
        input_column.image(noisy_image, caption=input_caption, clamp=True)
        restored_column.image(restored, caption="Adaptive restoration", clamp=True)

    expert_columns = st.columns(3)
    for column, name, result in zip(
        expert_columns, ("DnCNN", "U-Net", "Restormer"), experts
    ):
        column.image(result, caption=f"{name} output", clamp=True)

    st.subheader("Noise estimate and fusion")
    metrics_columns = st.columns(3)
    predicted_class = max(range(len(probabilities)), key=probabilities.__getitem__)
    class_names = [name for name, _ in sorted(NOISE_TO_ID.items(), key=lambda item: item[1])]
    metrics_columns[0].metric("Predicted noise", class_names[predicted_class].capitalize())
    metrics_columns[1].metric("Estimated severity (σ)", f"{severity:.4f}")
    metrics_columns[2].metric("Inference device", str(device))
    st.write(
        {
            "noise_probabilities": dict(zip(class_names, probabilities)),
            "adaptive_weights": dict(
                zip(("DnCNN", "U-Net", "Restormer"), weights)
            ),
        }
    )

    if reference is not None:
        try:
            target_slices, target_metadata = prepare_uploaded_volume(
                reference.getvalue(), reference.name
            )
        except Exception as exc:
            st.error(f"Could not read the clean reference volume: {exc}")
            st.stop()

        same_geometry = (
            target_metadata["shape"] == metadata["shape"]
            and np.allclose(
                target_metadata["voxel_spacing"], metadata["voxel_spacing"], atol=1e-4
            )
            and np.allclose(target_metadata["affine"], metadata["affine"], atol=1e-4)
        )
        same_slice_layout = (
            len(target_slices) == len(slices)
            and target_slices[index][0] == source_slice_index
            and target_slices[index][1].shape == image.shape
        )
        if same_geometry and same_slice_layout:
            target = torch.from_numpy(target_slices[index][1])[None, None]
            prediction = torch.from_numpy(restored)[None, None]
            st.subheader("Reference metrics for this slice")
            st.json(restoration_metrics(prediction, target))
        else:
            st.warning(
                "The clean reference must match the input volume's canonical geometry "
                "and contain the same usable axial slices."
            )
