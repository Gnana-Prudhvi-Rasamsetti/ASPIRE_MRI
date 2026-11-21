"""Non-diagnostic ASPIRE-MRI demo. Requires trained checkpoints for meaningful outputs."""
from pathlib import Path
import tempfile
import numpy as np, streamlit as st, torch
from datasets.ixi_loader import load_ixi_volume
from datasets.preprocessing import robust_normalize, axial_slices
from main import build_model
from training.common import load_checkpoint
from evaluation.metrics import restoration_metrics
st.set_page_config(page_title='ASPIRE-MRI',layout='wide'); st.title('ASPIRE-MRI Restoration Demo'); st.caption('Academic research demonstration only - not a clinical or diagnostic tool.')
uploaded=st.file_uploader('Upload NIfTI MRI volume',type=['nii','gz'])
checkpoint=st.text_input('Adaptive checkpoint path',value='checkpoints/adaptive/best.pt')
reference=st.file_uploader('Optional clean reference NIfTI volume',type=['nii','gz'],key='reference')

def read_uploaded_volume(file):
    suffix='.nii.gz' if file.name.lower().endswith('.nii.gz') else '.nii'
    handle=tempfile.NamedTemporaryFile(suffix=suffix,delete=False)
    try:
        handle.write(file.getvalue()); handle.close()
        return load_ixi_volume(Path(handle.name))
    finally:
        Path(handle.name).unlink(missing_ok=True)

if uploaded:
    if not Path(checkpoint).is_file():
        st.error('A trained adaptive checkpoint is required. Train the model or provide a valid checkpoint path.')
        st.stop()
    volume,metadata=read_uploaded_volume(uploaded); slices=list(axial_slices(robust_normalize(volume)))
    if not slices:
        st.error('No usable axial slices were found in this volume.')
        st.stop()
    index=st.slider('Axial slice',0,len(slices)-1,len(slices)//2); image=slices[index][1]
    model=build_model().eval(); load_checkpoint(checkpoint,model); st.success('Loaded trained checkpoint.')
    with torch.inference_mode():
        output=model(torch.from_numpy(image)[None,None]); restored=output['restored'][0,0].clamp(0,1).numpy()
        experts=output['experts'][0,:,0].clamp(0,1).numpy()
    left,right=st.columns(2); left.image(image,caption='Input',clamp=True); right.image(restored,caption='Adaptive restoration',clamp=True)
    expert_columns=st.columns(3)
    for column,name,result in zip(expert_columns,('DnCNN','U-Net','Restormer'),experts): column.image(result,caption=f'{name} output',clamp=True)
    st.write({'noise_probabilities':output['noise_logits'].softmax(1)[0].tolist(),'estimated_severity':float(output['severity'][0]),'adaptive_weights':output['weights'][0].tolist()})
    if reference:
        target_data,target_metadata=read_uploaded_volume(reference)
        target_volume=robust_normalize(target_data); target_slices=list(axial_slices(target_volume))
        same_geometry=(target_metadata['shape']==metadata['shape'] and
                       np.allclose(target_metadata['voxel_spacing'],metadata['voxel_spacing'],atol=1e-4) and
                       np.allclose(target_metadata['affine'],metadata['affine'],atol=1e-4))
        if same_geometry and len(target_slices)==len(slices) and target_slices[index][1].shape==image.shape:
            target=torch.from_numpy(target_slices[index][1])[None,None]
            prediction=torch.from_numpy(restored)[None,None]
            st.write({'reference_metrics':restoration_metrics(prediction,target)})
        else:
            st.warning('The clean reference must have the same slice geometry and usable slice count as the uploaded volume.')
