"""Result plots; values are supplied from completed experiments rather than fabricated."""
from __future__ import annotations
import numpy as np
def comparison_figure(clean, noisy, outputs):
    import matplotlib.pyplot as plt
    entries=[('Clean',clean),('Noisy',noisy),*outputs.items()]; fig,axes=plt.subplots(1,len(entries),figsize=(4*len(entries),4))
    for axis,(title,image) in zip(axes,entries): axis.imshow(np.squeeze(image),cmap='gray',vmin=0,vmax=1); axis.set_title(title); axis.axis('off')
    fig.tight_layout(); return fig
