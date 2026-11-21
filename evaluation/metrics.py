from __future__ import annotations
import math, torch
from losses.ssim_loss import SSIMLoss
def restoration_metrics(prediction, target):
    mse=torch.mean((prediction.clamp(0,1)-target.clamp(0,1))**2).item(); return {"rmse":math.sqrt(mse),"psnr":float('inf') if mse==0 else -10*math.log10(mse),"ssim":1-SSIMLoss()(prediction.clamp(0,1),target.clamp(0,1)).item()}
def parameter_count(model): return sum(p.numel() for p in model.parameters())

def characterization_metrics(logits, severity_prediction, classes, severity_target, class_count=3):
    """Dependency-free accuracy, macro precision/recall/F1, confusion matrix, MAE and RMSE."""
    predicted=logits.argmax(1).detach().cpu(); actual=classes.detach().cpu(); matrix=torch.zeros(class_count,class_count,dtype=torch.long)
    for truth,guess in zip(actual,predicted): matrix[truth,guess]+=1
    scores=[]
    for label in range(class_count):
        tp=matrix[label,label].float(); precision=tp/(matrix[:,label].sum().float()+1e-8); recall=tp/(matrix[label,:].sum().float()+1e-8); scores.append((precision,recall,2*precision*recall/(precision+recall+1e-8)))
    error=(severity_prediction.detach().cpu()-severity_target.detach().cpu()).float()
    return {'accuracy':float((predicted==actual).float().mean()),'macro_precision':float(torch.stack([x[0] for x in scores]).mean()),'macro_recall':float(torch.stack([x[1] for x in scores]).mean()),'macro_f1':float(torch.stack([x[2] for x in scores]).mean()),'confusion_matrix':matrix.tolist(),'severity_mae':float(error.abs().mean()),'severity_rmse':float(error.square().mean().sqrt())}
