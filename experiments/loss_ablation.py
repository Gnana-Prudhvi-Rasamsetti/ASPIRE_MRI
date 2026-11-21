from losses import StructurePreservingLoss
def configurations(): return {'mse':StructurePreservingLoss(1,0,0),'mse_ssim':StructurePreservingLoss(1,.2,0),'mse_ssim_gradient':StructurePreservingLoss(1,.2,.1)}
