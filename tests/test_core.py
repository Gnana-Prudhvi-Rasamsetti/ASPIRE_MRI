import numpy as np, torch
from datasets.noise_generator import corrupt
from datasets.preprocessing import robust_normalize
from models import DnCNN, UNet, Restormer, NoiseCharacterizationNet, AdaptiveGatingNetwork, AdaptiveRestoration
from losses import StructurePreservingLoss
from evaluation.metrics import characterization_metrics
def test_noise_reproducible():
    clean=np.ones((16,16),np.float32); a,m=corrupt(clean,'mixed',.05,7); b,_=corrupt(clean,'mixed',.05,7); assert np.array_equal(a,b) and m.severity==.05
def test_normalization_is_bounded(): assert robust_normalize(np.arange(64).reshape(8,8)).min()>=0
def test_models_and_fusion_shapes():
    x=torch.rand(1,1,32,32); model=AdaptiveRestoration(DnCNN(features=8,depth=3),UNet(base=8),Restormer(dim=8,blocks=1,heads=2),NoiseCharacterizationNet(embedding_dim=16),AdaptiveGatingNetwork(embedding_dim=16))
    output=model(x); assert output['restored'].shape==x.shape; assert torch.allclose(output['weights'].sum(1),torch.ones(1),atol=1e-6)
def test_structure_loss_is_differentiable():
    x=torch.rand(2,1,16,16,requires_grad=True); value=StructurePreservingLoss()(x,torch.rand_like(x)); value.backward(); assert x.grad is not None
def test_characterization_metrics_perfect_prediction():
    output=characterization_metrics(torch.tensor([[4.,0.,0.],[0.,3.,0.],[0.,0.,2.]]),torch.tensor([.02,.05,.1]),torch.tensor([0,1,2]),torch.tensor([.02,.05,.1])); assert output['accuracy']==1 and output['macro_f1']>0.99 and output['severity_mae']==0
