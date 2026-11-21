"""ASPIRE-MRI command-line smoke test and project entry point."""
from __future__ import annotations
import argparse, torch
from models import DnCNN, UNet, Restormer, NoiseCharacterizationNet, AdaptiveGatingNetwork, AdaptiveRestoration
from evaluation.metrics import parameter_count
def build_model(): return AdaptiveRestoration(DnCNN(),UNet(),Restormer(),NoiseCharacterizationNet(),AdaptiveGatingNetwork())
if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--smoke-test',action='store_true'); args=parser.parse_args(); model=build_model()
    if args.smoke_test:
        model.eval()
        with torch.inference_mode(): output=model(torch.rand(1,1,128,128))
        assert output['restored'].shape==(1,1,128,128) and torch.allclose(output['weights'].sum(1),torch.ones(1),atol=1e-5)
        print({'status':'ok','parameters':parameter_count(model),'weights':output['weights'].tolist()})
