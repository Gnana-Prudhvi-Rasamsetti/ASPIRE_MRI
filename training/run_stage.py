"""Command-line runner for the staged ASPIRE-MRI training workflow."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from datasets.dataloader import MRINoiseDataset
from models import AdaptiveGatingNetwork, AdaptiveRestoration, DnCNN, NoiseCharacterizationNet, Restormer, UNet
from training.common import load_checkpoint, seed_everything, write_log
from training.fine_tune import fine_tune
from training.train_adaptive import train as train_adaptive
from training.train_expert import train_expert
from training.train_noise_detector import train as train_noise_detector

def read_subjects(splits_dir: Path, name: str) -> list[str]:
    path=splits_dir/f'{name}_subjects.csv'
    if not path.exists(): raise FileNotFoundError(f'Missing {path}. Run datasets.prepare first.')
    with path.open(newline='',encoding='utf-8') as stream: return [row['subject_id'] for row in csv.DictReader(stream)]

def make_loader(clean_root: Path, subjects: list[str], batch_size: int, seed: int, shuffle: bool, num_workers: int = 2) -> DataLoader:
    return DataLoader(
        MRINoiseDataset(clean_root, subjects, seed=seed), batch_size=batch_size,
        shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

def checkpoint_path(stage: str) -> Path:
    return {'dncnn':Path('checkpoints/dncnn/best.pt'),'unet':Path('checkpoints/unet/best.pt'),'restormer':Path('checkpoints/restormer/best.pt'),'noise_detector':Path('checkpoints/noise_detector/best.pt'),'adaptive':Path('checkpoints/adaptive/best.pt'),'fine_tune':Path('checkpoints/adaptive/fine_tuned.pt')}[stage]

def latest_checkpoint_path(stage: str) -> Path:
    if stage == 'fine_tune':
        return Path('checkpoints/adaptive/fine_tuned_latest.pt')
    return checkpoint_path(stage).with_name('latest.pt')

def build_adaptive(device: torch.device) -> AdaptiveRestoration:
    parts=(DnCNN(),UNet(),Restormer(),NoiseCharacterizationNet(),AdaptiveGatingNetwork())
    for model,name in zip(parts[:4],('dncnn','unet','restormer','noise_detector')):
        path=checkpoint_path(name)
        if not path.exists(): raise FileNotFoundError(f'Required checkpoint missing: {path}. Train {name} first.')
        load_checkpoint(path,model,device)
    return AdaptiveRestoration(*parts)

def main() -> None:
    parser=argparse.ArgumentParser(description='Train one ASPIRE-MRI stage using subject-level splits.')
    parser.add_argument('stage',choices=('dncnn','unet','restormer','noise_detector','adaptive','fine_tune')); parser.add_argument('--clean-root',type=Path,default=Path('data/processed/clean')); parser.add_argument('--splits-dir',type=Path,default=Path('data/splits')); parser.add_argument('--epochs',type=int,default=50); parser.add_argument('--batch-size',type=int); parser.add_argument('--num-workers',type=int,default=2); parser.add_argument('--learning-rate',type=float); parser.add_argument('--seed',type=int,default=42); parser.add_argument('--resume',action='store_true',help='Continue from this stage\'s latest completed epoch checkpoint.'); args=parser.parse_args()
    seed_everything(args.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); train_name='fusion_train' if args.stage in ('adaptive','fine_tune') else 'expert_train'; train_subjects=read_subjects(args.splits_dir,train_name); validation_subjects=read_subjects(args.splits_dir,'validation'); batch_size=args.batch_size or (8 if args.stage=='restormer' else 16); learning_rate=args.learning_rate or (1e-5 if args.stage=='fine_tune' else 1e-4); train_loader=make_loader(args.clean_root,train_subjects,batch_size,args.seed,True,args.num_workers)
    write_log(f'stage={args.stage} device={device} train_subjects={len(train_subjects)} batch_size={batch_size} num_workers={args.num_workers} resume={args.resume}')
    if args.stage in ('dncnn','unet','restormer'):
        model={'dncnn':DnCNN(),'unet':UNet(),'restormer':Restormer()}[args.stage]; validation_loader=make_loader(args.clean_root,validation_subjects,batch_size,args.seed,False,args.num_workers); best=train_expert(model,train_loader,validation_loader,device,args.epochs,learning_rate,str(checkpoint_path(args.stage)),latest_checkpoint_path=latest_checkpoint_path(args.stage),resume=args.resume); write_log(f'completed stage={args.stage} best_validation_loss={best:.6f}')
    elif args.stage=='noise_detector':
        validation_loader=make_loader(args.clean_root,validation_subjects,batch_size,args.seed,False,args.num_workers)
        best=train_noise_detector(NoiseCharacterizationNet(),train_loader,validation_loader,device,args.epochs,learning_rate,checkpoint_path=checkpoint_path(args.stage),latest_checkpoint_path=latest_checkpoint_path(args.stage),resume=args.resume); write_log(f'completed stage=noise_detector best_validation_loss={best:.6f}')
    else:
        model=build_adaptive(device)
        validation_loader=make_loader(args.clean_root,validation_subjects,batch_size,args.seed,False,args.num_workers)
        if args.stage=='adaptive': best=train_adaptive(model,train_loader,validation_loader,device,args.epochs,learning_rate,checkpoint_path(args.stage),latest_checkpoint_path=latest_checkpoint_path(args.stage),resume=args.resume)
        else:
            frozen_path=checkpoint_path('adaptive')
            if not frozen_path.exists(): raise FileNotFoundError(f'Required checkpoint missing: {frozen_path}. Train adaptive fusion before fine-tuning.')
            load_checkpoint(frozen_path,model,device)
            best=fine_tune(model,train_loader,validation_loader,device,args.epochs,learning_rate,checkpoint_path(args.stage),latest_checkpoint_path=latest_checkpoint_path(args.stage),resume=args.resume)
        write_log(f'completed stage={args.stage} best_validation_loss={best:.6f}')
if __name__=='__main__': main()
