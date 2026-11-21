"""Create reproducible, subject-level ASPIRE-MRI splits before slice extraction."""
from __future__ import annotations
import argparse
from pathlib import Path
from .splitting import make_subject_splits, write_splits
from .ixi_loader import list_t1_volumes, nifti_stem
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('raw_dir',type=Path); parser.add_argument('splits_dir',type=Path); parser.add_argument('--seed',type=int,default=42); args=parser.parse_args()
    subjects=[nifti_stem(p) for p in list_t1_volumes(args.raw_dir)]
    if len(subjects)<4: raise ValueError('At least four volumes are required for non-overlapping subject-level splits.')
    write_splits(make_subject_splits(subjects,args.seed),args.splits_dir)
if __name__=='__main__': main()
