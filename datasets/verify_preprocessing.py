"""Verify ASPIRE-MRI's pre-training data boundary without training any model."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import json

from .ixi_loader import list_t1_volumes, nifti_stem
from .noise_generator import deterministic_corruption


def read_split(path: Path) -> set[str]:
    with path.open(newline='', encoding='utf-8') as stream:
        return {row['subject_id'] for row in csv.DictReader(stream)}


def main() -> None:
    parser=argparse.ArgumentParser(description='Validate raw IXI data, subject splits, clean slices, and persisted-pair coverage.')
    parser.add_argument('--raw-root',type=Path,default=Path('data/raw/IXI'))
    parser.add_argument('--clean-root',type=Path,default=Path('data/processed/clean'))
    parser.add_argument('--splits-dir',type=Path,default=Path('data/splits'))
    parser.add_argument('--pairs-root',type=Path,default=Path('data/processed'))
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    raw={nifti_stem(path) for path in list_t1_volumes(args.raw_root)}
    if not raw: raise ValueError(f'No T1-weighted NIfTI volumes found in {args.raw_root}.')
    split_names=('expert_train','fusion_train','validation','test')
    splits={name:read_split(args.splits_dir/f'{name}_subjects.csv') for name in split_names}
    union=set().union(*splits.values()); overlap=sum(len(splits[a]&splits[b]) for index,a in enumerate(split_names) for b in split_names[index+1:])
    if overlap or union!=raw: raise ValueError(f'Split failure: overlap={overlap}, raw={len(raw)}, split_union={len(union)}')
    clean_subjects={path.name for path in args.clean_root.iterdir() if path.is_dir()}
    missing_clean=raw-clean_subjects
    unexpected_clean=clean_subjects-raw
    if missing_clean or unexpected_clean:
        raise ValueError(f'Clean subject mismatch: missing={len(missing_clean)}, unexpected={len(unexpected_clean)}.')
    invalid=[]
    for subject in raw:
        folder=args.clean_root/subject
        if not (folder/'metadata.json').is_file() or not any(folder.glob('*.npy')): invalid.append(subject)
    if invalid: raise ValueError(f'Invalid clean output for {len(invalid)} subjects.')
    kinds_seen=set(); pair_count=0
    for clean_path in sorted(args.clean_root.glob('*/*.npy')):
        source_id=clean_path.relative_to(args.clean_root).as_posix()
        kind,severity,sample_seed=deterministic_corruption(source_id,args.seed)
        paired_dir=args.pairs_root/kind/clean_path.parent.name
        noisy_path=paired_dir/f'{clean_path.stem}_noisy.npy'
        metadata_path=paired_dir/f'{clean_path.stem}.json'
        if not all(path.is_file() for path in (clean_path,noisy_path,metadata_path)):
            raise ValueError(f'Missing persisted pair for {source_id} ({kind}, severity={severity}).')
        metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('source')!=source_id or metadata.get('noise_type')!=kind or float(metadata.get('severity',-1))!=severity or int(metadata.get('seed',-1))!=sample_seed:
            raise ValueError(f'Persisted pair metadata does not match deterministic training corruption for {source_id}.')
        kinds_seen.add(kind); pair_count+=1
    if pair_count==0 or kinds_seen!={'gaussian','rician','mixed'}:
        raise ValueError(f'Persisted pairs do not cover all required noise types: {sorted(kinds_seen)}.')
    print({'raw_t1_volumes':len(raw),'splits':{name:len(subjects) for name,subjects in splits.items()},'clean_subjects':len(clean_subjects),'persisted_pairs':pair_count,'noise_types':sorted(kinds_seen),'status':'preprocessing_ready'})


if __name__=='__main__': main()
