"""Persist noisy-clean pairs and one metadata JSON record per sample for auditability."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .noise_generator import corrupt, deterministic_corruption
def main():
    p=argparse.ArgumentParser(); p.add_argument('clean_root',type=Path); p.add_argument('output_root',type=Path); p.add_argument('--seed',type=int,default=42); p.add_argument('--overwrite',action='store_true',help='Regenerate pairs that already exist.'); p.add_argument('--progress-every',type=int,default=1000); args=p.parse_args()
    files=sorted(args.clean_root.glob('*/*.npy'))
    if not files: raise ValueError(f'No clean .npy slices found beneath {args.clean_root}. Run slice extraction first.')
    written=skipped=0
    for index,file in enumerate(files):
        source_id=file.relative_to(args.clean_root).as_posix()
        kind,severity,sample_seed=deterministic_corruption(source_id,args.seed)
        target=args.output_root/kind/file.parent.name; target.mkdir(parents=True,exist_ok=True); stem=file.stem
        noisy_path,metadata_path=target/f'{stem}_noisy.npy',target/f'{stem}.json'
        expected={'source':source_id,'noise_type':kind,'severity':severity,'seed':sample_seed}
        existing_matches=False
        if not args.overwrite and all(path.exists() for path in (noisy_path,metadata_path)):
            try:
                existing=json.loads(metadata_path.read_text(encoding='utf-8'))
                existing_matches=all(existing.get(key)==value for key,value in expected.items())
            except (OSError,json.JSONDecodeError):
                existing_matches=False
        if existing_matches:
            skipped+=1
        else:
            clean=np.load(file).astype(np.float32); noisy,meta=corrupt(clean,kind,severity,sample_seed)
            np.save(noisy_path,noisy)
            expected['seed']=meta.seed
            metadata_path.write_text(json.dumps(expected,indent=2),encoding='utf-8')
            written+=1
        if (index+1)%args.progress_every==0 or index+1==len(files): print(f'[{index+1}/{len(files)}] written={written} skipped={skipped}',flush=True)
    print(f'Completed: written={written}, skipped={skipped}, total={len(files)}.',flush=True)
if __name__=='__main__': main()
