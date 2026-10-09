"""Regenerate and refit in a new workspace, leaving the sealed study untouched.

Default verifies one recurrent seed and its correctness scorer. --all runs the
complete training/selection/evaluation protocol in the separate workspace.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .data import EXP,OUT,write,digest,now,config

def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',default='artifacts/experiment_05d/reproduction_workspace');p.add_argument('--all',action='store_true');args=p.parse_args()
    root=Path(args.destination)
    if root.exists():raise RuntimeError('Choose a new reproduction destination; existing work is preserved')
    root.mkdir(parents=True);shutil.copytree('src',root/'src',ignore=shutil.ignore_patterns('__pycache__'));exp=root/EXP;exp.mkdir(parents=True)
    for name in ('config.json','protocol.md'):shutil.copyfile(EXP/name,exp/name)
    program="""
from latent_loop.decision_bridge.data import generate,write,EXP,digest
from latent_loop.decision_bridge.train import main,fit_base,fit_scorer,checkpoints
generate()
"""
    if args.all:program+="""
main()
from latent_loop.decision_bridge.evaluate import freeze,evaluate,rollout,latency
freeze();evaluate();rollout();latency()
"""
    else:program+="""
r=fit_base('recurrent',401)
write(EXP/'base_freeze.json',{'checkpoints':{'recurrent_seed401':digest(checkpoints('recurrent_seed401')/'best.safetensors')}})
fit_scorer('recurrent',401,'correctness')
"""
    env=dict(os.environ);env['PYTHONPATH']='src'
    with (root/'execution.log').open('w') as log:
        # Whole owned subprocess budget; epoch checkpoints survive a timeout.
        subprocess.run([sys.executable,'-c',program],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=config()['max_active_seconds'])
    names=['recurrent_seed401','q_recurrent_correctness_seed401'] if not args.all else [x.stem for x in (OUT/'training').glob('*.json')]
    result={'completed':now(),'full_study':args.all,'verification_fits':len(names),'checkpoints':{},'fit_seconds':0}
    for name in names:
        r=json.loads((root/OUT/'training'/f'{name}.json').read_text());original=json.loads((OUT/'training'/f'{name}.json').read_text())
        result['checkpoints'][name]={'reproduced_sha256':r['checkpoint_sha256'],'original_sha256':original['checkpoint_sha256'],'byte_identical':r['checkpoint_sha256']==original['checkpoint_sha256']}
        result['fit_seconds']+=sum(e['seconds'] for e in r['epochs'])
    write(OUT/('full_retraining_verification.json' if args.all else 'seed401_retraining_verification.json'),result)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
