"""Separate frozen final4 training-depth diagnostic, never a selection candidate."""
import json
import numpy as np
from latent_loop.planning.grid_recurrence import GridPlanner
from latent_loop.planning.data import digest,write_json,save_rows,now
from latent_loop.planning.evaluate import correct_and_metrics
from latent_loop.scaling.data import EXP,OUT,load,config,read
from latent_loop.scaling.evaluate import infer,verify

def run():
    verify();f=json.load(open('experiments/experiment_05b/frozen_protocol.json'));result={}
    for split in ('size9','size15'):
        rows=load(split,True);vals=[]
        for s in config()['baseline_seeds']:
            p=f'artifacts/experiment_05b/checkpoints/rec_final4_seed{s}/best.safetensors';assert digest(p)==f['checkpoints'][p]
            m=GridPlanner(32);m.load_weights(p);m.eval();trace=infer(m,rows,tuple(config()['depths']));save_rows(OUT/'extrapolation'/f'final4_seed{s}_{split}.jsonl.gz',trace)
            vals.append({str(d):correct_and_metrics(rows,[r['depths'][str(d)]['logits'] for r in trace])[1] for d in config()['depths']})
        result[split]={'seed_metrics':vals,'mean':{str(d):{k:float(np.mean([v[str(d)][k] for v in vals])) for k in vals[0][str(d)]} for d in config()['depths']}}
    write_json(OUT/'train_depth_diagnostic.json',{'created':now(),'training':'Historical final-depth4-only; exact frozen checkpoints; same old training data but different objective from baseline','results':result,'status':'Secondary prespecified diagnostic; never model selection'})

if __name__=='__main__':run()
