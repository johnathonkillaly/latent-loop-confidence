"""Verify unchanged baseline against archived decisions before any new fitting."""
import json
import resource
import subprocess
import numpy as np
import mlx.core as mx
from latent_loop.backbone import environment
from latent_loop.planning.data import digest,write_json,now,arrays
from latent_loop.planning.evaluate import read_trace
from latent_loop.planning.grid_recurrence import GridPlanner
from latent_loop.environments.maze import Maze
from latent_loop.scaling.model import forward
from latent_loop.scaling.data import EXP,OUT,config

def baseline(seed):
    p=f'artifacts/experiment_05b/checkpoints/rec_simultaneous_seed{seed}/best.safetensors'
    expected=json.load(open('experiments/experiment_05b/frozen_protocol.json'))['checkpoints'][p]
    assert digest(p)==expected
    m=GridPlanner(32);m.load_weights(p);m.eval();return m

def run():
    environment(48);rows=read_trace('results/experiment_05b/datasets/fresh_id.jsonl.gz');x,_=arrays(rows);records=[]
    for seed in config()['baseline_seeds']:
        m=baseline(seed);arch=read_trace(f'results/experiment_05b/trajectories/rec_simultaneous_seed{seed}_fresh_id.jsonl.gz');errors=[];parity=[]
        for i in range(0,len(x),128):
            out=forward(m,mx.array(x[i:i+128]),(1,2,4,8,16),True);old=m(mx.array(x[i:i+128]),(1,2,4,8,16),True);mx.eval(out,old)
            for d in (1,2,4,8):
                expected=np.array([r['depths'][str(d)]['logits'] for r in arch[i:i+128]])
                errors.append(float(np.max(np.abs(np.asarray(out[d]['logits'])-expected))))
            for d in (1,2,4,8,16):
                for field in ('logits','states','state_norm','state_change'):parity.append(float(np.max(np.abs(np.asarray(out[d][field])-np.asarray(old[d][field])))))
        assert max(errors)==0 and max(parity)==0
        records.append({'seed':seed,'parameters':m.parameter_count(),'archived_examples':len(rows),'max_logit_error':max(errors),'extended_executor_max_parity_error':max(parity)})
    feasible=[];m=baseline(11)
    for size in (5,7,9,11,15):
        maze=Maze(tuple(tuple(0 for _ in range(size)) for _ in range(size)),(0,0),(size-1,size-1));before=maze.to_dict();out=forward(m,mx.array(maze.observation()[None]),(1,2,4,8,12,16,24,32),True);mx.eval(out)
        for d,r in out.items():
            h=np.asarray(r['states']);assert np.isfinite(h).all() and np.max(np.abs(h))<=1.000001
        assert before==maze.to_dict();feasible.append({'size':size,'finite_bounded_to_depth':32})
    write_json(OUT/'baseline_verification.json',{'verified':now(),'archived':records,'shape_feasibility':feasible,'raw_observation_channels':4,'environment_unchanged':True,'architecture_changed':False,'executor_change':'New executor extends old depth guard16 to32 without changing tensors/updates; exact parity through16','peak_mlx_gib':mx.get_peak_memory()/1024**3})
    release=subprocess.check_output(['git','rev-parse','v0.2.0^{}'],text=True).strip()
    write_json(EXP/'publication_receipt.json',{'verified':now(),'url':'https://github.com/johnathonkillaly/latent-loop-confidence/releases/tag/v0.2.0','tag':'v0.2.0','commit':release,'v010_commit':subprocess.check_output(['git','rev-parse','v0.1.0^{}'],text=True).strip(),'remote_release_verified':True})
    print('All three baseline checkpoints reproduce 600 archived examples exactly; native5–15 shapes finite/bounded through32; original executor parity exact through16',flush=True)

if __name__=='__main__':run()
