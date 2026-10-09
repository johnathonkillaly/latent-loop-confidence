"""Live replay of archived5B navigation using unchanged checkpoint identities."""
import time
import numpy as np
import mlx.core as mx
from latent_loop.environments.maze import Maze
from latent_loop.planning.rollout import navigate
from latent_loop.planning.data import write_json,now
from latent_loop.scaling.data import OUT,read
from latent_loop.scaling.evaluate import policy
from latent_loop.scaling.validate import baseline

def run():
    started=time.perf_counter();result={}
    for split in ('larger','long_paths'):
        rows=read(f'results/experiment_05b/datasets/{split}.jsonl.gz')[:60];success=[]
        for seed in (11,22,33):
            act=policy(baseline(seed),8);arch=read(f'results/experiment_05b/rollouts/rec_simultaneous_d8_seed{seed}_{split}.jsonl.gz');ok=[]
            for row,expected in zip(rows,arch):
                assert row['id']==expected['id'];actual=navigate(Maze.from_dict(row),act,max(24,4*row['labels']['distance']))
                for k,v in actual.items():
                    if k!='inference_seconds':assert v==expected[k],(split,seed,row['id'],k)
                ok.append(actual['success'])
            success.append(ok)
        result[split]={'episodes':60,'training_seeds':3,'completion':float(np.mean(success)),'all_actions_positions_outcomes_and_offline_metrics_exact':True}
    write_json(OUT/'archived_rollout_verification.json',{'verified':now(),'results':result,'live_inference_seconds_with_scoring':time.perf_counter()-started,'peak_mlx_gib':mx.get_peak_memory()/1024**3,'timings_not_expected_identical':True,'historical_artifacts_modified':False});print(result,flush=True)
if __name__=='__main__':run()
