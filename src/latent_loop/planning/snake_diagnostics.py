"""Export every Snake loop with frozen parameters and unchanged observations."""
import json
import numpy as np
from latent_loop.environments.snake import Snake
from latent_loop.planning.data import ROOT,RESULTS,ARTIFACTS,save_rows,digest
from latent_loop.planning.evaluate import read_trace
from latent_loop.planning.snake_training import build,observation,config

def run():
    import mlx.core as mx
    from latent_loop.backbone import environment
    frozen=json.loads((ROOT/'snake_frozen_protocol.json').read_text())
    assert all(digest(p)==h for p,h in frozen['hashes'].items())
    c=config();environment(c['memory_gib']);data=read_trace(RESULTS/'snake_neural'/'datasets'/'final.jsonl.gz')
    x=np.stack([observation(Snake.from_dict(r['state'])) for r in data])
    for name in ('snake_plan_rec','snake_safety_rec'):
        for seed in c['seeds']:
            model=build(c['conditions'][name],seed);model.load_weights(str(ARTIFACTS/'checkpoints'/f'{name}_seed{seed}'/'best.safetensors'))
            trace=model(mx.array(x),tuple(range(1,9)));mx.eval(trace)
            records=[{'id':r['id'],'depths':{str(d):{k:np.asarray(v)[i].tolist() for k,v in trace[d].items()} for d in trace}} for i,r in enumerate(data)]
            original=read_trace(RESULTS/'snake_neural'/'trajectories'/f'{name}_seed{seed}.jsonl.gz')
            for a,b in zip(records,original):
                assert a['id']==b['id']
                for d in (1,2,4,8):np.testing.assert_array_equal(a['depths'][str(d)]['logits'],b['depths'][str(d)]['logits'])
            save_rows(RESULTS/'snake_neural'/'all_loop_trajectories'/f'{name}_seed{seed}.jsonl.gz',records)
    print('Every-loop Snake exports exactly match original requested-depth logits')

if __name__=='__main__':run()
