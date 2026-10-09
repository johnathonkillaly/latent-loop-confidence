"""Supplementary interleaved timing to reduce serial-order/desktop-load bias.

Retains the originally prescribed timing archive; changes no model or selection.
"""
import json
import time
import resource
import numpy as np
import mlx.core as mx
from .data import EXP,OUT,load,arrays,choose,write,now,config
from .model import tensor_inputs,scorer_features
from .interface import fuse
from .train import load_base,load_scorer,BASES
from .evaluate import verify

def main():
    verify();cfg=config();selection=json.loads((EXP/'selection.json').read_text());rng=np.random.default_rng(530112);records=[]
    for split in ('id','size32','size64'):
        rows=load(split,final=True)
        for condition in (*BASES,'extra_budget'):
            base=load_base(condition,cfg['seeds'][0]);spec=selection['conditions'].get(condition);q=load_scorer(condition,cfg['seeds'][0],spec['objective']) if spec else None
            depths=(1,) if condition=='one_pass' else (1,2,4,8,16) if condition=='recurrent' else (8,)
            for depth in depths:
                for batch in (1,8):
                    sample=[{'observation':r['observation']} for r in rows[:batch]];modes=('base','scored') if q is not None else ('base',);times={m:[] for m in modes}
                    for rep in range(60):
                        for mode in rng.permutation(modes):
                            mx.synchronize();t=time.perf_counter();inputs,_,keys=arrays(sample);x=tensor_inputs(inputs);r=base(x,(depth,))[depth]
                            if mode=='scored':
                                quality=q(scorer_features(r,x['cm'],depth));mx.eval(quality,r['logits']);weight=spec['by_depth'].get(str(depth),spec)['lambda'];z=fuse(np.asarray(r['logits']),np.asarray(quality),weight,inputs['cm'])
                            else:mx.eval(r['logits']);z=np.asarray(r['logits'])
                            choose(z,keys,inputs['cm']);mx.synchronize();dt=(time.perf_counter()-t)*1000
                            if rep>=10:times[mode].append(dt)
                    for mode in modes:
                        records.append({'split':split,'nodes':max(r['observation']['n'] for r in sample),'condition':condition,'depth':depth,'mode':mode,'batch':batch,'median_ms':float(np.median(times[mode])),'iqr_ms':np.quantile(times[mode],[.25,.75]).tolist(),'samples_ms':times[mode], 'paired_added_ms':float(np.median(np.array(times['scored'])-times['base'])) if mode=='scored' else None})
    write(OUT/'latency_interleaved.json',{'completed':now(),'reason':'Original serial-order timings had scorer-faster artifacts; retain them and supplement with randomized paired order. No speedup inference from noisy estimates.','warmups':10,'repetitions':50,'records':records,'includes':'raw observation prep, MLX transfer/model/Q, CPU fusion/selection, synchronized; checkpoint I/O excluded','peak_mlx_gib':mx.get_peak_memory()/1024**3,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3})

if __name__=='__main__':main()
