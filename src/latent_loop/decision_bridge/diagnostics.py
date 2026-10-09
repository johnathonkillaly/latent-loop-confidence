"""Model invariance, locality and frozen-base checks, never model selection."""
import json
import numpy as np
import mlx.core as mx
from .data import EXP,OUT,load,arrays,write,adjacency,read_rows,save_rows
from .model import tensor_inputs,scorer_features
from .train import load_base,load_scorer,config
from .evaluate import verify
from .interface import fuse
from .data import choose

def relabel(row,rng):
    old=row['observation'];n=old['n'];p=rng.permutation(n);inverse=np.argsort(p)
    obs=dict(old);obs['start']=int(inverse[old['start']]);obs['goal']=int(inverse[old['goal']])
    obs['edges']=[[int(inverse[u]),int(inverse[v])] for u,v in old['edges']]
    candidates=[int(inverse[v]) if v>=0 else -1 for v in old['candidates']];order=rng.permutation(len(candidates))
    obs['candidates']=[candidates[j] for j in order];obs['tie_keys']=[old['tie_keys'][v] for v in p]+[old['tie_keys'][-1]]
    out={'observation':obs,'labels':{'optimal':[row['labels']['optimal'][j] for j in order]}}
    return out,p,order

def run():
    verify();selection=json.loads((EXP/'selection.json').read_text());cfg=config();rng=np.random.default_rng(53077);records=[]
    for split in ('id','size32','counterfactual'):
        rows=load(split,final=True)[:60];changed=[];maps=[]
        for r in rows:
            rr,p,c=relabel(r,rng);changed.append(rr);maps.append((p,c))
        x,y,keys=arrays(rows);xx,yy,kk=arrays(changed)
        for seed in cfg['seeds']:
            for condition in ('one_pass','recurrent','ff8'):
                depth=1 if condition=='one_pass' else 8;base=load_base(condition,seed);spec=selection['conditions'][condition];q=load_scorer(condition,seed,spec['objective'])
                tx=tensor_inputs(x);txx=tensor_inputs(xx);r=base(tx,(depth,))[depth];rr=base(txx,(depth,))[depth]
                score=q(scorer_features(r,tx['cm'],depth));score2=q(scorer_features(rr,txx['cm'],depth));mx.eval(r,rr,score,score2)
                z=np.asarray(r['logits']);zz=np.asarray(rr['logits']);qs=np.asarray(score);qq=np.asarray(score2)
                base_ids=choose(z,keys,x['cm']);new_ids=choose(zz,kk,xx['cm']);fused=choose(fuse(z,qs,spec['lambda'],x['cm']),keys,x['cm']);new_fused=choose(fuse(zz,qq,spec['lambda'],xx['cm']),kk,xx['cm'])
                errors=[];qerrors=[];semantic_base=0;semantic_fused=0
                for i,(_,order) in enumerate(maps):
                    nc=len(order);errors.append(float(np.max(np.abs(z[i,order]-zz[i,:nc]))));qerrors.append(float(np.max(np.abs(qs[i,order]-qq[i,:nc]))))
                    semantic_base+=int(order[new_ids[i]]==base_ids[i]);semantic_fused+=int(order[new_fused[i]]==fused[i])
                records.append({'split':split,'seed':seed,'condition':condition,'count':len(rows),'max_logit_error':max(errors),'max_score_error':max(qerrors),'same_base_action':semantic_base,'same_scored_action':semantic_fused})
    # All pairs must have identical candidate inputs before the first informative hop.
    locality=[];rows=load('counterfactual',final=True)
    for seed in cfg['seeds']:
        base=load_base('recurrent',seed)
        for arm in (3,5,9,13):
            subset=[r for r in rows if r['metadata']['arm_distance']==arm];x,_,_=arrays(subset);depths=tuple(k for k in cfg['depths'] if k<arm)
            out=base(tensor_inputs(x),depths)
            for k,r in out.items():
                mx.eval(r['logits']);z=np.asarray(r['logits']);errors=np.max(np.abs(z[::2]-z[1::2]),axis=1)
                assert np.max(errors)==0
                locality.append({'seed':seed,'arm_distance':arm,'depth':k,'pairs':len(subset)//2,'max_pair_logit_difference':float(max(errors))})
    write(OUT/'invariance_and_locality.json',{'permutations':records,'locality':locality,'interpretation':'Floating-point permutation tolerance is measured; random node tie keys are mapped with identities and never neural inputs.'})

if __name__=='__main__':run()
