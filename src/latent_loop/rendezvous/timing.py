"""Synchronized full-policy timing, including tokenization and actual selection."""

import json
import resource
import time

import numpy as np

from latent_loop.backbone import environment, get_model, tokenize
from .data import ROOT, read_rows
from .study import checkpoint, save


def live_policy(model, tokenizer, rows, config, head=None, condition="fusion", lam=0):
    import mlx.core as mx
    from .models import view
    encoded = [tokenize(tokenizer,r["prompt"],config["assistant_prefix"]) for r in rows]
    lengths = [len(t) for t in encoded]
    ids = mx.array([t+[tokenizer.pad_token_id]*(max(lengths)-len(t)) for t in encoded])
    h = model.model(ids)[mx.arange(len(rows)),mx.array(lengths)-1]
    labelids = mx.array([[tokenizer.encode(l,add_special_tokens=False)[0] for l in r["labels"]] for r in rows])
    label = model.model.embed_tokens.weight[labelids]
    z = mx.sum(h[:,None]*label,axis=-1)
    if head is not None:
        embedding = model.model.embed_tokens.weight
        candidate = mx.stack([mx.stack([mx.mean(embedding[mx.array(tokenizer.encode(d,add_special_tokens=False))],axis=0) for d in r["candidate_descriptions"]]) for r in rows])
        def norm(a):
            return a/mx.maximum(mx.sqrt(mx.mean(a*a,axis=-1,keepdims=True)),1e-6)
        centered=z-mx.mean(z,axis=-1,keepdims=True)
        lp=z-mx.logsumexp(z,axis=-1,keepdims=True);p=mx.exp(lp)
        entropy=-mx.sum(p*lp,axis=-1);sz=mx.sort(z,axis=-1);margin=sz[:,-1]-sz[:,-2]
        scalar=mx.stack([mx.clip(centered/5,-5,5),p,mx.clip(lp/5,-5,0),mx.repeat(entropy[:,None]/2,4,axis=1),mx.repeat(margin[:,None]/5,4,axis=1)],axis=-1)
        x=mx.concatenate([mx.repeat(norm(h)[:,None],4,axis=1),norm(candidate),norm(label),scalar],axis=-1)
        q=head(view(x,"c2" if condition=="c2" else "correctness"))
        if condition=="fusion":
            q=q-mx.mean(q,axis=-1,keepdims=True)
            q=q/mx.maximum(mx.sqrt(mx.mean(q*q,axis=-1,keepdims=True)),1)
            z=z+lam*q
        else:
            z=q
    selected=mx.argmax(z,axis=-1)
    mx.eval(z,selected)
    # Device-to-host and final semantic-action mapping included.
    return [rows[i]["candidate_descriptions"][int(a)] for i,a in enumerate(np.array(selected))], np.array(z)


def measure(config):
    import mlx.core as mx
    from .models import CandidateHead
    environment(config["memory_gb"])
    model,tokenizer,_=get_model()
    frozen=json.loads((ROOT/"protocol_freeze.json").read_text())
    seed=config["training_seeds"][0]
    dimension=3*model.model.embed_tokens.weight.shape[-1]+5
    q=CandidateHead(dimension,config["hidden_width"]);q.load_weights(str(checkpoint(frozen['selected']['objective'],seed)));q.eval()
    direct=CandidateHead(dimension,config["hidden_width"]);direct.load_weights(str(checkpoint('c2',seed)));direct.eval()
    rows=read_rows(ROOT/"datasets/validation.jsonl.gz")
    encoded=[len(tokenize(tokenizer,r['prompt'],config['assistant_prefix'])) for r in rows]
    # Predeclared standard length band, not chosen on final performance.
    selected=[r for r,n in zip(rows,encoded) if 140<=n<=200][:16]
    if len(selected)<8:
        raise RuntimeError('Insufficient standard-length timing prompts')
    result={'seed':seed,'scope':'tokenization (including token-boundary assertions), frozen full-prefix forward, four-label tied projection, candidate features if needed, head, actual selection, host transfer, Metal synchronization. No KV cache; excludes model loading.','length_band':[140,200],'warmups_per_method':6,'repetitions_per_method':36,'conditions_randomly_interleaved':True,'candidate_projection':'four permitted label tokens; unrestricted 151936-token diagnostic is separate from deployment timing','batch1':{},'batch8':{}}
    rng=np.random.default_rng(5419)
    for batch in (1,8):
        for method,head in [('base',None),('fusion',q),('c2',direct)]:
            for _ in range(6):
                live_policy(model,tokenizer,selected[:batch],config,head,method,frozen['selected']['lambda'])
        samples={m:[] for m in ('base','fusion','c2')}
        for repetition in range(36):
            items=[selected[(repetition+j)%len(selected)] for j in range(batch)]
            for method in rng.permutation(['base','fusion','c2']):
                head={'base':None,'fusion':q,'c2':direct}[method]
                mx.synchronize();tic=time.perf_counter()
                live_policy(model,tokenizer,items,config,head,method,frozen['selected']['lambda'])
                mx.synchronize();samples[method].append(1000*(time.perf_counter()-tic))
        for method,values in samples.items():
            result[f'batch{batch}'][method]={'median_ms':float(np.median(values)),'p10_ms':float(np.quantile(values,.1)),'p90_ms':float(np.quantile(values,.9)),'samples_ms':values}
    # Head-only diagnostic, explicitly not a full policy comparison.
    dummy=mx.ones((1,4,dimension));mx.eval(dummy)
    for _ in range(10):mx.eval(q(dummy))
    values=[]
    for _ in range(100):
        mx.synchronize();tic=time.perf_counter();mx.eval(q(dummy));mx.synchronize();values.append(1000*(time.perf_counter()-tic))
    result['head_only_median_ms']=float(np.median(values))
    result['peak_mlx_gib']=mx.get_peak_memory()/1024**3
    result['peak_rss_gib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3
    result['incremental_batch1_ms']=result['batch1']['fusion']['median_ms']-result['batch1']['base']['median_ms']
    save(ROOT/'timing.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('batch1','batch8')},indent=2))
