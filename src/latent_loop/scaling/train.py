"""Small controlled fits with per-epoch resumable weights and optimizer."""
import json
import time
import resource
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten
from latent_loop.backbone import environment
from latent_loop.planning.grid_recurrence import set_loss
from latent_loop.planning.data import arrays,write_json,digest,now
from latent_loop.scaling.data import EXP,OUT,ART,load,config
from latent_loop.scaling.model import build,forward

def groups(rows):
    return {s:([i for i,r in enumerate(rows) if len(r['walls'])==s],arrays([r for r in rows if len(r['walls'])==s])) for s in sorted({len(r['walls']) for r in rows})}

def validate(model,rows,depth):
    losses=[];correct=[]
    for _,(_, (x,y)) in groups(rows).items():
        for i in range(0,len(x),128):
            xx=mx.array(x[i:i+128]);yy=mx.array(y[i:i+128]);z=forward(model,xx,(depth,))[depth]['logits'];loss=set_loss(z,yy);ok=yy[mx.arange(len(xx)),mx.argmax(z,-1)];mx.eval(loss,ok)
            losses.extend([float(loss)]*len(xx));correct.extend(np.asarray(ok).tolist())
    return {'loss':float(np.mean(losses)),'accuracy':float(np.mean(correct))}

def train(condition,seed):
    if (EXP/'lockbox_opened.json').exists():raise RuntimeError('Final set opened; fitting locked')
    cfg=config();environment(cfg['memory_gib']);spec=cfg['conditions'][condition];rows=load('train');val=load('validation');data=groups(rows)
    model=build(spec,seed);opt=optim.AdamW(cfg['learning_rate'],weight_decay=cfg['weight_decay']);name=f'{condition}_seed{seed}';ck=ART/'checkpoints'/name;ck.mkdir(parents=True,exist_ok=True);path=OUT/'training'/f'{name}.json'
    signature={str(p):digest(p) for p in (EXP/'config.json',EXP/'dataset_manifest.json',__file__, 'src/latent_loop/scaling/model.py')}
    # Keep receipt paths relative, including module __file__.
    signature={k.replace(str(__import__('pathlib').Path.cwd())+'/',''):v for k,v in signature.items()}
    rec={'condition':condition,'seed':seed,'signature':signature,'parameters':model.parameter_count(),'epochs':[],'best_loss':1e30,'created':now()}
    if path.exists():
        rec=json.loads(path.read_text());assert rec['signature']==signature
        model.load_weights(str(ck/'last.safetensors'));opt.state=tree_unflatten(list(mx.load(str(ck/'optimizer.safetensors')).items()))
    vg=nn.value_and_grad(model,lambda m,x,y,depth:set_loss(forward(m,x,(depth,))[depth]['logits'],y))
    for epoch in range(len(rec['epochs']),cfg['epochs']):
        t=time.perf_counter();rng=np.random.default_rng(seed*100000+epoch);batches=[];counts={};losses=[];norms=[]
        for size,(indices,(x,y)) in data.items():
            eligible=np.arange(len(indices))
            if spec['objective']=='curriculum':
                maxd=8 if epoch<10 else 12 if epoch<20 else 999
                eligible=np.array([j for j,i in enumerate(indices) if rows[i]['labels']['distance']<=maxd]);order=rng.choice(eligible,len(indices),replace=True)
            else:order=rng.permutation(eligible)
            batches.extend((size,order[i:i+cfg['batch_size']]) for i in range(0,len(order),cfg['batch_size']))
        rng.shuffle(batches)
        for size,idx in batches:
            if spec['objective']=='variable':depth=int(rng.choice([1,2,4,8,16]))
            elif spec['objective']=='curriculum':depth=int(rng.choice([1,2,4] if epoch<10 else [2,4,8] if epoch<20 else [4,8,16]))
            else:depth=spec['depth']
            x,y=data[size][1];loss,g=vg(model,mx.array(x[idx]),mx.array(y[idx]),depth);norm=mx.sqrt(sum(mx.sum(a*a) for _,a in tree_flatten(g)));g,_=optim.clip_grad_norm(g,cfg['clip_norm']);opt.update(model,g);mx.eval(model.parameters(),opt.state,loss,norm)
            losses.append(float(loss));norms.append(float(norm));counts[str(depth)]=counts.get(str(depth),0)+len(idx)
        v=validate(model,val,spec['depth']);elapsed=time.perf_counter()-t
        if not np.isfinite(losses+norms).all():raise RuntimeError('Nonfinite gradients/loss')
        if v['loss']<rec['best_loss']:rec.update(best_loss=v['loss'],best_epoch=epoch+1);model.save_weights(str(ck/'best.safetensors'))
        model.save_weights(str(ck/'last.safetensors'));mx.save_safetensors(str(ck/'optimizer.safetensors'),dict(tree_flatten(opt.state)))
        rec['epochs'].append({'epoch':epoch+1,'loss':float(np.mean(losses)),'validation':v,'seconds':elapsed,'gradient_norm_mean':float(np.mean(norms)),'gradient_norm_max':max(norms),'depth_examples':counts,'examples':sum(counts.values())})
        rec.update(checkpoint_sha256=digest(ck/'best.safetensors'),peak_mlx_gib=mx.get_peak_memory()/1024**3,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,updated=now());write_json(path,rec)
        print(f'{name} {epoch+1}/{cfg["epochs"]} loss{np.mean(losses):.3f} val{v["accuracy"]:.3f} {elapsed:.2f}s',flush=True)
        total=sum(sum(e['seconds'] for e in json.loads(p.read_text())['epochs']) for p in (OUT/'training').glob('*.json'))
        if total>cfg['max_active_seconds']*.7:raise RuntimeError('Fit budget reached; preserve evaluation reserve')
    return rec

if __name__=='__main__':
    for c in config()['conditions']:
        for s in config()['seeds']:train(c,s)
