"""Separate resumable base-policy and frozen-representation scorer fitting."""
import json
import time
import resource
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten
from latent_loop.backbone import environment
from .data import EXP,OUT,ART,load,config,arrays,choose,write,digest,now
from .model import build,Scorer,tensor_inputs,scorer_features,set_loss,score_loss

BASES=('one_pass','recurrent','ff8','ff8_params')
def checkpoints(name):return ART/'checkpoints'/name
def deploy(condition):return 1 if condition=='one_pass' else 8
def assert_training():
    if (EXP/'lockbox_opened.json').exists():raise RuntimeError('Final opened; no further fitting')
def signature():
    return {str(p):digest(p) for p in [EXP/'config.json',EXP/'dataset_manifest.json',__import__('pathlib').Path(__file__).relative_to(__import__('pathlib').Path.cwd()),'src/latent_loop/decision_bridge/model.py']}

def predict(model,rows,depths,batch=128,with_features=False):
    inputs,y,keys=arrays(rows);out={d:{'logits':[],'norm':[],'drift':[],'features':[]} for d in depths}
    for start in range(0,len(rows),batch):
        x=tensor_inputs({k:v[start:start+batch] for k,v in inputs.items()});records=model(x,depths)
        for d,r in records.items():
            mx.eval(r)
            for key in ('logits','norm','drift'):out[d][key].append(np.asarray(r[key]))
            if with_features:out[d]['features'].append(np.asarray(scorer_features(r,x['cm'],d)))
    for d in depths:
        out[d]={k:np.concatenate(v) for k,v in out[d].items() if v}
        z=out[d]['logits'];out[d]['correct']=y[np.arange(len(y)),choose(z,keys,inputs['cm'])]
    return out,inputs['cm'],y,keys

def validation(model,rows,depth):
    predictions,mask,y,keys=predict(model,rows,(depth,));z=predictions[depth]['logits']
    loss=set_loss(mx.array(z),mx.array(y));mx.eval(loss)
    return {'loss':float(loss),'accuracy':float(predictions[depth]['correct'].mean())}

def fit_base(condition,seed):
    assert_training();cfg=config();environment(cfg['memory_gib']);name=f'{condition}_seed{seed}'
    ck=checkpoints(name);ck.mkdir(parents=True,exist_ok=True);path=OUT/'training'/f'{name}.json';model=build(condition,seed)
    if condition=='extra_budget':
        frozen=json.loads((EXP/'base_freeze.json').read_text());source=checkpoints(f'recurrent_seed{seed}')/'best.safetensors'
        assert frozen['checkpoints'][f'recurrent_seed{seed}']==digest(source);model.load_weights(str(source))
    data=load('scorer_train' if condition=='extra_budget' else 'base_train');val=load('validation');x,y,keys=arrays(data)
    opt=optim.AdamW(cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rec={'condition':condition,'seed':seed,'parameters':model.parameter_count(),'signature':signature(),'epochs':[],'best_loss':1e30,'created':now(),'data':'scorer_train' if condition=='extra_budget' else 'base_train'}
    if path.exists():
        rec=json.loads(path.read_text());assert rec['signature']==signature()
        model.load_weights(str(ck/'last.safetensors'));opt.state=tree_unflatten(list(mx.load(str(ck/'optimizer.safetensors')).items()))
    vg=nn.value_and_grad(model,lambda m,xx,yy,d:set_loss(m(xx,(d,))[d]['logits'],yy))
    for epoch in range(len(rec['epochs']),cfg['epochs']):
        t=time.perf_counter();rng=np.random.default_rng(seed*10000+epoch);order=rng.permutation(len(data));losses=[];norms=[];depth_counts={}
        for offset in range(0,len(data),cfg['batch_size']):
            ids=order[offset:offset+cfg['batch_size']];depth=int(rng.choice(cfg['train_depths'])) if condition in ('recurrent','extra_budget') else deploy(condition)
            loss,g=vg(model,tensor_inputs({k:v[ids] for k,v in x.items()}),mx.array(y[ids]),depth)
            norm=mx.sqrt(sum(mx.sum(v*v) for _,v in tree_flatten(g)));g,_=optim.clip_grad_norm(g,cfg['clip_norm']);opt.update(model,g);mx.eval(model.parameters(),opt.state,loss,norm)
            losses.append(float(loss));norms.append(float(norm));depth_counts[str(depth)]=depth_counts.get(str(depth),0)+len(ids)
        v=validation(model,val,deploy(condition));elapsed=time.perf_counter()-t
        assert np.isfinite(losses+norms).all()
        if v['loss']<rec['best_loss']:rec.update(best_loss=v['loss'],best_epoch=epoch+1);model.save_weights(str(ck/'best.safetensors'))
        model.save_weights(str(ck/'last.safetensors'));mx.save_safetensors(str(ck/'optimizer.safetensors'),dict(tree_flatten(opt.state)))
        rec['epochs'].append({'epoch':epoch+1,'loss':float(np.mean(losses)),'gradient_mean':float(np.mean(norms)),'gradient_max':max(norms),'validation':v,'seconds':elapsed,'examples':len(data),'depth_examples':depth_counts})
        rec.update(checkpoint_sha256=digest(ck/'best.safetensors'),peak_mlx_gib=mx.get_peak_memory()/1024**3,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)
        write(path,rec);print(f'{name} {epoch+1} loss{np.mean(losses):.3f} val{v["accuracy"]:.3f} {elapsed:.2f}s',flush=True)
    return rec

def freeze_bases():
    assert_training();receipt={'frozen':now(),'checkpoints':{}}
    for c in BASES:
        for seed in config()['seeds']:
            name=f'{c}_seed{seed}';r=json.loads((OUT/'training'/f'{name}.json').read_text())
            assert len(r['epochs'])==config()['epochs'];p=checkpoints(name)/'best.safetensors'
            assert digest(p)==r['checkpoint_sha256'];receipt['checkpoints'][name]=digest(p)
    write(EXP/'base_freeze.json',receipt)

def load_base(condition,seed):
    model=build(condition,seed);p=checkpoints(f'{condition}_seed{seed}')/'best.safetensors'
    record=json.loads((OUT/'training'/f'{condition}_seed{seed}.json').read_text());assert digest(p)==record['checkpoint_sha256'];model.load_weights(str(p));return model

def cache(condition,seed,split):
    freeze=json.loads((EXP/'base_freeze.json').read_text());name=f'{condition}_seed{seed}';p=checkpoints(name)/'best.safetensors';assert freeze['checkpoints'][name]==digest(p)
    path=ART/'features'/f'{name}_{split}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        cached=np.load(path);assert str(cached['base_sha256'])==digest(p)
        return {k:cached[k] for k in cached.files if k!='base_sha256'}
    model=load_base(condition,seed);rows=load(split);depths=(1,) if condition=='one_pass' else (1,2,4,8) if condition=='recurrent' else (8,)
    pred,mask,y,keys=predict(model,rows,depths,with_features=True)
    result={'features':np.stack([pred[d]['features'] for d in depths]),'logits':np.stack([pred[d]['logits'] for d in depths]),'mask':mask,'target':y,'keys':keys,'depths':np.array(depths)}
    np.savez(path,base_sha256=digest(p),**result);return result

def scorer_validation(model,data,objective,depth):
    index=list(data['depths']).index(depth);q=model(mx.array(data['features'][index]));loss=score_loss(q,mx.array(data['target']),mx.array(data['mask']),objective);mx.eval(q,loss)
    ids=choose(np.asarray(q),data['keys'],data['mask']);return {'loss':float(loss),'accuracy':float(data['target'][np.arange(len(ids)),ids].mean())}

def fit_scorer(condition,seed,objective):
    assert_training();cfg=config();environment(cfg['memory_gib']);train=cache(condition,seed,'scorer_train');val=cache(condition,seed,'validation')
    name=f'q_{condition}_{objective}_seed{seed}';ck=checkpoints(name);ck.mkdir(parents=True,exist_ok=True);path=OUT/'training'/f'{name}.json'
    base_hash=digest(checkpoints(f'{condition}_seed{seed}')/'best.safetensors');mx.random.seed(seed+1000);model=Scorer();opt=optim.AdamW(cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    sig={**signature(),'base_sha256':base_hash}
    rec={'condition':condition,'objective':objective,'seed':seed,'parameters':model.parameter_count(),'signature':sig,'epochs':[],'best_loss':1e30,'created':now(),'data':'scorer_train'}
    if path.exists():
        rec=json.loads(path.read_text());assert rec['signature']==sig;model.load_weights(str(ck/'last.safetensors'));opt.state=tree_unflatten(list(mx.load(str(ck/'optimizer.safetensors')).items()))
    vg=nn.value_and_grad(model,lambda m,f,y,mask:score_loss(m(f),y,mask,objective));n=len(train['target'])
    for epoch in range(len(rec['epochs']),cfg['scorer_epochs']):
        t=time.perf_counter();rng=np.random.default_rng(seed*10000+epoch);order=rng.permutation(n);losses=[];norms=[]
        for offset in range(0,n,cfg['batch_size']):
            ids=order[offset:offset+cfg['batch_size']];d=int(rng.integers(len(train['depths'])))
            loss,g=vg(model,mx.array(train['features'][d,ids]),mx.array(train['target'][ids]),mx.array(train['mask'][ids]))
            norm=mx.sqrt(sum(mx.sum(v*v) for _,v in tree_flatten(g)));g,_=optim.clip_grad_norm(g,cfg['clip_norm']);opt.update(model,g);mx.eval(model.parameters(),opt.state,loss,norm)
            losses.append(float(loss));norms.append(float(norm))
        v=scorer_validation(model,val,objective,deploy(condition));elapsed=time.perf_counter()-t;assert np.isfinite(losses+norms).all()
        if v['loss']<rec['best_loss']:rec.update(best_loss=v['loss'],best_epoch=epoch+1);model.save_weights(str(ck/'best.safetensors'))
        model.save_weights(str(ck/'last.safetensors'));mx.save_safetensors(str(ck/'optimizer.safetensors'),dict(tree_flatten(opt.state)))
        rec['epochs'].append({'epoch':epoch+1,'loss':float(np.mean(losses)),'gradient_mean':float(np.mean(norms)),'gradient_max':max(norms),'validation':v,'seconds':elapsed,'examples':n})
        rec.update(checkpoint_sha256=digest(ck/'best.safetensors'),peak_mlx_gib=mx.get_peak_memory()/1024**3,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3)
        write(path,rec);print(f'{name} {epoch+1} loss{np.mean(losses):.3f} val{v["accuracy"]:.3f} {elapsed:.2f}s',flush=True)
    assert digest(checkpoints(f'{condition}_seed{seed}')/'best.safetensors')==base_hash
    return rec

def load_scorer(condition,seed,objective):
    model=Scorer();name=f'q_{condition}_{objective}_seed{seed}';p=checkpoints(name)/'best.safetensors';rec=json.loads((OUT/'training'/f'{name}.json').read_text())
    assert digest(p)==rec['checkpoint_sha256'];model.load_weights(str(p));return model

def main():
    for condition in BASES:
        for seed in config()['seeds']:fit_base(condition,seed)
    freeze_bases()
    for seed in config()['seeds']:fit_base('extra_budget',seed)
    for condition in ('one_pass','recurrent','ff8'):
        for objective in (('direct',) if condition=='ff8' else config()['scorer_objectives']):
            for seed in config()['seeds']:fit_scorer(condition,seed,objective)

if __name__=='__main__':main()
