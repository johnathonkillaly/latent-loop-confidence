"""Validation-only selection, prospective freeze, paired decisions and episodes."""
import json
import time
import resource
from pathlib import Path
import numpy as np
import mlx.core as mx
from scipy.special import expit,logsumexp
from scipy.optimize import minimize
from .data import EXP,OUT,ART,config,load,arrays,choose,write,digest,now,save_rows,adjacency
from .model import tensor_inputs,scorer_features
from .interface import fuse
from .train import BASES,deploy,load_base,load_scorer,predict,cache,checkpoints

def calibration(q,y,mask):
    qq=q[mask];yy=y[mask].astype(float)
    def objective(v):
        z=np.exp(v[0])*qq+v[1]
        return np.mean(np.logaddexp(0,z)-yy*z)+1e-5*np.sum(v*v)
    fit=minimize(objective,np.array([0.,0.]),method='L-BFGS-B',bounds=[(-5,5),(-10,10)])
    return {'scale':float(np.exp(fit.x[0])),'offset':float(fit.x[1]),'converged':bool(fit.success)}

def select():
    if (EXP/'lockbox_opened.json').exists():raise RuntimeError('Cannot select after final opening')
    cfg=config();selection={'created':now(),'primary':{'condition':'recurrent','depth':8,'split':'id'},'conditions':{}}
    for condition in ('one_pass','recurrent','ff8'):
        records=[];depth=deploy(condition)
        for seed in cfg['seeds']:
            data=cache(condition,seed,'validation');d=list(data['depths']).index(depth)
            qs={o:np.asarray(load_scorer(condition,seed,o)(mx.array(data['features'][d]))) for o in (('direct',) if condition=='ff8' else cfg['scorer_objectives'])}
            records.append((data,d,qs))
        scores={};calibrators={}
        for objective in records[0][2]:
            vals=[]
            for weight in cfg['fusion_lambdas']:
                accuracy=[]
                for data,d,qs in records:
                    ids=choose(fuse(data['logits'][d],qs[objective],weight,data['mask']),data['keys'],data['mask'])
                    accuracy.append(float(data['target'][np.arange(len(ids)),ids].mean()))
                vals.append({'lambda':weight,'accuracy':float(np.mean(accuracy))})
            best=max(vals,key=lambda r:(r['accuracy'],-r['lambda']));scores[objective]={'grid':vals,**best}
            calibrators[objective]=calibration(np.concatenate([r[2][objective] for r in records]),np.concatenate([r[0]['target'] for r in records]),np.concatenate([r[0]['mask'] for r in records]))
        temperatures=[]
        for temperature in cfg['temperatures']:
            losses=[]
            for data,d,qs in records:
                z=data['logits'][d]/temperature
                losses.append(float(np.mean(logsumexp(z,-1)-logsumexp(np.where(data['target'],z,-1e9),-1))))
            temperatures.append({'temperature':temperature,'loss':float(np.mean(losses))})
        selected=min(temperatures,key=lambda r:(r['loss'],abs(np.log(r['temperature']))))
        candidates=['correctness','ranking'] if condition!='ff8' else ['direct']
        objective=max(candidates,key=lambda o:(scores[o]['accuracy'],o=='correctness'))
        by_depth={}
        for diagnostic_depth in records[0][0]['depths']:
            grid=[]
            for weight in cfg['fusion_lambdas']:
                acc=[]
                for data,_,_ in records:
                    di=list(data['depths']).index(diagnostic_depth)
                    # The checkpoint stays fixed; only a prespecified validation fusion grid.
                    model=load_scorer(condition,cfg['seeds'][len(acc)],objective)
                    quality=np.asarray(model(mx.array(data['features'][di])))
                    ids=choose(fuse(data['logits'][di],quality,weight,data['mask']),data['keys'],data['mask'])
                    acc.append(float(data['target'][np.arange(len(ids)),ids].mean()))
                grid.append({'lambda':weight,'accuracy':float(np.mean(acc))})
            by_depth[str(int(diagnostic_depth))]=max(grid,key=lambda r:(r['accuracy'],-r['lambda']))
        nonzero=max([v for v in scores[objective]['grid'] if v['lambda']>0],key=lambda r:(r['accuracy'],-r['lambda']))
        selection['conditions'][condition]={'objective':objective,'lambda':scores[objective]['lambda'],'validation_accuracy':scores[objective]['accuracy'],'objectives':scores,'calibration':calibrators,'temperature':selected['temperature'],'temperature_grid':temperatures,'by_depth':by_depth,'forced_nonzero':nonzero}
    selection['primary'].update({k:selection['conditions']['recurrent'][k] for k in ('objective','lambda')})
    write(EXP/'selection.json',selection);return selection

def freeze():
    selection=select();sources=list(Path('src/latent_loop/decision_bridge').glob('*.py'))+[EXP/'config.json',EXP/'protocol.md',EXP/'dataset_manifest.json',EXP/'base_freeze.json',EXP/'selection.json']
    records=list((OUT/'training').glob('*.json'));assert len(records)==36
    hashes={}
    for path in records:
        r=json.loads(path.read_text());assert len(r['epochs'])==config()['epochs']
        ck=checkpoints(path.stem)/'best.safetensors';assert digest(ck)==r['checkpoint_sha256'];hashes[path.stem]=digest(ck)
    receipt={'frozen':now(),'parent_commit':'1f2e4947689984643bbadccb3327a8baec4cba81','selection':selection,'source_hashes':{str(p):digest(p) for p in sources},'checkpoint_hashes':hashes,'neural_fits':36}
    write(EXP/'frozen_protocol.json',receipt)
    return receipt

def verify():
    r=json.loads((EXP/'frozen_protocol.json').read_text())
    for path,sha in r['source_hashes'].items():assert digest(path)==sha,path
    for name,sha in r['checkpoint_hashes'].items():assert digest(checkpoints(name)/'best.safetensors')==sha,name
    return r

def open_final():
    receipt=verify();path=EXP/'lockbox_opened.json'
    if not path.exists():write(path,{'opened':now(),'frozen_protocol_sha256':digest(EXP/'frozen_protocol.json'),'primary':receipt['selection']['primary']})

def q_predict(q,features,batch=256):
    out=[]
    for start in range(0,len(features),batch):
        result=q(mx.array(features[start:start+batch]));mx.eval(result);out.append(np.asarray(result))
    return np.concatenate(out)

def evaluate():
    open_final();cfg=config();selection=json.loads((EXP/'selection.json').read_text());splits=list(cfg['final_counts'])+['counterfactual']
    for split in splits:
        rows=load(split,final=True)
        for condition in (*BASES,'extra_budget'):
            depths=(1,) if condition=='one_pass' else tuple(cfg['depths']) if condition in ('recurrent','extra_budget') else (8,)
            for seed in cfg['seeds']:
                path=OUT/'decisions'/f'{split}_{condition}_seed{seed}.jsonl.gz'
                if path.exists():continue
                base=load_base(condition,seed);pred,mask,y,keys=predict(base,rows,depths,with_features=condition in selection['conditions']);all_records=[]
                qmodels={o:load_scorer(condition,seed,o) for o in selection['conditions'].get(condition,{}).get('objectives',{})}
                for depth,record in pred.items():
                    z=record['logits'];p=np.exp(z-logsumexp(z,-1,keepdims=True));qs={o:q_predict(model,record['features']) for o,model in qmodels.items()}
                    decisions={'base':choose(z,keys,mask),'constant':choose(fuse(z,np.ones_like(z),4,mask),keys,mask)}
                    assert np.array_equal(decisions['base'],decisions['constant'])
                    if qmodels:
                        spec=selection['conditions'][condition];decisions['temperature']=choose(z/spec['temperature'],keys,mask)
                        assert np.array_equal(decisions['base'],decisions['temperature'])
                        for o,q in qs.items():
                            decisions[f'{o}_only']=choose(q,keys,mask)
                            decisions[f'{o}_fused']=choose(fuse(z,q,spec['objectives'][o]['lambda'],mask),keys,mask)
                        weight=spec['by_depth'].get(str(depth),spec)['lambda']
                        decisions['scored']=choose(fuse(z,qs[spec['objective']],weight,mask),keys,mask)
                        decisions['forced_nonzero']=choose(fuse(z,qs[spec['objective']],spec['forced_nonzero']['lambda'],mask),keys,mask)
                        rng=np.random.default_rng(seed+depth*100+cfg['dataset_seed']);shuffle=qs[spec['objective']].copy()
                        for i in range(len(shuffle)):
                            ids=np.flatnonzero(mask[i]);shuffle[i,ids]=shuffle[i,rng.permutation(ids)]
                        decisions['shuffled']=choose(fuse(z,shuffle,weight,mask),keys,mask)
                    for i,row in enumerate(rows):
                        nc=int(mask[i].sum());rec={'id':row['id'],'seed':seed,'condition':condition,'depth':depth,'base_logits':z[i,:nc].tolist(),'base_probabilities':p[i,:nc].tolist(),'q':{o:q[i,:nc].tolist() for o,q in qs.items()},'decisions':{mode:int(ids[i]) for mode,ids in decisions.items()},'correct':{mode:bool(y[i,ids[i]]) for mode,ids in decisions.items()},'state_norm':float(record['norm'][i]),'state_drift':float(record['drift'][i])}
                        all_records.append(rec)
                save_rows(path,all_records);print('evaluated',path.name,len(all_records),flush=True)
    write(OUT/'evaluation_receipt.json',{'completed':now(),'peak_mlx_gib':mx.get_peak_memory()/1024**3,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,'files':len(list((OUT/'decisions').glob('*.gz')))})

def policy_action(base,q,obs,depth,weight):
    # No labels, distances, outcomes or trajectory history reach this function.
    inputs,_,keys=arrays([{'observation':obs}]);x=tensor_inputs(inputs);r=base(x,(depth,))[depth]
    if q is None:mx.eval(r['logits']);z=np.asarray(r['logits'])
    else:
        quality=q(scorer_features(r,x['cm'],depth));mx.eval(r['logits'],quality);z=fuse(np.asarray(r['logits']),np.asarray(quality),weight,inputs['cm'])
    index=int(choose(z,keys,inputs['cm'])[0]);return obs['candidates'][index]

def rollout():
    verify();cfg=config();selection=json.loads((EXP/'selection.json').read_text());rng=np.random.default_rng(53033)
    for split in cfg['final_counts']:
        rows=load(split,final=True)[:cfg['rollouts_per_split']]
        for condition in (*BASES,'extra_budget'):
            depths=(1,) if condition=='one_pass' else (1,2,4,8,16) if condition=='recurrent' else (8,)
            modes=('base','scored') if condition in ('one_pass','recurrent') else ('base',)
            for seed in cfg['seeds']:
                path=OUT/'rollouts'/f'{split}_{condition}_seed{seed}.jsonl.gz'
                if path.exists():continue
                base=load_base(condition,seed);spec=selection['conditions'].get(condition);q=load_scorer(condition,seed,spec['objective']) if condition in ('one_pass','recurrent') else None;records=[]
                for row in rows:
                    raw=row['observation'];a=adjacency(raw['n'],raw['edges'])
                    for depth in depths:
                        for mode in modes:
                            current=raw['start'];visited=set();positions=[current];actions=[];abstained=False;t=time.perf_counter()
                            for step in range(cfg['max_episode_steps']):
                                if current==raw['goal'] or current in visited:break
                                visited.add(current);obs=dict(raw);obs['start']=current
                                candidates=[int(v) for v in np.flatnonzero(a[current])]+[-1]
                                order=np.random.default_rng(int(row['id'][:8],16)+current).permutation(len(candidates));obs['candidates']=[candidates[j] for j in order]
                                weight=spec['by_depth'].get(str(depth),spec)['lambda'] if spec else 0
                                action=policy_action(base,q if mode=='scored' else None,obs,depth,weight);actions.append(action)
                                if action<0:abstained=True;break
                                assert a[current,action]>0;current=action;positions.append(current)
                            elapsed=time.perf_counter()-t;solved=current==raw['goal'];reachable=row['labels']['reachable'];distance=row['labels']['distance']
                            records.append({'id':row['id'],'seed':seed,'condition':condition,'depth':depth,'mode':mode,'goal_completed':solved,'reachable':reachable,'correct_abstain':bool(abstained and not reachable),'success':bool(solved or (abstained and not reachable)),'abstained':abstained,'actions':actions,'positions':positions,'repeats':len(positions)-len(set(positions)),'steps':len(actions),'path_ratio':len(actions)/distance if solved and distance>0 else None,'seconds':elapsed,'invalid':0})
                save_rows(path,records);print('rollout',path.name,len(records),flush=True)
    write(OUT/'rollout_receipt.json',{'completed':now(),'peak_mlx_gib':mx.get_peak_memory()/1024**3,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3})

def latency():
    verify();cfg=config();selection=json.loads((EXP/'selection.json').read_text());seed=cfg['seeds'][0];records=[]
    for split in ('id','size32','size64'):
        rows=load(split,final=True)
        for condition in (*BASES,'extra_budget'):
            base=load_base(condition,seed);spec=selection['conditions'].get(condition);q=load_scorer(condition,seed,spec['objective']) if spec else None
            depths=(1,) if condition=='one_pass' else (1,2,4,8,16) if condition=='recurrent' else (8,)
            for depth in depths:
                for mode in (('base','scored') if q is not None else ('base',)):
                    for batch in (1,8):
                        sample=[{'observation':r['observation']} for r in rows[:batch]];times=[]
                        for rep in range(23):
                            mx.synchronize();t=time.perf_counter();inputs,_,keys=arrays(sample);x=tensor_inputs(inputs);r=base(x,(depth,))[depth]
                            if mode=='scored':quality=q(scorer_features(r,x['cm'],depth));mx.eval(quality,r['logits']);z=fuse(np.asarray(r['logits']),np.asarray(quality),spec['by_depth'].get(str(depth),spec)['lambda'],inputs['cm'])
                            else:mx.eval(r['logits']);z=np.asarray(r['logits'])
                            choose(z,keys,inputs['cm']);mx.synchronize();elapsed=time.perf_counter()-t
                            if rep>=3:times.append(elapsed*1000)
                        records.append({'split':split,'nodes':max(r['observation']['n'] for r in sample),'condition':condition,'depth':depth,'mode':mode,'batch':batch,'median_ms':float(np.median(times)),'samples_ms':times})
    write(OUT/'latency.json',{'records':records,'warmups':3,'repetitions':20,'includes':'raw observation preparation, MLX transfer, model, scorer, fusion and selection; excludes checkpoint I/O','peak_mlx_gib':mx.get_peak_memory()/1024**3,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3})

if __name__=='__main__':
    import sys
    {'freeze':freeze,'evaluate':evaluate,'rollout':rollout,'latency':latency}[sys.argv[1]]()
