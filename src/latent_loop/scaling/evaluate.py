"""Prospective freeze, checkpoint inference, paired metrics and navigation."""
import json
import time
import resource
from pathlib import Path
import numpy as np
from scipy.special import softmax
from latent_loop.planning.data import arrays,write_json,save_rows,digest,now
from latent_loop.planning.evaluate import correct_and_metrics
from latent_loop.environments.maze import Maze,labels
from latent_loop.planning.oracles import bfs_action,greedy_action
from latent_loop.scaling.data import EXP,OUT,ART,config,load,read

def models():return ['frozen_05b',*config()['conditions']]
def seeds(c):return config()['baseline_seeds'] if c=='frozen_05b' else config()['seeds']
def depths(c):return tuple(range(1,33)) if c=='frozen_05b' or config()['conditions'][c]['shared'] else (config()['conditions'][c]['depth'],)
def model(c,s):
    from latent_loop.scaling.model import build
    if c=='frozen_05b':
        from latent_loop.scaling.validate import baseline
        return baseline(s)
    m=build(config()['conditions'][c],s);m.load_weights(str(ART/'checkpoints'/f'{c}_seed{s}'/'best.safetensors'));m.eval();return m

def paired(values,rows,repeats=4000):
    a=np.asarray(values,float);rng=np.random.default_rng(521099);bins=sorted({r['path_bin'] for r in rows});idx=[np.array([i for i,r in enumerate(rows) if r['path_bin']==b]) for b in bins];v=[]
    for _ in range(repeats):
        si=rng.integers(a.shape[0],size=a.shape[0]);ii=np.concatenate([rng.choice(j,len(j),replace=True) for j in idx]);v.append(a[si][:,ii].mean())
    return {'mean':float(a.mean()),'ci95':np.quantile(v,[.025,.975]).tolist(),'independent_mazes':len(rows),'training_seeds':a.shape[0],'bootstrap':'paired crossed training seeds and maze instances, within fixed path strata','replicates':repeats}

def freeze():
    from latent_loop.scaling.train import validate
    if (EXP/'frozen_protocol.json').exists():raise RuntimeError('Already frozen')
    cfg=config();scores={};checkpoints={};val=load('validation')
    for c in models():
        vals=[]
        for s in seeds(c):
            if c!='frozen_05b':
                fit=json.loads((OUT/'training'/f'{c}_seed{s}.json').read_text());assert len(fit['epochs'])==cfg['epochs'];p=ART/'checkpoints'/f'{c}_seed{s}'/'best.safetensors'
                assert digest(p)==fit['checkpoint_sha256']
            else:p=Path(f'artifacts/experiment_05b/checkpoints/rec_simultaneous_seed{s}/best.safetensors')
            checkpoints[str(p)]=digest(p);m=model(c,s);vals.append(validate(m,val,8 if c=='frozen_05b' else cfg['conditions'][c]['depth']))
        scores[c]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
    chosen=min(['rec_variable','rec_curriculum'],key=lambda c:(-scores[c]['accuracy'],scores[c]['loss']))
    ff=min(['ff4_params','ff8','global_cnn'],key=lambda c:(-scores[c]['accuracy'],scores[c]['loss']))
    paths=list(Path('src/latent_loop/scaling').glob('*.py'))+[EXP/'config.json',EXP/'plan.md',EXP/'dataset_manifest.json']
    paths=[p for p in paths if p.name not in ('report.py','__main__.py','audit.py')]
    write_json(EXP/'frozen_protocol.json',{'frozen':now(),'primary':cfg['primary'],'validation':scores,'selected_recurrent':chosen,'selected_feedforward':ff,'deployment_depths':{'frozen_05b':8,**{c:v['depth'] for c,v in cfg['conditions'].items()}},'checkpoints':checkpoints,'sources':{str(p):digest(p) for p in paths},'all_other_comparisons':'secondary/exploratory; no final-based model/depth selection'})
    print('FROZEN',chosen,ff,scores,flush=True)

def verify():
    f=json.loads((EXP/'frozen_protocol.json').read_text())
    for p,h in {**f['sources'],**f['checkpoints']}.items():assert digest(p)==h,p
    return f

def infer(m,rows,dd):
    import mlx.core as mx
    from latent_loop.scaling.model import forward
    x,_=arrays(rows);records=[]
    for i in range(0,len(rows),64):
        out=forward(m,mx.array(x[i:i+64]),dd);mx.eval(out);a={d:{k:np.asarray(v) for k,v in r.items()} for d,r in out.items()}
        assert all(np.isfinite(v).all() for r in a.values() for v in r.values())
        assert max(float(np.max(r['state_norm'])) for r in a.values())<=1.000001
        records.extend({'id':row['id'],'depths':{str(d):{k:v[j].tolist() for k,v in r.items()} for d,r in a.items()}} for j,row in enumerate(rows[i:i+64]))
    return records

def evaluate():
    import mlx.core as mx
    from latent_loop.backbone import environment
    environment(48);f=verify();cfg=config()
    if not (EXP/'lockbox_opened.json').exists():write_json(EXP/'lockbox_opened.json',{'opened':now(),'frozen_protocol_sha256':digest(EXP/'frozen_protocol.json')})
    summary={'selected_recurrent':f['selected_recurrent'],'selected_feedforward':f['selected_feedforward'],'splits':{}}
    for split,spec in cfg['datasets'].items():
        if not spec.get('final'):continue
        assert digest(ART/'sealed'/f'{split}.jsonl.gz')==json.loads((EXP/'dataset_manifest.json').read_text())['splits'][split]['sha256']
        rows=load(split,True);save_rows(OUT/'datasets'/f'{split}.jsonl.gz',rows);result={'count':len(rows),'models':{},'classical':{}};correctness={}
        for name,policy in [('bfs',bfs_action),('greedy',greedy_action)]:
            z=np.full((len(rows),4),-30.);z[np.arange(len(rows)),[policy(Maze.from_dict(r)) for r in rows]]=0;result['classical'][name]=correct_and_metrics(rows,z)[1]
        for c in models():
            metrics=[];oks=[];strata=[];changes=[]
            for s in seeds(c):
                path=OUT/'trajectories'/f'{c}_seed{s}_{split}.jsonl.gz'
                if path.exists():trace=read(path)
                else:trace=infer(model(c,s),rows,depths(c));save_rows(path,trace)
                assert [v['id'] for v in trace]==[r['id'] for r in rows];mm={};oo={};ss={};last=None;before_last=None;ch={}
                for d in depths(c):
                    z=np.array([v['depths'][str(d)]['logits'] for v in trace]);ok,stat=correct_and_metrics(rows,z);oo[str(d)]=ok;mm[str(d)]=stat
                    for field in ('state_norm','state_change','state_drift','state_cosine'):mm[str(d)][field]=float(np.mean([v['depths'][str(d)][field] for v in trace]))
                    ss[str(d)]={b:correct_and_metrics([r for r in rows if r['path_bin']==b],z[[i for i,r in enumerate(rows) if r['path_bin']==b]])[1] for b in sorted({r['path_bin'] for r in rows})}
                    pred=z.argmax(-1);ch[str(d)]={'prediction_change':float(np.mean(pred!=last)) if last is not None else 0,'return_to_two_loops_ago':float(np.mean((pred==before_last)&(pred!=last))) if before_last is not None else 0};before_last=last;last=pred
                metrics.append(mm);oks.append(oo);strata.append(ss);changes.append(ch)
            correctness[c]=oks;mean={str(d):{k:float(np.mean([v[str(d)][k] for v in metrics])) for k in metrics[0][str(d)]} for d in depths(c)}
            bybin={str(d):{b:{k:float(np.mean([v[str(d)][b][k] for v in strata])) for k in strata[0][str(d)][b]} for b in strata[0][str(d)]} for d in depths(c)}
            transitions={}
            if len(depths(c))>1:
                for a,b in zip(cfg['depths'][:-1],cfg['depths'][1:]):
                    old=np.array([o[str(a)] for o in oks]);new=np.array([o[str(b)] for o in oks]);transitions[f'{a}->{b}']={'wrong_right':float((~old&new).mean()),'right_wrong':float((old&~new).mean()),'difference':paired(new.astype(float)-old.astype(float),rows,1000)}
            result['models'][c]={'mean':mean,'seed_metrics':metrics,'by_path':bybin,'transitions':transitions,'oscillation':changes}
        old=np.array([o['4'] for o in correctness['frozen_05b']]);new=np.array([o['8'] for o in correctness['frozen_05b']]);result['frozen_8_minus_4']=paired(new.astype(float)-old.astype(float),rows)
        selected=f['selected_recurrent'];rec=np.array([o['8'] for o in correctness[selected]],float)
        result['selected_recurrent_vs_controls']={c:paired(rec-np.array([o[str(cfg['conditions'][c]['depth'])] for o in correctness[c]],float),rows) for c in ('ff4_params','ff8','global_cnn')}
        summary['splits'][split]=result;write_json(OUT/'summary.json',summary)
        print(split, {c:{d:round(v['accuracy'],3) for d,v in r['mean'].items() if int(d) in cfg['depths']} for c,r in result['models'].items()},flush=True)
    write_json(OUT/'evaluation_resources.json',{'completed':now(),'peak_mlx_gib':mx.get_peak_memory()/1024**3,'peak_rss_gib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3})

def policy(m,depth):
    import mlx.core as mx
    from latent_loop.scaling.model import forward
    def act(maze):
        # Raw board only. No labels, BFS, maps or state feedback enters this call.
        z=forward(m,mx.array(maze.observation()[None]),(depth,))[depth]['logits'];mx.eval(z);return int(mx.argmax(z,-1)[0])
    return act

def episode(initial,act,max_steps):
    m=initial;positions=[list(m.agent)];actions=[];invalid=0;regret=[];t=0.;seen={m.agent};done=False;cycle=False
    optimum=labels(m)['distance']
    for _ in range(max_steps):
        start=time.perf_counter();a=act(m);t+=time.perf_counter()-start
        # Offline labels are computed only after committed model action.
        regret.append(labels(m)['regret'][a]);m,_,done,info=m.step(a);invalid+=int(info['invalid']);actions.append(a);positions.append(list(m.agent))
        if done:break
        if m.agent in seen:cycle=True;break  # Deterministic stateless policy cannot escape.
        seen.add(m.agent)
    return {'done':done,'steps':len(actions),'optimum':optimum,'step_ratio':len(actions)/optimum,'invalid':invalid,'cycle':cycle,'positions':positions,'actions':actions,'regret':regret,'inference_seconds':t,'step_budget':max_steps}

def rollouts():
    from latent_loop.backbone import environment
    environment(48);f=verify();cfg=config();result={}
    for split,spec in cfg['datasets'].items():
        if not spec.get('final'):continue
        allrows=load(split,True);rows=[]
        for b in sorted({r['path_bin'] for r in allrows}):rows.extend([r for r in allrows if r['path_bin']==b][:cfg['rollout']['per_path_bin']])
        splitres={'maze_count':len(rows),'models':{}}
        for c in [*models(),'greedy','bfs']:
            dd=cfg['rollout']['depths'] if c=='frozen_05b' or (c in cfg['conditions'] and cfg['conditions'][c]['shared']) else [cfg['conditions'][c]['depth']] if c in cfg['conditions'] else [0]
            for d in dd:
                path=OUT/'rollouts'/f'{c}_depth{d}_{split}.jsonl.gz'
                if path.exists():records=read(path)
                else:
                    records=[]
                    for s in seeds(c) if c in models() else [0]:
                        act=policy(model(c,s),d) if c in models() else greedy_action if c=='greedy' else bfs_action
                        for row in rows:records.append({'id':row['id'],'seed':s,'path_bin':row['path_bin'],**episode(Maze.from_dict(row),act,cfg['rollout']['max_steps'])})
                    save_rows(path,records)
                success=np.array([[r['done'] for r in records if r['seed']==s] for s in sorted({r['seed'] for r in records})],float)
                splitres['models'][f'{c}@{d}']={'completion':float(success.mean()),'completion_interval':paired(success,rows,1000),'mean_steps':float(np.mean([r['steps'] for r in records])),'solved_step_ratio':float(np.mean([r['step_ratio'] for r in records if r['done']])) if success.any() else None,'mean_invalid':float(np.mean([r['invalid'] for r in records])),'cycle_rate':float(np.mean([r['cycle'] for r in records])),'mean_inference_seconds':float(np.mean([r['inference_seconds'] for r in records])),'by_path':{b:float(np.mean([r['done'] for r in records if r['path_bin']==b])) for b in sorted({r['path_bin'] for r in records})}}
        result[split]=splitres;write_json(OUT/'rollout_summary.json',result);print('rollouts',split,{c:round(v['completion'],3) for c,v in splitres['models'].items()},flush=True)

def latency():
    import mlx.core as mx
    from latent_loop.scaling.model import forward
    records=[]
    for size in (5,7,9,11,15):
        rows=load('validation');base=next(r for r in rows if len(r['walls'])==5)
        # Standardized open boards for native-size timing; no final selection.
        maze=Maze(tuple(tuple(0 for _ in range(size)) for _ in range(size)),(0,0),(size-1,size-1))
        for c in models():
            m=model(c,seeds(c)[0]);dd=config()['depths'] if c=='frozen_05b' or config()['conditions'][c]['shared'] else [config()['conditions'][c]['depth']]
            for d in dd:
                for batch in (1,8):
                    times=[]
                    for i in range(13):
                        t=time.perf_counter();x=mx.array(np.stack([maze.observation() for _ in range(batch)]));z=forward(m,x,(d,))[d]['logits'];mx.eval(z);a=np.asarray(mx.argmax(z,-1));elapsed=time.perf_counter()-t
                        if i>=3:times.append(elapsed)
                    records.append({'model':c,'size':size,'depth':d,'batch':batch,'milliseconds_per_item':1000*float(np.median(times))/batch,'samples_seconds':times})
    write_json(OUT/'latency.json',{'procedure':'3 warmups10 repeats, synchronized MLX full raw observation construction/transfer/policy/host argmax; median, first prespecified seed; serial condition order','rows':records,'peak_mlx_gib':mx.get_peak_memory()/1024**3})

if __name__=='__main__':
    import sys
    {'freeze':freeze,'final':evaluate,'rollout':rollouts,'latency':latency}[sys.argv[1]]()
