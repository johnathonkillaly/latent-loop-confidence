"""Prospective secondary Snake neural study; frozen maze models are untouched."""
import json
import time
import resource
import hashlib
from pathlib import Path
import numpy as np
from latent_loop.environments.snake import Snake,teacher
from latent_loop.environments.maze import ACTIONS
from latent_loop.planning.data import ROOT,RESULTS,ARTIFACTS,write_json,save_rows,digest,now
from latent_loop.planning.evaluate import read_trace,paired_interval

def config(): return json.loads((ROOT/"snake_config.json").read_text())

def canonical(state):
    x=np.zeros((state.size,state.size),dtype=np.uint8)
    for p in state.walls:x[p]=1
    for i,p in enumerate(state.body):x[p]=i+2
    if state.food is not None:x[state.food]=7
    # Heading is determined by the ordered head/neck edge for this generator.
    return hashlib.sha256(min(np.rot90(y,k).tobytes() for y in (x,np.fliplr(x)) for k in range(4))).hexdigest()

def generate():
    c=config();path=ROOT/"snake_dataset_manifest.json"
    if path.exists():raise RuntimeError("Snake datasets already exist")
    seen=set();manifest={"created":now(),"config_sha256":digest(ROOT/"snake_config.json"),"canonicalization":"D4 of walls, ordered body and food; heading determined by head/neck edge","splits":{}}
    for split,count,seed in zip(("train","validation","final"),(c["train_count"],c["validation_count"],c["final_count"]),c["split_seeds"]):
        rng=np.random.default_rng(seed);rows=[];attempts=0
        while len(rows)<count:
            attempts+=1;size=c["board_size"];wall=rng.random((size,size))<c["wall_probability"]
            free=[(r,col) for r in range(size) for col in range(size) if not wall[r,col]]
            if len(free)<6:continue
            body=[free[int(rng.integers(len(free)))]]
            for _ in range(int(rng.choice(c["body_length"]))-1):
                p=body[-1];choices=[(p[0]+dr,p[1]+dc) for dr,dc in ACTIONS if (p[0]+dr,p[1]+dc) in free and (p[0]+dr,p[1]+dc) not in body]
                if not choices:break
                body.append(choices[int(rng.integers(len(choices)))])
            if len(body)<2:continue
            heading=ACTIONS.index((body[0][0]-body[1][0],body[0][1]-body[1][1]))
            foodchoices=[p for p in free if p not in body]
            if not foodchoices:continue
            food=None if rng.random()<c["post_food_fraction"] else foodchoices[int(rng.integers(len(foodchoices)))]
            state=Snake(size,tuple(body),food,heading,tuple(map(tuple,np.argwhere(wall).tolist())),collected=int(food is None))
            h=canonical(state)
            if h in seen:continue
            seen.add(h);targets={str(k):teacher(state,k)["optimal"] for k in c["teacher_horizons"]};safe=[state.step(a)[0].alive for a in range(4)]
            rows.append({"id":f"snake-{split}-{len(rows):05d}","state":state.to_dict(),"hash":h,"teacher":targets,"safe":safe})
        target=(ARTIFACTS/"snake_sealed" if split=="final" else RESULTS/"snake_neural"/"datasets")/f"{split}.jsonl.gz";save_rows(target,rows)
        manifest["splits"][split]={"count":count,"seed":seed,"attempts":attempts,"sha256":digest(target)}
        print(f"Snake generated {split} {len(rows)}",flush=True)
    write_json(path,manifest)

def rows(split):
    if split=="final" and not (ROOT/"snake_frozen_protocol.json").exists():raise RuntimeError("Freeze Snake protocol first")
    return read_trace((ARTIFACTS/"snake_sealed" if split=="final" else RESULTS/"snake_neural"/"datasets")/f"{split}.jsonl.gz")

def observation(state):
    x=np.zeros((state.size,state.size,9),dtype=np.float32);x[...,1]=1
    for p in state.walls:x[p+(0,)]=1;x[p+(1,)]=0
    x[state.body[0]+(2,)]=1
    if state.food is not None:x[state.food+(3,)]=1
    for i,p in enumerate(state.body):x[p+(4,)]=(i+1)/4
    x[...,5+state.heading]=1
    return x

def arrays(data,objective):
    x=np.stack([observation(Snake.from_dict(r["state"])) for r in data]);y=[]
    for row in data:
        target=row["safe"] if objective=="safety" else row["teacher"]["6"]
        # Impossible safety states contribute zero training loss, not fake
        # successful actions. Evaluation reports actual collisions explicitly.
        y.append(target if any(target) else [True]*4)
    return x,np.array(y,dtype=bool)

def build(spec,seed):
    import mlx.core as mx
    import mlx.nn as nn
    from latent_loop.planning.grid_recurrence import GridPlanner
    mx.random.seed(seed);model=GridPlanner(spec["width"],spec["blocks"],spec["shared"])
    model.encoder=nn.Linear(9,spec["width"])
    return model

def train_all():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten,tree_unflatten
    from latent_loop.backbone import environment
    from latent_loop.planning.grid_recurrence import set_loss
    from latent_loop.planning.train import validate
    c=config();environment(c["memory_gib"])
    if (ROOT/"snake_frozen_protocol.json").exists():raise RuntimeError("Snake lockbox already frozen")
    for condition,spec in c["conditions"].items():
        x,y=arrays(rows("train"),spec["objective"]);vx,vy=arrays(rows("validation"),spec["objective"])
        for seed in c["seeds"]:
            model=build(spec,seed);opt=optim.AdamW(c["learning_rate"],weight_decay=.0001);depth=4 if spec["shared"] else spec["blocks"]
            folder=ARTIFACTS/"checkpoints"/f"{condition}_seed{seed}";folder.mkdir(parents=True,exist_ok=True)
            path=RESULTS/"snake_neural"/"training"/f"{condition}_seed{seed}.json"
            signature={"config_sha256":digest(ROOT/"snake_config.json"),"dataset_sha256":digest(ROOT/"snake_dataset_manifest.json"),"source_sha256":digest(__file__)}
            record={"condition":condition,"seed":seed,"parameters":model.parameter_count(),"signature":signature,"epochs":[],"best_loss":1e30}
            if path.exists():
                record=json.loads(path.read_text());assert record["signature"]==signature
                model.load_weights(str(folder/"last.safetensors"));opt.state=tree_unflatten(list(mx.load(str(folder/"optimizer.safetensors")).items()))
            vg=nn.value_and_grad(model,lambda m,xx,yy,d:set_loss(m(xx,(d,))[d]["logits"],yy))
            for epoch in range(len(record["epochs"]),c["epochs"]):
                t=time.perf_counter();rng=np.random.default_rng(seed*100000+epoch);order=rng.permutation(len(x));losses=[];norms=[]
                for i in range(0,len(order),c["batch_size"]):
                    idx=order[i:i+c["batch_size"]];d=int(rng.choice([1,2,4,8])) if spec["shared"] else depth
                    loss,g=vg(model,mx.array(x[idx]),mx.array(y[idx]),d);norm=mx.sqrt(sum(mx.sum(v*v) for _,v in tree_flatten(g)));g,_=optim.clip_grad_norm(g,1.);opt.update(model,g);mx.eval(model.parameters(),opt.state,loss,norm);losses.append(float(loss));norms.append(float(norm))
                val=validate(model,vx,vy,depth)
                if val["loss"]<record["best_loss"]:record["best_loss"]=val["loss"];record["best_epoch"]=epoch+1;model.save_weights(str(folder/"best.safetensors"))
                model.save_weights(str(folder/"last.safetensors"));mx.save_safetensors(str(folder/"optimizer.safetensors"),dict(tree_flatten(opt.state)))
                record["epochs"].append({"epoch":epoch+1,"training_loss":float(np.mean(losses)),"gradient_norm_mean":float(np.mean(norms)),"gradient_norm_max":max(norms),"validation":val,"seconds":time.perf_counter()-t});record["peak_mlx_gib"]=mx.get_peak_memory()/1024**3;record["checkpoint_sha256"]=digest(folder/"best.safetensors");write_json(path,record)
            print(f"Snake trained {condition} seed {seed} best val loss {record['best_loss']:.4f}",flush=True)

def evaluate():
    import mlx.core as mx
    from latent_loop.backbone import environment
    from latent_loop.planning.grid_recurrence import set_loss
    c=config();environment(c["memory_gib"]);frozen=ROOT/"snake_frozen_protocol.json"
    if not frozen.exists():
        paths=[ROOT/"snake_config.json",ROOT/"snake_dataset_manifest.json",Path(__file__).relative_to(Path.cwd()) if Path(__file__).is_absolute() else Path(__file__),Path("src/latent_loop/environments/snake.py"),Path("src/latent_loop/planning/grid_recurrence.py")]
        paths += [ARTIFACTS/"checkpoints"/f"{condition}_seed{seed}"/"best.safetensors" for condition in c["conditions"] for seed in c["seeds"]]
        write_json(frozen,{"frozen":now(),"hashes":{str(p):digest(p) for p in paths},"primary":c["primary"],"checkpoint_rule":c["checkpoint_rule"]})
    assert all(digest(p)==h for p,h in json.loads(frozen.read_text())["hashes"].items())
    data=rows("final");save_rows(RESULTS/"snake_neural"/"datasets"/"final.jsonl.gz",data);x,_=arrays(data,"planning");summary={"models":{},"paired":{},"config":c,"unavoidable_immediate_collision_fraction":float(np.mean([not any(r["safe"]) for r in data]))};correctness={}
    for condition,spec in c["conditions"].items():
        ds=(1,2,4,8) if spec["shared"] else (spec["blocks"],);seed_metrics=[];seed_correct=[];timings=[]
        for seed in c["seeds"]:
            model=build(spec,seed);model.load_weights(str(ARTIFACTS/"checkpoints"/f"{condition}_seed{seed}"/"best.safetensors"));model.eval();trace=model(mx.array(x),ds);mx.eval(trace)
            records=[{"id":r["id"],"depths":{str(d):{k:np.asarray(v)[i].tolist() for k,v in trace[d].items()} for d in ds}} for i,r in enumerate(data)];save_rows(RESULTS/"snake_neural"/"trajectories"/f"{condition}_seed{seed}.jsonl.gz",records)
            metrics={};correct={}
            for d in ds:
                logits=np.asarray(trace[d]["logits"]);pred=logits.argmax(-1);index=np.arange(len(data));targets={h:np.array([r["teacher"][str(h)] for r in data]) for h in (2,4,6)};safe=np.array([r["safe"] for r in data]);metrics[str(d)]={"horizon_accuracy":{str(h):float(targets[h][index,pred].mean()) for h in targets},"immediate_safety":float(safe[index,pred].mean()),"illegal_reverse":float(np.mean([pred[i]==(r["state"]["heading"]+2)%4 for i,r in enumerate(data)]))};correct[str(d)]=targets[6][index,pred]
                def once():
                    xx=mx.array(observation(Snake.from_dict(data[0]["state"]))[None]);tr=model(xx,(d,));mx.eval(tr);return int(mx.argmax(tr[d]["logits"],axis=-1)[0])
                for _ in range(3):once()
                times=[]
                for _ in range(5):
                    mx.synchronize();t=time.perf_counter();once();mx.synchronize();times.append(time.perf_counter()-t)
                timings.append({"seed":seed,"depth":d,"seconds":times})
            seed_metrics.append(metrics);seed_correct.append(correct)
        mean={str(d):{"horizon_accuracy":{str(h):float(np.mean([r[str(d)]["horizon_accuracy"][str(h)] for r in seed_metrics])) for h in (2,4,6)},"immediate_safety":float(np.mean([r[str(d)]["immediate_safety"] for r in seed_metrics])),"illegal_reverse":float(np.mean([r[str(d)]["illegal_reverse"] for r in seed_metrics]))} for d in ds}
        summary["models"][condition]={"parameters":model.parameter_count(),"mean":mean,"seed_metrics":seed_metrics,"latency_batch1":timings};correctness[condition]=seed_correct
    for a,b in ((1,2),(2,4),(1,4),(4,8)):
        old=np.array([r[str(a)] for r in correctness["snake_plan_rec"]]);new=np.array([r[str(b)] for r in correctness["snake_plan_rec"]]);summary["paired"][f"{a}->{b}"]={"accuracy":paired_interval(new.astype(float)-old.astype(float)),"wrong_right":float((~old&new).mean()),"right_wrong":float((old&~new).mean())}
    summary["paired"]["rec4_vs_ff4"]=paired_interval(np.array([r['4'] for r in correctness['snake_plan_rec']],float)-np.array([r['4'] for r in correctness['snake_plan_ff4']],float))
    write_json(RESULTS/"snake_neural_summary.json",summary);print({name:value['mean'] for name,value in summary['models'].items()},flush=True)

def episodes():
    import mlx.core as mx
    from latent_loop.backbone import environment
    c=config();environment(c["memory_gib"]);assert (ROOT/"snake_frozen_protocol.json").exists()
    data=[r for r in rows("final") if r["state"]["food"] is not None][:c["episodes"]];summary={}
    conditions=[(name,d) for name,spec in c['conditions'].items() for d in ((1,2,4,8) if spec['shared'] else (spec['blocks'],))]+[("random_legal",0),("safe_greedy",0),("oracle_h6",0)]
    for name,depth in conditions:
        all_records=[]
        for seed in (c['seeds'] if name in c['conditions'] else [0]):
            if name in c['conditions']:
                model=build(c['conditions'][name],seed);model.load_weights(str(ARTIFACTS/'checkpoints'/f'{name}_seed{seed}'/'best.safetensors'))
            for i,row in enumerate(data):
                state=Snake.from_dict(row['state']);rng=np.random.default_rng(559900+i);survived=0;illegal=0;actions=[];times=[]
                for _ in range(c['maximum_episode_steps']):
                    t=time.perf_counter()
                    if name in c['conditions']:
                        trace=model(mx.array(observation(state)[None]),(depth,));mx.eval(trace);a=int(mx.argmax(trace[depth]['logits'],axis=-1)[0])
                    elif name=='oracle_h6':a=teacher(state,6)['optimal'].index(True)
                    elif name=='random_legal':a=int(rng.choice([a for a in range(4) if a!=(state.heading+2)%4]))
                    else:
                        safe=[a for a in range(4) if state.step(a)[0].alive]
                        def key(a):
                            nxt,_=state.step(a);p=nxt.body[0];return (0 if nxt.food is None else abs(p[0]-nxt.food[0])+abs(p[1]-nxt.food[1]),a)
                        a=min(safe,key=key) if safe else state.heading
                    times.append(time.perf_counter()-t);state,info=state.step(a);actions.append(a);illegal+=int(info['illegal'])
                    if not state.alive:break
                    survived+=1
                all_records.append({'id':row['id'],'seed':seed,'survival_steps':survived,'food_collected':state.collected,'collision':not state.alive,'illegal_actions':illegal,'actions':actions,'inference_seconds':sum(times)})
        save_rows(RESULTS/'snake_neural'/'episodes'/f'{name}_d{depth}.jsonl.gz',all_records)
        summary[f'{name}@{depth}']={k:float(np.mean([r[k] for r in all_records])) for k in ('survival_steps','food_collected','collision','illegal_actions','inference_seconds')};print('Snake episodes',name,depth,summary[f'{name}@{depth}'],flush=True)
    write_json(RESULTS/'snake_neural_episodes.json',summary)

if __name__=='__main__':
    import sys
    {'generate':generate,'train':train_all,'evaluate':evaluate,'episodes':episodes}[sys.argv[1]]()
