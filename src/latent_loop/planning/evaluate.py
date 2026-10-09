"""Prospective selection freeze, paired maze-level metrics, and live rollouts."""
import gzip
import json
import time
import resource
from pathlib import Path
import numpy as np
from scipy.special import softmax,logsumexp
from latent_loop.environments.maze import Maze,labels
from latent_loop.planning.data import ROOT,RESULTS,ARTIFACTS,load_rows,arrays,write_json,save_rows,digest,now

def paired_interval(values,repeats=2000,seed=551999):
    """Crossed resampling of training seeds and independent initial maze layouts."""
    values=np.asarray(values,dtype=float)
    if values.ndim==1: values=values[None]
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(repeats):
        s=rng.integers(values.shape[0],size=values.shape[0]); i=rng.integers(values.shape[1],size=values.shape[1])
        samples.append(values[s][:,i].mean())
    return {"mean":float(values.mean()),"ci95":np.quantile(samples,[.025,.975]).tolist(),"training_seeds":values.shape[0],"independent_mazes":values.shape[1]}

def correct_and_metrics(rows,logits):
    logits=np.asarray(logits); p=softmax(logits,axis=-1); pred=p.argmax(-1)
    opt=np.array([r["labels"]["optimal"] for r in rows]); legal=np.array([r["labels"]["legal"] for r in rows]); regret=np.array([r["labels"]["regret"] for r in rows]); index=np.arange(len(rows))
    correct=opt[index,pred]; confidence=p.max(-1); ece=0.
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        selected=(confidence>=lo)&(confidence<hi if hi<1 else confidence<=hi)
        if selected.any(): ece+=selected.mean()*abs(correct[selected].mean()-confidence[selected].mean())
    nll=logsumexp(logits,axis=-1)-logsumexp(np.where(opt,logits,-1e9),axis=-1)
    return correct,{"accuracy":float(correct.mean()),"set_nll":float(nll.mean()),"invalid_rate":float((~legal[index,pred]).mean()),"mean_regret":float(regret[index,pred].mean()),"ece":float(ece)}

def get_grid(condition,seed,config):
    from latent_loop.planning.train import build
    model=build(config["conditions"][condition],seed)
    model.load_weights(str(ARTIFACTS/"checkpoints"/f"{condition}_seed{seed}"/"best.safetensors")); model.eval()
    return model

def predict_grid(model,rows,depths):
    import mlx.core as mx
    x,_=arrays(rows); chunks={d:[] for d in depths}
    for i in range(0,len(x),128):
        trace=model(mx.array(x[i:i+128]),depths); mx.eval(trace)
        for d in depths:
            chunks[d].extend({k:np.asarray(v)[j].tolist() for k,v in trace[d].items()} for j in range(len(x[i:i+128])))
    return chunks

def freeze():
    import mlx.core as mx
    from latent_loop.backbone import environment
    config=json.loads((ROOT/"config.json").read_text()); environment(config["memory_gib"])
    if (ROOT/"frozen_protocol.json").exists(): raise RuntimeError("Protocol already frozen")
    rows=load_rows("validation"); scores={}; checkpoints={}
    for condition,spec in config["conditions"].items():
        vals=[]
        for seed in config["seeds"]:
            path=ARTIFACTS/"checkpoints"/f"{condition}_seed{seed}"/"best.safetensors"
            train=json.loads((RESULTS/"training"/f"{condition}_seed{seed}.json").read_text())
            if train["completed_epochs"]!=config["epochs"]: raise RuntimeError("Incomplete fit")
            checkpoints[str(path)]=digest(path)
            model=get_grid(condition,seed,config)
            logits=[r["logits"] for r in predict_grid(model,rows,(spec["depth"],))[spec["depth"]]]
            _,metric=correct_and_metrics(rows,logits); vals.append(metric)
        scores[condition]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
    ranking=lambda c:(-scores[c]["accuracy"],scores[c]["set_nll"],c)
    selected=min([c for c in scores if c.startswith("rec_")],key=ranking)
    control=min([c for c in scores if c.startswith("ff4_")],key=ranking)
    paths=list(Path("src/latent_loop/planning").glob("*.py"))+list(Path("src/latent_loop/environments").glob("*.py"))+[ROOT/"config.json",ROOT/"plan.md",ROOT/"dataset_manifest.json"]
    record={"frozen":now(),"validation_scores":scores,"selected_recurrent":selected,"selected_feedforward":control,
            "primary_depth_pair":[1,4],"secondary_depth_pairs":[[1,2],[2,4]],"checkpoints":checkpoints,
            "sources":{str(p):digest(p) for p in paths if p.name not in ("report.py","snake_experiment.py","snake.py")},
            "qwen_protocol":"Fixed in original config/plan; trained/evaluated separately with its own checkpoint freeze, no grid final selection feedback"}
    write_json(ROOT/"frozen_protocol.json",record)
    print(json.dumps({k:record[k] for k in ("selected_recurrent","selected_feedforward","validation_scores")}),flush=True)

def verify_freeze():
    record=json.loads((ROOT/"frozen_protocol.json").read_text())
    for path,h in {**record["sources"],**record["checkpoints"]}.items():
        if digest(path)!=h: raise RuntimeError(f"Frozen artifact changed: {path}")
    return record

def read_trace(path):
    with gzip.open(path,"rt") as f: return [json.loads(line) for line in f]

def evaluate():
    import mlx.core as mx
    from latent_loop.backbone import environment
    from latent_loop.planning.oracles import bfs_action,greedy_action,propagation
    freeze_record=verify_freeze(); config=json.loads((ROOT/"config.json").read_text()); environment(config["memory_gib"])
    if not (ROOT/"lockbox_opened.json").exists(): write_json(ROOT/"lockbox_opened.json",{"opened":now(),"protocol_sha256":digest(ROOT/"frozen_protocol.json")})
    summary={"selected_recurrent":freeze_record["selected_recurrent"],"selected_feedforward":freeze_record["selected_feedforward"],"splits":{}}
    for split,spec in config["datasets"].items():
        if not spec.get("final"): continue
        source=ARTIFACTS/"sealed"/f"{split}.jsonl.gz"
        if digest(source)!=json.loads((ROOT/"dataset_manifest.json").read_text())["splits"][split]["sha256"]: raise RuntimeError("Dataset hash changed")
        rows=load_rows(split,True); save_rows(RESULTS/"datasets"/f"{split}.jsonl.gz",rows)
        split_result={"count":len(rows),"models":{},"baselines":{}}
        for name,policy in (("bfs",bfs_action),("greedy",greedy_action)):
            pred=[policy(Maze.from_dict(r)) for r in rows]
            logits=np.full((len(rows),4),-30.); logits[np.arange(len(rows)),pred]=0
            _,metric=correct_and_metrics(rows,logits); split_result["baselines"][name]=metric
        # Bellman reference commits from known finite neighbor distances; unknowns
        # use the documented Manhattan fallback. Never a learned input.
        prop={}
        for depth in (1,2,4,8):
            preds=[]
            for row in rows:
                maze=Maze.from_dict(row); d=propagation(maze,depth)[-1]
                choices=[(d[maze.neighbor(maze.agent,a)],a) for a in range(4) if maze.free(maze.neighbor(maze.agent,a))]
                distance,action=min(choices); preds.append(action if np.isfinite(distance) else greedy_action(maze))
            logits=np.full((len(rows),4),-30.); logits[np.arange(len(rows)),preds]=0
            prop[str(depth)]=correct_and_metrics(rows,logits)[1]
        split_result["baselines"]["classical_propagation"]=prop
        correctness={}
        for condition,model_spec in config["conditions"].items():
            depths=tuple(range(1,9)) if model_spec["shared"] else (model_spec["depth"],)
            seed_metrics=[]; seed_correct=[]; family_metrics=[]
            for seed in config["seeds"]:
                path=RESULTS/"trajectories"/f"{condition}_seed{seed}_{split}.jsonl.gz"
                if path.exists(): records=read_trace(path)
                else:
                    model=get_grid(condition,seed,config); trace=predict_grid(model,rows,depths)
                    records=[{"id":row["id"],"depths":{str(d):trace[d][i] for d in depths}} for i,row in enumerate(rows)]
                    save_rows(path,records)
                metrics={}; correct={}; per_family={}
                for d in depths:
                    logits=[r["depths"][str(d)]["logits"] for r in records]
                    ok,metric=correct_and_metrics(rows,logits); metrics[str(d)]=metric; correct[str(d)]=ok
                    per_family[str(d)]={str(f):correct_and_metrics([rows[i] for i in idx],np.asarray(logits)[idx])[1] for f in sorted({r["family"] for r in rows}) if len(idx:=[i for i,r in enumerate(rows) if r["family"]==f])}
                seed_metrics.append(metrics); seed_correct.append(correct); family_metrics.append(per_family)
            correctness[condition]=seed_correct
            average={str(d):{k:float(np.mean([v[str(d)][k] for v in seed_metrics])) for k in seed_metrics[0][str(d)]} for d in depths}
            transitions={}
            if model_spec["shared"]:
                for a,b in ((1,2),(2,4),(1,4),(4,8)):
                    old=np.array([v[str(a)] for v in seed_correct]); new=np.array([v[str(b)] for v in seed_correct])
                    transitions[f"{a}->{b}"]={"paired_accuracy":paired_interval(new.astype(float)-old.astype(float)),"wrong_right":float((~old&new).mean()),"right_wrong":float((old&~new).mean()),"conditional_wrong_right":float((~old&new).sum()/max(1,(~old).sum())),"conditional_right_wrong":float((old&~new).sum()/max(1,old.sum()))}
            split_result["models"][condition]={"mean":average,"seed_metrics":seed_metrics,"family_metrics":family_metrics,"transitions":transitions}
        r=correctness[freeze_record["selected_recurrent"]]; f=correctness[freeze_record["selected_feedforward"]]
        split_result["recurrent_vs_feedforward"]=paired_interval(np.array([v["4"] for v in r],float)-np.array([v["4"] for v in f],float))
        summary["splits"][split]=split_result
        write_json(RESULTS/"summary.json",summary)
        print(f"evaluated {split}: "+str({c:{d:round(v['accuracy'],3) for d,v in m['mean'].items() if d in ('1','2','4','8')} for c,m in split_result['models'].items()}),flush=True)
    write_json(RESULTS/"grid_evaluation_resources.json",{"peak_mlx_gib":mx.get_peak_memory()/1024**3,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,"completed":now()})
    latency(config)

def latency(config):
    import mlx.core as mx
    rows=load_rows("validation")[:config["latency"]["items"]]; results={}
    for condition,spec in config["conditions"].items():
        values=[]
        for seed in config["seeds"]:
            model=get_grid(condition,seed,config)
            for depth in ((1,2,4,8) if spec["shared"] else (spec["depth"],)):
                for batch in (1,8):
                    subset=rows[:batch]
                    def once():
                        x,_=arrays(subset); trace=model(mx.array(x),(depth,)); mx.eval(trace)
                        return np.asarray(mx.argmax(trace[depth]["logits"],axis=-1))
                    for _ in range(config["latency"]["warmups"]): once()
                    reps=[]
                    for _ in range(config["latency"]["repeats"]):
                        mx.synchronize(); t=time.perf_counter(); once(); mx.synchronize(); reps.append(time.perf_counter()-t)
                    values.append({"seed":seed,"depth":depth,"batch":batch,"seconds":reps,"per_item_ms":float(np.median(reps)*1000/batch)})
        results[condition]=values
    write_json(RESULTS/"latency.json",{"procedure":"Fresh four-channel observation construction, MLX transfer, full grid model, argmax host conversion, synchronization. No checkpoint load. First 1/8 validation examples; three warmups, five repeats per seed. Serial conditions; tiny timings include dispatch overhead.","models":results})

def rollouts():
    import mlx.core as mx
    from latent_loop.backbone import environment
    from latent_loop.planning.rollout import navigate
    from latent_loop.planning.oracles import bfs_action,greedy_action
    frozen=verify_freeze(); config=json.loads((ROOT/"config.json").read_text()); environment(config["memory_gib"])
    names=[frozen["selected_recurrent"],frozen["selected_feedforward"],"ff_onepass"]
    summary={}
    for split in ("fresh_id","natural","structure","larger","long_paths"):
        rows=load_rows(split,True)[:config["rollout"]["per_split"]]; summary[split]={}
        for name in ["bfs","greedy"]+names:
            depths=(1,2,4,8) if name.startswith("rec_") else (config["conditions"][name]["depth"],) if name.startswith("ff") else (0,)
            for depth in depths:
                seeds=config["seeds"] if name in names else [0]; seed_records=[]
                for seed in seeds:
                    path=RESULTS/"rollouts"/f"{name}_d{depth}_seed{seed}_{split}.jsonl.gz"
                    if path.exists(): records=read_trace(path)
                    else:
                        if name in names:
                            model=get_grid(name,seed,config)
                            def policy(maze):
                                trace=model(mx.array(maze.observation()[None]),(depth,)); mx.eval(trace)
                                return int(mx.argmax(trace[depth]["logits"],axis=-1)[0])
                        else: policy=bfs_action if name=="bfs" else greedy_action
                        records=[]
                        for row in rows:
                            budget=max(config["rollout"]["minimum_budget"],row["labels"]["distance"]*config["rollout"]["step_multiplier"])
                            records.append({"id":row["id"],"family":row["family"],**navigate(Maze.from_dict(row),policy,budget)})
                        save_rows(path,records)
                    seed_records.append(records)
                success=np.array([[r["success"] for r in records] for records in seed_records])
                flat=[r for records in seed_records for r in records]; solved=[r for r in flat if r["success"]]
                summary[split][f"{name}@{depth}"]={"completion":paired_interval(success),"steps_mean":float(np.mean([r["steps"] for r in flat])),"solved_steps_over_optimal":float(np.mean([r["steps_over_optimal"] for r in solved])) if solved else None,"invalid_moves_mean":float(np.mean([r["invalid_moves"] for r in flat])),"repeated_positions_mean":float(np.mean([r["repeated_positions"] for r in flat])),"inference_seconds_mean":float(np.mean([r["inference_seconds"] for r in flat])),"solved_inference_seconds_mean":float(np.mean([r["inference_seconds"] for r in solved])) if solved else None,"by_family":{str(f):float(np.mean([r["success"] for r in flat if r["family"]==f])) for f in sorted({r["family"] for r in flat})}}
                print(f"rollout {split} {name}@{depth}: {success.mean():.3f}",flush=True)
        write_json(RESULTS/"rollout_summary.json",summary)
