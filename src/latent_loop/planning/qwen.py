"""Maze adaptation of the existing frozen middle-layer Qwen thinker.

Separate labels and observations, no KV caches, unchanged positions per loop.
The Qwen study has its own smaller, equal data budget for one-pass/recurrent fits.
"""
import json
import time
import resource
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten,tree_unflatten
from latent_loop.backbone import environment,get_model,tokenize
from latent_loop.experiment05.model import Thinker
from latent_loop.planning.grid_recurrence import set_loss
from latent_loop.planning.data import ROOT,RESULTS,ARTIFACTS,load_rows,prompt,write_json,save_rows,digest,now
from latent_loop.planning.evaluate import correct_and_metrics,paired_interval,read_trace

def batch(tokenizer,rows,alternative=False):
    tokens=[tokenize(tokenizer,prompt(r,alternative)) for r in rows]
    lengths=[len(t) for t in tokens]; pad=tokenizer.pad_token_id or tokenizer.eos_token_id
    return mx.array([t+[pad]*(max(lengths)-len(t)) for t in tokens]),mx.array(lengths)

def build(base,tokenizer,config,condition,seed):
    mx.random.seed(seed); c=config["qwen"]
    model=Thinker(base,[tokenizer.encode(l,add_special_tokens=False)[0] for l in "ABCD"],c["start"],c["count"],c["rank"],"none")
    if condition=="qwen_onepass": model.freeze(keys=["update_logit"],recurse=False)
    elif condition!="qwen_recurrent": raise ValueError("Use qwen_onepass or qwen_recurrent")
    return model

def predict(model,tokenizer,rows,depths=(1,),alternative=False):
    trace={d:[] for d in depths}
    for start in range(0,len(rows),8):
        ids,lengths=batch(tokenizer,rows[start:start+8],alternative)
        result=model(ids,lengths,depths); mx.eval(result)
        for d in depths:
            trace[d].extend({k:np.asarray(v)[i].tolist() for k,v in result[d].items()} for i in range(len(rows[start:start+8])))
    return trace

def validation():
    config=json.loads((ROOT/"config.json").read_text()); env=environment(config["memory_gib"])
    base,tokenizer,manifest=get_model(); rows=load_rows("train")[:2]
    ids,lengths=batch(tokenizer,rows); label_ids=[tokenizer.encode(l,add_special_tokens=False)[0] for l in "ABCD"]
    h=base.model(ids)[mx.arange(len(rows)),lengths-1]; native=h@base.model.embed_tokens.weight[mx.array(label_ids)].T; mx.eval(native)
    model=build(base,tokenizer,config,"qwen_recurrent",11)
    parity=float(mx.max(mx.abs(model(ids,lengths,(1,))[1]["logits"]-native)))
    y=mx.array([r["labels"]["optimal"] for r in rows])
    loss,g=nn.value_and_grad(model,lambda m:set_loss(m(ids,lengths,(4,))[4]["logits"],y))(model); mx.eval(loss,g)
    norms={k:float(mx.linalg.norm(v)) for k,v in tree_flatten(g)}
    opt=optim.AdamW(.0005); opt.update(model,g); mx.eval(model.trainable_parameters(),opt.state)
    cut=24; full=model(ids[:1],depths=(1,4),return_tokens=True); prefix=model(ids[:1,:cut],depths=(1,4),return_tokens=True)
    errors={str(d):float(mx.max(mx.abs(full[d]["logits"][:,:cut]-prefix[d]["logits"]))) for d in (1,4)}
    expected=np.asarray(model(ids,lengths,(4,))[4]["logits"])
    deterministic=np.array_equal(expected,np.asarray(model(ids,lengths,(4,))[4]["logits"]))
    path=ARTIFACTS/"qwen_validation.safetensors"; path.parent.mkdir(parents=True,exist_ok=True)
    mx.save_safetensors(str(path),dict(tree_flatten(model.trainable_parameters())))
    restored=build(base,tokenizer,config,"qwen_recurrent",11); restored.load_weights(str(path),strict=False)
    exact=np.array_equal(expected,np.asarray(restored(ids,lengths,(4,))[4]["logits"]))
    assert parity<.001 and max(errors.values())<.002 and deterministic and exact
    assert all(np.isfinite(list(norms.values()))) and sum(v for k,v in norms.items() if k.endswith("lora_b"))>0 and norms["update_logit"]>0
    record={"all_passed":True,"depth1_max_abs_error":parity,"causal_prefix_max_abs_error":errors,"gradient_norms":norms,"deterministic":deterministic,"checkpoint_restoration_exact":exact,"parameters":restored.parameter_counts(),"model":manifest,"environment":env,"prompt_example":prompt(rows[0]),"token_lengths":np.asarray(lengths).tolist(),"cache_semantics":"Full prefix recomputed; cache=None per layer per loop; RoPE positions unchanged"}
    write_json(RESULTS/"qwen_validation.json",record)
    # Frozen competence on validation before any task fitting.
    frozen=Thinker(base,label_ids,config["qwen"]["start"],config["qwen"]["count"],0)
    val=load_rows("validation")[:config["qwen"]["validation_count"]]
    tr=predict(frozen,tokenizer,val,(1,4))
    write_json(RESULTS/"qwen_frozen_validation.json",{str(d):correct_and_metrics(val,[v["logits"] for v in tr[d]])[1] for d in (1,4)})
    print(json.dumps({k:record[k] for k in ("all_passed","depth1_max_abs_error","causal_prefix_max_abs_error","token_lengths")}),flush=True)

def train(condition,seed,epochs=None):
    if (ROOT/"qwen_frozen_protocol.json").exists(): raise RuntimeError("Qwen final protocol already frozen")
    config=json.loads((ROOT/"config.json").read_text()); c=config["qwen"]; env=environment(config["memory_gib"])
    if not (RESULTS/"qwen_validation.json").exists() or not json.loads((RESULTS/"qwen_validation.json").read_text())["all_passed"]: raise RuntimeError("Run qwen-validate first")
    base,tokenizer,manifest=get_model(); model=build(base,tokenizer,config,condition,seed)
    train=load_rows("train")[:c["train_count"]]; val=load_rows("validation")[:c["validation_count"]]
    depth=1 if condition=="qwen_onepass" else c["depth"]
    optimizer=optim.AdamW(c["learning_rate"],weight_decay=.01)
    ck=ARTIFACTS/"checkpoints"/f"{condition}_seed{seed}"; ck.mkdir(parents=True,exist_ok=True)
    out=RESULTS/"training"/f"{condition}_seed{seed}.json"
    signature={"condition":condition,"seed":seed,"config_sha256":digest(ROOT/"config.json"),"dataset_sha256":digest(ROOT/"dataset_manifest.json"),"source_sha256":digest(__file__)}
    record={**signature,"created":now(),"epochs":[],"best_validation_loss":1e30,"parameters":model.parameter_counts(),"environment":env}
    if out.exists():
        record=json.loads(out.read_text())
        if any(record[k]!=v for k,v in signature.items()): raise RuntimeError("Resume signature changed")
        model.load_weights(str(ck/"last.safetensors"),strict=False)
        optimizer.state=tree_unflatten(list(mx.load(str(ck/"optimizer.safetensors")).items()))
    def loss_fn(m,ids,lengths,y): return set_loss(m(ids,lengths,(depth,))[depth]["logits"],y)
    vg=nn.value_and_grad(model,loss_fn)
    for epoch in range(len(record["epochs"]),epochs or c["epochs"]):
        t=time.perf_counter(); order=np.random.default_rng(seed*100000+epoch).permutation(len(train)); losses=[]; norms=[]
        for i in range(0,len(order),c["batch_size"]):
            rows=[train[j] for j in order[i:i+c["batch_size"]]]; ids,lengths=batch(tokenizer,rows); y=mx.array([r["labels"]["optimal"] for r in rows])
            loss,g=vg(model,ids,lengths,y); norm=mx.sqrt(sum(mx.sum(v*v) for _,v in tree_flatten(g)))
            g,_=optim.clip_grad_norm(g,1.); optimizer.update(model,g); mx.eval(model.trainable_parameters(),optimizer.state,loss,norm)
            losses.append(float(loss)); norms.append(float(norm))
            if (i//c["batch_size"])%60==0: print(f"{condition} seed {seed} epoch {epoch+1} batch {i//c['batch_size']} loss {float(loss):.3f}",flush=True)
        tr=predict(model,tokenizer,val,(depth,)); metric=correct_and_metrics(val,[v["logits"] for v in tr[depth]])[1]
        if metric["set_nll"]<record["best_validation_loss"]:
            record["best_validation_loss"]=metric["set_nll"]; record["best_epoch"]=epoch+1
            mx.save_safetensors(str(ck/"best.safetensors"),dict(tree_flatten(model.trainable_parameters())))
        mx.save_safetensors(str(ck/"last.safetensors"),dict(tree_flatten(model.trainable_parameters())))
        mx.save_safetensors(str(ck/"optimizer.safetensors"),dict(tree_flatten(optimizer.state)))
        record["epochs"].append({"epoch":epoch+1,"training_loss":float(np.mean(losses)),"validation":metric,"gradient_norm_mean":float(np.mean(norms)),"gradient_norm_max":max(norms),"seconds":time.perf_counter()-t})
        record.update({"completed_epochs":epoch+1,"checkpoint_sha256":digest(ck/"best.safetensors"),"peak_mlx_gib":mx.get_peak_memory()/1024**3,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,"updated":now()})
        write_json(out,record); print(f"{condition} seed {seed} epoch {epoch+1} val {metric['accuracy']:.3f}",flush=True)

def evaluate():
    config=json.loads((ROOT/"config.json").read_text()); c=config["qwen"]; environment(config["memory_gib"])
    paths=[ROOT/"config.json",ROOT/"plan.md",ROOT/"dataset_manifest.json",Path(__file__),Path("src/latent_loop/experiment05/model.py")]
    for name in ("qwen_onepass","qwen_recurrent"):
        for seed in c["seeds"]:
            path=ARTIFACTS/"checkpoints"/f"{name}_seed{seed}"/"best.safetensors"; paths.append(path)
            record=json.loads((RESULTS/"training"/f"{name}_seed{seed}.json").read_text())
            if record["completed_epochs"]!=c["epochs"]: raise RuntimeError("Incomplete Qwen fit")
    frozen=ROOT/"qwen_frozen_protocol.json"
    if not frozen.exists(): write_json(frozen,{"frozen":now(),"hashes":{str(p):digest(p) for p in paths},"primary":"qwen_recurrent depth4 versus qwen_onepass depth1; fixed before training","depth_comparisons":"1/2/4/8 secondary; no final-depth selection"})
    for p,h in json.loads(frozen.read_text())["hashes"].items():
        if digest(p)!=h: raise RuntimeError(f"Frozen Qwen artifact changed {p}")
    base,tokenizer,_=get_model(); summary={}
    # Fixed subset keeps final inference bounded; exactly same examples for both.
    for split in ("fresh_id","natural","structure","larger"):
        rows=load_rows(split,True)[:120]; result={}; correctness={}
        for condition in ("qwen_frozen","qwen_onepass","qwen_recurrent"):
            depths=(1,2,3,4,5,6,7,8) if condition=="qwen_recurrent" else (1,)
            seeds=c["seeds"] if condition!="qwen_frozen" else [0]; seed_metrics=[]; seed_correct=[]
            for seed in seeds:
                path=RESULTS/"trajectories"/f"{condition}_seed{seed}_{split}.jsonl.gz"
                if path.exists(): records=read_trace(path)
                else:
                    model=(Thinker(base,[tokenizer.encode(l,add_special_tokens=False)[0] for l in "ABCD"],c["start"],c["count"],0) if condition=="qwen_frozen" else build(base,tokenizer,config,condition,seed))
                    if condition!="qwen_frozen": model.load_weights(str(ARTIFACTS/"checkpoints"/f"{condition}_seed{seed}"/"best.safetensors"),strict=False)
                    trace=predict(model,tokenizer,rows,depths)
                    records=[{"id":r["id"],"depths":{str(d):trace[d][i] for d in depths}} for i,r in enumerate(rows)]; save_rows(path,records)
                metrics={}; correct={}
                for d in depths:
                    correct[str(d)],metrics[str(d)]=correct_and_metrics(rows,[r["depths"][str(d)]["logits"] for r in records])
                seed_metrics.append(metrics); seed_correct.append(correct)
            result[condition]={"mean":{str(d):{k:float(np.mean([v[str(d)][k] for v in seed_metrics])) for k in seed_metrics[0][str(d)]} for d in depths},"seed_metrics":seed_metrics}
            correctness[condition]=seed_correct
        r=correctness["qwen_recurrent"]; f=correctness["qwen_onepass"]
        result["paired_recurrent_vs_onepass"]=paired_interval(np.array([v["4"] for v in r],float)-np.array([v["1"] for v in f],float))
        result["depth4_minus1"]=paired_interval(np.array([v["4"] for v in r],float)-np.array([v["1"] for v in r],float))
        summary[split]=result; write_json(RESULTS/"qwen_summary.json",summary); print(f"Qwen evaluated {split}: "+str({k:v['mean'] for k,v in result.items() if 'mean' in v and isinstance(v['mean'],dict)}),flush=True)
    timing=[]; rows=load_rows("validation")[:8]
    for condition,depth in (("qwen_onepass",1),("qwen_recurrent",1),("qwen_recurrent",2),("qwen_recurrent",4),("qwen_recurrent",8)):
        for seed in c["seeds"]:
            model=build(base,tokenizer,config,condition,seed); model.load_weights(str(ARTIFACTS/"checkpoints"/f"{condition}_seed{seed}"/"best.safetensors"),strict=False)
            for size in (1,8):
                def once():
                    ids,lengths=batch(tokenizer,rows[:size]); tr=model(ids,lengths,(depth,)); mx.eval(tr); return np.asarray(mx.argmax(tr[depth]["logits"],axis=-1))
                for _ in range(2): once()
                times=[]
                for _ in range(3):
                    mx.synchronize(); t=time.perf_counter(); once(); mx.synchronize(); times.append(time.perf_counter()-t)
                timing.append({"condition":condition,"depth":depth,"seed":seed,"batch":size,"seconds":times,"per_item_ms":float(np.median(times)*1000/size)})
    write_json(RESULTS/"qwen_latency.json",{"procedure":"Full tokenization, full-prefix Qwen forward, four-label scoring, synchronized argmax host conversion; 2 warmups/3 repeats; first 1/8 validation examples; no checkpoint load","measurements":timing,"peak_mlx_gib":mx.get_peak_memory()/1024**3,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3})

if __name__=="__main__":
    import sys
    if sys.argv[1]=="validate": validation()
    elif sys.argv[1]=="evaluate": evaluate()
    elif sys.argv[1]=="train-all":
        config=json.loads((ROOT/"config.json").read_text())
        for condition in ("qwen_onepass","qwen_recurrent"):
            for seed in config["qwen"]["seeds"]: train(condition,seed)
