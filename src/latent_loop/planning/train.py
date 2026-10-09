"""Resumable tiny-grid fits, preserving every seed and unsuccessful condition."""
import json
import time
import resource
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten
from latent_loop.backbone import environment
from latent_loop.planning.data import ROOT,RESULTS,ARTIFACTS,arrays,load_rows,write_json,digest,now
from latent_loop.planning.grid_recurrence import GridPlanner,set_loss

def build(spec,seed):
    mx.random.seed(seed)
    return GridPlanner(spec["width"],spec["blocks"],spec["shared"])

def validate(model,x,y,depth,batch=128):
    losses=[]; correct=[]
    for i in range(0,len(x),batch):
        xx=mx.array(x[i:i+batch]); yy=mx.array(y[i:i+batch]); logits=model(xx,(depth,))[depth]["logits"]
        loss=set_loss(logits,yy); pred=mx.argmax(logits,axis=-1)
        ok=yy[mx.arange(len(xx)),pred]; mx.eval(loss,ok)
        losses.append(float(loss)*len(xx)); correct.extend(np.asarray(ok).tolist())
    return {"loss":sum(losses)/len(x),"accuracy":float(np.mean(correct))}

def train(condition,seed,epochs=None,pilot=False):
    config=json.loads((ROOT/"config.json").read_text()); spec=config["conditions"][condition]
    env=environment(config["memory_gib"])
    rows=load_rows("train"); x,y=arrays(rows); vx,vy=arrays(load_rows("validation"))
    if pilot: x=x[:600]; y=y[:600]; rows=rows[:600]
    model=build(spec,seed); optimizer=optim.AdamW(config["learning_rate"],weight_decay=config["weight_decay"])
    name=f"{'pilot_' if pilot else ''}{condition}_seed{seed}"
    ck=ARTIFACTS/"checkpoints"/name; ck.mkdir(parents=True,exist_ok=True)
    out=RESULTS/"training"/f"{name}.json"
    signature={"config_sha256":digest(ROOT/"config.json"),"data_sha256":digest(ROOT/"dataset_manifest.json"),"condition":condition,"seed":seed,"pilot":pilot}
    record={**signature,"environment":env,"parameters":model.parameter_count(),"epochs":[],"best_validation_loss":1e30,"created":now()}
    start_epoch=0
    if out.exists():
        record=json.loads(out.read_text())
        if any(record[k]!=v for k,v in signature.items()): raise RuntimeError("Resume signature differs")
        start_epoch=len(record["epochs"])
        model.load_weights(str(ck/"last.safetensors"))
        optimizer.state=tree_unflatten(list(mx.load(str(ck/"optimizer.safetensors")).items()))
    total_epochs=epochs or config["epochs"]
    def loss_fn(m,xx,yy,depths):
        trace=m(xx,depths)
        return sum(set_loss(trace[d]["logits"],yy) for d in depths)/len(depths)
    vg=nn.value_and_grad(model,loss_fn)
    for epoch in range(start_epoch,total_epochs):
        t=time.perf_counter(); rng=np.random.default_rng(seed*100000+epoch)
        pool=np.arange(len(x)); objective=spec["objective"]
        if objective=="curriculum":
            max_family=1 if epoch<5 else 3 if epoch<10 else 5
            pool=np.asarray([i for i,r in enumerate(rows) if r["family"]<=max_family])
            order=rng.choice(pool,len(x),replace=True)
        else: order=rng.permutation(pool)
        losses=[]; norms=[]; depth_counts={str(k):0 for k in (1,2,4,8)}
        for offset in range(0,len(order),config["batch_size"]):
            idx=order[offset:offset+config["batch_size"]]
            if objective=="variable": depths=(int(rng.choice([1,2,4,8])),)
            elif objective=="simultaneous": depths=(1,2,4,8)
            elif objective=="curriculum": depths=(int(rng.choice([1] if epoch<5 else [1,2] if epoch<10 else [1,2,4])),)
            else: depths=(spec["depth"],)
            loss,grads=vg(model,mx.array(x[idx]),mx.array(y[idx]),depths)
            norm=mx.sqrt(sum(mx.sum(g*g) for _,g in tree_flatten(grads)))
            grads,_=optim.clip_grad_norm(grads,config["clip_norm"])
            optimizer.update(model,grads)
            mx.eval(model.parameters(),optimizer.state,loss,norm)
            losses.append(float(loss)); norms.append(float(norm))
            for d in depths: depth_counts[str(d)]+=len(idx)
        val=validate(model,vx,vy,spec["depth"])
        elapsed=time.perf_counter()-t
        if not np.isfinite(losses).all() or not np.isfinite(norms).all(): raise RuntimeError("Nonfinite training")
        if val["loss"]<record["best_validation_loss"]:
            record["best_validation_loss"]=val["loss"]; record["best_epoch"]=epoch+1
            model.save_weights(str(ck/"best.safetensors"))
        model.save_weights(str(ck/"last.safetensors"))
        mx.save_safetensors(str(ck/"optimizer.safetensors"),dict(tree_flatten(optimizer.state)))
        record["epochs"].append({"epoch":epoch+1,"training_loss":float(np.mean(losses)),"validation":val,"gradient_norm_mean":float(np.mean(norms)),"gradient_norm_max":max(norms),"depth_examples":depth_counts,"seconds":elapsed})
        record.update({"completed_epochs":epoch+1,"checkpoint_sha256":digest(ck/"best.safetensors"),"peak_mlx_gib":mx.get_peak_memory()/1024**3,"peak_rss_gib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3,"updated":now()})
        write_json(out,record)
        print(f"{name} epoch {epoch+1}/{total_epochs} loss {np.mean(losses):.4f} val {val['accuracy']:.3f} time {elapsed:.2f}s",flush=True)
    return record
