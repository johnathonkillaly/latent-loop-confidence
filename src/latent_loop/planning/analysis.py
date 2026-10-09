"""Offline representation probes and interventions; never deployed policy inputs."""
import json
from pathlib import Path
import numpy as np
from latent_loop.environments.maze import Maze,distances
from latent_loop.planning.data import ROOT,RESULTS,load_rows,arrays,write_json,save_rows

def gradient_conflicts():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from latent_loop.backbone import environment
    from latent_loop.planning.train import build
    from latent_loop.planning.grid_recurrence import set_loss
    from latent_loop.planning.evaluate import get_grid
    config=json.loads((ROOT/"config.json").read_text()); environment(config["memory_gib"])
    x,y=arrays(load_rows("train")[:64]); x=mx.array(x); y=mx.array(y); results=[]
    for condition in ("rec_final4","rec_variable","rec_curriculum","rec_simultaneous"):
        for stage in ("initial","trained"):
            for seed in config["seeds"]:
                model=build(config["conditions"][condition],seed) if stage=="initial" else get_grid(condition,seed,config)
                gradients={}; norms={}
                for depth in (1,2,4,8):
                    loss,g=nn.value_and_grad(model,lambda m:set_loss(m(x,(depth,))[depth]["logits"],y))(model)
                    flat=mx.concatenate([v.reshape(-1) for _,v in tree_flatten(g)]); mx.eval(flat)
                    gradients[depth]=np.asarray(flat); norms[str(depth)]=float(np.linalg.norm(gradients[depth]))
                cosine={f"{a}-{b}":float(np.dot(gradients[a],gradients[b])/max(1e-12,np.linalg.norm(gradients[a])*np.linalg.norm(gradients[b]))) for a,b in ((1,2),(1,4),(1,8),(2,4),(4,8))}
                results.append({"condition":condition,"stage":stage,"seed":seed,"gradient_norm":norms,"cosine":cosine})
    write_json(RESULTS/"gradient_conflicts.json",{"scope":"First 64 training examples, diagnostic only, all three seeds; raw per-depth gradients of set loss. No final labels used.","measurements":results})

def representations():
    import mlx.core as mx
    from latent_loop.backbone import environment
    from latent_loop.planning.evaluate import verify_freeze,get_grid,correct_and_metrics
    frozen=verify_freeze(); config=json.loads((ROOT/"config.json").read_text()); environment(config["memory_gib"])
    selected=frozen["selected_recurrent"]; records=[]
    train=load_rows("train")[:600]; val=load_rows("validation")[:120]
    final=load_rows("fresh_id",True); hard=load_rows("larger",True)
    def extract(model,rows,depth):
        x,_=arrays(rows); features=[]; targets=[]; reach=[]; mazeids=[]
        for i in range(0,len(x),64):
            state=model(mx.array(x[i:i+64]),(depth,),True)[depth]["states"]; mx.eval(state); state=np.asarray(state)
            for j,row in enumerate(rows[i:i+64]):
                maze=Maze.from_dict(row); d=distances(maze); free=np.asarray(maze.walls)==0
                features.append(state[j][free]); targets.append(d[free]); reach.append(d[free]>=0); mazeids.extend([row["id"]]*int(free.sum()))
        return np.concatenate(features),np.concatenate(targets),np.concatenate(reach),mazeids
    for seed in config["seeds"]:
        model=get_grid(selected,seed,config)
        for depth in (1,2,4,8):
            tx,ty,tr,_=extract(model,train,depth)
            mu=tx.mean(0); sd=np.maximum(tx.std(0),1e-5)
            def design(x): return np.column_stack([(x-mu)/sd,np.ones(len(x))])
            a=design(tx[tr]); b=ty[tr]; penalty=np.eye(a.shape[1]); penalty[-1,-1]=0
            weights=np.linalg.solve(a.T@a+10*penalty,a.T@b)
            reach_weights=np.linalg.solve(design(tx).T@design(tx)+10*penalty,design(tx).T@tr.astype(float))
            for split,rows in (("validation",val),("fresh_id",final),("larger",hard)):
                xx,yy,reachable,ids=extract(model,rows,depth); prediction=design(xx)@weights
                target=yy[reachable]; error=prediction[reachable]-target
                r2=1-float(np.sum(error**2)/max(1e-8,np.sum((target-target.mean())**2)))
                reachscore=design(xx)@reach_weights
                from scipy.stats import spearmanr
                from scipy.stats import rankdata
                # Reachability can be near-constant on connected generators.
                positive=int(reachable.sum()); negative=len(reachable)-positive
                auc=float((rankdata(reachscore)[reachable].sum()-positive*(positive+1)/2)/(positive*negative)) if positive and negative else None
                records.append({"seed":seed,"depth":depth,"split":split,"distance_r2":r2,"distance_mae":float(np.mean(abs(error))),"distance_spearman":float(spearmanr(target,prediction[reachable]).statistic),"reachability_auroc":auc,"cells":len(xx),"independent_mazes":len(rows)})
    # Label-free input intervention: remove the goal marker, without changing walls.
    interventions=[]
    from latent_loop.planning.evaluate import predict_grid
    for seed in config["seeds"]:
        model=get_grid(selected,seed,config)
        for split,rows in (("fresh_id",final),("larger",hard)):
            x,_=arrays(rows); removed=x.copy(); removed[...,3]=0
            for depth in (1,2,4,8):
                base=model(mx.array(x),(depth,))[depth]["logits"]; modified=model(mx.array(removed),(depth,))[depth]["logits"]; mx.eval(base,modified)
                a=np.asarray(base); b=np.asarray(modified)
                ok,metric=correct_and_metrics(rows,a); _,other=correct_and_metrics(rows,b)
                interventions.append({"seed":seed,"split":split,"depth":depth,"accuracy":metric["accuracy"],"goal_removed_accuracy":other["accuracy"],"decision_change_fraction":float(np.mean(a.argmax(-1)!=b.argmax(-1))),"max_abs_logit_change":float(np.max(abs(a-b)))})
    write_json(RESULTS/"representation_analysis.json",{"selected_condition":selected,"probe":"Ridge distance regression and linear reachability ranking; first 600 training layouts, fixed L2=10, normalized training features. Cells correlated within maze; these are descriptive metrics, no cell-level confidence intervals. No probes used by policies.","limitations":"Decodability and goal ablation are not evidence of a learned BFS algorithm. Goal removal is out-of-distribution; distance probes can exploit local topology and coordinates implicit in boundaries.","probes":records,"goal_ablation":interventions})

if __name__=="__main__":
    import sys
    if sys.argv[1]=="gradients": gradient_conflicts()
    elif sys.argv[1]=="representations": representations()
