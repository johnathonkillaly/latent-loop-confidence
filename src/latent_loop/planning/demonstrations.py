"""Post-evaluation descriptive cases and symmetry diagnostics, no model selection."""
import json
import numpy as np
from scipy.special import softmax
from latent_loop.environments.maze import Maze,labels,distances
from latent_loop.planning.data import ROOT,RESULTS,load_rows,write_json,save_rows
from latent_loop.planning.evaluate import verify_freeze,get_grid,read_trace,correct_and_metrics,paired_interval

def run():
    import mlx.core as mx
    from latent_loop.backbone import environment
    frozen=verify_freeze(); config=json.loads((ROOT/"config.json").read_text()); environment(config["memory_gib"])
    condition=frozen["selected_recurrent"]; seed=11; model=get_grid(condition,seed,config)
    cases=[]; strata={}; symmetries=[]
    for split in ("fresh_id","larger","long_paths"):
        rows=load_rows(split,True)
        traces=[read_trace(RESULTS/"trajectories"/f"{condition}_seed{s}_{split}.jsonl.gz") for s in config["seeds"]]
        correct={d:np.array([[r["labels"]["optimal"][int(np.argmax(t[i]["depths"][str(d)]["logits"]))] for i,r in enumerate(rows)] for t in traces]) for d in (1,2,4,8)}
        # Overlapping strata are descriptive, paired at independent maze level.
        masks={"single_legal":np.array([sum(r["labels"]["legal"])==1 for r in rows]),"branching":np.array([sum(r["labels"]["legal"])>=2 for r in rows]),"path_1_3":np.array([r["labels"]["distance"]<=3 for r in rows]),"path_4_7":np.array([4<=r["labels"]["distance"]<=7 for r in rows]),"path_8_plus":np.array([r["labels"]["distance"]>=8 for r in rows]),"branching_detour":np.array([sum(r["labels"]["legal"])>=2 and r["labels"]["detour_ratio"]>1 for r in rows]),"tied_optimal":np.array([sum(r["labels"]["optimal"])>1 for r in rows])}
        strata[split]={name:{"mazes":int(mask.sum()),"accuracy":{str(d):float(v[:,mask].mean()) for d,v in correct.items()},"depth4_minus1":paired_interval(correct[4][:,mask].astype(float)-correct[1][:,mask].astype(float))} for name,mask in masks.items() if mask.any()}
        selectors={"correction_1_to_4":(~correct[1][0])&correct[4][0],"degradation_4_to_8":correct[4][0]&~correct[8][0],"failure_at_4":~correct[4][0]}
        for kind,mask in selectors.items():
            idx=np.flatnonzero(mask)
            if not len(idx): continue
            # Deterministic first branching example, otherwise first available.
            i=next((int(j) for j in idx if sum(rows[j]["labels"]["legal"])>=2),int(idx[0])); row=rows[i]; maze=Maze.from_dict(row)
            x=maze.observation()[None]; removed=x.copy(); removed[...,3]=0
            trace=model(mx.array(x),tuple(range(1,9)),True); ablated=model(mx.array(removed),tuple(range(1,9)),True); mx.eval(trace,ablated)
            states={d:np.asarray(trace[d]["states"])[0] for d in trace}
            cases.append({"split":split,"kind":kind,"seed":seed,"condition":condition,"example":row,"bfs_distance":distances(maze).tolist(),"depths":{str(d):{"probabilities":softmax(np.asarray(trace[d]["logits"])[0]).tolist(),"prediction":int(np.argmax(np.asarray(trace[d]["logits"])[0])),"state_rms_map":np.sqrt(np.mean(states[d]**2,axis=-1)).tolist(),"state_change_map":np.sqrt(np.mean((states[d]-(states[d-1] if d>1 else 0))**2,axis=-1)).tolist(),"goal_effect_map":np.sqrt(np.mean((states[d]-np.asarray(ablated[d]["states"])[0])**2,axis=-1)).tolist()} for d in trace}})
    save_rows(RESULTS/"demonstrations.jsonl.gz",cases)
    # Explicit rotations/reflections of FINAL layouts are correlated robustness
    # diagnostics, never an expanded independent test sample.
    rows=load_rows("fresh_id",True)[:120]
    for seed in config["seeds"]:
        model=get_grid(condition,seed,config)
        all_correct=[]
        for reflect in (False,True):
            for rotation in range(4):
                transformed=[]
                for row in rows:
                    maze=Maze.from_dict(row); board=np.array(maze.walls); board[maze.agent]=2; board[maze.goal]=3
                    board=np.rot90(np.fliplr(board) if reflect else board,rotation)
                    m=Maze(tuple(map(tuple,(board==1).astype(int))),tuple(np.argwhere(board==2)[0]),tuple(np.argwhere(board==3)[0]))
                    transformed.append({**m.to_dict(),"labels":labels(m)})
                from latent_loop.planning.data import arrays
                x,_=arrays(transformed); logits=model(mx.array(x),(4,))[4]["logits"]; mx.eval(logits)
                ok,metric=correct_and_metrics(transformed,np.asarray(logits)); all_correct.append(ok)
                symmetries.append({"seed":seed,"reflection":reflect,"rotation_quarters":rotation,"independent_original_mazes":120,**metric})
        # Eight variants remain one cluster per original board.
    write_json(RESULTS/"descriptive_analysis.json",{"status":"Post-evaluation descriptive analysis; no training/checkpoint/depth choices changed","strata":strata,"symmetry":symmetries,"demonstration_count":len(cases),"symmetry_uncertainty":"No independent-variant intervals: eight views of the same 120 layouts are correlated"})

if __name__=="__main__": run()
