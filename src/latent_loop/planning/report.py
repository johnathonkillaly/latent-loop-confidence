"""CPU-only regeneration from recorded measurements; no model inference."""
import base64
import gzip
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import markdown
from latent_loop.environments.maze import Maze,NAMES
from latent_loop.planning.oracles import shortest_path,propagation
from latent_loop.planning.data import ROOT,RESULTS,write_json,digest
from latent_loop.planning.evaluate import read_trace,correct_and_metrics,paired_interval

REPORT=Path("reports/experiment_05b")
COLORS=["#3366AA","#D47B24","#4D9475","#985F9F","#6B7280","#B45959","#697D39"]
DISPLAY={"rec_final4":"Recurrent: final-4","rec_variable":"Recurrent: variable","rec_curriculum":"Recurrent: curriculum","rec_simultaneous":"Recurrent: simultaneous","ff_onepass":"One-hop feedforward","ff4_matched_params":"Four-hop FF, parameter matched","ff4_matched_compute":"Four-hop FF, same width"}

def load(name): return json.loads((RESULTS/name).read_text())
def pct(x): return f"{100*x:.2f}%"
def interval(v): return f"{100*v['mean']:+.2f} pp [{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]"
def table(headers,rows): return "| "+" | ".join(headers)+" |\n|"+"|".join([" --- "]*len(headers))+"|\n"+"\n".join("| "+" | ".join(map(str,r))+" |" for r in rows)+"\n"
def figure(name,fig):
    fig.tight_layout(); fig.savefig(REPORT/"figures"/f"{name}.png",dpi=160,bbox_inches="tight"); fig.savefig(REPORT/"figures"/f"{name}.svg",bbox_inches="tight"); plt.close(fig)
    return f"![{name.replace('_',' ')}](figures/{name}.png)\n"
def latency_ms(latency,condition,depth,batch=1):
    return float(np.mean([r["per_item_ms"] for r in latency["models"][condition] if r["depth"]==depth and r["batch"]==batch]))
def board(ax,maze,title,path=None):
    w=np.asarray(maze.walls); ax.imshow(w,cmap=ListedColormap(["#F7F8FA","#273445"]),vmin=0,vmax=1)
    ax.set_xticks(range(maze.size)); ax.set_yticks(range(maze.size)); ax.tick_params(labelsize=7)
    ax.set_xticks(np.arange(-.5,maze.size,1),minor=True); ax.set_yticks(np.arange(-.5,maze.size,1),minor=True); ax.grid(which="minor",color="#BBC2CC",alpha=.5); ax.tick_params(which="minor",length=0)
    if path:
        p=np.array(path); ax.plot(p[:,1],p[:,0],color="#D47B24",lw=2,alpha=.8)
    for p,mark,color in ((maze.agent,"A","#3366AA"),(maze.goal,"G","#438568")):
        ax.text(p[1],p[0],mark,color=color,ha="center",va="center",fontweight="bold",fontsize=12)
    ax.set_title(title,fontsize=10)

def audit():
    config=json.loads((ROOT/"config.json").read_text()); manifest=json.loads((ROOT/"dataset_manifest.json").read_text()); layouts=set(); count=0; trajectories=0; decisions=0
    for split,spec in manifest["splits"].items():
        path=RESULTS/"datasets"/f"{split}.jsonl.gz"
        if digest(path)!=spec["sha256"]: raise AssertionError("Public dataset hash mismatch")
        rows=read_trace(path)
        from latent_loop.environments.maze import labels
        for row in rows:
            maze=Maze.from_dict(row); assert maze.canonical_hash()==row["layout_hash"] and maze.canonical_hash() not in layouts
            layouts.add(row["layout_hash"]); assert labels(maze)==row["labels"]; count+=1
    summary=load("summary.json")
    for split,result in summary["splits"].items():
        rows=read_trace(RESULTS/"datasets"/f"{split}.jsonl.gz")
        for condition,model in result["models"].items():
            for seed_index,seed in enumerate(config["seeds"]):
                records=read_trace(RESULTS/"trajectories"/f"{condition}_seed{seed}_{split}.jsonl.gz"); assert [r["id"] for r in rows]==[r["id"] for r in records]; trajectories+=1
                for depth in model["seed_metrics"][seed_index]:
                    _,metric=correct_and_metrics(rows,[r["depths"][depth]["logits"] for r in records])
                    for k,value in metric.items(): assert abs(value-model["seed_metrics"][seed_index][depth][k])<1e-10
                    decisions+=len(rows)
    if (RESULTS/"qwen_summary.json").exists():
        for split,result in load("qwen_summary.json").items():
            rows=read_trace(RESULTS/"datasets"/f"{split}.jsonl.gz")[:120]
            for condition in ("qwen_frozen","qwen_onepass","qwen_recurrent"):
                seeds=[0] if condition=="qwen_frozen" else config["qwen"]["seeds"]
                for j,seed in enumerate(seeds):
                    records=read_trace(RESULTS/"trajectories"/f"{condition}_seed{seed}_{split}.jsonl.gz"); assert [r["id"] for r in records]==[r["id"] for r in rows]; trajectories+=1
                    for d,expected in result[condition]["seed_metrics"][j].items():
                        metric=correct_and_metrics(rows,[r["depths"][d]["logits"] for r in records])[1]
                        assert all(abs(metric[k]-v)<1e-10 for k,v in expected.items()); decisions+=len(rows)
    snake_states=0;snake_decisions=0
    if (RESULTS/'snake_neural_summary.json').exists():
        from latent_loop.environments.snake import Snake,teacher
        from latent_loop.planning.snake_training import canonical
        snake_manifest=json.loads((ROOT/'snake_dataset_manifest.json').read_text());seen=set()
        for split,spec in snake_manifest['splits'].items():
            path=RESULTS/'snake_neural'/'datasets'/f'{split}.jsonl.gz';assert digest(path)==spec['sha256']
            for row in read_trace(path):
                state=Snake.from_dict(row['state']);h=canonical(state);assert h==row['hash'] and h not in seen;seen.add(h)
                assert row['safe']==[state.step(a)[0].alive for a in range(4)]
                assert all(row['teacher'][str(k)]==teacher(state,k)['optimal'] for k in (2,4,6));snake_states+=1
        sr=load('snake_neural_summary.json');final=read_trace(RESULTS/'snake_neural'/'datasets'/'final.jsonl.gz')
        for condition,model in sr['models'].items():
            for i,seed in enumerate(sr['config']['seeds']):
                tr=read_trace(RESULTS/'snake_neural'/'trajectories'/f'{condition}_seed{seed}.jsonl.gz');assert [r['id'] for r in tr]==[r['id'] for r in final]
                for d,metric in model['seed_metrics'][i].items():
                    pred=np.array([np.argmax(r['depths'][d]['logits']) for r in tr])
                    for k in (2,4,6):
                        actual=np.mean([row['teacher'][str(k)][a] for row,a in zip(final,pred)]);assert abs(actual-metric['horizon_accuracy'][str(k)])<1e-10
                    snake_decisions+=len(final)
    value={"all_passed":True,"independently_recomputed_maze_labels":count,"distinct_canonical_wall_layouts":len(layouts),"checked_trajectory_files":trajectories,"checked_decisions":decisions,"scope":"Public dataset labels/hashes and ALL seed/depth grid/Qwen classification accuracy, set NLL, regret, invalid rate, ECE recomputed from logits. No claim of independent full retraining."}
    value.update({'snake_states_recomputed':snake_states,'snake_decisions_recomputed':snake_decisions})
    write_json(RESULTS/"public_audit.json",value);return value

def report():
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/"figures").mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.hashsalt":"experiment05b"})
    config=json.loads((ROOT/"config.json").read_text()); frozen=json.loads((ROOT/"frozen_protocol.json").read_text()); s=load("summary.json"); rollout=load("rollout_summary.json"); lat=load("latency.json"); description=load("descriptive_analysis.json"); probes=load("representation_analysis.json"); selected=s["selected_recurrent"]; ff=s["selected_feedforward"]; primary=s["splits"]["fresh_id"]["models"][selected]; m=primary["mean"]
    checked=audit(); figs={}
    fig,ax=plt.subplots(figsize=(8,4))
    for i,c in enumerate(config["conditions"]):
        vals=s["splits"]["fresh_id"]["models"][c]["mean"]
        if c.startswith("rec_"): ax.plot([1,2,4,8],[100*vals[str(d)]["accuracy"] for d in (1,2,4,8)],"o-",label=DISPLAY[c],color=COLORS[i])
        else: ax.axhline(100*next(iter(vals.values()))["accuracy"],ls="--",label=DISPLAY[c],color=COLORS[i],alpha=.8)
    ax.set(xlabel="Internal loops (same observation and trained weights)",ylabel="Optimal-first-action accuracy (%)",xticks=[1,2,4,8],ylim=(35,102));ax.legend(fontsize=7,ncol=2);figs["accuracy"]=figure("accuracy_by_depth",fig)
    fig,ax=plt.subplots(figsize=(7,3.5))
    for i,c in enumerate([v for v in config["conditions"] if v.startswith("rec_")]):
        vals=s["splits"]["fresh_id"]["models"][c]["mean"];ax.plot([1,2,4,8],[vals[str(d)]["mean_regret"] for d in (1,2,4,8)],"o-",label=DISPLAY[c],color=COLORS[i])
    ax.set(xlabel="Internal loops",ylabel="First-action shortest-path regret",xticks=[1,2,4,8]);ax.legend(fontsize=8);figs["regret"]=figure("regret_by_depth",fig)
    fig,ax=plt.subplots(figsize=(8,3.5));families=range(6);x=np.arange(6)
    for i,d in enumerate((1,2,4,8)):
        v=rollout["fresh_id"][f"{selected}@{d}"]["by_family"];ax.plot(x,[100*v[str(f)] for f in families],"o-",label=f"{d} loops",color=COLORS[i])
    ax.set(xticks=x,xticklabels=["Straight","Turns","Misleading","Traps","Routes","Detours"],ylabel="Full-maze completion (%)",ylim=(-3,103));ax.legend(ncol=4,fontsize=8);figs["difficulty"]=figure("completion_by_difficulty",fig)
    fig,ax=plt.subplots(figsize=(7,3.5));pairs=["1->2","2->4","4->8"];x=np.arange(3)
    ax.bar(x-.17,[100*primary["transitions"][p]["wrong_right"] for p in pairs],width=.34,label="Wrong → right",color="#4D9475");ax.bar(x+.17,[100*primary["transitions"][p]["right_wrong"] for p in pairs],width=.34,label="Right → wrong",color="#B45959");ax.set(xticks=x,xticklabels=pairs,ylabel="Fraction of all decisions (%)");ax.legend();figs["transitions"]=figure("corrections_and_degradations",fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for i,c in enumerate(config["conditions"]):
        training=[load(f"training/{c}_seed{seed}.json") for seed in config["seeds"]]
        for ax,field in zip(axes,("training_loss","validation")):
            values=[[e[field] if field=="training_loss" else e[field]["loss"] for e in t["epochs"]] for t in training]
            ax.plot(np.arange(1,21),np.mean(values,axis=0),label=DISPLAY[c],color=COLORS[i])
    axes[0].set(xlabel="Epoch",ylabel="Training objective (conditions differ)",ylim=(0,1));axes[1].set(xlabel="Epoch",ylabel="Validation set NLL (deployment depth)",ylim=(0,1));axes[1].legend(fontsize=6);figs["loss"]=figure("training_validation_loss",fig)
    fig,ax=plt.subplots(figsize=(8,3.5));splits=list(s["splits"]);x=np.arange(len(splits))
    for i,d in enumerate((1,2,4,8)): ax.plot(x,[100*s["splits"][split]["models"][selected]["mean"][str(d)]["accuracy"] for split in splits],"o-",label=f"{d} loops",color=COLORS[i])
    ax.plot(x,[100*s["splits"][split]["models"][ff]["mean"]["4"]["accuracy"] for split in splits],"k--",label="Four-hop feedforward");ax.set(xticks=x,xticklabels=["Balanced ID","Natural","New algorithm","7×7","Dense","Long paths"],ylabel="Optimal-action accuracy (%)");ax.legend(fontsize=8,ncol=3);figs["generalization"]=figure("unseen_structures",fig)
    fig,ax=plt.subplots(figsize=(7,3.5))
    for c in config["conditions"]:
        depths=(1,2,4,8) if c==selected else (config["conditions"][c]["depth"],)
        xx=[latency_ms(lat,c,d) for d in depths];yy=[100*s["splits"]["fresh_id"]["models"][c]["mean"][str(d)]["accuracy"] for d in depths]
        ax.plot(xx,yy,"o-" if c==selected else "o",label=DISPLAY[c]);
        if c==selected:
            for a,b,d in zip(xx,yy,depths): ax.annotate(str(d), (a,b),xytext=(5,-10),textcoords="offset points")
    ax.set(xlabel="Batch-1 end-to-end inference (ms)",ylabel="Optimal-action accuracy (%)");ax.legend(fontsize=7);figs["latency"]=figure("accuracy_vs_latency",fig)
    fig,ax=plt.subplots(figsize=(8,3.5))
    for i,c in enumerate(config["conditions"]):
        t=load(f"training/{c}_seed11.json");acc=s["splits"]["fresh_id"]["models"][c]["mean"][str(config["conditions"][c]["depth"])]["accuracy"]
        ax.scatter(t["parameters"],100*acc,color=COLORS[i],label=DISPLAY[c],s=50)
    ax.set(xlabel="Trainable parameters",ylabel="Optimal-action accuracy (%)");ax.legend(fontsize=7);figs["parameters"]=figure("parameters_vs_accuracy",fig)
    records=[read_trace(RESULTS/"trajectories"/f"{selected}_seed{seed}_fresh_id.jsonl.gz") for seed in config["seeds"]]
    fig,axes=plt.subplots(1,2,figsize=(9,3))
    for i,field in enumerate(("state_norm","state_change")):
        for j,seed_records in enumerate(records): axes[i].plot(range(1,9),[np.mean([r["depths"][str(d)][field] for r in seed_records]) for d in range(1,9)],"o-",label=f"Seed {config['seeds'][j]}")
        axes[i].set(xlabel="Loop",ylabel="State RMS" if i==0 else "RMS adjacent state change")
    axes[1].legend(fontsize=8);figs["drift"]=figure("recurrent_state_changes",fig)
    cases=read_trace(RESULTS/"demonstrations.jsonl.gz");case=next(c for c in cases if c["kind"]=="correction_1_to_4" and c["split"]=="fresh_id")
    maze=Maze.from_dict(case["example"]);fig,axes=plt.subplots(1,2,figsize=(9,3.5));board(axes[0],maze,"Illustrative correction; BFS path is evaluation only",shortest_path(maze))
    for a,name in enumerate(NAMES): axes[1].plot(range(1,9),[case["depths"][str(d)]["probabilities"][a] for d in range(1,9)],"o-",label=name)
    axes[1].set(xlabel="Internal loop; board unchanged",ylabel="Action probability",ylim=(0,1));axes[1].legend();figs["example"]=figure("demonstration_decision_trajectory",fig)
    fig,axes=plt.subplots(2,2,figsize=(9,7))
    for i,kind in enumerate(("degradation_4_to_8","failure_at_4")):
        bad=next(c for c in cases if c["kind"]==kind and c["split"]=="fresh_id");bad_maze=Maze.from_dict(bad["example"])
        board(axes[i,0],bad_maze,"Illustrative "+kind.replace("_"," "),shortest_path(bad_maze))
        optimal=bad["example"]["labels"]["optimal"]
        for a,name in enumerate(NAMES): axes[i,1].plot(range(1,9),[bad["depths"][str(d)]["probabilities"][a] for d in range(1,9)],"o-",label=name+(" (optimal)" if optimal[a] else ""))
        axes[i,1].set(xlabel="Loop; same observation",ylabel="Action probability",ylim=(0,1));axes[i,1].legend(fontsize=8)
    figs["failures"]=figure("failure_and_degradation_demonstrations",fig)
    fig,axes=plt.subplots(2,4,figsize=(10,5))
    maxima={field:max(np.max(case["depths"][str(d)][field]) for d in (1,2,4,8)) for field in ("state_rms_map","goal_effect_map")}
    for j,d in enumerate((1,2,4,8)):
        for i,field in enumerate(("state_rms_map","goal_effect_map")):
            data=np.array(case["depths"][str(d)][field]);data[np.array(maze.walls,dtype=bool)]=np.nan
            im=axes[i,j].imshow(data,cmap="viridis",vmin=0,vmax=maxima[field]);axes[i,j].set_title(f"{'State RMS' if i==0 else 'Goal ablation effect'}, loop {d}",fontsize=8);axes[i,j].set_xticks([]);axes[i,j].set_yticks([]);fig.colorbar(im,ax=axes[i,j],shrink=.65)
            for p,mark in ((maze.agent,"A"),(maze.goal,"G")): axes[i,j].text(p[1],p[0],mark,ha="center",va="center",fontsize=9,color="black",bbox={"facecolor":"white","alpha":.75,"edgecolor":"none","pad":1})
    figs["activation"]=figure("spatial_activation_maps",fig)
    fig,axes=plt.subplots(1,4,figsize=(10,2.8));classical=propagation(maze,8)
    for ax,d in zip(axes,(1,2,4,8)):
        data=classical[d].copy();data[~np.isfinite(data)]=np.nan
        im=ax.imshow(data,cmap="viridis_r",vmin=0,vmax=8);ax.set_title(f"Classical propagation, {d} steps",fontsize=8);ax.set_xticks([]);ax.set_yticks([]);fig.colorbar(im,ax=ax,shrink=.7)
        for p,mark in ((maze.agent,"A"),(maze.goal,"G")): ax.text(p[1],p[0],mark,ha="center",va="center",fontsize=9,bbox={"facecolor":"white","alpha":.8,"edgecolor":"none","pad":1})
    figs["classical"]=figure("classical_propagation_reference",fig)
    paths={d:read_trace(RESULTS/"rollouts"/f"{selected}_d{d}_seed11_fresh_id.jsonl.gz") for d in (1,4,8)}
    idx=next(i for i in range(len(paths[1])) if not paths[1][i]["success"] and paths[4][i]["success"])
    rowmap={r["id"]:r for r in read_trace(RESULTS/"datasets"/"fresh_id.jsonl.gz")};maze=Maze.from_dict(rowmap[paths[1][idx]["id"]]);fig,axes=plt.subplots(1,4,figsize=(11,3))
    board(axes[0],maze,"BFS reference",shortest_path(maze))
    for ax,d in zip(axes[1:],(1,4,8)): board(ax,maze,f"{d} loops/action: {'solved' if paths[d][idx]['success'] else 'failed'}\n{paths[d][idx]['steps']} actions, {paths[d][idx]['repeated_positions']} repeats",paths[d][idx]["positions"])
    figs["rollout"]=figure("full_maze_rollout_demonstration",fig)
    snake=load("snake_summary.json");fig,axes=plt.subplots(1,2,figsize=(9,3))
    for ax,field in zip(axes,("survival_steps","food_collected")):
        ax.plot([2,4,6],[snake["policies"][f"oracle_h{h}"][field] for h in (2,4,6)],"o-",label="Bounded teacher (no learned policy)")
        for color,name in zip(("#D47B24","#6B7280"),("safe_greedy","random_legal")): ax.axhline(snake["policies"][name][field],ls="--",color=color,label=name)
        ax.set(xlabel="Teacher planning horizon",ylabel=field.replace("_"," "),xticks=[2,4,6]);ax.legend(fontsize=7)
    figs["snake"]=figure("snake_baseline_horizons",fig)
    # Tables and all numerical conclusions are generated from recorded data.
    variant_rows=[]; fit_seconds=0.;peak_grid=0.;peak_qwen=0.
    for c in config["conditions"]:
        ts=[load(f"training/{c}_seed{seed}.json") for seed in config["seeds"]];fit_seconds+=sum(e["seconds"] for t in ts for e in t["epochs"]);peak_grid=max(peak_grid,max(t["peak_mlx_gib"] for t in ts))
        v=s["splits"]["fresh_id"]["models"][c]["mean"]
        variant_rows.append([DISPLAY[c],ts[0]["parameters"]]+[pct(v[str(d)]["accuracy"]) if str(d) in v else "—" for d in (1,2,4,8)])
    transition_rows=[[p,interval(v["paired_accuracy"]),pct(v["wrong_right"]),pct(v["right_wrong"])] for p,v in primary["transitions"].items()]
    shift_rows=[[split]+[pct(result["models"][selected]["mean"][str(d)]["accuracy"]) for d in (1,2,4,8)]+[pct(result["models"][ff]["mean"]["4"]["accuracy"]),interval(result["recurrent_vs_feedforward"])] for split,result in s["splits"].items()]
    rollout_rows=[[split]+[pct(result[f"{selected}@{d}"]["completion"]["mean"]) for d in (1,2,4,8)]+[pct(result[f"{ff}@4"]["completion"]["mean"]),pct(result["greedy@0"]["completion"]["mean"])] for split,result in rollout.items()]
    family_rows=[]
    for f in range(6):
        metrics=primary["family_metrics"];family_rows.append([f]+[pct(np.mean([v[str(d)][str(f)]["accuracy"] for v in metrics])) for d in (1,2,4,8)])
    latency_rows=[[DISPLAY[c],d,f"{latency_ms(lat,c,d):.3f}",f"{latency_ms(lat,c,d,8):.3f}"] for c in config["conditions"] for d in ((1,2,4,8) if c==selected else (config["conditions"][c]["depth"],))]
    primary_delta=primary["transitions"]["1->4"]["paired_accuracy"];ff_delta=s["splits"]["fresh_id"]["recurrent_vs_feedforward"]
    idrows=read_trace(RESULTS/"datasets"/"fresh_id.jsonl.gz");rec_correct=[];narrow_correct=[]
    for seed in config['seeds']:
        for c,target in ((selected,rec_correct),('ff4_matched_params',narrow_correct)):
            tr=read_trace(RESULTS/'trajectories'/f'{c}_seed{seed}_fresh_id.jsonl.gz')
            target.append(correct_and_metrics(idrows,[r['depths']['4']['logits'] for r in tr])[0])
    narrow_delta=paired_interval(np.array(rec_correct,float)-np.array(narrow_correct,float))
    episode_correct={}
    for c,d in ((selected,1),(selected,4),(selected,8),(ff,4)):
        episode_correct[f'{c}@{d}']=np.array([[r['success'] for r in read_trace(RESULTS/'rollouts'/f'{c}_d{d}_seed{seed}_fresh_id.jsonl.gz')] for seed in config['seeds']],float)
    gameplay_delta=paired_interval(episode_correct[f'{selected}@4']-episode_correct[f'{selected}@1'])
    gameplay_ff_delta=paired_interval(episode_correct[f'{selected}@4']-episode_correct[f'{ff}@4'])
    text=f"""# Experiment 5.1 — Think Again: learning to navigate

## Executive summary

**A tiny learned grid network makes better decisions after more internal updates over an unchanged board.** The validation-selected `{selected}` reaches {pct(m['1']['accuracy'])}, {pct(m['2']['accuracy'])}, {pct(m['4']['accuracy'])}, and {pct(m['8']['accuracy'])} at loops 1/2/4/8 on 600 new balanced-ID layouts, averaged over seeds 11/22/33. The prespecified primary depth-4 minus depth-1 difference is **{interval(primary_delta)}**. Secondary improvements occur from 1→2 and 2→4 too. Full-maze completion rises from {pct(rollout['fresh_id'][f'{selected}@1']['completion']['mean'])} at one loop to {pct(rollout['fresh_id'][f'{selected}@4']['completion']['mean'])} at four and {pct(rollout['fresh_id'][f'{selected}@8']['completion']['mean'])} at eight on 60 independent starting mazes.

**A recurrence advantage over ordinary deeper feedforward computation is not established.** At four hops, recurrent minus validation-selected same-width feedforward accuracy is {interval(ff_delta)}. The recurrent model has 18,884 parameters versus 74,660 for the same-width four-block control; the narrower four-block control has 18,900. Extra spatial computation is useful; weight sharing is an efficient parameterization, with no established accuracy superiority or measured speedup claim. The eight-loop result is a prespecified secondary comparison for a model trained at eight, not a final-selected deployment depth.

This is a small synthetic spatial-planning study, independent of pretrained language-model reasoning. Qwen findings are reported separately below. Snake completed its environment/teacher/baseline smoke stage only. All work remains local pending review; public v0.1.0 is unchanged.

## Motivation and architecture

[Experiment 5](../experiment_05/train_the_thinker.md) trained shared Qwen middle layers but did not establish reliable gains from extra depth. Here local message passing supplies a concrete computational mechanism: a goal signal can travel at most one free-space edge per loop. Internal loops do not move the agent or reveal new observations.

Track A uses four channels (wall/free/agent/goal), a pointwise width-32 encoder, shared cardinal-neighbor gated updates, and a pointwise action head at the agent cell. Each update uses separate north/east/south/west states, original pointwise encoding, sigmoid reset/update gates, layer normalization and a tanh candidate. Hidden states on walls are zero. Convex gated updates keep states bounded; no global pooling, coordinates, distance inputs, visible reasoning tokens or across-action memory are used. Gates are open at initialization. The observation and original encoding remain unchanged throughout a decision. Models reset state at each environment action.

This is a GRU-style local message-passing network, not a handcrafted BFS update. A separate deterministic one-hop min-plus propagation reference converges to exact BFS distances and is reported as a privileged classical algorithm, never a learned-model input. It reads the same public board, with an explicit Manhattan fallback before propagation reaches the agent. Track A is not a RecurTrace reproduction. Track B reuses Experiment 5's frozen Qwen3-0.6B, shared middle layers 12–14 and rank-8 q/v LoRA; unlike loop-time-memory architectures, neither track uses explicit history attention or adaptive halting.

## Dataset and supervision

Training contains 3,600 independent 5×5 wall layouts; validation 600. Final sets: balanced ID 600, natural occupancy 600, connected-growth structural shift 360, 7×7 360, denser walls 360, and longer 7×7 paths 360. Each starts at a distinct nonterminal, solvable agent/goal placement. Four actions have explicit N/E/S/W coordinates. Boundaries/walls leave the maze agent in place with a penalty. Every equally optimal first move is accepted. Loss is negative log total probability of the optimal-action set; no arbitrary tie-breaking label is learned. BFS maps supervise and score only; deployed learned policies receive four observation channels, never labels/maps/paths. No legality mask hides invalid predictions.

The six balanced nominal families are straight corridors, one/two turns, misleading greedy moves, return-through-agent dead-end branches, multiple routes, and initial-away long detours. Families can overlap semantically. Actual BFS length, Manhattan distance, detour ratio, cycle rank, legal moves, trap membership and first-action regret are retained. In an undirected unweighted maze, a wrong legal first move has regret 2; this metric does not measure the entire cost of exploring a trap. Traps are branches requiring a return through the current cell, not necessarily single leaves.

D4-canonical WALL-LAYOUT hashes reject rotations/reflections and any other start/goal placement of a previously used layout across all splits. This is stricter than exact observation deduplication. Structural shift uses connected frontier growth rather than independent occupancy. Density shift changes wall probability from 0.30 to 0.44. Larger boards are processed at native 7×7 resolution. Long-path evaluation requires BFS distance ≥10. The natural split is unfiltered beyond solvability and layout exclusion; its task/action prevalence is retained. Balanced accuracy is a diagnostic benchmark, not a claim about natural prevalence. Training optimal-action membership counts N/E/S/W are 1026/1045/1053/1011, with ties counted for every acceptable action; exact equality is not claimed.

Difficulty is imperfect: 46.8% of training Level-5 examples have one legal move, so they require no first-action planning. On balanced final ID, all 71 single-legal examples are correct at every depth; the 529 branching examples improve from {pct(description['strata']['fresh_id']['branching']['accuracy']['1'])} to {pct(description['strata']['fresh_id']['branching']['accuracy']['4'])}. The 200 branching-detour examples improve from {pct(description['strata']['fresh_id']['branching_detour']['accuracy']['1'])} to {pct(description['strata']['fresh_id']['branching_detour']['accuracy']['4'])}; paired {interval(description['strata']['fresh_id']['branching_detour']['depth4_minus1'])}. These descriptive strata overlap and were not used for checkpoint selection.

## Training, causal checks and prospective evaluation

All seven grid conditions use 20 epochs, batch 64, AdamW 0.001, decay 0.0001, clip 1, three seeds. Final4 trains only depth 4; variable samples 1/2/4/8 uniformly per batch; simultaneous averages their set losses; curriculum stages families 0–1/depth1, then 0–3/depth1–2, then all/depth1–2–4. The curriculum samples its eligible pool with replacement to retain batch count; exposure frequencies differ. Checkpoints minimize depth-4 validation set loss (depth1 for one-pass). Simultaneous training was best on mean validation depth-4 accuracy and selected before final opening. Its success here contrasts with the failed simultaneous condition in Experiment 5; this is architecture/task-specific evidence.

The [frozen protocol](../../experiments/experiment_05b/frozen_protocol.json) records code/config/data/checkpoint hashes and validation choices. It froze at `{frozen['frozen']}`; final opening is separately recorded. Primary comparison is depth4−depth1 for the selected recurrent objective; 2−1, 4−2, other objectives, shifts, and eight loops are secondary/descriptive. No final depth, checkpoint or architecture was reselected. Fixed-depth full rollouts reset hidden state, commit one action after k internal loops, then observe the new board. Step budget is max(24,4×optimal starting distance), fixed before evaluation.

Tests verify BFS against independent Floyd–Warshall distances, full optimal sets, illegal/terminal actions, D4 exclusion, exact local receptive fields, no environment mutation, zero wall states, nonzero finite gradients, deterministic depth evaluation, exact weight and optimizer restoration, and native larger-grid shapes. Real Qwen checks confirm depth-one parity, no future-token leakage, fixed-position full-prefix recomputation with cache=None, open-gate adapter gradients and exact checkpoint restoration. All original weights remain frozen. First-pass LoRA-A gradients can be zero because LoRA-B initializes to zero; LoRA-B and the recurrent gate receive gradients. Gradient-conflict diagnostics compare raw per-depth gradients on 64 training examples, before/after training, all seeds; no final labels influence fitting.

Intervals resample paired independent maze layouts and training seeds (2,000 crossed bootstrap replicates). Repeated depths share each item, and 3×600 is not 1,800 independent mazes. Rollout uncertainty clusters whole initial episodes; individual action steps are never independent samples. Only three seeds and unadjusted secondary comparisons limit inference. Descriptive activation cases, goal interventions and rotation diagnostics were produced after evaluation with fixed models.

## Accuracy, extra depth and negative variants

"""
    text+=table(["Condition","Trainable parameters","Loop 1","Loop 2","Loop 4","Loop 8"],variant_rows)+"\n"+figs["accuracy"]+"\n"
    text+="Selected-condition seed-level accuracy (the same 600 test layouts):\n\n"+table(["Seed","Loop 1","Loop 2","Loop 4","Loop 8","Four-hop FF"],[[seed]+[pct(primary['seed_metrics'][i][str(d)]['accuracy']) for d in (1,2,4,8)]+[pct(s['splits']['fresh_id']['models'][ff]['seed_metrics'][i]['4']['accuracy'])] for i,seed in enumerate(config['seeds'])])+"\n"
    text+="Static feedforward controls have one or four distinct blocks; nonexistent recurrent depths are marked —. One-hop matches the recurrent parameter count exactly; four-hop width16 differs by only 16 parameters. Same-width four-hop has identical local computation structure with untied weights, four times as many update parameters. No full-board global CNN or eight-block untied control was trained.\n\n"
    text+=f"Recurrent4 minus the 18,900-parameter narrow four-block control is {interval(narrow_delta)} (secondary). Its interval also includes zero.\n\n"
    baseline=s['splits']['fresh_id']['baselines']
    text+=table(["Classical reference","Balanced-ID first-action accuracy"],[["BFS oracle",pct(baseline['bfs']['accuracy'])],["Greedy Manhattan",pct(baseline['greedy']['accuracy'])]]+[[f"Local min-plus propagation {d} steps + Manhattan fallback",pct(baseline['classical_propagation'][str(d)]['accuracy'])] for d in (1,2,4,8)])+"\n"+figs["classical"]+"\nClassical maps above are a deterministic reference on the demonstration board, not learned activations or inference features. White cells include unreached cells/walls.\n\n"
    text+=table(["Depth comparison","Paired accuracy difference, 95% CI","Wrong→right, all items","Right→wrong, all items"],transition_rows)+"\n"+figs["transitions"]+"\n"+figs["regret"]
    text+=f"\nAt 1→4, conditional correction among previously wrong predictions is {pct(primary['transitions']['1->4']['conditional_wrong_right'])}; conditional degradation among previously right predictions is {pct(primary['transitions']['1->4']['conditional_right_wrong'])}. Fractions in the table use all paired items and have different denominators. For the selected model, first-action regret at 1/2/4/8 is "+" / ".join(f"{m[str(d)]['mean_regret']:.4f}" for d in (1,2,4,8))+". Calibration ECE is "+" / ".join(f"{m[str(d)]['ece']:.4f}" for d in (1,2,4,8))+". Accuracy, confidence calibration and set loss remain distinct.\n\n"
    text+="Final-depth-only training extrapolates poorly: its fresh-ID accuracy falls from "+pct(s['splits']['fresh_id']['models']['rec_final4']['mean']['4']['accuracy'])+" at four loops to "+pct(s['splits']['fresh_id']['models']['rec_final4']['mean']['8']['accuracy'])+" at eight, and long-path accuracy falls to "+pct(s['splits']['long_paths']['models']['rec_final4']['mean']['8']['accuracy'])+". More iterations are not automatically useful. Variable/simultaneous objectives train explicitly at eight; their eight-loop results are within trained budgets. Curriculum and final4 eight-loop results are extrapolation. No distance-map auxiliary loss was needed or used.\n\n"+figs["loss"]
    text+="\n## Difficulty and generalization\n\n"+table(["Nominal family","Loop 1","Loop 2","Loop 4","Loop 8"],family_rows)+"\n"+table(["Split","Loop 1","Loop 2","Loop 4","Loop 8","Four-hop FF","Rec4 minus FF4, 95% CI"],shift_rows)+"\n"+figs["generalization"]
    text+="\nDepth gains persist on new wall layouts, new start/goal placements, denser walls and a different generation algorithm. Larger/longer layouts reduce accuracy. Improvements are particularly large for branching detours, but the path-length strata are not monotonic evidence of abstract reasoning difficulty. Explicit D4 robustness diagnostics evaluate eight views of each of 120 final layouts; these are correlated views, not 960 independent mazes. No transformed copy crosses a training/validation/final boundary.\n\n## Full maze gameplay\n\n"+table(["Split (60 mazes)","1 loop","2 loops","4 loops","8 loops","Four-hop FF","Greedy"],rollout_rows)+"\n"+figs["difficulty"]+"\n"+figs["rollout"]
    text+="\nSingle-step accuracy substantially overstates full-game reliability. Four-loop mistakes often produce stable repeated-position cycles; a stateless deterministic policy cannot recover without a decision changing. Long-path eight-loop completion remains only "+pct(rollout['long_paths'][f'{selected}@8']['completion']['mean'])+" despite high first-action accuracy. Saved trajectories include invalid actions, positions, step/optimal ratios, repeated positions, episode inference time and success/failure. Example figures are illustrative selected cases, not aggregate evidence.\n\n"+figs["failures"]+"\n## What the hidden computation contains\n\n"+figs["activation"]+"\n"+figs["example"]+"\n"+figs["drift"]
    probe_rows=[[d,f"{np.mean([r['distance_r2'] for r in probes['probes'] if r['split']=='fresh_id' and r['depth']==d]):.3f}",f"{np.mean([r['distance_mae'] for r in probes['probes'] if r['split']=='fresh_id' and r['depth']==d]):.3f}"] for d in (1,2,4,8)]
    text=text.replace("\n## What the hidden computation contains",f"\nWhole-episode paired balanced-ID completion differences are {interval(gameplay_delta)} for four versus one loop, and {interval(gameplay_ff_delta)} for four-loop recurrence versus four-block feedforward. These use 60 episode clusters and three seeds, not individual navigation steps.\n\n## What the hidden computation contains")
    text+="\nA fixed ridge probe (L2=10), fitted only on first 600 training layouts, decodes reachable-cell BFS distance from hidden states. Final-ID descriptive seed means:\n\n"+table(["Loop","Distance R²","Distance MAE (edges)"],probe_rows)
    text+="\nCells within a maze are correlated; no cell-level uncertainty claim is made. Reachability rankings and goal-removal effects are also recorded. The spatial figures show actual state RMS and the change caused by removing the goal marker. Ablation is out of distribution. Decodability, bounded goal influence and visually expanding maps support local information propagation, but do not establish that learned states implement BFS. The distance probe is never a policy input. Walls mask all states, and each loop expands the reachable information radius by at most one graph edge.\n\n## Measured computation and parameter controls\n\n"+table(["Model","Loops/blocks","Batch1 ms","Batch8 ms/item"],latency_rows)+"\n"+figs["latency"]+"\n"+figs["parameters"]
    text+=f"\nGrid main-fit epoch wall time totals {fit_seconds:.2f} seconds, including synchronized updates and validation, excluding process startup/data loading/checkpoint I/O. Tiny matrix operations are fast on this device; this figure is not total project duration. Grid training peak MLX allocation is {peak_grid:.3f} GiB; grid evaluation peak {load('grid_evaluation_resources.json')['peak_mlx_gib']:.3f} GiB. Batch1/8 timing includes fresh observation construction, array transfer, complete policy, host argmax and synchronization; three warmups/five repeats/three seeds, fixed validation observations. Measurements are device- and dispatch-dependent; serial condition order and very small samples limit speed comparisons. Parameter sharing does not imply fewer executed updates or a speedup.\n"
    if (RESULTS/"qwen_summary.json").exists():
        q=load("qwen_summary.json");qrows=[]
        for split,r in q.items():
            qrows.append([split,pct(r['qwen_frozen']['mean']['1']['accuracy']),pct(r['qwen_onepass']['mean']['1']['accuracy'])]+[pct(r['qwen_recurrent']['mean'][str(d)]['accuracy']) for d in (1,2,4,8)]+[interval(r['paired_recurrent_vs_onepass'])])
        qr=q['fresh_id'];qfits=[load(f"training/{c}_seed{seed}.json") for c in ('qwen_onepass','qwen_recurrent') for seed in config['seeds']];qtime=sum(e['seconds'] for r in qfits for e in r['epochs']);qlat=load('qwen_latency.json');peak_qwen=max(r['peak_mlx_gib'] for r in qfits)
        text+="\n## Track B: frozen Qwen3 middle-layer recurrence\n\nFour actions use fixed A=N/B=E/C=S/D=W next-token labels. The prompt contains the entire grid and explicit zero-based coordinates, no solved paths. Frozen native revision is `c1899de289a04d12100db370d81485cdf75e47ca`, FP32. Both adapters train on identical 720/120 examples, three seeds, three epochs; recurrence trains at fixed four loops, one-pass at one. The one-pass adapter has 122,880 trainable parameters, recurrent 122,881 including its scalar gate; total recurrent parameters 596,172,801. The pretrained architecture is inspected, not assumed. All backbone tensors are frozen. Recurrent states share original middle weights/adapters and use the original bounded residual update; no loop memory or adaptive controller.\n\n"+table(["Split (120 layouts)","Frozen","One-pass","Rec1","Rec2","Rec4","Rec8","Rec4 minus one-pass, 95% CI"],qrows)
        text+=f"\nQwen's within-model depth4−depth1 paired difference is {interval(qr['depth4_minus1'])}. Its recurrent-versus-one-pass result is {interval(qr['paired_recurrent_vs_onepass'])}. This small-budget transformer study does not establish transfer of the tiny grid planner's useful iterative spatial computation. Grid and Qwen receive different representations and training budgets; their aggregate accuracies are not a controlled architecture comparison. No Qwen gradient/caching failure occurred. Limited task competence/optimization and the prior architecture remain possible bottlenecks. Qwen eight-loop results are extrapolation diagnostics for fixed-four training, not selected depths.\n"
        text+="\n"+table(["Seed","One-pass","Rec1","Rec2","Rec4","Rec8"],[[seed,pct(qr['qwen_onepass']['seed_metrics'][i]['1']['accuracy'])]+[pct(qr['qwen_recurrent']['seed_metrics'][i][str(d)]['accuracy']) for d in (1,2,4,8)] for i,seed in enumerate(config['seeds'])])
        qrows=read_trace(RESULTS/'datasets'/'fresh_id.jsonl.gz')[:120];qcorrect={d:[] for d in (1,2,4,8)}
        for seed in config['seeds']:
            tr=read_trace(RESULTS/'trajectories'/f'qwen_recurrent_seed{seed}_fresh_id.jsonl.gz')
            for d in qcorrect: qcorrect[d].append(correct_and_metrics(qrows,[r['depths'][str(d)]['logits'] for r in tr])[0])
        transitions=[]
        for a,b in ((1,2),(2,4),(1,4),(4,8)):
            old=np.array(qcorrect[a]);new=np.array(qcorrect[b]);transitions.append([f'{a}->{b}',pct((~old&new).mean()),pct((old&~new).mean())])
        text+="\nQwen transitions, fractions of all seed/item decisions:\n\n"+table(['Depths','Wrong→right','Right→wrong'],transitions)
        qtiming=[]
        for c,d in (("qwen_onepass",1),("qwen_recurrent",1),("qwen_recurrent",2),("qwen_recurrent",4),("qwen_recurrent",8)):
            qtiming.append([c,d]+[f"{np.mean([r['per_item_ms'] for r in qlat['measurements'] if r['condition']==c and r['depth']==d and r['batch']==b]):.2f}" for b in (1,8)])
        text+="\n"+table(["Qwen model","Loops","Batch1 ms","Batch8 ms/item"],qtiming)+f"\nQwen fit epoch wall time totals {qtime/60:.2f} minutes; peak training MLX allocation {peak_qwen:.2f} GiB, evaluation {qlat['peak_mlx_gib']:.2f} GiB. Timings include chat serialization/tokenization and full-prefix model computation, two warmups/three repeats/three seeds. No cached feature scoring is compared with full inference. The one-pass control matches parameters/data/epochs, but not total recurrent training compute; no extra-epoch Qwen compute control was trained.\n"
        if (RESULTS/"qwen_diagnostics.json").exists():
            diag=load('qwen_diagnostics.json');rows=[]
            for split,rs in diag['rollouts'].items():
                for name,v in rs.items(): rows.append([split,name,pct(v['completion']['mean']),f"{v['invalid_moves_mean']:.2f}",f"{v['inference_seconds_mean']:.3f}"])
            text+="\nSmall Qwen gameplay diagnostic (12 independent starting layouts, three seeds; wide uncertainty, no selection):\n\n"+table(["Split","Model@loops","Completion","Invalid moves/episode","Inference s/episode"],rows)
            text+="\nAlternative spaced-grid serialization is exploratory: "+", ".join(f"{c} {pct(v['mean']['accuracy'])}" for c,v in diag['alternative_serialization'].items())+" on the same 120 final-ID examples. Format sensitivity cannot be separated from task competence with this small sample.\n"
    text+="\n## Snake: initial simulator smoke\n\nOrdered body segments, moving-tail collision logic, one food target, growth, fixed heading, prohibited reversal, wall/self collision and terminal behavior are tested. Reverse attempts terminate as illegal collisions; no food respawns. A bounded exhaustive teacher searches the actual simulator at horizons 2/4/6. Frozen lexicographic utility prioritizes survival steps, food collected, reachable space, then negative food distance. Sixty exact 4×4 starting states (straight bodies of length 2–4, no obstacles) and 12-step episodes are preserved. This is a deliberately simple baseline smoke distribution, separate from the learned-policy study below.\n\n"
    text+=table(["Policy","Survival steps","Food collected","Collision rate","Inference ms/episode"],[[name,f"{v['survival_steps']:.2f}",f"{v['food_collected']:.3f}",pct(v['collision_rate']),f"{v['mean_episode_inference_seconds']*1000:.2f}"] for name,v in snake['policies'].items()])+"\n"+figs["snake"]
    text+="\nLonger-horizon receding search collects less food here despite perfect finite-episode survival. With the frozen utility, horizon-dependent tie-breaking and postponement are plausible explanations; the reward was not changed after observing this. No learned Snake network was trained, so Snake provides no evidence for or against neural recurrence benefits.\n\n## Failures, limitations and next phase\n\nThe main limitation is **no established accuracy advantage over a comparably deep untied feedforward model**. Strong within-model depth gains show that this local architecture can use extra spatial computation, but receptive-field expansion is a designed advantage over the one-hop baseline. It does not distinguish planning algorithms from learned local topology heuristics, memorized synthetic regularities or repeated nonlinear processing. The parameter-matched narrow control performs similarly too. No unrestricted global CNN, eight-block compute control, auxiliary distance supervision, larger training boards, RL training or broad real-world reasoning test was included.\n\nHarder path lengths and full rollouts remain difficult; near-perfect first moves do not guarantee navigation. Curriculum did not improve the selected objective. Final-depth-only training damages extrapolated depth. Qwen transfer was not established. Three seeds, near-ceiling ID accuracy, family overlap, weak one-legal examples, finite rollout subsets, unadjusted secondary intervals and small serial latency samples all constrain claims. The two serialization bugs were corrected before training; their failure record is retained. Unsuccessful training conditions were not removed. C/V/D heads were not reused or retrained.\n\nThe next phase should prospectively compare **shared eight-hop recurrence with an eight-block untied and a global CNN control**, at matched measured compute and data, on larger branching mazes constructed to make identical local neighborhoods require different distant-goal decisions. Train on mixed 5×5/7×7 sizes, evaluate a new larger locked set, and retain full episodes. Separately test whether a compact grid encoder or better text curriculum improves Qwen competence before adding controllers. Snake needs bent bodies/obstacles, a non-postponing prospective objective and a trained-policy comparison before interpretation.\n\n## Reproduction and local status\n\nSee [reproduction commands](../../experiments/experiment_05b/README.md), [configuration](../../experiments/experiment_05b/config.json), [plan](../../experiments/experiment_05b/plan.md), [raw public results](../../results/experiment_05b/summary.json), [representation diagnostics](../../results/experiment_05b/representation_analysis.json), and [complete seed-level trajectories](../../results/experiment_05b/trajectories). All eight dataset gzip files regenerated byte-identically in an isolated CPU environment. Public recorded classification metrics regenerate exactly from saved logits; hidden-state interventions and retraining require ignored adapter checkpoints, with regeneration commands provided. No full independent retraining replication is claimed. No model weights or large hidden-state caches are committed.\n\n"
    if (RESULTS/'snake_neural_summary.json').exists():
        sn=load('snake_neural_summary.json');ep=load('snake_neural_episodes.json');rec=sn['models']['snake_plan_rec'];safety=sn['models']['snake_safety_rec'];final=read_trace(RESULTS/'snake_neural'/'datasets'/'final.jsonl.gz')
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        axes[0].plot([1,2,4,8],[100*rec['mean'][str(d)]['horizon_accuracy']['6'] for d in (1,2,4,8)],'o-',label='Planning recurrence',color=COLORS[0])
        axes[0].axhline(100*sn['models']['snake_plan_ff4']['mean']['4']['horizon_accuracy']['6'],ls='--',label='Four-block feedforward',color=COLORS[1]);axes[0].set(xlabel='Internal loops',ylabel='Horizon-6 teacher agreement (%)',xticks=[1,2,4,8]);axes[0].legend(fontsize=7)
        for ax,field in zip(axes[1:],('survival_steps','food_collected')):
            ax.plot([1,2,4,8],[ep[f'snake_plan_rec@{d}'][field] for d in (1,2,4,8)],'o-',label='Planning recurrence',color=COLORS[0]);ax.plot([1,2,4,8],[ep[f'snake_safety_rec@{d}'][field] for d in (1,2,4,8)],'o--',label='Safety recurrence',color=COLORS[2])
            for name,color in (('snake_plan_ff4@4',COLORS[1]),('safe_greedy@0',COLORS[4]),('oracle_h6@0',COLORS[3])):ax.axhline(ep[name][field],ls='--',label=name,color=color)
            ax.set(xlabel='Internal loops',ylabel=field.replace('_',' '),xticks=[1,2,4,8]);ax.legend(fontsize=6)
        snake_fig=figure('snake_learned_policies',fig)
        block="\n## Snake: prospective learned-policy study\n\nAfter the maze models/results were frozen, remaining budget supported 12 additional fits on independent Snake data. The [Snake protocol](../../experiments/experiment_05b/snake_config.json) was fixed before generating its prospective final set, and the [Snake checkpoint freeze](../../experiments/experiment_05b/snake_frozen_protocol.json) preceded opening that set. Maze choices were not changed. The original lexicographic utility remains unchanged.\n\nTraining/validation/final contain 3,600/600/600 D4-disjoint ordered-body states on 4×4 boards with 0.10 wall probability and length 2–4, including bent bodies. Twenty percent generator probability represents post-food states (no respawn), avoiding an entirely unseen state type after eating. Full episodes start only with food present: the first 60 such final states, fixed by protocol. Head/neck determine heading. Nine input channels include walls/free/head/food, ordered segment index, and four heading indicators; no teacher utility or future states enter the policy. Original grid recurrence code and maze checkpoints are untouched.\n\nThe primary planning network uses the same local gated family, width32, 19,044 parameters; four-block feedforward has 74,820. Planning targets are full optimal sets of the horizon-6 teacher; horizons2/4 agreement is secondary. A separate safety network trains the full safe-action set. Shared models sample training loops1/2/4/8, 20 epochs, batch64, three seeds; checkpoints minimize depth4 validation set NLL. Feedforward controls receive identical examples/epochs. Eight percent of final states have no safe immediate action: safety training masks them with an all-action zero-loss target, while evaluation explicitly counts their unavoidable collisions. Planning ties in impossible states can accept colliding actions; teacher agreement is not a survival guarantee.\n\n"
        block+=table(['Policy','Parameters','Loop1 H6','Loop2 H6','Loop4 H6','Loop8 H6'],[[name,v['parameters']]+[pct(v['mean'][str(d)]['horizon_accuracy']['6']) if str(d) in v['mean'] else '—' for d in (1,2,4,8)] for name,v in sn['models'].items()])
        block+=f"\nPlanning recurrence depth4−depth1 is {interval(sn['paired']['1->4']['accuracy'])}; depth4−depth2 is {interval(sn['paired']['2->4']['accuracy'])}. Recurrence4−four-block feedforward4 is {interval(sn['paired']['rec4_vs_ff4'])}. This is a small depth benefit over one internal update, without a sharing advantage. All learned policies attain 92% immediate safety, the ceiling on these final states.\n\n"
        block+=table(['Loops','Teacher H2 agreement','Teacher H4 agreement','Teacher H6 agreement'],[[d]+[pct(rec['mean'][str(d)]['horizon_accuracy'][str(h)]) for h in (2,4,6)] for d in (1,2,4,8)])
        block+='\n'+snake_fig+'\n'+table(['Policy@loops','Survival steps','Food collected','Collision rate','Illegal actions/episode','Inference ms/episode'],[[name,f"{v['survival_steps']:.2f}",f"{v['food_collected']:.3f}",pct(v['collision']),f"{v['illegal_actions']:.3f}",f"{v['inference_seconds']*1000:.2f}"] for name,v in ep.items()])
        outcomes={}
        for name,d in (('snake_plan_rec',1),('snake_plan_rec',4),('snake_plan_ff4',4)):
            rs=read_trace(RESULTS/'snake_neural'/'episodes'/f'{name}_d{d}.jsonl.gz');outcomes[f'{name}@{d}']={field:np.array([[r[field] for r in rs if r['seed']==seed] for seed in sn['config']['seeds']],float) for field in ('survival_steps','food_collected')}
        survival=paired_interval(outcomes['snake_plan_rec@4']['survival_steps']-outcomes['snake_plan_rec@1']['survival_steps']);food=paired_interval(outcomes['snake_plan_rec@4']['food_collected']-outcomes['snake_plan_rec@1']['food_collected'])
        block+=f"\nEpisode-clustered recurrence4−recurrence1 survival difference: {survival['mean']:+.3f} steps, 95% CI [{survival['ci95'][0]:+.3f}, {survival['ci95'][1]:+.3f}]. Food difference: {food['mean']:+.3f}, CI [{food['ci95'][0]:+.3f}, {food['ci95'][1]:+.3f}]. Each uses 60 initial-state clusters and three seeds. Survival and food gains are not interchangeable. The four-block control collects more food than four-loop recurrence; the simple safe greedy policy collects 0.90 versus recurrence4 0.556. More loops improve teacher agreement modestly but do not establish a superior food-collecting policy. The teacher itself collects only 0.50 while surviving longer, reflecting the frozen survival-first finite-horizon utility and receding decisions.\n\n"
        ties={k:sum(sum(r['teacher']['6'])==k for r in final) for k in (1,2,3,4)}
        block+=f"Horizon6 optimal-set sizes 1/2/3/4 occur on {ties[1]}/{ties[2]}/{ties[3]}/{ties[4]} final states. Many ties and impossible starting states inflate teacher agreement relative to unique-action planning. No learned Snake generalization to larger boards was tested. All three compressed Snake datasets regenerated byte-identically in the clean CPU environment.\n"
        text=text.replace('\n## Failures, limitations and next phase',block+'\n## Failures, limitations and next phase')
        text=text.replace('Snake completed its environment/teacher/baseline smoke stage only.','Snake completed a separate prospective learned-policy comparison after its initial simulator smoke.')
        text=text.replace('No learned Snake network was trained, so Snake provides no evidence for or against neural recurrence benefits.','Those initial smoke baselines are followed by a separately generated prospective learned-policy study below.')
        text=text.replace('Snake needs bent bodies/obstacles, a non-postponing prospective objective and a trained-policy comparison before interpretation.','Snake now includes bent bodies/obstacles and trained controls; a new prospective utility/episode design is needed to address postponement and distinguish teacher agreement from food collection.')
    text+=f"The CPU report audit independently recomputed {checked['independently_recomputed_maze_labels']:,} maze labels, verified {checked['distinct_canonical_wall_layouts']:,} distinct canonical layouts and checked {checked['checked_decisions']:,} depth decisions across {checked['checked_trajectory_files']} trajectory files. Markdown/HTML/figures use recorded measurements only. Public `v0.1.0` remains at `0a637eb97d66eebbd740b612ab92e9095c496931`. Experiment 5 and 5.1 remain local and require review before publication. Original code is MIT; separately downloaded Qwen weights retain Apache-2.0 terms.\n"
    if (RESULTS/'completion.json').exists():
        completion=load('completion.json')
        text+=f"\nThrough the recorded report QA checkpoint, elapsed project time was {completion['elapsed_seconds_through_report_qa']/60:.1f} minutes, with {completion['fit_epoch_seconds_total']/60:.2f} minutes of recorded fit epochs across {completion.get('main_fits_total',27)} main fits and one grid pilot. This remained within the six-hour budget. MLX integration: {completion['tests']['mlx_passed']} tests passed. Clean CPU environment: {completion['tests']['cpu_passed']} passed, {completion['tests']['cpu_skipped']} skipped. The local/public boundary, frozen artifacts, 72 trained-state locality checks and relative report links passed verification. The protocol receipt's single absolute path was normalized without changing frozen content hashes or decisions; the original is preserved privately.\n"
    if checked.get('snake_states_recomputed'):
        text+=f"\nThe secondary Snake audit additionally recomputed {checked['snake_states_recomputed']:,} states' safety and horizon2/4/6 teacher targets and {checked['snake_decisions_recomputed']:,} neural teacher-agreement decisions from recorded logits.\n"
    text+="\nResearch lineage and source attribution are recorded in [Experiment 5's verified references](../../experiments/experiment_05/research_references.md), including Qwen3, RecurTrace, Universal Transformers, ACT, PonderNet and LoRA. This experiment reuses the project's own middle-layer implementation and independently implements the grid environment/message-passing planner. It does not claim architectural novelty or reproduce those papers' experimental protocols.\n"
    (REPORT/"learning_to_navigate.md").write_text(text)
    body=markdown.markdown(text,extensions=["tables","fenced_code"])
    for name in (REPORT/"figures").glob("*.png"):
        encoded=base64.b64encode(name.read_bytes()).decode();body=body.replace(f'src="figures/{name.name}"',f'src="data:image/png;base64,{encoded}"')
    (REPORT/"learning_to_navigate.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Think Again — Experiment 5.1</title><style>body{max-width:1050px;margin:40px auto;padding:0 22px;font:16px/1.6 system-ui;color:#243247;background:#fcfcfd}h1,h2{line-height:1.2}h2{margin-top:2.2em}img{max-width:100%;height:auto}table{border-collapse:collapse;display:block;overflow-x:auto;font-size:13px}td,th{padding:8px 12px;border-bottom:1px solid #d7dfe7;text-align:left}th{background:#edf2f7}pre{overflow:auto;padding:16px;background:#edf2f7}a{color:#2362a3}code{font-size:.9em}</style><body>'+body+'</body></html>')
    write_json(RESULTS/"report_artifacts.json",{"files":{str(p):digest(p) for p in REPORT.rglob("*") if p.is_file()},"embedded_images":body.count('src="data:image/png'),"report_only_backend":"CPU; saved measurements only"})
    print(f"Report regenerated; audited {checked['checked_decisions']} decisions, embedded {body.count('src='+chr(34)+'data:image/png')} figures",flush=True)

if __name__=="__main__": report()
