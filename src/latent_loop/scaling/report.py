"""Self-contained scientific report from recorded measurements, CPU only."""
import base64
import json
import re
from pathlib import Path
import numpy as np
from scipy.special import softmax
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown
from latent_loop.scaling.data import EXP,OUT,read,config
from latent_loop.planning.data import write_json,digest,now
from latent_loop.environments.maze import Maze,NAMES
from latent_loop.planning.oracles import shortest_path
from latent_loop.planning.report import board,table,pct,interval

REPORT=Path('reports/experiment_05c');DD=[1,2,4,8,12,16,24,32];SIZES=[5,7,9,11,15]
NAMES_MODEL={'frozen_05b':'Frozen 5B','rec_variable':'Variable recurrence','rec_curriculum':'Curriculum recurrence','ff4_params':'Four-block FF (narrow)','ff8':'Eight-block FF','global_cnn':'Global-context CNN'}
def get(name):return json.loads((OUT/name).read_text())
def fig(name,f):
    f.tight_layout();f.savefig(REPORT/'figures'/f'{name}.png',dpi=160,bbox_inches='tight');f.savefig(REPORT/'figures'/f'{name}.svg',bbox_inches='tight',metadata={'Date':None});plt.close(f)
    return f'![{name.replace("_"," ")}](figures/{name}.png)\n'
def heat(ax,a,x,y,title):
    im=ax.imshow(a,vmin=0,vmax=100,cmap='viridis',aspect='auto');ax.set(xticks=range(len(x)),xticklabels=x,yticks=range(len(y)),yticklabels=y,title=title)
    for i in range(len(y)):
        for j in range(len(x)):ax.text(j,i,f'{a[i,j]:.1f}',ha='center',va='center',color='white' if a[i,j]<55 else 'black',fontsize=8)
    return im

def run():
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'figures').mkdir(exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'exp05c'})
    cfg=config();s=get('summary.json');roll=get('rollout_summary.json');desc=get('descriptive_analysis.json');prop=get('propagation_analysis.json');lat=get('latency.json');ex=get('train_depth_diagnostic.json');freeze=json.loads((EXP/'frozen_protocol.json').read_text());selected=freeze['selected_recurrent'];long_rows=read(OUT/'datasets/long15.jsonl.gz');long_min=min(r['labels']['distance'] for r in long_rows);long_max=max(r['labels']['distance'] for r in long_rows);primary=s['splits']['size9'];pics={}
    def acc(split,c,d):return s['splits'][split]['models'][c]['mean'][str(d)]['accuracy']
    def completion(split,c,d):return roll[split]['models'][f'{c}@{d}']['completion']
    def ms(c,d,size=9,batch=1):return next(r['milliseconds_per_item'] for r in lat['rows'] if r['model']==c and r['depth']==d and r['size']==size and r['batch']==batch)
    f,ax=plt.subplots(figsize=(8,4))
    for size in SIZES:ax.plot(DD,[100*acc(f'size{size}','frozen_05b',d) for d in DD],'o-',label=f'{size}×{size}')
    ax.axvline(8,color='gray',ls=':',label='Largest trained budget');ax.set(xlabel='Internal iterations, unchanged weights/observation',ylabel='Optimal-first-action accuracy (%)',xticks=DD,ylim=(35,102));ax.legend(ncol=3,fontsize=8);pics['loops']=fig('accuracy_vs_iterations',f)
    f,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axes,('size9','long15')):
        bins=sorted(s['splits'][split]['models']['frozen_05b']['by_path']['4'],key=lambda b:int(b.split('-')[0]));x=np.arange(len(bins))
        for d in (4,8,16,32):ax.plot(x,[100*s['splits'][split]['models']['frozen_05b']['by_path'][str(d)][b]['accuracy'] for b in bins],'o-',label=f'{d} loops')
        ax.set(xticks=x,xticklabels=bins,xlabel='Actual BFS shortest-path length stratum',ylabel='Accuracy (%)',title=split,ylim=(30,103));ax.legend(fontsize=8)
    pics['path']=fig('accuracy_vs_path_length',f)
    f,ax=plt.subplots(figsize=(8,4));a=np.array([[100*acc(f'size{z}','frozen_05b',d) for d in DD] for z in SIZES]);im=heat(ax,a,DD,[f'{z}×{z}' for z in SIZES],'Frozen planner: board size × internal loops');ax.set_xlabel('Loops');f.colorbar(im,ax=ax,label='Accuracy (%)');pics['size']=fig('size_loop_accuracy_heatmap',f)
    f,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axes,('size9','long15')):
        bins=sorted(roll[split]['models']['frozen_05b@4']['by_path'],key=lambda b:int(b.split('-')[0]));a=np.array([[100*roll[split]['models'][f'frozen_05b@{d}']['by_path'][b] for d in (4,8,16,32)] for b in bins]);im=heat(ax,a,[4,8,16,32],bins,split+' completion');ax.set(xlabel='Loops per decision',ylabel='Starting BFS path length');f.colorbar(im,ax=ax,shrink=.8)
    pics['completion']=fig('path_loop_completion_heatmap',f)
    f,ax=plt.subplots(figsize=(9,4))
    for c in NAMES_MODEL:
        d=8 if c=='frozen_05b' else cfg['conditions'][c]['depth'];ax.plot(SIZES,[100*acc(f'size{z}',c,d) for z in SIZES],'o-',label=NAMES_MODEL[c]+f' @{d}')
    ax.set(xlabel='Native board side length',ylabel='First-action accuracy (%)',xticks=SIZES,ylim=(40,103));ax.legend(fontsize=8,ncol=2);pics['controls']=fig('recurrent_vs_feedforward',f)
    f,ax=plt.subplots(figsize=(8,4))
    for c in NAMES_MODEL:
        dd=DD if c=='frozen_05b' or cfg['conditions'][c]['shared'] else [cfg['conditions'][c]['depth']];xx=[ms(c,d) for d in dd];yy=[100*acc('size9',c,d) for d in dd];ax.plot(xx,yy,'o-',label=NAMES_MODEL[c])
        if c in ('frozen_05b',selected):
            for x,y,d in zip(xx,yy,dd):ax.annotate(str(d),(x,y),xytext=(2,4),textcoords='offset points',fontsize=7)
    ax.set(xlabel='Measured batch-1 end-to-end inference (ms)',ylabel='9×9 action accuracy (%)');ax.legend(fontsize=8);pics['latency']=fig('accuracy_vs_actual_latency',f)
    f,ax=plt.subplots(figsize=(8,3.5));pairs=['4->8','8->12','12->16','16->24','24->32'];x=np.arange(len(pairs));tt=primary['models']['frozen_05b']['transitions'];ax.bar(x-.18,[100*tt[p]['wrong_right'] for p in pairs],.36,label='Wrong → right');ax.bar(x+.18,[100*tt[p]['right_wrong'] for p in pairs],.36,label='Right → wrong');ax.set(xticks=x,xticklabels=pairs,ylabel='Fraction of all paired instances (%)',xlabel='Frozen planner, 9×9');ax.legend();pics['transitions']=fig('corrected_vs_degraded',f)
    cases=read(OUT/'propagation_examples.jsonl.gz');case=max([c for c in cases if c['model']=='frozen_05b'],key=lambda c:c['example']['labels']['distance']);maze=Maze.from_dict(case['example']);f,axes=plt.subplots(3,4,figsize=(11,8.5));dd=(1,4,8,32)
    for j,d in enumerate(dd):
        for i,field in enumerate(('state_rms','goal_effect','classical_distance')):
            a=np.array(case['depths'][str(d)][field]);a[np.array(maze.walls,bool)|(a<0)]=np.nan;im=axes[i,j].imshow(a,cmap='magma' if i==1 else 'viridis_r' if i==2 else 'viridis',vmin=0,vmax=max(np.nanmax(np.array(case['depths'][str(k)][field])) for k in dd));axes[i,j].set_title(f'{field.replace("_"," ")} @ {d}',fontsize=9);axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
            for p,mark in ((maze.agent,'A'),(maze.goal,'G')):axes[i,j].text(p[1],p[0],mark,ha='center',va='center',bbox={'facecolor':'white','alpha':.7,'pad':1},fontsize=9)
            f.colorbar(im,ax=axes[i,j],shrink=.65)
    pics['propagation']=fig('hidden_state_propagation',f)
    f,axes=plt.subplots(2,2,figsize=(11,9));failures=[]
    for i,split in enumerate(('size9','long15')):
        rows={r['id']:r for r in read(OUT/'datasets'/f'{split}.jsonl.gz')};episodes=read(OUT/'rollouts'/f'frozen_05b_depth8_{split}.jsonl.gz');traces={r['id']:r for r in read(OUT/'trajectories'/f'frozen_05b_seed11_{split}.jsonl.gz')};bad=[e for e in episodes if e['seed']==11 and not e['done']];chosen=next((e for e in bad if rows[e['id']]['labels']['optimal'][int(np.argmax(traces[e['id']]['depths']['8']['logits']))]),bad[0]);r=rows[chosen['id']];m=Maze.from_dict(r);board(axes[i,0],m,f'Demonstration: {split}, BFS {r["labels"]["distance"]}, cycle',shortest_path(m));path=np.array(chosen['positions']);axes[i,0].plot(path[:,1],path[:,0],'r.-',alpha=.8,label='Model rollout @8');axes[i,0].legend(fontsize=7)
        for a,name in enumerate(NAMES):axes[i,1].plot(DD,[softmax(traces[r['id']]['depths'][str(d)]['logits'])[a] for d in DD],'o-',label=name+(' optimal' if r['labels']['optimal'][a] else ''))
        axes[i,1].set(xlabel='Internal loop, initial board unchanged',ylabel='Action probability',ylim=(0,1));axes[i,1].legend(fontsize=8);failures.append({'split':split,'id':r['id'],'seed':11,'rollout_depth':8,'episode':chosen})
    pics['failures']=fig('new_failure_examples',f);write_json(OUT/'illustrative_failure_selection.json',{'selection':'First seed11 failure with correct initial action if available; descriptive, not aggregate','examples':failures})
    f,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axes,('size9','size15')):
        matrix=np.array([[100*ex['results'][split]['mean'][str(d)]['accuracy'] for d in DD],*[ [100*acc(split,c,d) for d in DD] for c in ('frozen_05b','rec_variable','rec_curriculum')]])
        im=heat(ax,matrix,DD,['Old final4 (max4)','Old simultaneous (max8)','New variable (max16)','New curriculum (max16)'],split);ax.set_xlabel('Test depth');f.colorbar(im,ax=ax,shrink=.8)
    pics['extrapolation']=fig('train_depth_test_depth_generalization',f)
    f,axes=plt.subplots(1,2,figsize=(10,3.8))
    for c in ('frozen_05b','rec_variable','rec_curriculum'):
        for ax,field in zip(axes,('state_norm','state_change')):ax.plot(DD,[primary['models'][c]['mean'][str(d)][field] for d in DD],'o-',label=NAMES_MODEL[c]);ax.set(xlabel='Internal loop',ylabel=field.replace('_',' '))
    axes[1].legend(fontsize=8);pics['states']=fig('bounded_states_and_drift',f)
    f,axes=plt.subplots(1,2,figsize=(11,4))
    for c in cfg['conditions']:
        fits=[get(f'training/{c}_seed{s}.json') for s in cfg['seeds']]
        for ax,field in zip(axes,('loss','validation')):
            vals=[[e[field] if field=='loss' else e[field]['loss'] for e in t['epochs']] for t in fits];ax.plot(range(1,cfg['epochs']+1),np.mean(vals,0),label=NAMES_MODEL[c]);ax.set(xlabel='Epoch',ylabel='Training objective' if field=='loss' else 'Validation set loss',ylim=(0,.6))
    axes[1].legend(fontsize=7);pics['training']=fig('training_validation_loss',f)
    primary_effect=primary['frozen_8_minus_4'];fits=[get(f'training/{c}_seed{seed}.json') for c in cfg['conditions'] for seed in cfg['seeds']];fit_seconds=sum(e['seconds'] for t in fits for e in t['epochs']);peak=max([t['peak_mlx_gib'] for t in fits]+[get('evaluation_resources.json')['peak_mlx_gib'],get('diagnostic_resources.json')['peak_mlx_gib'],lat['peak_mlx_gib']])
    text=f'''# Experiment 5.2 — Break the Thinker

## Executive summary

**More loops help on new larger mazes, but the learned policy does not scale into a reliable shortest-path planner.** The unchanged18,884-parameter5B network improves on the prespecified600 branching9×9 mazes from {pct(acc('size9','frozen_05b',4))} at four loops to {pct(acc('size9','frozen_05b',8))} at eight: **{interval(primary_effect)}**, paired across maze instances and three historical training seeds. This is the only confirmatory comparison; other comparisons are secondary/exploratory.

Full navigation is much harder. Eight-loop frozen completion is {pct(completion('size9','frozen_05b',8))} on30 new9×9 occupancy episodes and {pct(completion('structure9','frozen_05b',8))} on24 DFS-shift episodes. On24 long15×15 DFS episodes it is zero at4/8/16/32 loops; BFS completes every episode. The new variable model solves only one seed/maze pair at16 and32. Extra loops improve some decisions while persistent local mistakes create inescapable cycles.

Validation selected **{selected}**, deployment depth8, and feedforward **`{freeze['selected_feedforward']}`** at eight blocks. On9×9, new recurrence versus eight-block feedforward is {interval(primary['selected_recurrent_vs_controls']['ff8'])}; no first-action superiority is established. Episode performance differs and is separately reported. Sharing reduces parameters, not executed hops. Longer paths are not uniformly harder in occupancy mazes; local topology and generator selection create shortcuts.

Completed15 new fits, five conditions×three seeds101/202/303,30 epochs each. Synchronized training/validation epoch time totals **{fit_seconds:.2f}seconds ({fit_seconds/60:.2f}minutes)**; this excludes generation, checkpoint I/O, evaluation, reporting and process startup. Maximum recorded MLX allocation **{peak:.3f}GiB**, under48GiB. Total study duration is in the completion receipt. No Qwen/Snake training was added. `v0.2.0` is published; this study remains local.

## Reproduction of Experiment5B

All three historical simultaneous checkpoints reproduce all600 archived ID examples exactly at1/2/4/8 loops: maximum logit error0. Original source/weights remain unchanged. The new executor extends the old guard16→32, with exact logit/state/norm/change parity through16. Native5/7/9/11/15 shapes pass finite/bounded32-loop feasibility. Original5B recorded98.28/99.56% at4/8, with tied feedforward controls and negative Qwen transfer; those historical metrics are not overwritten or presented as this new benchmark.

Live QA also replayed360 seed/maze episodes over120 archived initial mazes, reproducing95.00%7×7 completion and54.44%long-path completion. Every action, position, outcome and non-timing metric matched the archive exactly. These easier historical populations differ from the new branching/distance-conditioned tests. [Archived gameplay verification](../../results/experiment_05c/archived_rollout_verification.json).

[Baseline verification](../../results/experiment_05c/baseline_verification.json) · [Publication receipt](../../experiments/experiment_05c/publication_receipt.json)

## New dataset methodology and leakage checks

Fresh6,000 training and960 validation examples use mixed5×5/7×7 occupancy boards. New final sets:5×5=400,7×7=500,9×9=600,11×11=600,15×15=600,DFS9×9=400,dense9×9=400,longDFS15×15=400; total3,900 final. Every decision is solvable/nonterminal with at least two legal actions. All equally optimal actions are accepted; the loss is negative log summed probability over the optimal-action set. No distance-map auxiliary loss is used. Agent/start selection conditions on distance strata, making this a diagnostic distribution rather than natural task prevalence. Long paths on small occupancy boards require substantial rejection sampling; proposal counts and rejections are retained.

Distance ranges are2–4/5–8/9–12/13–20/21–32/33–64 where feasible; longDFS uses21–32/33–48/49–80/81–126. Training's longest target bin is13–24. Board size alone is not a difficulty measure. Reports retain BFS distance, Manhattan detour, goal direction, traps, degree, junction/dead-end counts, cycle rank and optimal-action ties. A goal can influence an agent through this designed local mechanism no earlier than its graph distance in updates; this is an information bound, not a learned-algorithm claim.

Every new split excludes all6,840 historical5B wall layouts and conservative graph fingerprints. D4 wall hashes reject rotation/reflection and changed start/goal placements of a layout. Unrooted reachable-component WL8 invariants additionally reject differently embedded isomorphic graphs. Every matching invariant is rejected, including possible non-isomorphic collisions; equality is not asserted to prove isomorphism. Distinct invariants certify non-isomorphism, while conservative rejection can distort prevalence. Isolated unreachable components are ignored by this topology exclusion. No graph fingerprint repeats across new splits.

Queue-BFS labels were independently checked using heap Dijkstra during generation and in the public audit. All10 dataset gzip files regenerated byte-identically in an independent CPU-only environment. Primary population and comparison were fixed before final opening; final outcomes did not alter fitting, depth or checkpoint selection. [Manifest](../../experiments/experiment_05c/dataset_manifest.json) · [Frozen protocol](../../experiments/experiment_05c/frozen_protocol.json)

## Architectures and training procedures

Frozen baseline and new recurrent models use the original width32 pointwise encoder, shared one-hop cardinal gated update, wall masking and pointwise agent head. States remain convex combinations of tanh-bounded values. Hidden state resets for every decision; all loops receive the unchanged four-channel board. Environment advances once after the committed action. No labels, BFS values, future states, controller or visible reasoning tokens enter learned inference.

Variable training samples1/2/4/8/16 uniformly per batch. Curriculum spends10epochs on distances≤8/depths1/2/4,10 on distances≤12/depths2/4/8, then10 on all data/depths4/8/16. Eligible examples are resampled to retain update counts. Both use action supervision at one sampled depth. AdamW0.001, decay0.0001, clip1, batch64,30epochs. All models share data and nominal optimizer-update budgets; computational budgets differ and are measured. Best checkpoint minimizes validation set loss at depth8 for recurrence/FF8 or4 for other controls. Model-family selection uses validation accuracy, then loss. No final-set re-selection occurs.

The narrow four-block control nearly matches parameters. FF8 uses eight independent width32 updates with the same per-hop computation as recurrence. The global-context control has four width32 untied blocks and broadcasts a learned pooled feature/agent-goal coordinate summary derived only from raw channels before local processing. It has a full-board input dependency but only a compressed global summary, not unrestricted global attention or an exact graph algorithm. Its larger capacity and positional summary are disclosed.

'''
    text+=table(['Model','Parameters','Training seeds','Epoch fit seconds','Selection/deploy depth'],[[NAMES_MODEL[c],18884 if c=='frozen_05b' else next(t['parameters'] for t in fits if t['condition']==c),'11/22/33 (frozen)' if c=='frozen_05b' else '101/202/303','—' if c=='frozen_05b' else f'{sum(e["seconds"] for t in fits if t["condition"]==c for e in t["epochs"]):.2f}',8 if c=='frozen_05b' else cfg['conditions'][c]['depth']] for c in NAMES_MODEL])
    text+='\n'+pics['training']+'\n## Accuracy by loop count and scale\n\n'
    text+=table(['Final split','Mazes',*[str(d) for d in DD]],[[split,r['count'],*[pct(r['models']['frozen_05b']['mean'][str(d)]['accuracy']) for d in DD]] for split,r in s['splits'].items()])
    text+='\n'+pics['loops']+'\n'+pics['size']+'\nThe size aggregates have different feasible path-bin mixtures. Comparisons of board size therefore combine size and difficulty composition. The descriptive analysis also reports a common2–20-distance population across occupancy sizes. Every seed-specific metric and all32 recurrent predictions are saved; repeated seed/example decisions are not independent mazes.\n\n## Accuracy by shortest-path length\n\n'+pics['path']+'\nLonger occupancy paths can be easier to choose initially because winding, corridor-like topology narrows the useful local branch. Conditioning start/goal choice on a long distance can introduce local structural clues. More loops help most where newly available information changes action selection, not monotonically with raw path length. The DFS generator changes this relationship. High first-action accuracy beyond the goal influence radius demonstrates useful distributional/local heuristics, not full goal-to-agent planning.\n\n'
    text+=table(['9×9 path stratum','Mazes','Frozen4','Frozen8','Frozen16','Frozen32'],[[b,sum(r['path_bin']==b for r in read(OUT/'datasets/size9.jsonl.gz')),*[pct(primary['models']['frozen_05b']['by_path'][str(d)][b]['accuracy']) for d in (4,8,16,32)]] for b in sorted(primary['models']['frozen_05b']['by_path']['4'],key=lambda b:int(b.split('-')[0]))])
    text+='\n## Loss, calibration and shortest-path regret\n\n'
    text+=table(['Frozen9×9 loop','Set NLL','ECE','Invalid','First-action regret','Mean confidence'],[[d,f'{primary["models"]["frozen_05b"]["mean"][str(d)]["set_nll"]:.3f}',f'{primary["models"]["frozen_05b"]["mean"][str(d)]["ece"]:.3f}',pct(primary['models']['frozen_05b']['mean'][str(d)]['invalid_rate']),f'{primary["models"]["frozen_05b"]["mean"][str(d)]["mean_regret"]:.3f}',f'{desc["splits"]["size9"]["models"]["frozen_05b"][str(d)]["confidence"]:.3f}'] for d in DD])
    text+='\nA wrong legal first move has regret2 in this undirected unweighted graph; an invalid move stays put and has regret1. This metric does not capture the total cost of entering and exploring a trap. Full episodes provide that complementary failure evidence. Confidence/calibration and decision correction remain separate.\n'
    text+='\n## Feedforward comparisons\n\n'
    text+=table(['Split','Frozen8','New variable8','Curriculum8','NarrowFF4','UntiedFF8','GlobalCNN4'],[[split,*[pct(acc(split,c,8 if c=='frozen_05b' else cfg['conditions'][c]['depth'])) for c in NAMES_MODEL]] for split in s['splits']])
    text+='\n'+pics['controls']+'\n'+table(['9×9 selected recurrence8 minus control','Paired95% interval'],[[f'`{c}`',interval(v)] for c,v in primary['selected_recurrent_vs_controls'].items()])
    text+='\nThe narrower control is parameter-matched but has fewer hops/channels. Its gap does not isolate weight sharing. Recurrence and FF8 exchange local information the same number of times; FF8 has149,028 parameters versus18,884. GlobalCNN4 has75,844 and a different global-information route. Equal epochs/data do not equalize fit seconds. Neither fewer parameters nor these accuracy comparisons establish a runtime speedup.\n\n## Full navigation and maze completion\n\n'
    text+=table(['Split','Initial mazes','Frozen4','Frozen8','Frozen16','Frozen32','Variable8','Variable16','FF8','BFS'],[[split,r['maze_count'],*[pct(completion(split,'frozen_05b',d)) for d in (4,8,16,32)],pct(completion(split,selected,8)),pct(completion(split,selected,16)),pct(completion(split,'ff8',8)),pct(completion(split,'bfs',0))] for split,r in roll.items()])
    text+='\n'+pics['completion']+'\nRollouts use the first six independent layouts per distance stratum, all seeds, fixed budgets. Maximum256 environmental steps; a repeated position terminates as failure because a deterministic stateless policy will repeat that cycle forever. Failed episodes therefore record time/steps to detected cycle, not256 wasted actions. This shortcut preserves completion outcomes but changes failure durations; do not compare those durations with a full256-step benchmark. Successful step ratios, invalid moves, repeated positions, action regret, inference time and whole trajectories are retained. BFS is used only as separately labeled reference/offline scorer.\n\n'
    text+=table(['9×9 episode comparison (30 maze clusters)','Paired completion difference95%'],[['Frozen8 minus frozen4',interval(desc['episode_8_minus_4']['size9'])],['New recurrence8 minus FF8',interval(desc['selected_recurrent_episode_vs_ff8']['size9'])]])
    text+='\nFirst-action ties do not guarantee safe future decisions. New recurrence and feedforward models with similar initial accuracy can differ on later agent positions in the same maze. Any episode advantage is secondary, with small episode samples and unadjusted comparisons. No rollout-based model/depth selection was made.\n\n'+pics['failures']+'\n## Hidden-state propagation and interventions\n\n'+pics['propagation']+'\nThe first two rows show actual learned states/goal effects; the third shows recorded nonlearned one-hop min-plus distance propagation, never model input. White reference cells are walls or not yet reached. These are different quantities with separate color scales. Illustrative examples were selected separately from population summaries. A fixed ridge probe(L2=10) fits BFS distance from hidden channels plus intercept on the first200 training mazes only. It is evaluated on40 independent mazes per split/seed; cells are correlated, and no cell-level significance is claimed. Goal removal is out of distribution. Hidden-message cuts on a shortest-path intermediate cell are compared with approximately agent-distance-matched off-path sham cuts, also under goal removal to isolate goal-dependent effects. These are controlled perturbations, not equivalent natural inputs. A single cell cut can fail to block alternate routes; reduced/greater logit effects alone do not establish algorithm identity.\n\n'
    prs=prop['probe_results'];text+=table(['Frozen probe, split','Loop4 R²','Loop8 R²','Loop32 R²','Loop32 MAE'],[[split,*[f'{np.mean([r["r2"] for r in prs if r["model"]=="frozen_05b" and r["split"]==split and r["depth"]==d]):.3f}' for d in (4,8,32)],f'{np.mean([r["mae"] for r in prs if r["model"]=="frozen_05b" and r["split"]==split and r["depth"]==32]):.2f}'] for split in ('size5','size9','size15','structure9','long15')])
    text+='\nGoal effects are exactly zero outside each loop’s permitted graph radius, verifying designed locality. Same-board distant-goal counterfactual pairs require disjoint optimal actions while both goals are outside radius4. Their loop1/2/4 logits are exactly identical; no policy with those local observations can choose both correctly. Later differences show that goal information can become available, without proving that it is used optimally.\n\n'
    text+=table(['Model, paired far-goal diagnostic','Loop1','Loop2','Loop4','Loop8','Loop16','Loop32'],[[NAMES_MODEL[c],*[pct(v[str(d)]['paired_goal_accuracy']) for d in (1,2,4,8,16,32)]] for c,v in desc['far_goal_counterfactuals'].items()])
    text+='\n'+table(['Goal-dependent logit effect: intervention diagnostic','Loop','Mazes×seeds','Uncut mean norm','Path-cut mean norm','Sham-cut mean norm'],[[NAMES_MODEL[c],d,len(v:=[r for r in prop['interventions'] if r['model']==c and r['depth']==d]),*[f'{np.mean([r[field] for r in v]):.3f}' for field in ('goal_logit_effect','path_goal_effect','sham_goal_effect')]] for c in ('frozen_05b',selected) for d in (8,16,32)])
    text+='\nThese paired diagnostics reuse30 original boards per model and perturb goal placement within each board; they are not new independent aggregate-test mazes. Decodability and propagation fronts support reusable local information exchange. Distance-probe failure on longer structures, imperfect counterfactual decisions and failed rollouts constrain a shortest-path-procedure interpretation.\n\n## Generalization beyond trained depth\n\n'+pics['extrapolation']+'\nHistorical final4-only, historical simultaneous(max8), new variable(max16) and new curriculum(max16) share the small architecture, but differ in data/objective; the heatmap is descriptive and cannot isolate one training ingredient. Beyond8, frozen accuracy can plateau or decline; new variable training is more stable beyond16, while curriculum often deteriorates. Extrapolated24/32 budgets were not selected from final performance. More loops need not improve decisions even though state values remain bounded.\n\n'+pics['states']+'\n'+pics['transitions']+'\nThe saved trajectories include norm, adjacent-state cosine, change/drift, logits, loss, confidence/calibration and prediction oscillation. Bounded activations rule out amplitude explosion in these tests; they do not rule out harmful attractors, oscillation or representational degradation.\n\n## Failure frontier and negative findings\n\nEngineering thresholds were fixed prospectively:90% first-action accuracy and80% completion. These are descriptive thresholds, not significance tests. All tested size×distance×junction×loop cells and counts appear in [failure frontier](../../results/experiment_05c/failure_frontier.json). Six rollout mazes per distance bin are too few for precise frontier estimation. No continuous untested boundary is inferred.\n\n'
    text+=table(['Population','Frozen8 action','Frozen8 completion','Pass90% /80%','Frozen32 action','Frozen32 completion'],[[split,pct(acc(split,'frozen_05b',8)),pct(completion(split,'frozen_05b',8)),f'{acc(split,"frozen_05b",8)>=.9} / {completion(split,"frozen_05b",8)>=.8}',pct(acc(split,'frozen_05b',32)),pct(completion(split,'frozen_05b',32))] for split in s['splits']])
    text+=f'\nObserved long-path distances range from {long_min} to {long_max} edges.\n'
    text+='\nA goal farther than32 edges cannot influence this strictly local policy within the tested maximum32 loops. Failure beyond that bound alone cannot distinguish poor learning from insufficient communication time. More diagnostic: in the21–32-edge DFS stratum, frozen32-loop accuracy is64.33% and completion is0% on six independent mazes×three seeds. The new variable32-loop condition completes one of18 seed/maze rollouts(5.56%). Those cases fit inside the nominal information radius and still expose a learned-computation failure.\n'
    text+='\nThe sharpest failure is near-zero complete navigation on the long-path DFS population, despite70–78% first-action accuracy. Geometry/structure matters as much as scale. New curriculum does not robustly improve extrapolation; frozen/final-depth-only models can degrade beyond training. A compressed global-context CNN does not solve the scaling problem. No model demonstrates a reusable exact BFS procedure or reliable large-maze navigation.\n\n## Latency, parameters and actual computation\n\n'+pics['latency']+'\n'
    text+=table(['Model@loops','Parameters','9×9 B1 ms','9×9 B8 ms/item','9×9 accuracy'],[[NAMES_MODEL[c]+f'@{d}',18884 if c=='frozen_05b' else next(t['parameters'] for t in fits if t['condition']==c),f'{ms(c,d):.3f}',f'{ms(c,d,batch=8):.3f}',pct(acc('size9',c,d))] for c in NAMES_MODEL for d in (DD if c=='frozen_05b' else [cfg['conditions'][c]['depth']])])
    text+=f'''\nTiming uses three warmups/ten synchronized full-policy repetitions, raw observation construction, transfer and host argmax; medians, serial model order, one prespecified seed. Model loading is excluded. Batch8 is amortized per item. Timing uses standardized native-size open boards, while accuracy uses the locked maze population; size/batch are matched, topology is not. Small dispatch/device-load differences do not establish speedups. Maximum recorded MLX {peak:.3f}GiB is distinct from process RSS; counters are not added as physical memory. Per-stage resource records retain peak counters and synchronized fit durations.

## Statistical and scientific limitations

Primary inference resamples paired maze instances within fixed path strata and historical training seeds jointly,4,000 replicates. New-condition comparisons pair the same maze and matched new seed indices; three seeds limit precision. Episode intervals resample whole initial mazes and seeds, never navigation steps. Secondary/exploratory comparisons are unadjusted; no final-best depth is designated. All final examples are a one-time prospective evaluation with checkpoint selection frozen beforehand.

The benchmark is synthetic, heavily distance-conditioned and branching-only; rejecting conservative graph invariants further alters prevalence. Occupancy path strata contain local shortcuts. Different size mixtures limit raw size comparisons. No21×21 test,16-block untied model, reinforcement learning, persistent across-action memory, adaptive stopping, new C/V/D probe, Snake/Qwen expansion or full independent retraining replication was run. Global control is a compressed summary, not an exhaustive global architecture comparison. Baseline and new models trained on different data; their difference cannot isolate curriculum/extra depth alone. Hidden probes/interventions are diagnostic and imperfect. Failures and all training variants remain recorded.

Two NumPy metadata export failures and an empty-stratum analysis-export failure were preserved. The empty stratum is represented as null, with no model or metric change. The generation error was fixed before fitting. The post-freeze diagnostic coordinate export uses a documented native-scalar boundary wrapper, leaving every frozen scientific source/hash and weight unchanged. The complete diagnostic computation was repeated and its numeric measurements matched exactly; no evaluation or selection rule changed. A receipt-counter shadowing error was also corrected and the full public audit rerun. See [amendment](../../experiments/experiment_05c/diagnostic_export_amendment.json) and [failure record](../../results/experiment_05c/implementation_failures.json).

## Answers and next research step

1. **Does thinking longer help bigger mazes?** Yes for the prespecified9×9 four→eight comparison, with limited gains elsewhere; it does not yield reliable large-maze navigation.
2. **Does benefit scale with shortest-path length?** Not monotonically. Occupancy topology/selection shortcuts and DFS shifts show that raw distance is insufficient.
3. **Does shared recurrence outperform feedforward at useful budgets?** No established first-action advantage over equally trainedFF8 at eight hops. Some episode comparisons favor recurrence; they are secondary and do not establish broad superiority or speedup.
4. **Does it generalize beyond trained depth?** Partially and objective-dependently. Variable-depth training can remain useful beyond16; frozen/curriculum/final4 models may deteriorate. No universal extrapolation guarantee.
5. **Where does it break?** StructuralDFS shifts and long-path complete navigation; the tested threshold frontier above is more informative than a single board-size cutoff.
6. **Reusable iteration or depth-specific transformation?** Evidence supports reusable bounded local information exchange and some extrapolation, combined with strong local heuristics and training-depth sensitivity. It does not demonstrate a scalable shortest-path planning algorithm.

Recommended next study: prospective **local-alias paired-goal/branch-balanced mazes**, where identical local neighborhoods require opposite distant-goal actions. Train/test separate topologies, compare shared versus untied equal-hop updates, and require complete navigation improvement. A separately labeled distance-supervised variant could test whether missing planning representations, rather than recurrence stability, limit learning. Do not tune this consumed lockbox or launch that phase automatically.

## Reproduction and publication status

[Commands](../../experiments/experiment_05c/README.md) · [Raw summary](../../results/experiment_05c/summary.json) · [Public audit](../../results/experiment_05c/public_audit.json) · [Seed traces](../../results/experiment_05c/trajectories) · [Full episodes](../../results/experiment_05c/rollouts)

```sh
python -m latent_loop.scaling audit
python -m latent_loop.scaling report
LATENT_LOOP_MLX_TESTS=1 python -m pytest -q
```

CPU audit/report regeneration requires no checkpoints or MLX. Model inference/interventions require separately regenerated ignored weights; exact original identities are preserved locally. The final suite passed54 tests with MLX; CPU-only34 passed/14 skipped. The metadata export regression check is included. All10 datasets regenerated byte-identically. Original1–5B scientific files and both public tags remain unchanged. `v0.2.0`: [public release](https://github.com/johnathonkillaly/latent-loop-confidence/releases/tag/v0.2.0), commit `1ebd435b94448d45e42141b49808839218c2b7b2`. **Experiment5.2 stays local pending review.**
'''
    # Normalize prose spacing without changing paths/identifiers/measurement values.
    parts=re.split(r'(\[[^]]*\]\([^)]*\)|`[^`]*`|https?://\S+)',text)
    for i,part in enumerate(parts):
        if part.startswith(('[','`','http')):continue
        part=re.sub(r'(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])',' ',part)
        part=re.sub(r'(?<=\d)(?=GiB\b)',' ',part)
        for old,new in [('v 0.2.0','v0.2.0'),('AdamW0.001','AdamW 0.001'),('seeds 101/202/303,30','seeds 101/202/303, 30'),('batch 64,30','batch 64, 30'),('jointly,4,000','jointly, 4,000'),('StructuralDFS','Structural DFS'),('equally trainedFF8','equally trained FF8'),('local.53','local. 53'),('before outcomes.','before outcomes.'),('longDFS','long DFS'),('DFS9','DFS 9'),('DFS15','DFS 15'),('FF8','FF8'),('FF4','FF4')]:part=part.replace(old,new)
        parts[i]=part
    text=''.join(parts)
    (REPORT/'scaling_and_failure_frontier.md').write_text(text)
    body=markdown.markdown(text,extensions=['tables','fenced_code'])
    for p in (REPORT/'figures').glob('*.png'):body=body.replace(f'src="figures/{p.name}"',f'src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}"')
    html='<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Break the Thinker — Experiment5.2</title><style>body{max-width:1120px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#233143;background:#fafbfd}h1,h2{line-height:1.25}h2{margin-top:2em}table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto;margin:20px 0}td,th{padding:7px 10px;border-bottom:1px solid #d9dfe8;white-space:nowrap}th{background:#eaf0f7}img{max-width:100%;height:auto;background:white}a{color:#285d9d}code{font-size:13px}pre{padding:16px;background:#edf1f7;overflow:auto}</style></head><body>'+body+'</body></html>'
    (REPORT/'scaling_and_failure_frontier.html').write_text(html)
    write_json(OUT/'report_artifacts.json',{'generated':now(),'embedded_figures':len(list((REPORT/'figures').glob('*.png'))),'inputs':{str(OUT/p):digest(OUT/p) for p in ('summary.json','rollout_summary.json','descriptive_analysis.json','propagation_analysis.json','latency.json','train_depth_diagnostic.json')},'outputs':{str(p):digest(p) for p in REPORT.rglob('*') if p.is_file()}})
    print('Report/HTML generated from measurements,',len(pics),'figures',flush=True)
if __name__=='__main__':run()
