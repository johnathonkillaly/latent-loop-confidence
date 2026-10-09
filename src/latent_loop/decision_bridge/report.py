"""Generate scientific figures and self-contained reports from archived metrics."""
import base64
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import markdown
from .data import EXP,OUT,REPORT,config,read_rows,write

COLORS=['#2463a6','#dd7b22','#33906e','#8b5aa2','#b84251']
plt.rcParams.update({'figure.dpi':120,'savefig.dpi':160,'axes.spines.top':False,'axes.spines.right':False,'font.size':10,'svg.hashsalt':'decision-bridge-53'})

def pct(x):return f'{100*x:.2f}%'
def pp(r):return f'{100*r["estimate"]:+.2f} pp [{100*r["ci95"][0]:+.2f}, {100*r["ci95"][1]:+.2f}]'

def main():
    analysis=json.loads((OUT/'analysis.json').read_text());selection=json.loads((EXP/'selection.json').read_text());audit=json.loads((OUT/'dataset_audit.json').read_text());cfg=config();figdir=REPORT/'figures';figdir.mkdir(parents=True,exist_ok=True);figures=[]
    latency=json.loads((OUT/'latency_interleaved.json').read_text())
    def get(split,condition,depth,mode='base'):
        return next(r for r in analysis['summary'] if (r['split'],r['condition'],r['depth'],r['mode'])==(split,condition,depth,mode))
    def effect(split,condition,depth,mode='scored'):
        return next(r for r in analysis['scoring_effects'] if (r['split'],r['condition'],r['depth'],r['mode'])==(split,condition,depth,mode))
    def cost(split,condition,depth,mode='base',batch=1):
        return next(r for r in latency['records'] if (r['split'],r['condition'],r['depth'],r['mode'],r['batch'])==(split,condition,depth,mode,batch))
    def save(name,title):
        plt.gcf().suptitle(title);plt.tight_layout();plt.savefig(figdir/f'{name}.png',metadata={'Software':'Latent Loop Confidence'})
        svg=figdir/f'{name}.svg';plt.savefig(svg,metadata={'Date':None})
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close();figures.append((name,title))

    models=[('one_pass',1),('recurrent',1),('recurrent',2),('recurrent',4),('recurrent',8)];labels=['Trained one-hop','Recurrent 1','Recurrent 2','Recurrent 4','Recurrent 8']
    fig,ax=plt.subplots(figsize=(9,4));xx=np.arange(len(models));width=.36
    for j,mode in enumerate(('base','scored')):
        values=[100*get('id',c,d,mode)['accuracy'] for c,d in models];ax.bar(xx+(j-.5)*width,values,width,label=mode,color=COLORS[j])
        for i,(c,d) in enumerate(models):ax.scatter(np.full(3,xx[i]+(j-.5)*width),np.array(get('id',c,d,mode)['seed_accuracy'])*100,s=15,color='black',alpha=.6,zorder=4)
    ax.set(xticks=xx,xticklabels=labels,ylabel='Optimal-action accuracy (%)',ylim=(0,103));ax.legend();save('base_vs_scored','Fresh-ID decisions; black dots are the three seeds')

    fig,ax=plt.subplots(figsize=(8,4));effects=[effect('id',c,d) for c,d in models]
    ax.bar(xx-.18,[100*r['wrong_right'] for r in effects],.36,label='Wrong → right',color=COLORS[2]);ax.bar(xx+.18,[100*r['right_wrong'] for r in effects],.36,label='Right → wrong',color=COLORS[4]);ax.set(xticks=xx,xticklabels=labels,ylabel='Fraction of all decisions (%)');ax.legend();save('corrections_degradations','Real changes from frozen-base reranking')

    fig,ax=plt.subplots(figsize=(7,4));depths=cfg['depths']
    for split,color in zip(('id','community','size64','long'),COLORS):
        vals=[effect(split,'recurrent',d)['estimate']*100 for d in depths];ax.plot(depths,vals,'o-',label=split,color=color)
    id_effect=[effect('id','recurrent',d) for d in depths]
    ax.errorbar(depths,[100*r['estimate'] for r in id_effect],yerr=np.array([[100*(r['estimate']-r['ci95'][0]) for r in id_effect],[100*(r['ci95'][1]-r['estimate']) for r in id_effect]]),fmt='none',color=COLORS[0],capsize=3)
    ax.axhline(0,color='gray',lw=1);ax.set(xlabel='Shared recurrent iterations',ylabel='Scored − base accuracy (pp)',xticks=depths);ax.legend();save('scoring_benefit_by_depth','Validation-frozen fusion; ID intervals crossed by seed and graph')

    splits=['id','size16','size32','size64'];names=['ID 10/12','16','32','64'];fig,ax=plt.subplots(figsize=(7,4))
    for condition,depth,mode,label,color in [('one_pass',1,'base','One-hop',COLORS[0]),('recurrent',8,'base','Rec 8 (Q fusion identical)',COLORS[1]),('recurrent',16,'base','Rec 16 extrapolation',COLORS[2]),('ff8',8,'base','FF 8 width32',COLORS[3]),('ff8_params',8,'base','FF 8 width11',COLORS[4])]:ax.plot(names,[100*get(s,condition,depth,mode)['accuracy'] for s in splits],'o-',label=label,color=color)
    ax.set(xlabel='Graph nodes; fixed distance quotas, changing topology mix',ylabel='Optimal-action accuracy (%)',ylim=(0,103));ax.legend(fontsize=8);save('accuracy_by_graph_size','Size generalization does not establish arbitrary-topology planning')

    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,split in zip(axes,('id','long')):
        for depth,color in zip((1,4,8,16),COLORS):
            rows=sorted([r for r in analysis['strata'] if r['split']==split and r['condition']=='recurrent' and r['depth']==depth and r['mode']=='base' and r['statistic']=='distance' and r['level']>=0],key=lambda r:r['level'])
            ax.plot([r['level'] for r in rows],[100*r['accuracy'] for r in rows],'o-',label=f'{depth} loops',color=color)
        ax.set(title=split,xlabel='Actual BFS distance');ax.legend(fontsize=8)
    axes[0].set_ylabel('Optimal-action accuracy (%)');save('accuracy_by_distance','Distance strata; ID and long-path generators reported separately')

    fig,ax=plt.subplots(figsize=(7,4))
    for arm,color in zip((3,5,9,13),COLORS):
        rows=sorted([r for r in analysis['counterfactual'] if r['condition']=='recurrent' and r['mode']=='base' and r['arm_distance']==arm],key=lambda r:r['depth'])
        ax.plot([r['depth'] for r in rows],[100*r['both_correct'] for r in rows],'o-',label=f'Goal distance {arm+1}',color=color)
    ax.set(xticks=depths,xlabel='Internal loops',ylabel='Both counterfactual members correct (%)',ylim=(-3,103));ax.legend();save('counterfactual_pair_success','60 independent pairs per distance stratum, three shared seeds')

    fig,ax=plt.subplots(figsize=(7,4))
    for condition,mode,color,label in [('recurrent','base',COLORS[0],'Recurrent base'),('recurrent','scored',COLORS[1],'Recurrent + Q'),('one_pass','base',COLORS[2],'One-hop'),('ff8','base',COLORS[3],'FF 8 width32'),('ff8_params','base',COLORS[4],'FF 8 width11')]:
        ds=depths if condition=='recurrent' else [1] if condition=='one_pass' else [8]
        ax.plot([cost('id',condition,d,mode)['median_ms'] for d in ds],[100*get('id',condition,d,mode)['accuracy'] for d in ds],'o-',label=label,color=color)
    ax.set(xlabel='End-to-end batch-one latency (ms), 10-node timing case',ylabel='Fresh-ID accuracy (%)');ax.legend(fontsize=8);save('accuracy_latency','Randomized interleaved timing; normal desktop load')

    fig,ax=plt.subplots(figsize=(8,4));controls=[('one_pass',1,'base','One-hop'),('recurrent',8,'base','Rec8'),('recurrent',8,'scored','Rec8+Q (λ0)'),('ff8',8,'base','Untied FF8'),('ff8_params',8,'base','Narrow FF8'),('extra_budget',8,'base','Extra training')];x=np.arange(len(controls))
    for j,split in enumerate(('id','size64','long')):ax.bar(x+(j-1)*.25,[100*get(split,c,d,m)['accuracy'] for c,d,m,_ in controls],.25,label=split,color=COLORS[j])
    ax.set(xticks=x,xticklabels=[r[3] for r in controls],ylabel='Optimal-action accuracy (%)');ax.legend();save('recurrent_vs_feedforward','Ordinary feedforward depth remains a strong control')

    fig,axes=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True)
    for ax,depth in zip(axes,(1,8)):
        r=next(r for r in analysis['calibration'] if (r['split'],r['condition'],r['depth'],r['objective'])==('id','recurrent',depth,'correctness'))
        for mode,color in [('raw',COLORS[0]),('calibrated',COLORS[1])]:
            bins=r[mode]['bins'];ax.plot([b['probability'] for b in bins],[b['empirical'] for b in bins],'o-',label=f'{mode} ECE {r[mode]["ece"]:.3f}',color=color)
        ax.plot([0,1],[0,1],'--',color='gray');ax.set(title=f'{depth} loops',xlabel='Candidate correctness probability');ax.legend(fontsize=8)
    axes[0].set_ylabel('Observed optimal-set membership');save('candidate_calibration','Candidate calibration is separate from deployed action improvement')

    datasets={s:{r['id']:r for r in read_rows(OUT/'datasets'/f'{s}.jsonl.gz')} for s in cfg['final_counts']};examples=[]
    fig,axes=plt.subplots(2,2,figsize=(10,9))
    targets=[(401,1,False,True),(401,1,True,False),(502,2,False,True),(502,2,True,False)]
    for ax,(seed,depth,before,after) in zip(axes.flat,targets):
        records=read_rows(OUT/'decisions'/f'id_recurrent_seed{seed}.jsonl.gz');match=next((r for r in records if r['depth']==depth and r['correct']['base']==before and r['correct']['scored']==after),None)
        if match is None:ax.axis('off');ax.text(.1,.5,'No case in this prespecified category');continue
        row=datasets['id'][match['id']];o=row['observation'];n=o['n'];angle=np.arange(n)*2*np.pi/n;xy=np.column_stack([np.cos(angle),np.sin(angle)])
        for u,v in o['edges']:ax.plot(xy[[u,v],0],xy[[u,v],1],color='#b8c1ca',lw=1,zorder=1)
        colors=['#e1e7ed']*n;colors[o['start']]=COLORS[1];colors[o['goal']]=COLORS[2]
        ax.scatter(xy[:,0],xy[:,1],s=240,c=colors,edgecolors='white',zorder=2)
        for i in range(n):ax.text(*xy[i],str(i),ha='center',va='center',fontsize=9,zorder=3)
        annotations=[]
        for mode,color in [('base',COLORS[0]),('scored',COLORS[4])]:
            v=o['candidates'][match['decisions'][mode]];annotations.append(f'{mode}: {v if v>=0 else "abstain"} ({"correct" if match["correct"][mode] else "wrong"})')
            if v>=0:ax.annotate('',xy=xy[v],xytext=xy[o['start']],arrowprops={'arrowstyle':'->','color':color,'lw':2.5},zorder=4)
        ax.set_title(f'Seed {seed}, loop {depth}\n'+', '.join(annotations),fontsize=10);ax.set_aspect('equal');ax.axis('off');examples.append({'id':row['id'],'seed':seed,'depth':depth,'base_correct':before,'scored_correct':after,'source':f'id_recurrent_seed{seed}.jsonl.gz'})
    save('changed_decisions','Demonstration cases, first match per category; circular layout is display only');write(OUT/'representative_examples.json',examples)

    fig,ax=plt.subplots(figsize=(8,4));splits=['id','community','size16','size32','size64','long'];x=np.arange(len(splits))
    for j,(condition,depth,label) in enumerate([('recurrent',8,'Rec8 (scored identical)'),('recurrent',16,'Rec16'),('ff8',8,'FF8'),('ff8_params',8,'Narrow FF8')]):
        vals=[next(r['goal_completion_reachable'] for r in analysis['rollouts'] if (r['split'],r['condition'],r['depth'],r['mode'])==(s,condition,depth,'base')) for s in splits]
        ax.bar(x+(j-1.5)*.2,np.array(vals)*100,.2,label=label,color=COLORS[j])
    ax.set(xticks=x,xticklabels=splits,ylabel='Reachable-episode goal completion (%)');ax.legend(fontsize=8);save('navigation_completion','30 initial graphs per split; repeated positions terminate deterministic cycles')

    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for condition,color in [('one_pass',COLORS[0]),('recurrent',COLORS[1]),('ff8',COLORS[2])]:
        records=[json.loads((OUT/'training'/f'{condition}_seed{s}.json').read_text()) for s in cfg['seeds']]
        for ax,kind in zip(axes,('loss','validation')):
            vals=np.array([[e['loss'] if kind=='loss' else e['validation']['loss'] for e in r['epochs']] for r in records]);ax.plot(np.arange(1,31),vals.mean(0),label=condition,color=color)
            ax.fill_between(np.arange(1,31),vals.min(0),vals.max(0),color=color,alpha=.12);ax.set(xlabel='Epoch',ylabel='Optimal-set NLL',title='Training' if kind=='loss' else 'Validation');ax.legend(fontsize=8)
    save('training_validation_loss','Seed range bands; sampled-depth training and fixed deployment validation')

    e1=effect('id','recurrent',1);e2=effect('id','recurrent',2);resources=analysis['resources'];comparison=analysis['comparisons'];primary=analysis['primary'];manifest=json.loads((EXP/'dataset_manifest.json').read_text())
    memory=max(json.loads((OUT/name).read_text()).get('peak_mlx_gib',0) for name in ('evaluation_receipt.json','rollout_receipt.json','latency.json','latency_interleaved.json'))
    memory=max(memory,resources['max_training_mlx_gib']);rss=max(resources['max_training_rss_gib'],*[json.loads((OUT/name).read_text()).get('peak_rss_gib',0) for name in ('evaluation_receipt.json','rollout_receipt.json','latency_interleaved.json')])
    tables=[]
    for split in cfg['final_counts']:
        tables.append('| '+split+' | '+str(manifest['splits'][split]['count'])+' | '+' | '.join(pct(get(split,c,d,m)['accuracy']) for c,d,m in [('one_pass',1,'base'),('one_pass',1,'scored'),('recurrent',8,'base'),('recurrent',16,'base'),('ff8',8,'base'),('ff8_params',8,'base')])+' |')
    depth_table='\n'.join('| '+str(d)+' | '+' | '.join(pct(get('id','recurrent',d,m)['accuracy']) for m in ('base','scored','correctness_only','ranking_only','direct_only'))+' | '+str(selection['conditions']['recurrent']['by_depth'].get(str(d),selection['conditions']['recurrent'])['lambda'])+' |' for d in cfg['depths'])
    correction_table='\n'.join(f'| {d} | {pct(effect("id","recurrent",d)["wrong_right"])} | {pct(effect("id","recurrent",d)["right_wrong"])} | {pct(effect("id","recurrent",d)["both_right"])} | {pct(effect("id","recurrent",d)["both_wrong"])} | {pp(effect("id","recurrent",d))} |' for d in cfg['depths'])
    negative_table='\n'.join('| '+name+' | '+' | '.join(pct(get('id',c,d,m)['accuracy']) for m in ('base','temperature','constant','shuffled','scored','forced_nonzero'))+' |' for c,d,name in [('one_pass',1,'Trained one-hop'),('recurrent',1,'Rec1'),('recurrent',2,'Rec2'),('recurrent',8,'Rec8')])
    ranking_table='\n'.join(f'| {r["depth"]} | {r["objective"]} | {r["base_mean_within_graph_ranking_auc"]:.3f} | {r["mean_within_graph_ranking_auc"]:.3f} | {r["raw"]["ece"]:.3f} | {r["calibrated"]["ece"]:.3f} |' for r in analysis['calibration'] if r['split']=='id' and r['condition']=='recurrent' and r['depth'] in (1,2,8) and r['objective'] in ('correctness','ranking'))
    rollout_table='\n'.join('| '+s+' | '+' | '.join(pct(next(r['goal_completion_reachable'] for r in analysis['rollouts'] if (r['split'],r['condition'],r['depth'],r['mode'])==(s,c,d,'base'))) for c,d in [('recurrent',8),('recurrent',16),('ff8',8),('ff8_params',8)])+' |' for s in cfg['final_counts'])
    cf_table='\n'.join('| '+str(arm+1)+' | '+' | '.join(pct(next(r['both_correct'] for r in analysis['counterfactual'] if (r['condition'],r['depth'],r['mode'],r['arm_distance'])==('recurrent',d,'base',arm))) for d in cfg['depths'])+' |' for arm in (3,5,9,13))
    paramtable='''| One-hop / shared recurrent / extra-budget base | 11,681 | all during base fitting; frozen for Q |
| Candidate correctness, ranking or direct head | 2,241 | head only; frozen base |
| Base plus candidate head | 13,922 | 2,241 in the second stage |
| Untied eight-block width32 | 77,313 | all |
| Narrow untied eight-block width11 | 10,008 | all |'''
    text=f'''# Experiment 5.3 — The Decision Bridge

## Executive summary

**A candidate scorer can repair a frozen shallow readout, but this study does not establish a special scoring-plus-recurrence advantage.** The prespecified primary eight-loop frozen recurrent policy scores {pct(get('id','recurrent',8)['accuracy'])} on 600 fresh topology-disjoint graphs. Validation selected correctness scoring with **lambda 0**, so its fused policy makes identical decisions: {pp(primary)}. This degenerate interval reflects an identity intervention, not proof that every scorer is equivalent. The forced nonzero diagnostic also has zero net ID gain. No primary scoring improvement is established.

At one loop of the *same recurrent model*, scoring improves {pct(get('id','recurrent',1)['accuracy'])} to {pct(get('id','recurrent',1,'scored')['accuracy'])}: **{pp(e1)}**, secondary and unadjusted. Corrections are {pct(e1['wrong_right'])} and degradations {pct(e1['right_wrong'])} of all decisions. Seed gains are +14.33/+0.33/+1.83 pp: one seed dominates. A matched direct classifier reaches {pct(get('id','recurrent',1,'direct_only')['accuracy'])}; scored minus direct is {pp(comparison['rec1_scored_vs_direct'])}. This is readout repair with extra supervision, not evidence of a uniquely valuable auxiliary architecture. The independently trained one-hop base is stronger than the shallow recurrent base and its fusion gain is {pp(comparison['one_pass_scoring'])}.

Repeated local computation is useful. On the prespecified distance-six counterfactual population, both members are correct in 0% of pairs at two loops and 100% at eight (60 independent pairs, three seeds). Yet eight-loop recurrence achieves only {pct(get('size64','recurrent',8)['accuracy'])} on 64 nodes and {pct(get('long','recurrent',8)['accuracy'])} on long 32-node paths; sixteen loops reach {pct(get('long','recurrent',16)['accuracy'])}. Ordinary feedforward controls remain competitive or better. No LLM was trained or integrated. Nothing was pushed or published.

Completed **36 main fits** (15 base/control + 21 frozen-representation heads), **36 easy-distribution pilot fits**, and **two independent seed-401 verification refits** (74 total). The independent base and scorer checkpoints reproduce byte-for-byte; [verification receipt](../../results/experiment_05d/seed401_retraining_verification.json). Main synchronized epoch fitting is {resources['main_epoch_fit_seconds']:.2f} seconds; pilot fitting {resources['pilot_epoch_fit_seconds']:.2f} seconds, verification refitting3.86 seconds. Peak main measured MLX allocation {memory:.3f} GiB; stage RSS maximum {rss:.3f} GiB (different accounting, not additive). [Complete metrics](../../results/experiment_05d/analysis.json).

## Original motivation and relationship to C/V/D

[Historical C/V/D](../experiment_04b/decision_value_and_trajectory_selection.md) predicted correctness, loss improvement and decision transitions without reliable deployed gains. [Experiment 5](../experiment_05/train_the_thinker.md) did not establish useful extra Qwen recurrence; [5B](../experiment_05b/learning_to_navigate.md) demonstrated useful tiny-grid depth but tied feedforward controls. [5.2](../experiment_05c/scaling_and_failure_frontier.md) separated information propagation from successful long-horizon navigation. Those negative records remain untouched.

Here Q evaluates **each candidate**, including alternatives not selected by the base. C estimates selected-answer correctness; V and D concern future computation. Q is a separate candidate-supervision objective. Successful graph Q does not establish old C transfer, language-model reasoning, or an LLM decision improvement.

## Graph construction, splits and leakage checks

Base training: 3,000 graphs; disjoint scorer training: 3,000; validation: 600. Training uses 10/12 nodes, trees, sparse random-tree-plus-edge graphs, denser cyclic graphs and disconnected components. Starts have at least two neighbors. Quotas per size: 25% distance 1–2, 40% distance 3–4, 20% distance 5–9, 15% unreachable. These are diagnostic prevalence choices. Community shift uses two dense connected modules with a bridge (400 graphs, distance at least two); long shift uses a chain plus sparse chords (300 32-node graphs, distance at least ten). All labels use BFS, independently checked against Floyd–Warshall. Every equally optimal action is accepted; no-route cases target abstain. BFS itself is the exact 100% reference, never a neural feature.

Final sets contain 2,240 independent ordinary graphs and 480 counterfactual members (240 independent pairs). Total main data: 9,320 observations. Node IDs and presentation order are random; candidates are random-order neighbors plus abstain. Features are constant/start/goal/degree divided by N−1. No coordinates, paths, distances, correctness, outcomes or IDs enter neural inference. Candidate tie keys are independent random metadata, never learned features; they follow semantic identity for presentation-independent numerical tie resolution.

Whole unrooted graph WL invariants reject all matching fingerprints across splits, including non-isomorphic collisions. This certifies exclusion of isomorphic copies, not that equal fingerprints prove isomorphism. Counterfactual members intentionally share a group; groups never cross splits. Conservatism and finite topology inventories alter family prevalence: base/scorer/final-ID tree counts are **250/94/13**, despite identical nominal proposal rules. ID is fresh under the same quota/generator procedure, not exactly matched family proportions. Larger sparse graphs also change degree/topology composition. Report size and distance strata separately; do not attribute every shift solely to N.

The initial unconditioned pilot put 77% of examples at distance 1–2. Its 36 fits were completed before recognizing near-ceiling one-hop validation; all data/source/metrics are preserved at [pilot receipt](../../experiments/experiment_05d/pilot_easy_distribution/receipt.json) and `results/experiment_05d_pilot_easy/`. No pilot final neural predictions were made. The quota amendment preceded all main fitting and final comparisons. [Dataset manifest](../../experiments/experiment_05d/dataset_manifest.json) · [independent CPU byte regeneration](../../results/experiment_05d/independent_data_reproduction.json).

## Base and recurrent architectures

Pointwise four-feature encoder, width32; bounded shared GRU-style update using current state, original encoding and mean neighbor states. Sigmoid gates mix the previous bounded state with a layer-normalized tanh proposal. One update crosses one graph edge. The ordinary shared candidate MLP reads start state, neighbor state and abstain flag; abstain has zero neighbor state. The head sees information one edge beyond the start, so a goal at graph distance d cannot influence a candidate before d−1 updates. Raw degree is observed initially; paired constructions match those local degree features too.

Shared weights train at sampled depths 1/2/4/8. Hidden state resets each decision; internal loops never move the environment. There is no global pooling, controller, across-step memory, shortest-path auxiliary supervision or deterministic planning map. The untrained inputs and observation remain unchanged through recurrence. Eight untied blocks provide the same hop count with distinct weights. Width11 is a **narrow, approximately capacity-matched control with 14.3% fewer parameters**, not an exact parameter match. Candidate/direct-head comparisons are exactly matched at 2,241 parameters.

| Component | Parameters | Trainable stage |
|---|---:|---|
{paramtable}

## Separate training and candidate scoring

Seeds 401/502/603; 30 epochs, batch64, AdamW 0.001, decay0.0001, global gradient clip1. All base conditions receive 90,000 example presentations per seed; recurrence samples one depth per batch. Best base checkpoint minimizes validation optimal-set NLL (depth1 one-hop; depth8 others). All selected base hashes are frozen before any Q fit. [Base freeze](../../experiments/experiment_05d/base_freeze.json).

Q fits on **disjoint graph topologies** using cached frozen representations; no gradient reaches the base. It receives the same start/candidate/abstain representation as the action head, plus base probability, centered clipped base logit and known depth/8. A shared width32 SiLU MLP returns one score per option, with variable candidate counts. No fixed action-letter output parameters exist.

Correctness: candidate BCE, candidate-weighted. Ranking: mean positive-versus-negative pairwise logistic loss, graph-weighted. Direct classifier: negative log total probability of the optimal set, graph-weighted. These heads have identical observable inputs, architecture, size and training presentation budgets; the direct head is a decisive control for extra supervised capacity. Scorers sample one cached depth 1/2/4/8 per batch on recurrence (one-hop uses only depth1). No scorer at depth16 was trained. Checkpoint selection minimizes the corresponding depth-eight validation objective, depth-one for one-hop.

A separate recurrent **extra-budget** control starts from the selected frozen checkpoint, then trains all 11,681 parameters on scorer-training graphs for another 90,000 presentations. It does not replace the frozen base or contribute features to Q. Equal presentations do not mean equal wall time or optimization difficulty. No joint base/Q fit, dependency task or adaptive stopping was added.

![training and validation](figures/training_validation_loss.png)

## Fusion, calibration and prospective selection

Base chooses argmax P. Fusion chooses argmax(log P + lambda Q_norm). Q_norm subtracts the candidate mean and divides by max(1, candidate RMS); normalization removes arbitrary offsets and caps score-scale sensitivity without making Q a probability. Temperature is selected separately by validation optimal-set NLL and cannot improve argmax accuracy. Candidate probabilities use raw sigmoid and a validation-only positive-scale/intercept calibration; calibration metrics do not establish ranking or decision gains.

Validation chooses between correctness and ranking and the lambda grid 0/.1/.25/.5/1/2/4; ties prefer lower lambda, then correctness. Primary recurrent depth8 selects correctness/lambda0. Its secondary depth1/2/4 weights are 4/2/.25, fixed on validation. Depth16 reuses depth8 lambda0. The one-hop model selects ranking/.1. Direct and score-only policies, a validation-selected nonzero intervention, shuffled scores and constant scores are retained. [Selection grids](../../experiments/experiment_05d/selection.json).

The [prospective protocol](../../experiments/experiment_05d/protocol.md), [freeze](../../experiments/experiment_05d/frozen_protocol.json) and [opening receipt](../../experiments/experiment_05d/lockbox_opened.json) precede final comparisons. No model, checkpoint, objective, depth or lambda changed after opening. A numerical tie bug made the initial temperature control change a near-tied decision. Three partial trace files were archived, then **all** evaluations rerun after float64 tie comparisons and retaining unscaled base tie resolution for positive temperature. Original frozen source, exact corrected hashes and failure are preserved in the [evaluation amendment](../../experiments/experiment_05d/evaluation_amendment.json). This is an implementation correction, not test-set selection.

## Decision results and correction analysis

Primary: {pp(primary)}, with no actual primary intervention because lambda0. A nonzero weight .1 changes one seed/example decision between equally optimal options, with no correction or degradation. Extra Q compute is not justified at the selected eight-loop operating point.

| ID recurrent depth | Base | Fused | Q correctness only | Q ranking only | Direct classifier | Frozen lambda |
|---|---:|---:|---:|---:|---:|---:|
{depth_table}

![base versus scored](figures/base_vs_scored.png)

| Depth | Wrong→right | Right→wrong | Both right | Both wrong | Net paired effect, 95% CI |
|---|---:|---:|---:|---:|---|
{correction_table}

![correction and damage](figures/corrections_degradations.png)

| ID model | Base | Temperature | Constant Q | Shuffled Q | Validated fusion | Forced nonzero |
|---|---:|---:|---:|---:|---:|---:|
{negative_table}

Temperature and constant scores preserve every base choice. Shuffling uses a fixed
independent RNG and the same validated fusion weight; when lambda0 it is necessarily
inert. A nonzero intervention can change semantics without changing correctness
when both options are optimal.

The shallow gain is secondary, varies markedly by seed, and does not beat the matched classifier. Depth2 fusion gives {pp(e2)} (inconclusive); scored minus direct at depth2 is {pp(comparison['rec2_scored_vs_direct'])}. One-hop scoring is {pp(comparison['one_pass_scoring'])}. Q can alter real outputs and sometimes repair a biased frozen readout; no advantage over ordinary candidate classification is established.

Expected uniform-candidate accuracy is {pct(analysis['classical_baselines']['id']['uniform_candidate_expected_accuracy'])} on ID, {pct(analysis['classical_baselines']['size64']['uniform_candidate_expected_accuracy'])} on64 nodes and {pct(analysis['classical_baselines']['long']['uniform_candidate_expected_accuracy'])} on long paths. Tied optimal actions and candidate count change this reference across populations. Always abstaining achieves15% in the quota-based sets and0% on community/long/counterfactual sets; BFS achieves100% in every population.

![scoring benefit versus depth](figures/scoring_benefit_by_depth.png)

Three-seed/graph crossed bootstrap intervals use 3,000 replicates. Repeated model outputs on one graph do not increase the independent graph count. Counterfactuals cluster both members. Secondary intervals are unadjusted, and three seeds poorly characterize optimization variability. Exact 0/100% sample rates yield degenerate bootstrap intervals; these do not imply certainty on unobserved graphs or equivalence across populations.

## Counterfactual distant-dependency test

Two identical start arms end in interchangeable goal/distractor attachments. Swapping only the distant attachments flips the optimal first candidate while preserving local degrees, flags and candidate ordering. A random remote branch module supplies topology diversity; some graphs have irrelevant isolated nodes. Both pair members are independently solver-verified, with unique opposite optimal actions. There are 60 pairs per goal distance 4/6/10/14. Pair inputs are exactly indistinguishable to the local readout before the prescribed information radius; shallow logits are exactly equal in all recorded locality checks.

| Goal distance | Both correct at 1 | 2 | 4 | 8 | 16 |
|---|---:|---:|---:|---:|---:|
{cf_table}

![counterfactual pair success](figures/counterfactual_pair_success.png)

The prespecified distance-six 8−2 comparison is {pp(comparison['counterfactual_confirmation'])}, 60 pair clusters. This supports H1: additional local updates can incorporate distant goal information. It does not prove a learned BFS algorithm, H2 candidate-score benefit, or long-horizon planning. Before the signal arrives, the model often wrongly abstains rather than guessing an arm. Even after more loops, longer pairs and unfamiliar graph structures remain difficult. No scorer creates missing information at shallow depths.

## Generalization and feedforward controls

| Split | Graphs | Trained one-hop | One-hop+Q | Rec8 (Q identical) | Rec16 | FF8 width32 | Narrow FF8 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(tables)}

![graph size](figures/accuracy_by_graph_size.png)

![dependency distance](figures/accuracy_by_distance.png)

Rec8 minus independently trained one-hop ID is {pp(comparison['recurrence_only'])}; combined versus one-hop is identical. The scoring/recurrence interaction is {pp(comparison['interaction'])}, inconclusive. Both feedforward controls score 100% on ID. The same-width eight-block model reaches {pct(get('size64','ff8',8)['accuracy'])} on64; the narrower control reaches {pct(get('size64','ff8_params',8)['accuracy'])}. These final descriptive results do not re-select the deployment model. Extra-base training scores {pct(get('size64','extra_budget',8)['accuracy'])} on64, illustrating harmful shift sensitivity despite perfect ID validation.

![feedforward controls](figures/recurrent_vs_feedforward.png)

H1 is supported within the paired mechanism test. H2 has a limited secondary shallow-readout result, with no primary gain. H3 and H5 are not established. H4 is partial: permutations and the small community shift work, but scalable long-distance planning and robust scorer benefit do not follow.

## Full graph navigation

At each environmental action, reset hidden state, observe raw adjacency/start/goal, run a fixed internal budget, optionally score the *current* neighbor set, commit one action, then move. No labels/BFS distances or future states enter the policy. Same stable tie keys and deterministic candidate shuffling are used across models. Stop at goal, abstain, repeated node (deterministic cycle), or128 actions. Stopping at a cycle preserves deterministic completion; measured failure time is time to detect it, not wasted timeout iterations. Across-action memory is absent.

Thirty initial graphs per split, three seeds; goal completion below conditions on reachable instances. These are the first30 deterministic examples fixed by the evaluator, not a random episode sample. In quota-based splits they contain10 near/13 medium/4 far/3 unreachable cases, underrepresenting the far/unreachable strata relative to the full population. Completion estimates apply to this small subset. No-route abstention and total task success are separately retained. Successful-path ratios and action traces are public; ratios omit failures and must not be read alone.

| Split | Rec8 reachable completion | Rec16 | FF8 | Narrow FF8 |
|---|---:|---:|---:|---:|
{rollout_table}

![navigation](figures/navigation_completion.png)

Rec8+Q navigation is identical to Rec8 because lambda0. Long-path recurrence completes no eight-loop episodes and only22.22% at16; first-action accuracy alone is insufficient. The rollout subset is small and not a new model-selection population. Individual environmental actions are not independent statistical trials. [All episode summaries](../../results/experiment_05d/analysis.json).

## Permutations, calibration and representative decisions

All 1,620 relabeling/candidate-order checks preserve the semantic base and scored choice. Maximum observed FP32 logit change is8.59e−6; mathematical equivariance is tested numerically with tolerances, not bitwise node-summation identity. No inference observation mutates during loops. Gradient tests verify finite nonzero recurrent updates and bounded states; weight and optimizer restoration reproduce subsequent updates exactly. [Permutation/locality records](../../results/experiment_05d/invariance_and_locality.json).

![candidate calibration](figures/candidate_calibration.png)

| ID depth | Objective | Base within-graph ranking AUC | Q within-graph ranking AUC | Raw candidate ECE | Validation-calibrated ECE |
|---|---|---:|---:|---:|---:|
{ranking_table}

Calibration and ranking are reported separately for each objective/depth/split. High candidate AUROC does not demonstrate better deployed decisions; raw sigmoid of ranking/direct scores is not intrinsically a calibrated correctness probability. Validation calibration is shared across depths and can fail under shift.

![representative changed decisions](figures/changed_decisions.png)

Cases are the first archived correction/degradation in each labeled seed/depth category, not a claim about prevalence. Orange marks start, green goal; blue/red arrows are base/scored choices. The circular drawing is solely visualization, not a model coordinate input. [Exact case IDs](../../results/experiment_05d/representative_examples.json).

## Runtime and memory

MLX0.32.3/Metal, FP32, Python3.12.15, Apple M4 Max/128GiB. No new dependencies, Qwen loading/training, remote workstation access, or interruption of unrelated processes. Budget24GiB; main allocator maximum {memory:.3f}GiB. Trainable tiny models are under80,000 parameters. Main training/validation epoch time {resources['main_epoch_fit_seconds']:.2f}s, pilot {resources['pilot_epoch_fit_seconds']:.2f}s; generation, evaluation, checkpoint I/O, reporting and verification refits are separate. No idle calendar time is counted as training.

Randomized interleaved batch-one medians for the same ten-node timing case: one-hop {cost('id','one_pass',1)['median_ms']:.3f}ms, recurrent1 {cost('id','recurrent',1)['median_ms']:.3f}ms, recurrent1+Q {cost('id','recurrent',1,'scored')['median_ms']:.3f}ms, recurrent8 {cost('id','recurrent',8)['median_ms']:.3f}ms, recurrent8+Q {cost('id','recurrent',8,'scored')['median_ms']:.3f}ms, untiedFF8 {cost('id','ff8',8)['median_ms']:.3f}ms. Q adds a paired median {cost('id','recurrent',8,'scored')['paired_added_ms']:.3f}ms at8. A deployment could skip Q when lambda0; the benchmark deliberately measures its execution. Batch8 and32/64-node timings are included in the archive.

Timing includes raw observation preparation, transfer, synchronized model/Q, CPU fusion and action selection; excludes checkpoint loading. Ten warmups/50 paired random-order repetitions, first prespecified seed, normal desktop load. The original 3-warmup/20-repeat serial-order archive had scorer-faster artifacts and remains preserved; interleaved measurements supplement it. No speedup or sharing advantage is inferred from parameter count or noisy timing alone.

![latency](figures/accuracy_latency.png)

## Negative findings and limitations

1. Primary Q fusion is disabled by validation and supplies no decision gain at8. No architectural complementarity is established.
2. Direct classification matches the useful shallow scorer; a single seed contributes most of the mean shallow repair. That result does not beat the independently trained one-hop model.
3. Distant absent goal signals are confused with no route. On long graphs, extra computation can still fail to deliver reliable optimal actions or completion.
4. Narrow and untied feedforward controls remain strong, including better64-node and long-path performance. Recurrence is not uniquely effective.
5. Perfect small-graph ID accuracy is a ceiling, not general reasoning. Topology rejection changes family prevalence; larger graphs change degree distributions and normalized-degree features. Long and community shifts change generator, not only distance.
6. Three seeds, small episode samples, many unadjusted secondary comparisons, tied optimal sets, diagnostic prevalence and bounded graph families limit statistical/external inference. No general theorem or learned-BFS claim is supported.
7. No dependency task, adaptive controller, learned recurrent scorer, language-model integration or claim of direct Q-weight transfer was attempted. Full independent36-fit replication is not claimed unless separately recorded; a seed-specific verification is provided.

## Reproduction commands and publication status

Run from repository root with existing isolated Apple environment. No model downloads are needed for5.3. Original code/synthetic data use the project MIT license; no third-party model/code is vendored. Checkpoints, optimizer states and feature caches stay ignored under artifacts. Recreate them before live neural evaluation; public scalar audits and reports do not need them.

```sh
LATENT_LOOP_MLX_TESTS=1 PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src artifacts/publication/cpu-env/bin/python -m latent_loop.decision_bridge.public_audit
PYTHONPATH=src artifacts/publication/cpu-env/bin/python -m latent_loop.decision_bridge.reproduce_data
PYTHONPATH=src artifacts/publication/cpu-env/bin/python -m latent_loop.decision_bridge analyze
PYTHONPATH=src artifacts/publication/cpu-env/bin/python -m latent_loop.decision_bridge report
PYTHONPATH=src .venv/bin/python -m latent_loop.decision_bridge example
```

The sealed historical directory refuses new fitting. To reproduce training without deleting receipts or overwriting results:

```sh
# New empty destination; default verifies recurrent seed401 plus its BCE scorer.
PYTHONPATH=src .venv/bin/python -m latent_loop.decision_bridge.retrain --destination artifacts/experiment_05d/new_verification
# Full independent protocol in another fresh workspace (not run automatically).
PYTHONPATH=src .venv/bin/python -m latent_loop.decision_bridge.retrain --all --destination artifacts/experiment_05d/full_reproduction
```

For a fresh study the staged CLI is generate → train → freeze → evaluate → rollout → latency → diagnostics → analyze → report. Individual train fits accept --condition, --seed and --objective; every epoch saves last/best weights and optimizer state with source/data signatures. Fits stop after lockbox opening. Hash checks permit only the documented evaluation correction; model/config/data selection remains immutable.

Public reports/charts and all ten datasets regenerated on CPU; source/outcome checks and actual test counts are in [verification](../../results/experiment_05d/verification.json). Live MLX checks and seed-specific independent retraining are separately recorded. Public release tags and all historical research artifacts remain unchanged. This branch is local, **not published**; review the research caveats before any future publication.

## Rendezvous handoff

The reusable contribution is the candidate-scoring contract, frozen-base supervision and decision-impact evaluation. Graph message passing/adjacency/readouts are task-specific. An LLM would require fresh hidden/candidate features, new labels and newly trained/calibrated scoring weights. [Minimal proposed5.4 design and answers](rendezvous_handoff.md). No5.4 implementation was started.
'''
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'decision_bridge.md').write_text(text)
    handoff=f'''# Experiment 5.4 — Rendezvous handoff (design only)

1. **Does scoring improve decisions?** Not on the primary8-loop operating point: {pp(primary)}, lambda0. A secondary frozen recurrent1 readout improves {pp(e1)}, with one seed dominating. Independently trained one-hop fusion is inconclusive.
2. **Beyond direct classification?** No. Recurrent1 scored versus matched direct is {pp(comparison['rec1_scored_vs_direct'])}; both have2,241 extra trainable parameters, identical observations/data budgets. Score-only direct classification is a necessary baseline, not a lesser mechanism.
3. **Does recurrence supply information?** Yes in the specified paired task: distance-six pairs go0→100% both-correct from2→8 loops, and shallow logits are identical before information arrives. This demonstrates useful local communication, not general BFS or reasoning.
4. **Complementarity?** Not established. Eight-loop base is at the small-graph ceiling, so validation disables fusion. Shallow repair is consistent with a biased readout being retrained, and does not beat controls.
5. **Topology generalization?** Small community shift is strong, but64-node rec8 accuracy {pct(get('size64','recurrent',8)['accuracy'])} and long-path rec8 {pct(get('long','recurrent',8)['accuracy'])} show limits. Extra loops help some examples without delivering reliable scalable planning. Node/candidate permutations preserve all1,620 tested semantic choices.
6. **Reusable interfaces?** StateEncoder, BasePolicy, CandidateScorer, RecurrentUpdater and DecisionController protocols in [interface.py](../../src/latent_loop/decision_bridge/interface.py). Variable candidate sequences, arbitrary observable representations, original logits, known budget/depth, explicit fusion and stable tie handling. The example is a toy contract demonstration, not a trained LLM result.
7. **Graph-specific parts?** Adjacency batching, node flags/degrees, message-passing recurrence, start/neighbor extraction, BFS labels and navigation simulator. None should be presented as a transformer reasoning mechanism. Q weights expect68 features and graph-specific distributions; direct transfer to LLM hidden dimensions or token embeddings is unjustified.
8. **Best objective?** Validation selects ranking for separately trained one-hop fusion, correctness for recurrence (at8 with zero weight). Shallow BCE/ranking/listwise heads are competitive, with no generally winning objective. Calibration is separate from ranking; validation lambda0 is an acceptable result.
9. **First LLM objective?** Candidate optimal-set supervision with a frozen small shared scorer, comparing BCE against a matched direct/listwise head. Start with BCE for interpretable per-candidate correctness, plus positive/negative ranking as one controlled alternative. Never assume these graph weights transfer.
10. **C/V/D ideas?** C remains a selected-answer calibration diagnostic; Q explicitly evaluates alternatives. V and D can later label prospective loss/decision transitions, but require fresh representations/splits and cannot justify an ineffective recurrent mechanism. No adaptive controller is warranted by these results alone.
11. **Failure modes?** Missing information cannot be recovered by a new readout; absent goal propagation can induce false abstention; reranking damages some correct answers; scorer training helps mostly one shallow seed; direct heads tie; larger-degree/topology shifts hurt; extra training harms extrapolation; runtime overhead can exceed any accuracy gain.
12. **What justifies5.4?** A tightly bounded falsification pilot, not an expectation of success: demonstrate held-out net corrections over the frozen model **and** a matched direct classifier, with prospective candidate coverage, disjoint head data, paired intervals and full-model latency. Stop if gains disappear with the additional-supervision control or on topology/task shift.

## One minimal LLM experiment to review

Use the existing pinned frozen Qwen3-0.6B on one objectively labeled small structured-action task with variable candidates. Preserve the original candidate logits and observable hidden state; do not add recurrence initially. Split fresh problems into base-feature/scorer training, validation and locked structural shift; do not reuse consumed maze/graph tests. Train only a small shared candidate scorer, plus an identically featured listwise classifier and extra-head/data-budget control. Fix candidate generation/coverage, lambda (including0), calibration and a single primary population before final opening. Compare base argmax, temperature-only unchanged argmax, Q-only and fusion; report corrections/degradations, seed variability, calibration and end-to-end model latency. Only if scoring beats these controls should a second prespecified stage add safe middle-layer computation. No labels or future states at runtime.

This is a proposal only. No LLM integration, new transformer training, controller or5.4 code was executed. Experiment5.2/5.3 remain local pending separate review.
'''
    (REPORT/'rendezvous_handoff.md').write_text(handoff)
    for name,content in [('decision_bridge',text),('rendezvous_handoff',handoff)]:
        body=markdown.markdown(content,extensions=['tables','fenced_code'])
        def embed(match):
            p=REPORT/match.group(1)
            return 'src="data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()+'"' if p.exists() and p.suffix=='.png' else match.group(0)
        body=re.sub(r'src="([^"]+)"',embed,body)
        (REPORT/f'{name}.html').write_text('<!doctype html><meta charset="utf-8"><title>Latent Loop Confidence — '+name+'</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:48px auto;padding:0 24px;color:#1c2b38}img{max-width:100%}table{border-collapse:collapse;font-size:14px}th,td{border:1px solid #ccd5de;padding:8px}pre{overflow:auto;background:#f1f4f7;padding:18px}h1,h2{line-height:1.25}a{color:#2463a6}</style>'+body)
    write(OUT/'figure_manifest.json',{'figures':[{'name':n,'title':t,'source':'analysis.json; latency_interleaved.json; archived raw decisions/training'} for n,t in figures],'count':len(figures)})
    print('Generated',len(figures),'figures and Markdown/self-contained HTML reports')

if __name__=='__main__':main()
