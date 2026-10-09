"""CPU-only analysis of archived decisions; never fits neural models or selects."""
import json
from collections import defaultdict
import numpy as np
from scipy.special import expit,logsumexp
from scipy.stats import rankdata
from .data import EXP,OUT,config,read_rows,write,digest

def paired_interval(values,reps=3000,seed=53091):
    """Crossed seed/independent graph bootstrap; pairs must already be grouped."""
    a=np.asarray(values,dtype=float);rng=np.random.default_rng(seed);samples=[]
    for _ in range(reps):
        ss=rng.integers(a.shape[0],size=a.shape[0]);ii=rng.integers(a.shape[1],size=a.shape[1])
        samples.append(float(a[ss][:,ii].mean()))
    return {'estimate':float(a.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),'seeds':a.shape[0],'independent_instances':a.shape[1],'replicates':reps}

def auc(y,scores):
    y=np.asarray(y,bool);z=np.asarray(scores);n1=y.sum();n0=len(y)-n1
    return float((rankdata(z)[y].sum()-n1*(n1+1)/2)/(n1*n0)) if n1 and n0 else None

def reliability(prob,target,bins=10):
    p=np.asarray(prob);y=np.asarray(target,dtype=float);records=[];ece=0
    for lo in np.linspace(0,1,bins,endpoint=False):
        mask=(p>=lo)&(p<lo+1/bins if lo<.9 else p<=1)
        if np.any(mask):
            conf=float(p[mask].mean());rate=float(y[mask].mean());count=int(mask.sum());ece+=count/len(p)*abs(conf-rate)
            records.append({'probability':conf,'empirical':rate,'count':count})
    return {'ece':ece,'brier':float(np.mean((p-y)**2)),'bins':records}

def archive():
    cfg=config();datasets={s:read_rows(OUT/'datasets'/f'{s}.jsonl.gz') for s in [*cfg['final_counts'],'counterfactual']};groups={};details={}
    for split,rows in datasets.items():
        details[split]={r['id']:r for r in rows}
        for path in sorted((OUT/'decisions').glob(f'{split}_*.jsonl.gz')):
            for r in read_rows(path):
                key=(split,r['condition'],r['depth']);groups.setdefault(key,{}).setdefault(r['seed'],{})[r['id']]=r
    return datasets,details,groups

def correctness(group,ids,mode):
    return np.array([[group[s][i]['correct'][mode] for i in ids] for s in sorted(group)],dtype=float)

def main():
    cfg=config();datasets,details,groups=archive();selection=json.loads((EXP/'selection.json').read_text());summaries=[];effects=[];strata=[];cal=[];counter=[]
    for (split,condition,depth),g in groups.items():
        ids=[r['id'] for r in datasets[split]];base=correctness(g,ids,'base');all_modes=next(iter(next(iter(g.values())).values()))['correct']
        for mode in all_modes:
            a=correctness(g,ids,mode);summary={'split':split,'condition':condition,'depth':depth,'mode':mode,'accuracy':float(a.mean()),'seed_accuracy':[float(x.mean()) for x in a],'count':len(ids)}
            if mode=='base':
                ll=[];conf=[];rank=[]
                for seed in sorted(g):
                    for i in ids:
                        r=g[seed][i];target=np.array(details[split][i]['labels']['optimal']);z=np.array(r['base_logits']);ll.append(float(logsumexp(z)-logsumexp(z[target])));conf.append(max(r['base_probabilities']))
                summary.update(nll=float(np.mean(ll)),calibration=reliability(conf,base.reshape(-1)))
            summaries.append(summary)
            if mode in ('scored','forced_nonzero','correctness_only','ranking_only','direct_only','direct_fused','shuffled'):
                effect=paired_interval(a-base,cfg['bootstrap_replicates']);effect.update(split=split,condition=condition,depth=depth,mode=mode,wrong_right=float(((base==0)&(a==1)).mean()),right_wrong=float(((base==1)&(a==0)).mean()),both_right=float(((base==1)&(a==1)).mean()),both_wrong=float(((base==0)&(a==0)).mean()),changed=float(np.mean([[g[s][i]['decisions'][mode]!=g[s][i]['decisions']['base'] for i in ids] for s in sorted(g)])))
                effects.append(effect)
            if mode in ('base','scored','direct_only'):
                for statistic in ('nodes','distance','family'):
                    levels=sorted({details[split][i]['labels']['distance'] if statistic=='distance' else details[split][i]['metadata'][statistic] for i in ids},key=str)
                    for level in levels:
                        indices=[j for j,i in enumerate(ids) if (details[split][i]['labels']['distance'] if statistic=='distance' else details[split][i]['metadata'][statistic])==level]
                        strata.append({'split':split,'condition':condition,'depth':depth,'mode':mode,'statistic':statistic,'level':level,'count':len(indices),'accuracy':float(a[:,indices].mean())})
        if condition in selection['conditions']:
            spec=selection['conditions'][condition]
            for obj in spec['objectives']:
                scores=[];target=[];ranks=[];base_ranks=[]
                for seed in sorted(g):
                    for i in ids:
                        row=g[seed][i];z=np.array(row['q'][obj]);y=np.array(details[split][i]['labels']['optimal'],bool);scores.extend(z.tolist());target.extend(y.tolist())
                        ranks.append(float(np.mean((z[y,None]>z[None,~y])+.5*(z[y,None]==z[None,~y]))))
                        bz=np.array(row['base_logits']);base_ranks.append(float(np.mean((bz[y,None]>bz[None,~y])+.5*(bz[y,None]==bz[None,~y]))))
                params=spec['calibration'][obj];raw=expit(scores);prob=expit(params['scale']*np.asarray(scores)+params['offset'])
                cal.append({'split':split,'condition':condition,'depth':depth,'objective':obj,'candidate_auroc':auc(target,scores),'mean_within_graph_ranking_auc':float(np.mean(ranks)),'base_mean_within_graph_ranking_auc':float(np.mean(base_ranks)),'prevalence':float(np.mean(target)),'raw':reliability(raw,target),'calibrated':reliability(prob,target),'count':len(target)})
        if split=='counterfactual':
            for arm in (3,5,9,13):
                pairrows=[r for r in datasets[split] if r['metadata']['arm_distance']==arm];pairids=[r['id'] for r in pairrows]
                for mode in ('base','scored') if 'scored' in all_modes else ('base',):
                    a=correctness(g,pairids,mode);both=a[:,::2]*a[:,1::2]
                    counter.append({'condition':condition,'depth':depth,'mode':mode,'arm_distance':arm,'goal_distance':arm+1,'pair_count':len(pairids)//2,'both_correct':float(both.mean()),'seed_both_correct':[float(x.mean()) for x in both]})
    ids=[r['id'] for r in datasets['id']];g=groups['id','recurrent',8];primary=paired_interval(correctness(g,ids,'scored')-correctness(g,ids,'base'),cfg['bootstrap_replicates'])
    primary.update(protocol=selection['primary'],status='inconclusive' if primary['ci95'][0]<=0<=primary['ci95'][1] else 'positive' if primary['estimate']>0 else 'negative')
    comparisons={}
    rec=correctness(groups['id','recurrent',8],ids,'base');recq=correctness(groups['id','recurrent',8],ids,'scored');one=correctness(groups['id','one_pass',1],ids,'base');oneq=correctness(groups['id','one_pass',1],ids,'scored')
    for name,value in {'recurrence_only':rec-one,'one_pass_scoring':oneq-one,'combined_vs_one_pass':recq-one,'interaction':(recq-rec)-(oneq-one),'scored_vs_direct':recq-correctness(g,ids,'direct_only'),'rec_vs_ff8':rec-correctness(groups['id','ff8',8],ids,'base'),'rec_vs_narrow_ff8':rec-correctness(groups['id','ff8_params',8],ids,'base'),'scored_vs_extra_budget':recq-correctness(groups['id','extra_budget',8],ids,'base')}.items():comparisons[name]=paired_interval(value,cfg['bootstrap_replicates'])
    for depth in (1,2,4):
        shallow=groups['id','recurrent',depth]
        comparisons[f'rec{depth}_scored_vs_direct']=paired_interval(correctness(shallow,ids,'scored')-correctness(shallow,ids,'direct_only'),cfg['bootstrap_replicates'])
    comparisons['one_pass_scored_vs_direct']=paired_interval(oneq-correctness(groups['id','one_pass',1],ids,'direct_only'),cfg['bootstrap_replicates'])
    for split in ('community','size16','size32','size64','long'):
        ids2=[r['id'] for r in datasets[split]]
        comparisons[f'{split}_rec8_vs_ff8']=paired_interval(correctness(groups[split,'recurrent',8],ids2,'base')-correctness(groups[split,'ff8',8],ids2,'base'),cfg['bootstrap_replicates'])
    for split in cfg['final_counts']:
        ids2=[r['id'] for r in datasets[split]]
        for a,b in ((2,1),(4,2),(8,4),(16,8)):
            comparisons[f'{split}_rec{a}_minus_{b}']=paired_interval(correctness(groups[split,'recurrent',a],ids2,'base')-correctness(groups[split,'recurrent',b],ids2,'base'),cfg['bootstrap_replicates'])
    pairrows=[r for r in datasets['counterfactual'] if r['metadata']['arm_distance']==5];ids2=[r['id'] for r in pairrows];early=correctness(groups['counterfactual','recurrent',2],ids2,'base');late=correctness(groups['counterfactual','recurrent',8],ids2,'base')
    comparisons['counterfactual_confirmation']=paired_interval(late[:,::2]*late[:,1::2]-early[:,::2]*early[:,1::2],cfg['bootstrap_replicates'])
    # Episode-level clusters; no individual action contributes an independent sample.
    episodes={};rollout_summary=[]
    for split in cfg['final_counts']:
        for path in sorted((OUT/'rollouts').glob(f'{split}_*.jsonl.gz')):
            for r in read_rows(path):episodes.setdefault((split,r['condition'],r['depth'],r['mode']),{}).setdefault(r['seed'],{})[r['id']]=r
    for (split,condition,depth,mode),g in episodes.items():
        records=[r for perseed in g.values() for r in perseed.values()];reachable=[r for r in records if r['reachable']];unreachable=[r for r in records if not r['reachable']];success=[r for r in records if r['goal_completed']]
        rollout_summary.append({'split':split,'condition':condition,'depth':depth,'mode':mode,'mazes':len(next(iter(g.values()))),'seed_episodes':len(records),'reachable_seed_episodes':len(reachable),'goal_completion_reachable':float(np.mean([r['goal_completed'] for r in reachable])) if reachable else None,'correct_abstain_unreachable':float(np.mean([r['correct_abstain'] for r in unreachable])) if unreachable else None,'success':float(np.mean([r['success'] for r in records])),'mean_actions':float(np.mean([r['steps'] for r in records])),'path_ratio_solved':float(np.mean([r['path_ratio'] for r in success])) if success else None,'cycle_rate':float(np.mean([r['repeats']>0 for r in records])),'mean_seconds':float(np.mean([r['seconds'] for r in records]))})
    for split in cfg['final_counts']:
        g=episodes[split,'recurrent',8,'base'];q=episodes[split,'recurrent',8,'scored'];ids2=list(next(iter(g.values())))
        diff=np.array([[q[s][i]['success']-int(g[s][i]['success']) for i in ids2] for s in sorted(g)])
        comparisons[f'{split}_rollout_scoring']=paired_interval(diff,cfg['bootstrap_replicates'])
    training=[json.loads(p.read_text()) for p in (OUT/'training').glob('*.json')];pilot=json.loads((EXP/'pilot_easy_distribution/receipt.json').read_text())
    resources={'main_neural_fits':len(training),'pilot_neural_fits':pilot['neural_fits'],'main_epoch_fit_seconds':sum(e['seconds'] for r in training for e in r['epochs']),'pilot_epoch_fit_seconds':pilot['epoch_fit_seconds'],'models':[{'name':p.stem,'parameters':json.loads(p.read_text())['parameters'],'best_epoch':json.loads(p.read_text())['best_epoch']} for p in sorted((OUT/'training').glob('*.json'))],'max_training_mlx_gib':max(r['peak_mlx_gib'] for r in training),'max_training_rss_gib':max(r['peak_rss_gib'] for r in training)}
    baselines={s:{'bfs':1.0,'uniform_candidate_expected_accuracy':float(np.mean([r['labels']['optimal_count']/len(r['observation']['candidates']) for r in rows])),'always_abstain':float(np.mean([not r['labels']['reachable'] for r in rows]))} for s,rows in datasets.items()}
    write(OUT/'analysis.json',{'primary':primary,'comparisons':comparisons,'summary':summaries,'scoring_effects':effects,'strata':strata,'calibration':cal,'counterfactual':counter,'rollouts':rollout_summary,'resources':resources,'classical_baselines':baselines,'data_hashes':{str(p):digest(p) for p in sorted((OUT/'decisions').glob('*.gz'))}})
    print(json.dumps({'primary':primary,'comparisons':comparisons,'resources':resources},indent=2))

if __name__=='__main__':main()
