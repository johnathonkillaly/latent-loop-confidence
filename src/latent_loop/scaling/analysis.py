"""Post-freeze descriptive analysis; no model selection or fitting."""
import json
import numpy as np
from scipy.special import softmax
from latent_loop.scaling.data import OUT,EXP,read,config
from latent_loop.scaling.evaluate import seeds,paired
from latent_loop.planning.data import write_json,now

def run():
    cfg=config();summary=json.loads((OUT/'summary.json').read_text());result={'created':now(),'status':'Secondary descriptive summaries; final configuration unchanged','splits':{}}
    for split,spec in cfg['datasets'].items():
        if not spec.get('final'):continue
        rows=read(OUT/'datasets'/f'{split}.jsonl.gz');rr={'difficulty':{'path_lengths':[r['labels']['distance'] for r in rows],'junctions':[r['topology']['junctions'] for r in rows],'initial_away_fraction':float(np.mean([r['labels']['initial_away'] for r in rows])),'greedy_trap_fraction':float(np.mean([any(a and b for a,b in zip(r['labels']['greedy'],r['labels']['dead_end'])) for r in rows])),'multiple_optimal_fraction':float(np.mean([sum(r['labels']['optimal'])>1 for r in rows])),'action_memberships':np.sum([r['labels']['optimal'] for r in rows],axis=0).tolist(),'graph_nodes_mean':float(np.mean([r['topology']['component_nodes'] for r in rows]))},'models':{}}
        for c,m in summary['splits'][split]['models'].items():
            trace=[read(OUT/'trajectories'/f'{c}_seed{s}_{split}.jsonl.gz') for s in seeds(c)];stats={}
            for d in m['mean']:
                z=np.array([[t['depths'][d]['logits'] for t in tt] for tt in trace]);p=softmax(z,-1);pred=z.argmax(-1);opt=np.array([r['labels']['optimal'] for r in rows]);ok=opt[np.arange(len(rows))[None,:],pred];far=np.array([r['labels']['distance']>int(d) for r in rows]);common=np.array([r['labels']['distance']<=20 for r in rows]);ps=np.sort(p,axis=-1)
                stats[d]={'confidence':float(p.max(-1).mean()),'entropy':float(-(p*np.log(p+1e-30)).sum(-1).mean()),'margin':float((ps[...,-1]-ps[...,-2]).mean()),'goal_outside_radius_count':int(far.sum()),'accuracy_goal_outside_radius':float(ok[:,far].mean()) if far.any() else None,'common_distance_2_to_20_accuracy':float(ok[:,common].mean()) if common.any() else None,'beyond_goal_radius_fraction':float(far.mean())}
            rr['models'][c]=stats
        result['splits'][split]=rr
    if (OUT/'far_goal_counterfactuals.jsonl.gz').exists():
        cases=read(OUT/'far_goal_counterfactuals.jsonl.gz');diag={}
        for c in sorted({r['model'] for r in cases}):
            subset=[r for r in cases if r['model']==c];diag[c]={}
            for d in subset[0]['depths']:
                correct=[];diff=[]
                for r in subset:
                    a,b=r['depths'][d]['predictions'];correct.append((r['original_optimal'][a]+r['alternative_optimal'][b])/2);diff.append(a!=b)
                diag[c][d]={'paired_goal_accuracy':float(np.mean(correct)),'different_decision_fraction':float(np.mean(diff)),'pairs_across_seeds':len(subset),'independent_original_mazes':len({r['id'] for r in subset})}
        result['far_goal_counterfactuals']=diag
    if (OUT/'rollout_summary.json').exists():
        roll=json.loads((OUT/'rollout_summary.json').read_text());comparisons={}
        for split in roll:
            a=read(OUT/'rollouts'/f'frozen_05b_depth4_{split}.jsonl.gz');b=read(OUT/'rollouts'/f'frozen_05b_depth8_{split}.jsonl.gz');n=len(a)//3
            vals=np.array([r['done'] for r in b],float).reshape(3,n)-np.array([r['done'] for r in a],float).reshape(3,n);base=[{'path_bin':r['path_bin']} for r in a[:n]];comparisons[split]=paired(vals,base)
        result['episode_8_minus_4']=comparisons
        comparisons={}
        selected=summary['selected_recurrent']
        for split in roll:
            a=read(OUT/'rollouts'/f'{selected}_depth8_{split}.jsonl.gz');b=read(OUT/'rollouts'/f'ff8_depth8_{split}.jsonl.gz');n=len(a)//3
            vals=np.array([r['done'] for r in a],float).reshape(3,n)-np.array([r['done'] for r in b],float).reshape(3,n)
            comparisons[split]=paired(vals,[{'path_bin':r['path_bin']} for r in a[:n]])
        result['selected_recurrent_episode_vs_ff8']=comparisons
    write_json(OUT/'descriptive_analysis.json',result);print('Descriptive path/confidence/aliasing/episode analysis saved',flush=True)
if __name__=='__main__':run()
