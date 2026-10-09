"""Descriptive probes and controlled hidden-message interventions, never policies."""
import json
import numpy as np
import mlx.core as mx
from scipy.special import softmax
from latent_loop.environments.maze import Maze,distances,labels
from latent_loop.planning.oracles import shortest_path,propagation
from latent_loop.planning.data import write_json,save_rows,now
from latent_loop.scaling.data import EXP,OUT,load,config
from latent_loop.scaling.model import forward
from latent_loop.scaling.evaluate import model,seeds,verify

def design(h):return np.column_stack([h,np.ones(len(h))])
def extract(m,rows,dd):
    features={d:[] for d in dd};targets=[]
    for r in rows:
        maze=Maze.from_dict(r);distance=distances(maze);mask=distance>=0;out=forward(m,mx.array(maze.observation()[None]),dd,True);mx.eval(out)
        targets.extend(distance[mask].tolist())
        for d in dd:features[d].extend(np.asarray(out[d]['states'])[0][mask])
    return {d:np.array(h) for d,h in features.items()},np.array(targets)

def run():
    f=verify();selected=f['selected_recurrent'];dd=(1,2,4,8,16,32);probe=[];effects=[];examples=[];counterfactual=[]
    for c in ('frozen_05b',selected):
        for seed in seeds(c):
            m=model(c,seed);tx,ty=extract(m,load('train')[:200],dd);penalty=np.eye(m.width+1);penalty[-1,-1]=0
            weights={d:np.linalg.solve(design(tx[d]).T@design(tx[d])+10*penalty,design(tx[d]).T@ty) for d in dd}
            for split in ('size5','size9','size15','structure9','long15'):
                rows=load(split,True)[:40];xx,yy=extract(m,rows,dd)
                for d in dd:
                    pred=design(xx[d])@weights[d];probe.append({'model':c,'seed':seed,'split':split,'depth':d,'maze_count':len(rows),'cell_count':len(yy),'r2':float(1-np.sum((yy-pred)**2)/np.sum((yy-yy.mean())**2)),'mae':float(np.mean(np.abs(yy-pred))),'correlation':float(np.corrcoef(yy,pred)[0,1])})
            for row in load('size9',True)[:30]:
                maze=Maze.from_dict(row);x=maze.observation();no=x.copy();no[...,3]=0;distance=distances(maze);path=shortest_path(maze);mid=path[len(path)//2];da=distances(maze,maze.agent)
                sham_candidates=[tuple(p) for p in np.argwhere(da>=0) if tuple(p) not in path and tuple(p)!=maze.agent and tuple(p)!=maze.goal]
                if not sham_candidates:continue
                sham=min(sham_candidates,key=lambda p:abs(da[p]-da[mid]));normal=forward(m,mx.array(x[None]),dd,True);absent=forward(m,mx.array(no[None]),dd,True);mx.eval(normal,absent)
                cuts={}
                for name,cell in [('path',mid),('sham',sham)]:
                    cut=np.ones((1,maze.size,maze.size,1),np.float32);cut[0,*cell,0]=0
                    cuts[name]=(forward(m,mx.array(x[None]),dd,cut=mx.array(cut)),forward(m,mx.array(no[None]),dd,cut=mx.array(cut)));mx.eval(cuts[name])
                frame={}
                for d in dd:
                    hs=np.asarray(normal[d]['states'])[0];eff=np.linalg.norm(hs-np.asarray(absent[d]['states'])[0],axis=-1)
                    outside=(distance<0)|(distance>d);assert np.max(eff[outside],initial=0)==0
                    goalz=np.asarray(normal[d]['logits'])[0]-np.asarray(absent[d]['logits'])[0];rec={'model':c,'seed':seed,'id':row['id'],'depth':d,'goal_distance':row['labels']['distance'],'goal_logit_effect':float(np.linalg.norm(goalz)),'max_effect_outside_causal_radius':float(np.max(eff[outside],initial=0))}
                    for name,(yes,noo) in cuts.items():
                        z=np.asarray(yes[d]['logits'])[0]-np.asarray(noo[d]['logits'])[0];rec[name+'_goal_effect']=float(np.linalg.norm(z));rec[name+'_prediction_changed']=bool(np.argmax(np.asarray(yes[d]['logits']))!=np.argmax(np.asarray(normal[d]['logits'])))
                    effects.append(rec)
                    frame[str(d)]={'state_rms':np.sqrt(np.mean(hs*hs,-1)).tolist(),'goal_effect':eff.tolist(),'probabilities':softmax(np.asarray(normal[d]['logits'])[0]).tolist(),'classical_distance':np.where(np.isfinite(propagation(maze,d)[-1]),propagation(maze,d)[-1],-1).tolist()}
                if seed==seeds(c)[0] and len([e for e in examples if e['model']==c])<3:examples.append({'model':c,'seed':seed,'example':row,'path_cut':list(mid),'sham_cut':list(sham),'depths':frame})
            # Same-wall paired distant goals requiring disjoint optimal actions.
            count=0
            for row in load('size9',True):
                maze=Maze.from_dict(row)
                if labels(maze)['distance']<=4:continue
                da=distances(maze,maze.agent);other=None
                for p in np.argwhere(da>4):
                    alt=Maze(maze.walls,maze.agent,tuple(p));opt=labels(alt)['optimal']
                    if not any(a and b for a,b in zip(row['labels']['optimal'],opt)):other=alt;break
                if other is None:continue
                out=forward(m,mx.array(np.stack([maze.observation(),other.observation()])),dd);mx.eval(out)
                record={'model':c,'seed':seed,'id':row['id'],'alternative_goal':list(other.goal),'original_goal':list(maze.goal),'original_optimal':row['labels']['optimal'],'alternative_optimal':labels(other)['optimal'],'depths':{}}
                for d in dd:
                    z=np.asarray(out[d]['logits']);delta=float(np.max(np.abs(z[0]-z[1])));record['depths'][str(d)]={'logit_difference':delta,'predictions':z.argmax(-1).tolist()}
                    if d<=4:assert delta==0
                counterfactual.append(record);count+=1
                if count>=30:break
    write_json(OUT/'propagation_analysis.json',{'completed':now(),'probe_ridge':10,'probe_training_mazes':200,'probe_features':'hidden channels plus intercept only; labels supervise probe only; never deployed','probe_results':probe,'interventions':effects,'limitations':'Cells correlated; no cell-level CIs. Goal removal and hidden cuts are out of distribution. Path/sham cuts matched approximately by agent distance, not causal equivalence. Same-board goal counterfactuals are paired diagnostics, excluded from aggregate test counts.'})
    save_rows(OUT/'propagation_examples.jsonl.gz',examples);save_rows(OUT/'far_goal_counterfactuals.jsonl.gz',counterfactual)
    print('Propagation probes/interventions/far-goal counterfactuals complete',flush=True)

if __name__=='__main__':run()
