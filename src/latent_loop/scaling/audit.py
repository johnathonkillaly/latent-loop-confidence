"""CPU-only public numerical integrity, privacy and historical-preservation audit."""
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from latent_loop.environments.maze import Maze,labels
from latent_loop.planning.data import digest,write_json,now
from latent_loop.planning.evaluate import correct_and_metrics
from latent_loop.scaling.data import EXP,OUT,read,graph_key,independent_distance

def run():
    f=json.loads((EXP/'frozen_protocol.json').read_text());manifest=json.loads((EXP/'dataset_manifest.json').read_text());summary=json.loads((OUT/'summary.json').read_text());cfg=json.loads((EXP/'config.json').read_text())
    for p,h in f['sources'].items():assert digest(p)==h,p
    assert json.loads((EXP/'lockbox_opened.json').read_text())['opened']>f['frozen']
    seen=set();layouts=set();n=0;decisions=0;files=0;frontier=[]
    for p in Path('results/experiment_05b/datasets').glob('*.gz'):
        for row in read(p):m=Maze.from_dict(row);seen.add(graph_key(m)[0]);layouts.add(m.canonical_hash())
    for split,spec in manifest['splits'].items():
        p=OUT/'datasets'/f'{split}.jsonl.gz';assert digest(p)==spec['sha256'];rows=read(p)
        for row in rows:
            m=Maze.from_dict(row);g,topo=graph_key(m);assert g==row['graph_fingerprint'] and g not in seen;seen.add(g)
            assert m.canonical_hash()==row['layout_hash'] and m.canonical_hash() not in layouts;layouts.add(m.canonical_hash());assert labels(m)==row['labels'];assert int(independent_distance(m)[m.agent])==row['labels']['distance'];assert sum(row['labels']['legal'])>=2;n+=1
        if not spec.get('final'):continue
        for c,res in summary['splits'][split]['models'].items():
            seeds=cfg['baseline_seeds'] if c=='frozen_05b' else cfg['seeds'];traces=[read(OUT/'trajectories'/f'{c}_seed{s}_{split}.jsonl.gz') for s in seeds]
            for i,trace in enumerate(traces):
                assert [t['id'] for t in trace]==[r['id'] for r in rows];files+=1
                for d,expected in res['seed_metrics'][i].items():
                    _,metric=correct_and_metrics(rows,[t['depths'][d]['logits'] for t in trace]);assert all(abs(expected[k]-v)<1e-10 for k,v in metric.items());decisions+=len(rows)
                    for t in trace:
                        state=t['depths'][d];assert all(np.isfinite(state[k]) for k in ('state_norm','state_change','state_cosine','state_drift')) and state['state_norm']<=1.000001
            for d in map(str,cfg['depths'] if c=='frozen_05b' or cfg['conditions'][c]['shared'] else [cfg['conditions'][c]['depth']]):
                for b in sorted({r['path_bin'] for r in rows}):
                    for j in ('all','0-5','6-15','16+'):
                        idx=[i for i,r in enumerate(rows) if r['path_bin']==b and (j=='all' or (r['topology']['junctions']<=5 if j=='0-5' else 6<=r['topology']['junctions']<=15 if j=='6-15' else r['topology']['junctions']>=16))]
                        if not idx:continue
                        metrics=[correct_and_metrics([rows[i] for i in idx],[trace[i]['depths'][d]['logits'] for i in idx])[1] for trace in traces]
                        acc=float(np.mean([v['accuracy'] for v in metrics]));frontier.append({'split':split,'size':len(rows[0]['walls']),'path_bin':b,'junctions':j,'model':c,'loops':int(d),'independent_mazes':len(idx),'accuracy':acc,'mean_regret':float(np.mean([v['mean_regret'] for v in metrics])),'passes_90pct_action':acc>=.9})
    changed=subprocess.check_output(['git','diff','v0.2.0','--name-only'],text=True).splitlines();assert all(p.startswith(('src/latent_loop/scaling/','experiments/experiment_05c/','results/experiment_05c/','reports/experiment_05c/','tests/test_scaling')) for p in changed),changed
    findings=[];scanned=0
    patterns={'token':r'(?:github_pat_[A-Za-z0-9_]{25,}|gh[pousr]_[A-Za-z0-9]{25,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{25,})','private_path':r'/(?:Users|home)/[^/\s"<>]+/','key':r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'}
    for root in (EXP,OUT,Path('src/latent_loop/scaling'),Path('reports/experiment_05c')):
        for p in root.rglob('*'):
            if not p.is_file() or p.suffix in ('.png','.pyc') or p.name=='audit.py':continue
            try:text='\n'.join(json.dumps(r) for r in read(p)) if p.suffix=='.gz' else p.read_text()
            except UnicodeError:continue
            text=re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+','<image>',text);scanned+=1
            for kind,pattern in patterns.items():
                if re.search(pattern,text):findings.append({'path':str(p),'category':kind})
    assert not findings,findings
    write_json(OUT/'failure_frontier.json',{'thresholds':cfg['thresholds'],'status':'Descriptive engineering thresholds, not hypothesis tests; small cells retained with counts','cells':frontier})
    write_json(OUT/'public_audit.json',{'checked':now(),'all_passed':True,'new_labels_and_topologies_checked':n,'historical_topologies_excluded':manifest['historical_graph_fingerprints'],'trajectory_files':files,'decisions_recomputed':decisions,'source_freeze_verified':True,'historical_files_unchanged':True,'security_files_scanned':scanned,'security_findings':findings,'scope':'CPU metric/solver/graph-exclusion/finite-state audit. Exact historical inference separately verified. Full independent model retraining not claimed.'})
    print('Public audit:',n,'new labels,',decisions,'decisions,',files,'trace files, no security findings',flush=True)
if __name__=='__main__':run()
