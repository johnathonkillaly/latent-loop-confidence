"""Exploratory paired novel-candidate descriptions, frozen after the primary study.

No fitting, architecture selection or lambda changes. --audit needs only CPU.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from latent_loop.rendezvous.data import ROOT, generate_split, read_rows, solve, write_rows
from latent_loop.rendezvous.features import extract, load_features
from latent_loop.rendezvous.study import normalized, paired_ci, predictions, save

DIRECTORY=ROOT/'description_shift'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--audit',action='store_true');args=parser.parse_args()
    config=json.loads(Path('experiments/experiment_05e/config.json').read_text())
    frozen=json.loads((ROOT/'protocol_freeze.json').read_text())
    if args.audit:
        manifest=json.loads((DIRECTORY/'protocol.json').read_text())
        original=read_rows(ROOT/'datasets/description_natural.jsonl.gz')
        novel=read_rows(ROOT/'datasets/description_novel.jsonl.gz')
        seen={r['semantic_hash'] for split in config['sizes'] for r in read_rows(ROOT/'datasets'/f'{split}.jsonl.gz')}
        assert len({r['semantic_hash'] for r in original})==600
        assert not seen & {r['semantic_hash'] for r in original}
        for a,b in zip(original,novel):
            assert a['id']==b['id'] and a['raw']==b['raw'] and a['labels']==b['labels']
            assert a['optimal']==b['optimal']==solve(a)==solve(a,True)
        for name,sha in manifest['dataset_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha
        for view in ('natural','novel'):
            for r in read_rows(DIRECTORY/f'{view}_decisions.jsonl.gz'):
                for seed in r['seeds']:
                    assert all(int(np.argmax(z))==seed['choices'][method] for method,z in seed['scores'].items())
        save(DIRECTORY/'audit.json',{'status':'passed','paired_independent_instances':600,'observations':1200,'labels_verified_by_both_solvers':True,'semantic_identity_and_label_mapping_preserved':True,'no_new_fits':True,'interpretation':'Exploratory supplement frozen after the original primary results; no original protocol reselection.'})
        return
    if not (DIRECTORY/'protocol.json').exists():
        seen={r['semantic_hash'] for split in config['sizes'] for r in read_rows(ROOT/'datasets'/f'{split}.jsonl.gz')}
        rows,_=generate_split('description',600,641722,seen)
        novel=[]
        for row in rows:
            descriptions=[]
            for d in row['candidate_descriptions']:
                if row['family']=='routing':
                    descriptions.append('decline this request because no tool is eligible' if d.startswith('abstain') else 'dispatch this request through the named tool '+d.removeprefix('use '))
                elif row['family']=='graph':
                    descriptions.append('decline movement because the destination is unreachable' if d.startswith('abstain') else 'take the edge toward neighboring node '+d.removeprefix('move to '))
                else:
                    descriptions.append('report the integer value '+d)
            prompt=row['prompt'].split('\nCandidates:\n')[0]+'\nCandidates:\n'+'\n'.join(f"({row['labels'][i]}) {descriptions[i]}" for i in row['presentation_order'])+'\nRespond with one candidate label only.'
            novel.append({**row,'candidate_descriptions':descriptions,'prompt':prompt,'style':'novel_candidate_descriptions'})
        paths={}
        for name,items in [('natural',rows),('novel',novel)]:
            path=ROOT/'datasets'/f'description_{name}.jsonl.gz';write_rows(path,items);paths[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        save(DIRECTORY/'protocol.json',{'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'stage':'after primary opening; supplementary exploratory evaluation only','instances':600,'seed':641722,'family_counts':{'routing':200,'state':200,'graph':200},'comparison':'identical raw tasks, labels and candidate order; only option descriptions change','selected_original_fusion':frozen['selected'],'dataset_sha256':paths,'original_checkpoint_sha256':frozen['checkpoint_sha256'],'no_reselection':True})
    extract(config,['description_natural','description_novel'])
    result={'status':'Exploratory post-primary paired description shift; no fitting or reselection','views':{}}
    correctness={}
    for view in ('natural','novel'):
        rows,f=load_features('description_'+view)
        qs={name:[predictions(name,seed,f['x']) for seed in config['training_seeds']] for name in [frozen['selected']['objective'],'c2','description_only']}
        records=[]
        for i,r in enumerate(rows):
            seeds=[]
            for j,seed in enumerate(config['training_seeds']):
                q=qs[frozen['selected']['objective']][j][i]
                scores={'base':f['logits'][i],'fusion':f['logits'][i]+frozen['selected']['lambda']*normalized(q),'q_only':q,'c2':qs['c2'][j][i],'description_only':qs['description_only'][j][i]}
                seeds.append({'seed':seed,'choices':{m:int(np.argmax(z)) for m,z in scores.items()},'scores':{m:z.tolist() for m,z in scores.items()}})
            records.append({'id':r['id'],'family':r['family'],'optimal':r['optimal'],'seeds':seeds})
        write_rows(DIRECTORY/f'{view}_decisions.jsonl.gz',records)
        correctness[view]={m:np.array([[r['seeds'][j]['choices'][m] in r['optimal'] for r in records] for j in range(3)],dtype=float) for m in ['base','fusion','q_only','c2','description_only']}
        result['views'][view]={m:{'accuracy':float(a.mean()),'seed_accuracy':a.mean(-1).tolist(),'family_accuracy':{family:float(a[:,[r['family']==family for r in records]].mean()) for family in ['routing','state','graph']}} for m,a in correctness[view].items()}
    result['novel_minus_natural']={m:paired_ci(correctness['novel'][m]-correctness['natural'][m]) for m in correctness['natural']}
    result['novel_fusion_minus_base']=paired_ci(correctness['novel']['fusion']-correctness['novel']['base'])
    save(DIRECTORY/'summary.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
