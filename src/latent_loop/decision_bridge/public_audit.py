"""Verify public outcomes from scalar traces without MLX or ignored checkpoints."""
import json
from collections import defaultdict
import numpy as np
from .data import EXP,OUT,config,read_rows,digest,write,now,choose,audit
from .interface import fuse

def main():
    total=audit();frozen=json.loads((EXP/'frozen_protocol.json').read_text());amend=json.loads((EXP/'evaluation_amendment.json').read_text())['changes'];checks={}
    for path,sha in frozen['source_hashes'].items():
        if path in amend:
            assert digest(amend[path]['original_snapshot'])==sha
            assert digest(path)==amend[path]['corrected_sha256']
        else:assert digest(path)==sha
        checks[path]=True
    selection=json.loads((EXP/'selection.json').read_text());cfg=config();decision_count=0;record_count=0;episode_count=0
    for split in [*cfg['final_counts'],'counterfactual']:
        rows=read_rows(OUT/'datasets'/f'{split}.jsonl.gz');byid={r['id']:r for r in rows}
        for path in (OUT/'decisions').glob(f'{split}_*.jsonl.gz'):
            records=read_rows(path);seen=set();groups=defaultdict(list)
            for r in records:
                key=(r['id'],r['depth']);assert key not in seen;seen.add(key);groups[r['depth']].append(r)
                row=byid[r['id']];o=row['observation'];target=row['labels']['optimal'];keys=np.array([[o['tie_keys'][v if v>=0 else o['n']] for v in o['candidates']]])
                z=np.array([r['base_logits']]);base=int(choose(z,keys)[0]);assert base==r['decisions']['base']
                assert r['decisions']['constant']==base
                if 'temperature' in r['decisions']:assert r['decisions']['temperature']==base
                spec=selection['conditions'].get(r['condition'])
                for objective,quality in r['q'].items():
                    q=np.array([quality]);assert int(choose(q,keys)[0])==r['decisions'][objective+'_only']
                    assert int(choose(fuse(z,q,spec['objectives'][objective]['lambda']),keys)[0])==r['decisions'][objective+'_fused']
                if r['q']:
                    q=np.array([r['q'][spec['objective']]]);weight=spec['by_depth'].get(str(r['depth']),spec)['lambda']
                    assert int(choose(fuse(z,q,weight),keys)[0])==r['decisions']['scored']
                    assert int(choose(fuse(z,q,spec['forced_nonzero']['lambda']),keys)[0])==r['decisions']['forced_nonzero']
                for mode,index in r['decisions'].items():assert bool(target[index])==r['correct'][mode];decision_count+=1
                assert np.isfinite(r['base_logits']).all();assert r['state_norm']<=1.00001
                record_count+=1
            expected_depths={1} if r['condition']=='one_pass' else {1,2,4,8,16} if r['condition'] in ('recurrent','extra_budget') else {8}
            assert set(groups)==expected_depths
            for depth,rr in groups.items():assert {r['id'] for r in rr}==set(byid)
        for path in (OUT/'rollouts').glob(f'{split}_*.jsonl.gz'):
            for r in read_rows(path):
                o=byid[r['id']]['observation'];edges={tuple(sorted(e)) for e in o['edges']};positions=[o['start']]
                for action in r['actions']:
                    if action<0:assert action==-1;break
                    assert tuple(sorted((positions[-1],action))) in edges;positions.append(action)
                assert positions==r['positions'];assert (positions[-1]==o['goal'])==r['goal_completed']
                assert r['steps']==len(r['actions']);assert r['invalid']==0
                target=byid[r['id']]['labels'];assert r['reachable']==target['reachable']
                correct_abstain=bool(r['actions'] and r['actions'][-1]==-1 and not target['reachable'])
                assert correct_abstain==r['correct_abstain']
                assert r['success']==bool(r['goal_completed'] or correct_abstain)
                assert r['repeats']==len(positions)-len(set(positions))
                expected_ratio=len(r['actions'])/target['distance'] if r['goal_completed'] else None
                assert expected_ratio==r['path_ratio']
                episode_count+=1
    write(OUT/'public_audit.json',{'completed':now(),'graph_labels_verified':total,'decision_records':record_count,'policy_decisions':decision_count,'episodes':episode_count,'frozen_sources_and_amendment_verified':checks,'weights_needed':False,'limitation':'Validates recorded outcomes and provenance, not an independent neural refit or re-execution without ignored checkpoints.'})
    print(total,record_count,decision_count,episode_count)

if __name__=='__main__':main()
