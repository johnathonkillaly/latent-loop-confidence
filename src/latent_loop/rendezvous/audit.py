"""Public-artifact and deterministic-oracle checks; never print sensitive matches."""

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from . import data
from .data import ROOT, read_rows, solve
from .study import CONDITIONS, checkpoint, save


def audit(config):
    manifest=json.loads((ROOT/'dataset_manifest.json').read_text())
    seen=set(); count=0; families={}
    for split,meta in manifest['splits'].items():
        path=ROOT/'datasets'/f'{split}.jsonl.gz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
        rows=read_rows(path);assert len(rows)==meta['count']
        for r in rows:
            assert solve(r)==solve(r,independent=True)==r['optimal']
            if split!='permutations':
                assert r['semantic_hash'] not in seen
                seen.add(r['semantic_hash'])
            assert len(set(r['labels']))==4
            assert set(r['presentation_order'])==set(range(4))
            count+=1;families[r['family']]=families.get(r['family'],0)+1
    original=data.ROOT
    with tempfile.TemporaryDirectory(prefix='rendezvous-reproduction-') as directory:
        data.ROOT=Path(directory)
        regenerated=data.generate(config)
        assert regenerated==manifest
    data.ROOT=original
    result={'independent_solver_verified_observations':count,'family_counts':families,'ordinary_semantic_hashes':len(seen),'cross_split_duplicate_hashes':0,'datasets_byte_identical_regeneration':True,'deduplication_limitations':manifest['deduplication'],'historical_data':'New generation seeds and textual tasks; old graph model states/weights and historical LM feature caches are never used. Full graph-isomorphism/program-equivalence certification is not claimed.'}
    if (ROOT/'protocol_freeze.json').exists():
        frozen=json.loads((ROOT/'protocol_freeze.json').read_text())
        result['frozen_checkpoints_unchanged']=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen['checkpoint_sha256'].items())
        result['frozen_scientific_sources_unchanged']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen['source_sha256'].items()}
        if not result['frozen_checkpoints_unchanged']:
            raise AssertionError('Changed selected weights')
    if (ROOT/'decisions/structure.jsonl.gz').exists():
        records=0;temperature_changed=0
        for p in (ROOT/'decisions').glob('*.jsonl.gz'):
            for r in read_rows(p):
                for s in r['seeds']:
                    assert s['choices']['constant']==s['choices']['base']
                    temperature_changed+=s['choices']['temperature']!=s['choices']['base']
                    for method,z in s['scores'].items():
                        assert np.isfinite(z).all() and int(np.argmax(z))==s['choices'][method]
                    records+=1
        assert temperature_changed==0
        result.update({'seed_instance_decision_records':records,'temperature_changed_decisions':0})
    save(ROOT/'dataset_and_protocol_audit.json',result)
    print(json.dumps(result,indent=2))
