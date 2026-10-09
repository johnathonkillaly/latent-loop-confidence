"""CPU-only audit of a weight-free checkout, including deterministic regeneration."""
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from latent_loop.rendezvous import data
from latent_loop.rendezvous.data import ROOT, read_rows, solve
from latent_loop.rendezvous.report import correctness
from latent_loop.rendezvous.study import paired_ci, save


def main():
    config=json.loads(Path('experiments/experiment_05e/config.json').read_text())
    manifest=json.loads((ROOT/'dataset_manifest.json').read_text())
    counts=0;seen=set()
    for split,meta in manifest['splits'].items():
        path=ROOT/'datasets'/f'{split}.jsonl.gz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['sha256']
        for r in read_rows(path):
            assert solve(r)==solve(r,True)==r['optimal']
            if split!='permutations':
                assert r['semantic_hash'] not in seen;seen.add(r['semantic_hash'])
            counts+=1
    original=data.ROOT
    try:
        with tempfile.TemporaryDirectory(prefix='rendezvous-public-audit-') as directory:
            data.ROOT=Path(directory)
            assert data.generate(config)==manifest
    finally:
        data.ROOT=original
    pilot_count=0
    for name in ('pilot','pilot_unordered'):
        for r in read_rows(ROOT/'datasets'/f'{name}.jsonl.gz'):
            assert solve(r)==solve(r,True)==r['optimal'];pilot_count+=1
    frozen=json.loads((ROOT/'protocol_freeze.json').read_text())
    source_ok={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in frozen['source_sha256'].items()}
    assert all(source_ok.values())
    rows=read_rows(ROOT/'decisions/structure.jsonl.gz')
    expected=json.loads((ROOT/'summary.json').read_text())['primary']
    assert paired_ci(correctness(rows,'fusion')-correctness(rows,'base'))==expected['fusion_minus_base']
    assert paired_ci(correctness(rows,'fusion')-correctness(rows,'c2'))==expected['fusion_minus_c2']
    decision_count=0
    for p in (ROOT/'decisions').glob('*.jsonl.gz'):
        for r in read_rows(p):
            for seed in r['seeds']:
                assert seed['choices']['base']==seed['choices']['constant']==seed['choices']['temperature']
                for method,z in seed['scores'].items():
                    assert np.isfinite(z).all() and np.argmax(z)==seed['choices'][method]
                    decision_count+=1
    save(ROOT/'public_analysis_audit.json',{'status':'passed','cpu_only':True,'main_observations_verified':counts,'pilot_observations_verified':pilot_count,'ordinary_unique_semantic_hashes':len(seen),'byte_identical_main_datasets':True,'frozen_source_hashes':source_ok,'primary_paired_estimates_recomputed':True,'scalar_method_choices_checked':decision_count,'private_checkpoints_required':False,'checkpoint_refitting_scope':'Independent cached-feature verification receipt, not independently repeated by this CPU public audit.'})
    print('Public analysis audit passed:',counts,'main observations;',pilot_count,'pilot observations;',decision_count,'method decisions')


if __name__=='__main__':main()
