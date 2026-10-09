"""Post-freeze descriptive checks. Never fits or selects a model on final data."""
import hashlib
import json
from pathlib import Path

import numpy as np

from latent_loop.rendezvous.data import ROOT, read_rows
from latent_loop.rendezvous.report import correctness
from latent_loop.rendezvous.study import paired_ci, predictions, save, checkpoint


def main():
    config=json.loads(Path('experiments/experiment_05e/config.json').read_text())
    frozen=json.loads((ROOT/'protocol_freeze.json').read_text())
    diagnostics={'status':'Post-freeze descriptive diagnostics; no new model selection or fits','splits':{}}
    for split in ('id','wording','format','structure','harder'):
        rows=read_rows(ROOT/'decisions'/f'{split}.jsonl.gz')
        diagnostics['splits'][split]={'q_only_minus_c2':paired_ci(correctness(rows,'q_only')-correctness(rows,'c2')),'fusion_minus_description_only':paired_ci(correctness(rows,'fusion')-correctness(rows,'description_only')),'no_logits_minus_description_only':paired_ci(correctness(rows,'no_logits')-correctness(rows,'description_only')),'full_q_only_minus_logits_only':paired_ci(correctness(rows,'q_only')-correctness(rows,'logits_only')),'families':{}}
        raw=read_rows(ROOT/'datasets'/f'{split}.jsonl.gz')
        for family in ('routing','state','graph'):
            rr=[r for r in rows if r['family']==family]
            xx=[r for r in raw if r['family']==family]
            y=np.array([[i in r['optimal'] for i in range(4)] for r in xx])
            diagnostics['splits'][split]['families'][family]={'fusion_minus_base':paired_ci(correctness(rr,'fusion')-correctness(rr,'base')),'abstain_prevalence':float(np.mean([r['optimal']==[3] for r in xx])) if family!='state' else None,'uniform_random_accuracy':float(y.mean()),'difficulty_min_max':[min(r['difficulty'] for r in xx),max(r['difficulty'] for r in xx)]}
    save(ROOT/'secondary_diagnostics.json',diagnostics)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from latent_loop.backbone import environment,get_model
    from latent_loop.rendezvous.models import CandidateHead
    from latent_loop.rendezvous.features import load_features
    from latent_loop.rendezvous.timing import live_policy
    environment(config['memory_gb']);model,tokenizer,_=get_model()
    rows,f=load_features('validation');rows=rows[:16]
    live_policy(model,tokenizer,rows[:1],config)
    mx.synchronize();mx.clear_cache();before=mx.get_active_memory()
    dimension=f['x'].shape[-1]
    head=CandidateHead(dimension,config['hidden_width']);head.load_weights(str(checkpoint(frozen['selected']['objective'],config['training_seeds'][0])));head.eval();mx.eval(head.parameters())
    after=mx.get_active_memory()
    _,base_live=live_policy(model,tokenizer,rows,config)
    _,fusion_live=live_policy(model,tokenizer,rows,config,head,'fusion',frozen['selected']['lambda'])
    q=predictions(frozen['selected']['objective'],config['training_seeds'][0],f['x'][:16])
    from latent_loop.rendezvous.study import normalized
    fusion_cached=f['logits'][:16]+frozen['selected']['lambda']*normalized(q)
    assert np.max(np.abs(base_live-f['logits'][:16]))<1e-3
    assert np.max(np.abs(fusion_live-fusion_cached))<1e-3
    assert np.array_equal(base_live.argmax(-1),f['logits'][:16].argmax(-1))
    assert np.array_equal(fusion_live.argmax(-1),fusion_cached.argmax(-1))
    perm=[2,0,3,1]
    original=head(mx.array(f['x'][:16]));changed=head(mx.array(f['x'][:16,perm]));mx.eval(original,changed)
    permutation_error=float(np.max(np.abs(np.array(original)[:,perm]-np.array(changed))))
    assert permutation_error<1e-5
    counts={}
    for condition in ('correctness','ranking','c1','c2','no_logits','logits_only','description_only','residual'):
        h=CandidateHead(dimension,config['hidden_width']);h.load_weights(str(checkpoint(condition,config['training_seeds'][0])))
        counts[condition]=sum(v.size for _,v in tree_flatten(h.parameters()))
    result={'parameter_counts':counts,'candidate_head_row_permutation_max_error':permutation_error,'live_vs_cached_base_max_error':float(np.max(np.abs(base_live-f['logits'][:16]))),'live_vs_cached_fusion_max_error':float(np.max(np.abs(fusion_live-fusion_cached))),'live_vs_cached_decisions_identical':True,'validation_prompts_checked':16,'incremental_head_active_metal_bytes':int(after-before),'memory_scope':'Active Metal allocation increase after loading one frozen selected head; measured separately from full-policy peak and excludes backbone residency.','model_trainable_parameters':sum(v.size for _,v in tree_flatten(model.trainable_parameters()))}
    save(ROOT/'deployment_verification.json',result)
    # Rehash required checkpoint files after all model operations, without recording local cache paths.
    from huggingface_hub import snapshot_download
    from latent_loop.backbone import MODEL_ID,REVISION
    path=Path(snapshot_download(MODEL_ID,revision=REVISION,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja'],local_files_only=True))
    verified=json.loads((ROOT/'model_checkpoint_verification.json').read_text())
    assert all(hashlib.sha256((path/name).read_bytes()).hexdigest()==meta['sha256'] for name,meta in verified['files'].items())
    verified['post_evaluation_file_hashes_unchanged']=True
    save(ROOT/'model_checkpoint_verification.json',verified)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
