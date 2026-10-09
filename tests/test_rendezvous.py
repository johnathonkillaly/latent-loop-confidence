import copy
import json
import os

import numpy as np
import pytest

from latent_loop.rendezvous.data import generate_split, graph_fingerprint, render, solve
from latent_loop.rendezvous.features import design
from latent_loop.rendezvous.study import accuracy, normalized, softmax


def test_independent_solvers_and_split_uniqueness():
    seen = set()
    groups = []
    for i, split in enumerate(("train", "structure", "harder")):
        rows, _ = generate_split(split, 120, 9187+i, seen)
        assert all(r["optimal"] == solve(r, independent=True) for r in rows)
        assert all(r["optimal"] and len(r["labels"]) == len(set(r["labels"])) == 4 for r in rows)
        groups.append({r["semantic_hash"] for r in rows})
    assert not groups[0] & groups[1] and not groups[1] & groups[2]


def test_graph_invariant_and_ties_and_abstain():
    edges = [(0,1),(0,2),(1,3),(2,3)]
    p = [3,2,0,1]
    assert graph_fingerprint(4, edges, False) == graph_fingerprint(4, [(p[a],p[b]) for a,b in edges], False)
    row = {"family":"graph", "raw":{"n":5,"edges":edges,"neighbors":[1,2,4],"goal":3,"directed":False}}
    assert solve(row) == solve(row, True) == [0,1]
    row["raw"]["goal"] = 4
    assert solve(row) == solve(row, True) == [3]


def test_labels_never_enter_prompt_renderer():
    rows, _ = generate_split("train", 12, 781, set())
    for row in rows:
        changed = copy.deepcopy(row); changed["optimal"] = [0,1,2,3]
        a = render(row, np.random.default_rng(99))
        b = render(changed, np.random.default_rng(99))
        assert a["prompt"] == b["prompt"]


def test_feature_and_fusion_permutation():
    rng = np.random.default_rng(91)
    h, c, l, z = rng.normal(size=(5,7)), rng.normal(size=(5,4,7)), rng.normal(size=(5,4,7)), rng.normal(size=(5,4))
    p = [2,0,3,1]
    assert np.allclose(design(h,c[:,p],l[:,p],z[:,p]), design(h,c,l,z)[:,p])
    assert np.allclose(normalized(z[:,p]), normalized(z)[:,p])
    assert np.array_equal(z.argmax(-1), (z/.73).argmax(-1))
    assert np.allclose(softmax(z).sum(-1), 1)


def test_epoch_journal_recovers_torn_checkpoint_set(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'experiments/experiment_05e/run_resumable.py'
    spec=importlib.util.spec_from_file_location('rendezvous_resumable',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    weights,optimizer=tmp_path/'weights',tmp_path/'optimizer'
    weights.write_bytes(b'epoch one weights');optimizer.write_bytes(b'epoch one optimizer')
    monkeypatch.chdir(tmp_path)
    journal=module.EpochJournal('journal')
    journal.commit('fit',{'weights':weights,'optimizer':optimizer})
    weights.write_bytes(b'epoch two weights without corresponding optimizer')
    journal.recover()
    assert weights.read_bytes()==b'epoch one weights'
    assert optimizer.read_bytes()==b'epoch one optimizer'


@pytest.mark.skipif(os.environ.get("RUN_MLX_TESTS") != "1", reason="opt-in MLX integration")
def test_candidate_head_gradient_checkpoint_and_permutation(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from latent_loop.rendezvous.models import CandidateHead, objective, view
    mx.random.seed(11)
    model = CandidateHead(26, 64)
    x, y, z = mx.random.normal((8,4,26)), mx.array([[1.,0.,0.,0.]]*8), mx.zeros((8,4))
    a, b = model(x), model(x[:, [2,0,3,1]])
    assert np.allclose(np.array(a)[:,[2,0,3,1]], np.array(b), atol=1e-6)
    loss, g = nn.value_and_grad(model, lambda m: objective(m(x),y,"correctness",z))(model)
    assert np.isfinite(loss.item())
    assert sum(float(mx.sum(v*v).item()) for _,v in tree_flatten(g)) > 0
    for condition in ("ranking","c1","c2","no_logits","logits_only","residual"):
        assert np.isfinite(objective(model(view(x,condition)),y,condition,z).item())
    path = tmp_path / "head.safetensors"; model.save_weights(str(path))
    restored = CandidateHead(26,64); restored.load_weights(str(path))
    assert np.array_equal(np.array(a),np.array(restored(x)))
    assert sum(v.size for _,v in tree_flatten(model.parameters())) == (26+1)*64+65


@pytest.mark.skipif(os.environ.get("RUN_MLX_TESTS") != "1", reason="opt-in MLX integration")
def test_optimizer_resume_exact(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    from latent_loop.rendezvous.models import CandidateHead, objective
    mx.random.seed(17)
    a=CandidateHead(26);optimizer=optim.AdamW(.001)
    x=mx.random.normal((5,4,26));y=mx.array([[1.,0.,0.,0.]]*5);z=mx.zeros((5,4))
    def step(m,o):
        _,g=nn.value_and_grad(m,lambda model:objective(model(x),y,"correctness",z))(m)
        o.update(m,g);mx.eval(m.parameters(),o.state)
    step(a,optimizer)
    a.save_weights(str(tmp_path/'head.safetensors'))
    mx.savez(str(tmp_path/'optimizer.npz'),**dict(tree_flatten(optimizer.state)))
    b=CandidateHead(26);b.load_weights(str(tmp_path/'head.safetensors'))
    other=optim.AdamW(.001);other.state=tree_unflatten(list(mx.load(str(tmp_path/'optimizer.npz')).items()))
    step(a,optimizer);step(b,other)
    for (_,u),(_,v) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters())):
        np.testing.assert_array_equal(np.array(u),np.array(v))
