import numpy as np
import pytest
mx=pytest.importorskip('mlx.core');nn=pytest.importorskip('mlx.nn')
from mlx.utils import tree_flatten
from latent_loop.decision_bridge.data import adjacency,make_row,arrays,counterfactual
from latent_loop.decision_bridge.model import GraphPolicy,Scorer,tensor_inputs,scorer_features,set_loss

def test_gradients_bounded_states_and_checkpoint(tmp_path):
    row=make_row(adjacency(6,[(0,1),(0,2),(1,3),(3,4),(2,5)]),0,4,np.random.default_rng(3),'test')
    x,y,k=arrays([row]);x=tensor_inputs(x);model=GraphPolicy();out=model(x,(1,2,4,8,16),True)
    assert all(np.max(np.abs(np.asarray(r['states'])))<=1 for r in out.values())
    loss,g=nn.value_and_grad(model,lambda m:set_loss(m(x,(4,))[4]['logits'],mx.array(y)))(model)
    grads=[np.asarray(v) for _,v in tree_flatten(g)];assert all(np.isfinite(v).all() for v in grads)
    assert sum(np.abs(v).sum() for v in grads)>0
    for name,v in tree_flatten(g):
        if name.startswith('updates') and name.endswith('weight'):assert np.abs(np.asarray(v)).sum()>0
    path=str(tmp_path/'model.safetensors');model.save_weights(path);other=GraphPolicy();other.load_weights(path)
    np.testing.assert_array_equal(np.asarray(model(x,(4,))[4]['logits']),np.asarray(other(x,(4,))[4]['logits']))

def test_node_and_candidate_equivariance():
    row=make_row(adjacency(7,[(0,1),(0,2),(1,3),(3,4),(2,5),(5,6)]),0,4,np.random.default_rng(3),'test')
    x,y,keys=arrays([row]);model=GraphPolicy();original=model(tensor_inputs(x),(4,))[4]
    rng=np.random.default_rng(7);p=rng.permutation(7);inverse=np.argsort(p);c=rng.permutation(len(keys[0]))
    xx={k:v.copy() for k,v in x.items()};xx['a']=x['a'][:,p][:,:,p];xx['x']=x['x'][:,p];xx['valid']=x['valid'][:,p]
    xx['starts']=inverse[x['starts']];xx['idx']=inverse[x['idx']][:,c]
    for k in ('cm','ab'):xx[k]=x[k][:,c]
    changed=model(tensor_inputs(xx),(4,))[4]
    np.testing.assert_allclose(np.asarray(original['logits'])[:,c],np.asarray(changed['logits']),atol=2e-6)
    scorer=Scorer();q=scorer(scorer_features(original,mx.array(x['cm']),4));qq=scorer(scorer_features(changed,mx.array(xx['cm']),4))
    np.testing.assert_allclose(np.asarray(q)[:,c],np.asarray(qq),atol=2e-6)

def test_local_counterfactual_bound_and_unchanged_observation():
    rows=counterfactual(np.random.default_rng(33),5,'p');x,y,k=arrays(rows);before={key:v.copy() for key,v in x.items()}
    model=GraphPolicy();out=model(tensor_inputs(x),(1,2,4,8),True)
    for depth in (1,2,4):np.testing.assert_array_equal(np.asarray(out[depth]['logits'])[0],np.asarray(out[depth]['logits'])[1])
    for key in x:np.testing.assert_array_equal(before[key],x[key])
    assert not np.array_equal(np.asarray(out[8]['logits'])[0],np.asarray(out[8]['logits'])[1])

def test_graph_optimizer_exact_restore(tmp_path):
    import mlx.optimizers as optim
    from mlx.utils import tree_unflatten
    row=make_row(adjacency(5,[(0,1),(0,2),(1,3),(2,4)]),0,3,np.random.default_rng(17),'test')
    x,y,_=arrays([row]);x=tensor_inputs(x);y=mx.array(y);a=GraphPolicy();opt=optim.AdamW(.001)
    def step(model,optimizer):
        _,g=nn.value_and_grad(model,lambda m:set_loss(m(x,(4,))[4]['logits'],y))(model)
        optimizer.update(model,g);mx.eval(model.parameters(),optimizer.state)
    step(a,opt);a.save_weights(str(tmp_path/'a.safetensors'));mx.save_safetensors(str(tmp_path/'o.safetensors'),dict(tree_flatten(opt.state)))
    b=GraphPolicy();b.load_weights(str(tmp_path/'a.safetensors'));other=optim.AdamW(.001);other.state=tree_unflatten(list(mx.load(str(tmp_path/'o.safetensors')).items()))
    step(a,opt);step(b,other)
    for (_,aa),(_,bb) in zip(tree_flatten(a.parameters()),tree_flatten(b.parameters())):np.testing.assert_array_equal(np.asarray(aa),np.asarray(bb))
