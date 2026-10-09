import numpy as np
import pytest
from latent_loop.decision_bridge.data import adjacency,label,topology,make_row,arrays,counterfactual,choose,observation
from latent_loop.decision_bridge.interface import fuse,DecisionController,example

def test_ties_and_abstain():
    a=adjacency(5,[(0,1),(0,2),(1,3),(2,3)])
    assert label(a,0,3,[1,2,-1])['optimal']==[True,True,False]
    assert label(a,0,4,[1,2,-1])['optimal']==[False,False,True]

def test_permutation_invariant_topology_and_labels():
    rng=np.random.default_rng(25);a=adjacency(9,[(0,1),(0,2),(1,3),(3,4),(4,5),(2,6),(6,7),(7,8)])
    for _ in range(10):
        r=make_row(a,0,5,rng,'test');assert r['topology']==topology(a)
        assert r['labels']['distance']==4
        assert sum(r['labels']['optimal'])==1

def test_oracle_never_enters_inputs():
    r=make_row(adjacency(4,[(0,1),(1,2),(1,3)]),0,2,np.random.default_rng(2),'test')
    x,_,_=arrays([r]);r['labels']={'optimal':[False]*len(r['observation']['candidates']),'distance':100000}
    r['metadata']={'oracle':'malicious'};xx,_,_=arrays([r])
    assert set(observation(r))=={'n','edges','start','goal','candidates','tie_keys'}
    for k in x:np.testing.assert_array_equal(x[k],xx[k])

def test_counterfactual_local_identity():
    for arm in (3,5,9,13):
        a,b=counterfactual(np.random.default_rng(arm),arm,'p')
        oa,ob=a['observation'],b['observation'];assert oa['start']==ob['start']
        aa=adjacency(oa['n'],oa['edges']);bb=adjacency(ob['n'],ob['edges'])
        region={oa['start']}
        for _ in range(arm-1):region|={int(v) for u in region for v in np.flatnonzero(aa[u])}
        np.testing.assert_array_equal(aa[list(region)],bb[list(region)])
        assert a['labels']['optimal']!=b['labels']['optimal']

def test_fusion_controls_and_tie_resolution():
    z=np.array([[1.,2.,2.],[.4,.3,-2.]]);keys=np.array([[7,11,3],[8,2,5]])
    np.testing.assert_array_equal(choose(z,keys),choose(z/2,keys))
    np.testing.assert_array_equal(choose(z,keys),choose(fuse(z,np.ones_like(z),8),keys))
    perm=[2,0,1];selected=choose(z,keys)
    mapped=np.asarray(perm)[choose(z[:,perm],keys[:,perm])];np.testing.assert_array_equal(selected,mapped)
    assert set(example())=={'base','scored','after_optional_update'}

def test_lambda_zero_and_dtype_cannot_change_ties():
    z=np.array([[5.,5.-9.6e-7,1.]],dtype=np.float32);keys=np.array([[7,2,3]])
    np.testing.assert_array_equal(choose(z,keys),choose(z.astype(np.float64),keys))
    np.testing.assert_array_equal(choose(z,keys),choose(fuse(z,np.array([[2.,-9.,1.]]),0),keys))

def test_final_api_guard_and_cluster_intervals(tmp_path,monkeypatch):
    from latent_loop.decision_bridge import data
    from latent_loop.decision_bridge.analysis import paired_interval
    monkeypatch.setattr(data,'EXP',tmp_path)
    with pytest.raises(RuntimeError,match='sealed'):data.load('id')
    r=paired_interval(np.array([[1,0,-1],[1,0,-1],[1,0,-1]]),100)
    assert r['independent_instances']==3 and r['seeds']==3
    assert r['estimate']==0 and r['ci95'][0]<0<r['ci95'][1]
