import numpy as np
import pytest
mx=pytest.importorskip('mlx.core')
import mlx.nn as nn
from mlx.utils import tree_flatten
from latent_loop.environments.maze import Maze
from latent_loop.planning.grid_recurrence import GridPlanner,set_loss
from latent_loop.scaling.model import forward,GlobalPlanner

def test_extended_forward_exact_parity_bounded_locality():
    mx.random.seed(101);m=GridPlanner(8);maze=Maze(tuple(tuple(0 for _ in range(9)) for _ in range(9)),(0,0),(8,8));x=mx.array(maze.observation()[None]);old=m(x,(1,4,16),True);new=forward(m,x,(1,4,16,32),True)
    for d in old:
        for f in old[d]:np.testing.assert_array_equal(np.asarray(old[d][f]),np.asarray(new[d][f]))
    assert np.max(np.abs(np.asarray(new[32]['states'])))<=1
    no_goal=maze.observation();no_goal[...,3]=0
    np.testing.assert_array_equal(np.asarray(forward(m,x,(8,))[8]['logits']),np.asarray(forward(m,mx.array(no_goal[None]),(8,))[8]['logits']))
    with pytest.raises(ValueError):forward(m,x,(33,))

def test_new_forward_gradient_and_global_context():
    maze=Maze(((0,0,0),(0,1,0),(0,0,0)),(0,0),(2,2));x=mx.array(maze.observation()[None]);y=mx.array([[False,True,True,False]])
    for m in (GridPlanner(8),GlobalPlanner(8,4)):
        d=16 if m.shared else 4
        loss,g=nn.value_and_grad(m,lambda model:set_loss(forward(model,x,(d,))[d]['logits'],y))(m);mx.eval(loss,g)
        assert np.isfinite(float(loss))
        assert all(np.isfinite(np.asarray(v)).all() for _,v in tree_flatten(g))
        assert float(mx.linalg.norm(g['encoder']['weight']))>0
        if hasattr(m,'context'):assert float(mx.linalg.norm(g['context']['weight']))>0

def test_diagnostic_metadata_export_preserves_values():
    import json
    from latent_loop.scaling.diagnostic_export import native
    v={'coordinates':[np.int64(2),np.int32(3)],'measurement':np.float64(.25),'flag':np.bool_(True)}
    assert json.loads(json.dumps(native(v)))=={'coordinates':[2,3],'measurement':.25,'flag':True}
