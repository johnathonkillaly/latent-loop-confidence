import numpy as np
import pytest
mx=pytest.importorskip("mlx.core")
import mlx.nn as nn
from mlx.utils import tree_flatten
from latent_loop.environments.maze import Maze
from latent_loop.planning.grid_recurrence import GridPlanner,set_loss,neighbors

def test_no_future_environment_and_local_receptive_field():
    # Goals beyond radius two cannot influence a two-loop agent readout.
    mx.random.seed(7); model=GridPlanner(8)
    walls=tuple(tuple(0 for _ in range(5)) for _ in range(5))
    m=Maze(walls,(0,0),(4,4)); other=Maze(walls,(0,0),(4,3))
    x=mx.array(np.stack([m.observation(),other.observation()]))
    before=m.to_dict(); trace=model(x,(1,2,4,8))
    for d in (1,2): np.testing.assert_allclose(np.asarray(trace[d]["logits"])[0],np.asarray(trace[d]["logits"])[1],atol=0,rtol=0)
    assert before==m.to_dict()
    n=np.asarray(neighbors(mx.ones((1,3,3,1))))
    np.testing.assert_array_equal(n[0,0,0],[0,1,1,0])

def test_gradients_determinism_checkpoint_and_tied_loss(tmp_path):
    mx.random.seed(9); m=GridPlanner(8)
    maze=Maze(((0,0,0),(0,1,0),(0,0,0)),(0,0),(2,2))
    x=mx.array(maze.observation()[None]); y=mx.array([[False,True,True,False]])
    loss,g=nn.value_and_grad(m,lambda model:set_loss(model(x,(4,))[4]["logits"],y))(m)
    mx.eval(loss,g)
    flat=tree_flatten(g)
    assert np.isfinite(float(loss)) and all(np.isfinite(np.asarray(v)).all() for _,v in flat)
    assert all(float(mx.linalg.norm(v))>0 for k,v in flat if k.endswith("weight"))
    a=np.asarray(m(x,(1,2,4,8))[4]["logits"])
    np.testing.assert_array_equal(a,np.asarray(m(x,(4,))[4]["logits"]))
    path=tmp_path/"weights.safetensors"; m.save_weights(str(path)); restored=GridPlanner(8); restored.load_weights(str(path))
    np.testing.assert_array_equal(a,np.asarray(restored(x,(4,))[4]["logits"]))
    assert float(set_loss(mx.array([[0.,10.,10.,0.]]),y))<.001

def test_wall_states_zero_and_larger_boards():
    m=GridPlanner(8); maze=Maze(((0,0,0),(0,1,0),(0,0,0)),(0,0),(2,2))
    out=m(mx.array(maze.observation()[None]),(1,8),True)
    for record in out.values(): np.testing.assert_array_equal(np.asarray(record["states"])[0,1,1],0)
    assert m(mx.zeros((2,7,7,4)),(4,))[4]["logits"].shape==(2,4)

def test_optimizer_restoration_exact(tmp_path):
    import mlx.optimizers as optim
    from mlx.utils import tree_unflatten
    mx.random.seed(12); full=GridPlanner(8); optimizer=optim.AdamW(.001)
    x=mx.array(Maze(((0,0),(0,0)),(0,0),(1,1)).observation()[None]); y=mx.array([[False,True,True,False]])
    def step(model,opt):
        loss,g=nn.value_and_grad(model,lambda m:set_loss(m(x,(4,))[4]["logits"],y))(model)
        opt.update(model,g); mx.eval(model.parameters(),opt.state)
    step(full,optimizer)
    full.save_weights(str(tmp_path/"model.safetensors"))
    mx.save_safetensors(str(tmp_path/"optimizer.safetensors"),dict(tree_flatten(optimizer.state)))
    step(full,optimizer)
    restored=GridPlanner(8); restored.load_weights(str(tmp_path/"model.safetensors")); resumed=optim.AdamW(.001)
    resumed.state=tree_unflatten(list(mx.load(str(tmp_path/"optimizer.safetensors")).items()))
    step(restored,resumed)
    for (_,a),(_,b) in zip(tree_flatten(full.parameters()),tree_flatten(restored.parameters())):
        np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
