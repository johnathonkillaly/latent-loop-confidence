import numpy as np
from latent_loop.environments.maze import Maze,distances
from latent_loop.planning.oracles import propagation,shortest_path,greedy_action

def test_propagation_is_one_hop_and_converges():
    m=Maze(((0,0,0),(0,1,0),(0,0,0)),(0,0),(2,2))
    trace=propagation(m,9); bfs=distances(m)
    for k,d in enumerate(trace):
        assert np.array_equal(np.isfinite(d),(bfs>=0)&(bfs<=k))
        assert np.all(d[np.isfinite(d)]==bfs[np.isfinite(d)])
    assert len(shortest_path(m))-1==bfs[m.agent]
    assert greedy_action(m)==1
