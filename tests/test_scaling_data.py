import numpy as np
import pytest
from latent_loop.environments.maze import Maze,distances
from latent_loop.scaling.data import graph_key,independent_distance,dfs_board,load

def test_isomorphic_graphs_in_different_embeddings_share_conservative_key():
    # Five-node path straight versus bent: distinct spatial layouts, same graph.
    a=Maze(((0,0,0,0,0),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1),(1,1,1,1,1)),(0,0),(0,4))
    b=Maze(((0,0,0,1,1),(1,1,0,1,1),(1,1,0,1,1),(1,1,1,1,1),(1,1,1,1,1)),(0,0),(2,2))
    assert a.canonical_hash()!=b.canonical_hash()
    assert graph_key(a)[0]==graph_key(b)[0]
    moved=Maze(a.walls,(0,1),(0,3));assert graph_key(a)==graph_key(moved)

def test_solver_and_structural_generator():
    for s in (5,7,9,15):
        w=dfs_board(np.random.default_rng(s),s);m=Maze(tuple(map(tuple,w)),(0,0),(s-1,s-1))
        np.testing.assert_array_equal(independent_distance(m),distances(m))
        assert graph_key(m)[1]['edges']==graph_key(m)[1]['component_nodes']-1

def test_final_api_refuses_unfrozen_access():
    with pytest.raises(PermissionError):load('size9',False)
