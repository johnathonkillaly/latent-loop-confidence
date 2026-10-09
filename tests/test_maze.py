import numpy as np
import pytest
from latent_loop.environments.maze import Maze,labels,distances,candidate

def test_moves_and_observation_no_labels():
    m=Maze(((0,0,0),(0,1,0),(0,0,0)),(0,0),(2,2))
    unchanged,reward,done,info=m.step(0)
    assert unchanged==m and reward<0 and info["invalid"] and not done
    moved,_,_,_=m.step(1); assert moved.agent==(0,1) and m.agent==(0,0)
    assert m.observation().shape==(3,3,4)
    assert m.observation()[0,0,2]==1 and m.observation()[2,2,3]==1
    assert labels(m)["optimal"]==[False,True,True,False]

def test_terminal_and_validation():
    m=Maze(((0,0),(0,0)),(0,0),(0,0))
    assert m.step(3)[2]
    with pytest.raises(ValueError): labels(m)
    with pytest.raises(ValueError): Maze(((1,0),(0,0)),(0,0),(1,1))

def test_canonical_layout_excludes_placement_and_dihedral_copies():
    w=np.array([[0,0,1],[0,1,0],[0,0,0]])
    m=Maze(tuple(map(tuple,w)),(0,0),(2,2))
    rotated=Maze(tuple(map(tuple,np.rot90(w))),(0,1),(2,2))
    moved=Maze(m.walls,(0,1),(2,0))
    assert m.canonical_hash()==rotated.canonical_hash()==moved.canonical_hash()
    assert m.canonical_hash(False)!=moved.canonical_hash(False)

def test_bfs_independent_floyd_warshall():
    rng=np.random.default_rng(14)
    for _ in range(30):
        m=candidate(rng)
        if m is None: continue
        nodes=[(r,c) for r in range(m.size) for c in range(m.size) if m.free((r,c))]
        matrix=np.full((len(nodes),len(nodes)),999); np.fill_diagonal(matrix,0)
        for i,p in enumerate(nodes):
            for j,q in enumerate(nodes):
                if abs(p[0]-q[0])+abs(p[1]-q[1])==1: matrix[i,j]=1
        for k in range(len(nodes)): matrix=np.minimum(matrix,matrix[:,k,None]+matrix[None,k,:])
        d=distances(m)
        for i,p in enumerate(nodes): assert d[p]==(matrix[i,nodes.index(m.goal)] if matrix[i,nodes.index(m.goal)]<999 else -1)
        if d[m.agent]>0:
            y=labels(m)
            assert any(y["optimal"])
            assert all(y["regret"][a]==0 for a in range(4) if y["optimal"][a])
