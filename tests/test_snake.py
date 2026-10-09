import pytest
from latent_loop.environments.snake import Snake,teacher

def test_tail_vacates_and_reverse_rule():
    body=((1,1),(1,0),(0,0),(0,1))
    s=Snake(4,body,(3,3),1,prohibit_reverse=False)
    nxt,info=s.step(0)
    assert nxt.alive and nxt.body==((0,1),(1,1),(1,0),(0,0)) and not info["collision"]
    short=Snake(4,((1,1),(1,0)),(3,3),1,prohibit_reverse=False)
    assert short.step(3)[0].alive
    prohibited=Snake(4,short.body,short.food,1)
    assert not prohibited.step(3)[0].alive and prohibited.step(3)[1]["illegal"]

def test_growth_self_wall_and_terminal():
    s=Snake(4,((1,1),(1,0)),(1,2),1)
    nxt,info=s.step(1)
    assert info["ate"] and len(nxt.body)==3 and nxt.food is None and nxt.collected==1
    wall=Snake(4,((0,0),(1,0)),(3,3),0)
    dead,info=wall.step(0); assert not dead.alive and info["collision"]
    assert dead.step(1)[0]==dead and dead.step(1)[1]["terminal"]
    loop=Snake(4,((1,1),(1,0),(2,0),(2,1),(2,2)),(3,3),1,prohibit_reverse=False)
    assert not loop.step(2)[0].alive
    obstacle=Snake(4,((1,1),(1,0)),(3,3),1,walls=((1,2),))
    assert not obstacle.step(1)[0].alive

def test_teacher_dynamics_and_ties():
    s=Snake(4,((1,1),(1,0)),(1,2),1)
    assert teacher(s,2)["optimal"][1]
    t=teacher(s,6); assert not t["optimal"][3] and t["states_searched"]<20000
    assert Snake.from_dict(s.to_dict())==s
    with pytest.raises(ValueError): teacher(s,7)

def test_teacher_against_exhaustive_two_action_sequences():
    import itertools
    from latent_loop.environments.snake import utility
    s=Snake(4,((1,1),(1,0),(2,0)),(0,2),1,walls=((2,2),))
    first_scores=[]
    for first in range(4):
        scores=[]
        for second in range(4):
            state=s;survived=0
            for a in (first,second):
                if not state.alive:break
                state,_=state.step(a);survived+=int(state.alive)
            scores.append(utility(state,survived))
        first_scores.append(max(scores))
    assert teacher(s,2)['scores']==[list(v) for v in first_scores]

def test_neural_observation_has_order_and_heading_without_teacher():
    from latent_loop.planning.snake_training import observation,canonical
    s=Snake(4,((1,1),(1,0),(2,0)),(3,3),1,walls=((0,0),))
    x=observation(s)
    assert x.shape==(4,4,9) and x[1,1,2]==1 and x[3,3,3]==1
    assert x[1,1,4]==.25 and x[1,0,4]==.5 and x[2,0,4]==.75
    assert x[0,0,0]==1 and x[0,0,1]==0 and x[...,6].min()==1
    assert len(canonical(s))==64
