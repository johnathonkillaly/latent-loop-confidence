"""Deterministic ordered-body Snake. Food is consumed once, never respawned."""
from dataclasses import dataclass,replace
from collections import deque
from functools import lru_cache
from latent_loop.environments.maze import ACTIONS

@dataclass(frozen=True)
class Snake:
    size: int
    body: tuple
    food: tuple | None
    heading: int
    walls: tuple=()
    alive: bool=True
    collected: int=0
    prohibit_reverse: bool=True

    def __post_init__(self):
        if not self.body or len(set(self.body))!=len(self.body): raise ValueError("Invalid ordered body")
        if self.heading not in range(4): raise ValueError("Invalid heading")
        if any(not self.free_boundary(p) or p in self.walls for p in self.body): raise ValueError("Body outside board or on wall")
        if any(abs(a[0]-b[0])+abs(a[1]-b[1])!=1 for a,b in zip(self.body,self.body[1:])): raise ValueError("Body must be cardinally contiguous")
        if self.food is not None and (not self.free_boundary(self.food) or self.food in self.body or self.food in self.walls): raise ValueError("Invalid food")

    def free_boundary(self,p): return 0<=p[0]<self.size and 0<=p[1]<self.size

    def step(self,action):
        if action not in range(4): raise ValueError("Action outside N/E/S/W")
        if not self.alive: return self,{"collision":False,"illegal":False,"ate":False,"terminal":True}
        reverse=self.prohibit_reverse and len(self.body)>1 and action==(self.heading+2)%4
        dr,dc=ACTIONS[action]; head=(self.body[0][0]+dr,self.body[0][1]+dc); eating=head==self.food
        occupied=self.body if eating else self.body[:-1]
        collision=reverse or not self.free_boundary(head) or head in self.walls or head in occupied
        info={"collision":collision,"illegal":reverse,"ate":eating and not collision,"terminal":collision}
        if collision: return replace(self,alive=False),info
        body=(head,)+self.body if eating else (head,)+self.body[:-1]
        return replace(self,body=body,heading=action,food=None if eating else self.food,collected=self.collected+int(eating)),info

    def reachable(self):
        occupied=set(self.body[1:])|set(self.walls); seen={self.body[0]}; q=deque(seen)
        while q:
            p=q.popleft()
            for dr,dc in ACTIONS:
                u=(p[0]+dr,p[1]+dc)
                if self.free_boundary(u) and u not in occupied and u not in seen: seen.add(u); q.append(u)
        return len(seen)

    def to_dict(self):
        return {"size":self.size,"body":[list(p) for p in self.body],"food":list(self.food) if self.food else None,"heading":self.heading,"walls":[list(p) for p in self.walls],"alive":self.alive,"collected":self.collected,"prohibit_reverse":self.prohibit_reverse}

    @classmethod
    def from_dict(cls,d):
        return cls(d["size"],tuple(map(tuple,d["body"])),tuple(d["food"]) if d["food"] is not None else None,d["heading"],tuple(map(tuple,d.get("walls",[]))),d.get("alive",True),d.get("collected",0),d.get("prohibit_reverse",True))

    def observation(self):
        """Explicit ordered body and heading, with no teacher utility."""
        return self.to_dict()

def utility(state,survival_steps):
    p=state.body[0]; distance=0 if state.food is None else abs(p[0]-state.food[0])+abs(p[1]-state.food[1])
    return survival_steps,state.collected,state.reachable(),-distance

def teacher(state,horizon):
    """Bounded exhaustive teacher over ACTUAL moving-tail dynamics.

    Lexicographic utility is frozen: survival steps, food, reachable space,
    negative Manhattan food distance. No unbounded rollouts or food respawn.
    """
    if not 1<=horizon<=6: raise ValueError("Bounded teacher horizon must be 1..6")
    @lru_cache(None)
    def search(s,remaining):
        if remaining==0 or not s.alive: return utility(s,0)
        best=None
        for a in range(4):
            next_state,_=s.step(a); result=search(next_state,remaining-1)
            score=(result[0]+int(next_state.alive),)+result[1:]
            if best is None or score>best: best=score
        return best
    scores=[]
    for a in range(4):
        nxt,_=state.step(a); value=search(nxt,horizon-1); scores.append((value[0]+int(nxt.alive),)+value[1:])
    maximum=max(scores)
    return {"optimal":[s==maximum for s in scores],"scores":[list(s) for s in scores],"utility_order":["survival_steps","food_collected","reachable_free_space","negative_manhattan_to_food"],"states_searched":search.cache_info().currsize}
