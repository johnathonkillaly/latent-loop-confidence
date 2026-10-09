"""Secondary minimal Snake stage: exact teacher and deterministic baselines only."""
import time
import numpy as np
from latent_loop.environments.snake import Snake,teacher
from latent_loop.environments.maze import ACTIONS
from latent_loop.planning.data import RESULTS,write_json,save_rows,now

def generate(seed,count):
    rng=np.random.default_rng(seed); states=[]
    while len(states)<count:
        head=tuple(int(v) for v in rng.integers(0,4,2)); heading=int(rng.integers(4)); dr,dc=ACTIONS[heading]
        length=int(rng.integers(2,5)); body=tuple((head[0]-i*dr,head[1]-i*dc) for i in range(length))
        if any(not(0<=r<4 and 0<=c<4) for r,c in body): continue
        free=[(r,c) for r in range(4) for c in range(4) if (r,c) not in body]
        food=free[int(rng.integers(len(free)))]; states.append(Snake(4,body,food,heading))
    return states

def run():
    start=time.perf_counter(); states=generate(559001,60); records=[]; trajectories=[]
    for i,initial in enumerate(states):
        safety=[initial.step(a)[0].alive for a in range(4)]
        for h in (2,4,6):
            t=time.perf_counter(); target=teacher(initial,h)
            records.append({"id":i,"initial":initial.to_dict(),"horizon":h,"safety":safety,"teacher":target,"teacher_seconds":time.perf_counter()-t})
        for name in ("random_legal","safe_greedy","oracle_h2","oracle_h4","oracle_h6"):
            s=initial; rng=np.random.default_rng(559100+i); steps=0; illegal=0; decisions=[]; inference=0.
            for _ in range(12):
                t=time.perf_counter(); safe=[a for a in range(4) if s.step(a)[0].alive]
                if name.startswith("oracle"):
                    options=teacher(s,int(name[-1]))["optimal"]; a=options.index(True)
                elif name=="random_legal":
                    legal=[a for a in range(4) if a!=(s.heading+2)%4]; a=int(rng.choice(legal))
                else:
                    def key(a):
                        nxt,_=s.step(a); p=nxt.body[0]
                        return (0 if nxt.food is None else abs(p[0]-nxt.food[0])+abs(p[1]-nxt.food[1]),a)
                    a=min(safe,key=key) if safe else s.heading
                inference+=time.perf_counter()-t; s,info=s.step(a); decisions.append(a); illegal+=int(info["illegal"])
                if not s.alive: break
                steps+=1
            trajectories.append({"id":i,"policy":name,"initial":initial.to_dict(),"survival_steps":steps,"food_collected":s.collected,"collision":not s.alive,"illegal_actions":illegal,"decisions":decisions,"inference_seconds":inference})
    save_rows(RESULTS/"snake"/"teacher_smoke.jsonl.gz",records); save_rows(RESULTS/"snake"/"episodes.jsonl.gz",trajectories)
    summary={"status":"Environment, simulator tests, exact bounded teacher and baseline smoke only; no Snake neural model trained", "seed":559001,"episodes":60,"board":4,"maximum_steps":12,"food_respawn":False,"reverse":"prohibited; attempting reverse terminates as illegal collision","utility_order":["survival_steps","food_collected","reachable_free_space","negative_manhattan_to_food"],"completed":now(),"seconds":time.perf_counter()-start,"policies":{}}
    for name in sorted({r["policy"] for r in trajectories}):
        group=[r for r in trajectories if r["policy"]==name]
        summary["policies"][name]={"survival_steps":float(np.mean([r["survival_steps"] for r in group])),"food_collected":float(np.mean([r["food_collected"] for r in group])),"collision_rate":float(np.mean([r["collision"] for r in group])),"illegal_actions_mean":float(np.mean([r["illegal_actions"] for r in group])),"mean_episode_inference_seconds":float(np.mean([r["inference_seconds"] for r in group]))}
    write_json(RESULTS/"snake_summary.json",summary); print(summary,flush=True)
