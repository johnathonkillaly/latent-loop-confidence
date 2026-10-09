"""Deterministic, conservatively topology-disjoint prospective maze data."""
import gzip
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path
import numpy as np
from latent_loop.environments.maze import Maze, candidate, distances, labels
from latent_loop.planning.data import write_json, save_rows, digest, now

EXP=Path('experiments/experiment_05c'); OUT=Path('results/experiment_05c'); ART=Path('artifacts/experiment_05c')
def config(): return json.loads((EXP/'config.json').read_text())
def read(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]
def load(split,final=False):
    spec=config()['datasets'][split]
    if spec.get('final'):
        if not final or not (EXP/'lockbox_opened.json').exists():raise PermissionError('Final data sealed until protocol freeze')
        path=ART/'sealed'/f'{split}.jsonl.gz'
    else:path=OUT/'datasets'/f'{split}.jsonl.gz'
    return read(path)
def independent_distance(m):
    """Heap Dijkstra independently checks queue-BFS implementation."""
    d=np.full((m.size,m.size),-1,dtype=int); q=[(0,*m.goal)]
    while q:
        cost,r,c=heapq.heappop(q)
        if d[r,c]>=0:continue
        d[r,c]=cost
        for dr,dc in ((-1,0),(0,1),(1,0),(0,-1)):
            rr,cc=r+dr,c+dc
            if 0<=rr<m.size and 0<=cc<m.size and not m.walls[rr][cc] and d[rr,cc]<0:heapq.heappush(q,(cost+1,rr,cc))
    return d

def graph_key(m):
    """Unrooted component WL invariant, NOT a graph-isomorphism certificate.

    Reject ALL equal fingerprints, including possible non-isomorphic collisions.
    Isomorphic components always have equal fingerprints and cannot cross splits.
    """
    d=distances(m,m.agent); nodes=[tuple(p) for p in np.argwhere(d>=0)]; index={p:i for i,p in enumerate(nodes)}
    adj=[[index[q] for a in range(4) if (q:=m.neighbor(p,a)) in index] for p in nodes]
    colors=[str(len(n)) for n in adj]
    for _ in range(8):
        colors=[hashlib.sha256((colors[i]+'|'+'|'.join(sorted(colors[j] for j in n))).encode()).hexdigest() for i,n in enumerate(adj)]
    key=hashlib.sha256(('WL8|'+str(len(nodes))+'|'+str(sum(map(len,adj)))+'|'+'|'.join(sorted(colors))).encode()).hexdigest()
    return key, {'component_nodes':len(nodes),'edges':sum(map(len,adj))//2,'junctions':sum(len(n)>=3 for n in adj),'dead_ends':sum(len(n)==1 for n in adj),'degree_histogram':dict(Counter(map(len,adj)))}

def dfs_board(rng,size):
    walls=np.ones((size,size),int); rooms=[(r,c) for r in range(0,size,2) for c in range(0,size,2)]
    p=rooms[int(rng.integers(len(rooms)))];walls[p]=0;visited={p};stack=[p]
    while stack:
        r,c=stack[-1];choices=[(r+dr,c+dc) for dr,dc in ((-2,0),(0,2),(2,0),(0,-2)) if 0<=r+dr<size and 0<=c+dc<size and (r+dr,c+dc) not in visited]
        if not choices:stack.pop();continue
        q=choices[int(rng.integers(len(choices)))];walls[(r+q[0])//2,(c+q[1])//2]=0;walls[q]=0;visited.add(q);stack.append(q)
    return walls

def generate():
    if (EXP/'dataset_manifest.json').exists():raise RuntimeError('Refuse overwrite of existing prospective data')
    cfg=config();seen_layout=set();seen_graph=set();historical=0
    for path in sorted(Path('results/experiment_05b/datasets').glob('*.gz')):
        for row in read(path):
            m=Maze.from_dict(row);seen_layout.add(m.canonical_hash());seen_graph.add(graph_key(m)[0]);historical+=1
    manifest={'created':now(),'historical_layouts_excluded':historical,'historical_graph_fingerprints':len(seen_graph),'method':'D4 wall-layout and conservative unrooted component WL8 fingerprint exclusion; equality is not proof of isomorphism','splits':{}}
    for split,spec in cfg['datasets'].items():
        rng=np.random.default_rng(spec['seed']);rows=[];reject=Counter();attempt=0
        while len(rows)<spec['count']:
            attempt+=1
            if attempt>spec['count']*10000:raise RuntimeError(f'Generator exhausted {split} {len(rows)}; preserve partial diagnostics')
            cell=len(rows)%(len(spec['sizes'])*len(spec['bins']));size=spec['sizes'][cell%len(spec['sizes'])];bin_idx=cell//len(spec['sizes']);lo,hi=spec['bins'][bin_idx]
            if spec['algorithm']=='dfs':walls=dfs_board(rng,size)
            else:
                m=candidate(rng,size,'occupancy',spec.get('density',float(rng.uniform(.30,.45))))
                if m is None:reject['empty']+=1;continue
                walls=np.array(m.walls)
            free=np.argwhere(walls==0)
            if len(free)<2:continue
            goal=tuple(free[int(rng.integers(len(free)))]);m=Maze(tuple(map(tuple,walls)),goal,goal);d=distances(m)
            starts=[]
            for p in np.argwhere((d>=lo)&(d<=hi)):
                p=tuple(p)
                if sum(m.free(m.neighbor(p,a)) for a in range(4))>=2:starts.append(p)
            if not starts:reject['distance_or_branching']+=1;continue
            m=Maze(m.walls,starts[int(rng.integers(len(starts)))],goal);h=m.canonical_hash()
            if h in seen_layout:reject['layout']+=1;continue
            g,topo=graph_key(m)
            if g in seen_graph:reject['graph_invariant']+=1;continue
            y=labels(m);dd=independent_distance(m);assert np.array_equal(d,dd)
            opt=[m.free(q:=m.neighbor(m.agent,a)) and dd[q]==y['distance']-1 for a in range(4)]
            assert opt==y['optimal']
            topo.update({'goal_influence_min_hops':y['distance'],'agent_degree':sum(y['legal']),'junctions_radius4':int(sum(sum(m.free(m.neighbor(tuple(p),a)) for a in range(4))>=3 for p in np.argwhere((distances(m,m.agent)>=0)&(distances(m,m.agent)<=4)))), 'geometrically_close_topologically_far':y['manhattan']<=4 and y['distance']>=13})
            rows.append({**m.to_dict(),'id':f'{split}-{len(rows):05d}','layout_hash':h,'graph_fingerprint':g,'source_generator':spec['algorithm'],'path_bin':f'{lo}-{hi}','topology':topo,'labels':y})
            seen_layout.add(h);seen_graph.add(g)
        path=(ART/'sealed' if spec.get('final') else OUT/'datasets')/f'{split}.jsonl.gz';save_rows(path,rows)
        manifest['splits'][split]={**spec,'sha256':digest(path),'attempts':attempt,'rejections':dict(reject)};write_json(EXP/'generation_progress.json',manifest)
        print(f'{split}: {len(rows)} unique topologies, {attempt} proposals',flush=True)
    write_json(EXP/'dataset_manifest.json',manifest)

if __name__=='__main__':generate()
