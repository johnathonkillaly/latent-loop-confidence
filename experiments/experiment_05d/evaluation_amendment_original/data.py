"""Pure CPU synthetic graphs, independent labels and topology-group splits.

Only observation() crosses the inference boundary. No coordinates or IDs become
learned features. Labels and solver-derived metadata remain offline.
"""
from pathlib import Path
from collections import deque
from datetime import datetime, timezone
import gzip
import hashlib
import json
import numpy as np

EXP=Path('experiments/experiment_05d'); OUT=Path('results/experiment_05d')
ART=Path('artifacts/experiment_05d'); REPORT=Path('reports/experiment_05d')

def now(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
def config(): return json.loads((EXP/'config.json').read_text())
def save_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=''.join(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n' for r in rows).encode()
    path.write_bytes(gzip.compress(raw,mtime=0))
def read_rows(path): return [json.loads(x) for x in gzip.decompress(Path(path).read_bytes()).splitlines()]
def load(split,final=False):
    if split not in ('base_train','scorer_train','validation'):
        if not final or not (EXP/'lockbox_opened.json').exists(): raise RuntimeError('Final data are sealed')
    return read_rows(OUT/'datasets'/f'{split}.jsonl.gz')

def adjacency(n,edges):
    a=np.zeros((n,n),dtype=np.float32)
    for u,v in edges:a[u,v]=a[v,u]=1
    return a

def bfs(a,goal):
    dist=[-1]*len(a);dist[goal]=0;q=deque([goal])
    while q:
        v=q.popleft()
        for u in np.flatnonzero(a[v]):
            if dist[u]<0:dist[u]=dist[v]+1;q.append(int(u))
    return dist

def independent_distances(a):
    """Floyd–Warshall, independently implemented without BFS traversal."""
    d=np.where(a>0,1,10000).astype(np.int32);np.fill_diagonal(d,0)
    for k in range(len(a)):d=np.minimum(d,d[:,k,None]+d[None,k,:])
    return d

def topology(a):
    """Unrooted WL invariant. Reject equal invariants, never assert isomorphism."""
    colors=[str(int(x)) for x in a.sum(1)]
    for _ in range(len(a)):
        colors=[hashlib.sha256((colors[v]+'|'+','.join(sorted(colors[u] for u in np.flatnonzero(a[v])))).encode()).hexdigest() for v in range(len(a))]
    return hashlib.sha256(('|'.join(sorted(colors))).encode()).hexdigest()

def label(a,start,goal,candidates):
    dist=bfs(a,goal);n=len(a)
    optimal=[(v==-1 if dist[start]<0 else v>=0 and dist[v]==dist[start]-1) for v in candidates]
    independent=independent_distances(a)[:,goal]
    assert all((d<0 and z>=10000) or d==z for d,z in zip(dist,independent))
    assert any(optimal)
    return {'optimal':optimal,'distance':dist[start],'distances':dist,
            'reachable':dist[start]>=0,'optimal_count':sum(optimal),
            'candidate_distances':[dist[v] if v>=0 else -1 for v in candidates]}

def make_row(a,start,goal,rng,family,pair_id=None,arm=None):
    n=len(a);perm=rng.permutation(n);a=a[perm][:,perm]
    start=int(np.flatnonzero(perm==start)[0]);goal=int(np.flatnonzero(perm==goal)[0])
    candidates=[int(v) for v in np.flatnonzero(a[start])]+[-1];rng.shuffle(candidates)
    edges=np.argwhere(np.triu(a,1)>0).tolist()
    obs={'n':n,'edges':edges,'start':start,'goal':goal,'candidates':candidates,
         'tie_keys':rng.permutation(n+1).tolist()}
    labels=label(a,start,goal,candidates);deg=a.sum(1).astype(int).tolist()
    record={'observation':obs,'labels':labels,'topology':topology(a),
            'metadata':{'family':family,'nodes':n,'edges':len(edges),'degrees':deg,
                        'junctions':sum(x>=3 for x in deg),'dead_ends':sum(x==1 for x in deg),
                        'candidate_count':len(candidates)}}
    if pair_id is not None:record['pair_id']=pair_id;record['metadata']['arm_distance']=arm
    record['id']=hashlib.sha256(json.dumps(obs,sort_keys=True).encode()).hexdigest()[:24]
    return record

def observation(row):
    """Whitelisted raw observation only. Solvers/metadata never consulted."""
    return dict(row['observation'])

def arrays(rows):
    obs=[observation(r) for r in rows];n=max(o['n'] for o in obs);c=max(len(o['candidates']) for o in obs);b=len(rows)
    a=np.zeros((b,n,n),np.float32);x=np.zeros((b,n,4),np.float32);valid=np.zeros((b,n,1),np.float32)
    idx=np.zeros((b,c),np.int32);cm=np.zeros((b,c),bool);ab=np.zeros((b,c,1),np.float32);starts=np.zeros(b,np.int32)
    keys=np.full((b,c),100000,np.int32);y=np.zeros((b,c),bool)
    for i,(o,r) in enumerate(zip(obs,rows)):
        nn=o['n'];aa=adjacency(nn,o['edges']);a[i,:nn,:nn]=aa;valid[i,:nn]=1
        x[i,:nn,0]=1;x[i,o['start'],1]=1;x[i,o['goal'],2]=1
        x[i,:nn,3]=aa.sum(1)/max(1,nn-1);starts[i]=o['start']
        for j,v in enumerate(o['candidates']):
            cm[i,j]=True;idx[i,j]=max(0,v);ab[i,j,0]=v<0;keys[i,j]=o['tie_keys'][v if v>=0 else nn]
        if 'labels' in r:y[i,:len(o['candidates'])]=r['labels']['optimal']
    inputs={'a':a,'x':x,'valid':valid,'idx':idx,'cm':cm,'ab':ab,'starts':starts}
    return inputs,y,keys

def choose(scores,keys,mask=None):
    """Presentation-independent tie resolution using nonlearned random node keys.

    Keys follow node identity under relabeling, never enter neural features, and
    carry no label information. A 1e-6 numerical tie tolerance is fixed prospectively.
    """
    z=np.asarray(scores);m=np.ones_like(z,dtype=bool) if mask is None else mask
    near=m & (z>=np.max(np.where(m,z,-1e30),axis=-1,keepdims=True)-1e-6)
    return np.argmin(np.where(near,keys,1000000),axis=-1)

def graph(rng,n,family):
    a=np.zeros((n,n),np.float32)
    def edge(u,v): a[u,v]=a[v,u]=1
    if family in ('tree','sparse','cyclic','long'):
        for v in range(1,n):edge(v,int(rng.integers(v)) if family!='long' else v-1)
        p={'tree':0,'sparse':.08,'cyclic':.22,'long':.012}[family]
        for u in range(n):
            for v in range(u+1,n):
                if rng.random()<p:edge(u,v)
    elif family=='disconnected':
        cut=int(rng.integers(2,n-2))
        for lo,hi in ((0,cut),(cut,n)):
            for v in range(lo+1,hi):edge(v,int(rng.integers(lo,v)))
            for u in range(lo,hi):
                for v in range(u+1,hi):
                    if rng.random()<.16:edge(u,v)
    elif family=='community':
        # Held-out block-structured generator; a single bridge connects two dense modules.
        cut=n//2
        for lo,hi in ((0,cut),(cut,n)):
            for v in range(lo+1,hi):edge(v,int(rng.integers(lo,v)))
            for u in range(lo,hi):
                for v in range(u+1,hi):
                    if rng.random()<.45:edge(u,v)
        edge(int(rng.integers(cut)),int(rng.integers(cut,n)))
    else:raise ValueError(family)
    return a

def counterfactual(rng,arm,pair_id):
    # Two equal arms from start 0. Goal is a separate node attached to one distant
    # endpoint. The other arm has a distractor leaf, so no local degree cue flips.
    n=2*arm+9+int(rng.integers(0,5));goal=2*arm+1;other=goal+1
    a=np.zeros((n,n),np.float32)
    for offset in (0,arm):
        last=0
        for j in range(1,arm+1):v=offset+j;a[last,v]=a[v,last]=1;last=v
    # Same random branches on both pair members; attached past the initial fork.
    anchor=int(rng.choice(list(range(2,arm+1))+list(range(arm+2,2*arm+1))))
    for v in range(goal+3,n):
        u=anchor if v==goal+3 else int(rng.integers(goal+3,v))
        a[u,v]=a[v,u]=1
        for w in range(goal+3,v):
            if rng.random()<.2:a[w,v]=a[v,w]=1
    b=a.copy();a[arm,goal]=a[goal,arm]=1;a[2*arm,other]=a[other,2*arm]=1
    b[2*arm,goal]=b[goal,2*arm]=1;b[arm,other]=b[other,arm]=1
    # Shared presentation within a pair ensures exact local-input comparisons.
    state=rng.bit_generator.state
    r1=make_row(a,0,goal,rng,'counterfactual',pair_id,arm)
    rng.bit_generator.state=state;r2=make_row(b,0,goal,rng,'counterfactual',pair_id,arm)
    assert r1['labels']['distance']==r2['labels']['distance']==arm+1
    o1={v for v,y in zip(r1['observation']['candidates'],r1['labels']['optimal']) if y}
    o2={v for v,y in zip(r2['observation']['candidates'],r2['labels']['optimal']) if y}
    assert not o1&o2
    return [r1,r2]

def generate():
    if (EXP/'lockbox_opened.json').exists():raise RuntimeError('Cannot replace opened datasets')
    cfg=config();rng=np.random.default_rng(cfg['dataset_seed']);used=set();manifest={'seed':cfg['dataset_seed'],'splits':{},'created':now()}
    def produce(split,counts,force=None):
        rows=[];attempts=0;local=set()
        for n,count in counts.items():
            n=int(n);done=0
            while done<count:
                attempts+=1
                if attempts>300000:raise RuntimeError(f'Topology rejection exhausted {split}')
                bucket=done%20
                family=force or ('disconnected' if bucket>=17 else str(rng.choice(['tree','sparse','cyclic'],p=[.12,.7,.18])))
                a=graph(rng,n,family)
                distances=independent_distances(a);degree=a.sum(1)
                if force=='long':eligible=(distances>=max(6,n//3))&(distances<10000)
                elif force=='community':eligible=(distances>=2)&(distances<10000)
                elif bucket<5:eligible=(distances>=1)&(distances<=2)
                elif bucket<13:eligible=(distances>=3)&(distances<=4)
                elif bucket<17:eligible=(distances>=5)&(distances<=9)
                else:eligible=distances>=10000
                pairs=np.argwhere(eligible&(degree[:,None]>=2))
                if not len(pairs):continue
                fp=topology(a)
                if fp in used or fp in local:continue
                s,g=map(int,pairs[int(rng.integers(len(pairs)))])
                rows.append(make_row(a,s,g,rng,family));local.add(fp);done+=1
        used.update(local);path=OUT/'datasets'/f'{split}.jsonl.gz';save_rows(path,rows)
        manifest['splits'][split]={'count':len(rows),'sha256':digest(path),'proposals':attempts,'topologies':len(local)}
        print(split,len(rows),'proposals',attempts,flush=True)
    for split,key in [('base_train','base_counts'),('scorer_train','scorer_counts'),('validation','validation_counts')]:produce(split,cfg[key])
    # Locked final data are generated here, but no fitting API can open them.
    for split,count in cfg['final_counts'].items():
        counts=({'10':160,'12':440} if split=='id' else {str({'community':12,'size16':16,'size32':32,'size64':64,'long':32}[split]):count})
        produce(split,counts,'community' if split=='community' else 'long' if split=='long' else None)
    pairs=[];groups=[]
    for arm in (3,5,9,13):
        done=0;tries=0
        while done<60:
            tries+=1
            if tries>100000:raise RuntimeError('Counterfactual rejection exhausted')
            rr=counterfactual(rng,arm,f'arm{arm}_{done:03d}');fps={r['topology'] for r in rr}
            if fps&used:continue
            # Intentional matching/isomorphic topology within pair, never across pairs.
            used.update(fps);pairs.extend(rr);done+=1
    path=OUT/'datasets/counterfactual.jsonl.gz';save_rows(path,pairs)
    manifest['splits']['counterfactual']={'count':len(pairs),'pairs':240,'sha256':digest(path)}
    write(EXP/'dataset_manifest.json',manifest)

def audit():
    manifest=json.loads((EXP/'dataset_manifest.json').read_text());groups={};total=0;summary={}
    for split,entry in manifest['splits'].items():
        path=OUT/'datasets'/f'{split}.jsonl.gz';assert digest(path)==entry['sha256']
        rows=read_rows(path);assert len(rows)==entry['count'];stats={}
        for r in rows:
            o=r['observation'];a=adjacency(o['n'],o['edges']);assert topology(a)==r['topology']
            assert label(a,o['start'],o['goal'],o['candidates'])==r['labels']
            group=split+':'+r.get('pair_id',r['id']);old=groups.setdefault(r['topology'],group);assert old==group
            assert set(o['candidates'])==set(np.flatnonzero(a[o['start']]))|{-1}
            family=r['metadata']['family'];stats[family]=stats.get(family,0)+1;total+=1
        summary[split]={'count':len(rows),'families':stats,'reachable':sum(r['labels']['reachable'] for r in rows),
                        'ties':sum(r['labels']['optimal_count']>1 for r in rows),
                        'mean_distance_reachable':float(np.mean([r['labels']['distance'] for r in rows if r['labels']['reachable']]))}
    write(OUT/'dataset_audit.json',{'labels_verified':total,'topology_groups':len(set(groups.values())),'summary':summary,'solver':'BFS independently checked against Floyd–Warshall','verified':now()})
    return total
