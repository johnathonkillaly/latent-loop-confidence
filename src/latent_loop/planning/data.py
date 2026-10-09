"""Prospective maze splits with D4 wall-layout exclusion across every split."""
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from latent_loop.environments.maze import Maze, candidate, labels, family_matches

ROOT=Path("experiments/experiment_05b")
RESULTS=Path("results/experiment_05b")
ARTIFACTS=Path("artifacts/experiment_05b")

def now(): return datetime.now(timezone.utc).isoformat()
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n"); temporary.replace(path)
def save_rows(path,rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    # Timestamp-free gzip makes public regeneration byte-identical.
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw,mode="wb",mtime=0,filename="") as f:
            for row in rows: f.write((json.dumps(row,sort_keys=True)+"\n").encode())
def load_rows(split,final=False):
    if final and not (ROOT/"frozen_protocol.json").exists(): raise RuntimeError("Freeze protocol before opening final data")
    path=(ARTIFACTS/"sealed" if final else RESULTS/"datasets")/f"{split}.jsonl.gz"
    with gzip.open(path,"rt") as f: return [json.loads(line) for line in f]
def arrays(rows):
    return np.stack([Maze.from_dict(r).observation() for r in rows]), np.asarray([r["labels"]["optimal"] for r in rows],dtype=bool)

def generate(config):
    if (ROOT/"dataset_manifest.json").exists(): raise RuntimeError("Datasets already exist; refuse overwrite")
    seen=set(); manifest={"created":now(),"canonicalization":"D4 wall-layout only: excludes any start/goal relabeling across splits", "splits":{}}
    for split,spec in config["datasets"].items():
        rng=np.random.default_rng(spec["seed"]); rows=[]; attempts=0; rejects={"unsolvable":0,"family":0,"layout":0}
        while len(rows)<spec["count"]:
            attempts+=1
            if attempts>spec["count"]*20000: raise RuntimeError(f"Generator exhausted for {split}: {len(rows)}")
            family=len(rows)%6 if spec.get("balanced",False) else None
            m=candidate(rng,spec.get("size",5),spec.get("algorithm","occupancy"),spec.get("density",.30))
            if m is None: continue
            try: y=labels(m)
            except ValueError: rejects["unsolvable"]+=1; continue
            if family is not None and not family_matches(m,y,family): rejects["family"]+=1; continue
            if spec.get("min_distance",0)>y["distance"]: rejects["family"]+=1; continue
            h=m.canonical_hash()
            if h in seen: rejects["layout"]+=1; continue
            seen.add(h)
            flags=[f for f in range(6) if family_matches(m,y,f)]
            row={**m.to_dict(),"id":f"{split}-{len(rows):05d}","layout_hash":h,"family":family if family is not None else (max(flags) if flags else -1),"family_flags":flags,"labels":y}
            rows.append(row)
        final=spec.get("final",False)
        path=(ARTIFACTS/"sealed" if final else RESULTS/"datasets")/f"{split}.jsonl.gz"
        save_rows(path,rows)
        # No final outcome summaries before freeze. Generator constraints are public.
        manifest["splits"][split]={**spec,"sha256":digest(path),"attempts":attempts,"rejections":rejects}
        print(f"generated {split}: {len(rows)} unique layouts ({attempts} attempts)",flush=True)
    manifest["total_distinct_layouts"]=len(seen)
    write_json(ROOT/"dataset_manifest.json",manifest)

def prompt(row,alternative=False):
    m=Maze.from_dict(row)
    grid=m.text()
    if alternative: grid="\n".join(" ".join(r) for r in grid.splitlines())
    return (f"Choose an optimal first move along a shortest traversable route to G. Board {m.size} by {m.size}; rows increase south, columns east. "
            f"# wall, . free, A agent at row {m.agent[0]}, column {m.agent[1]}, G goal at row {m.goal[0]}, column {m.goal[1]}. "
            "No diagonal moves; walls and boundaries block movement.\n"+grid+"\nActions: (A) N north (B) E east (C) S south (D) W west. Reply with the option letter.")
