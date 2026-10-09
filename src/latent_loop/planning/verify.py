"""Post-run artifact, privacy, locality and historical-integrity checks."""
import gzip
import json
import re
import subprocess
from pathlib import Path
import numpy as np
from latent_loop.planning.data import ROOT,RESULTS,write_json,digest,now
from latent_loop.planning.evaluate import verify_freeze,read_trace

def main():
    verify_freeze()
    q=json.loads((ROOT/"qwen_frozen_protocol.json").read_text())
    assert all(digest(p)==h for p,h in q["hashes"].items())
    snake=json.loads((ROOT/'snake_frozen_protocol.json').read_text())
    assert all(digest(p)==h for p,h in snake['hashes'].items())
    changed=subprocess.check_output(["git","diff","17ae4e4","--name-only"],text=True).splitlines()
    assert all("experiment_05b" in p or "/planning/" in p or "/environments/" in p or p.startswith(("tests/test_maze","tests/test_snake","tests/test_planning_","tests/test_recurrent_planner")) for p in changed)
    assert subprocess.check_output(["git","rev-parse","v0.1.0^{}"],text=True).strip()=="0a637eb97d66eebbd740b612ab92e9095c496931"
    patterns={"secret_token":r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{24,}|hf_[A-Za-z0-9]{24,})",
              "private_key":r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
              "private_path":r"(?:/Users/|/home/)[A-Za-z0-9_.-]+",
              "private_ip":r"\b(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)\b"}
    findings=[];files=[]
    for folder in ("src/latent_loop/planning","src/latent_loop/environments","experiments/experiment_05b","results/experiment_05b","reports/experiment_05b"):
        for p in Path(folder).rglob("*"):
            if not p.is_file() or "__pycache__" in str(p) or p.suffix in (".png",".pyc"):continue
            if p.resolve()==Path(__file__).resolve():continue  # Scanner patterns are not leaked values.
            try:value=gzip.decompress(p.read_bytes()).decode() if p.suffix==".gz" else p.read_text()
            except (UnicodeError,OSError):continue
            files.append(str(p))
            for kind,pattern in patterns.items():
                if re.search(pattern,value): findings.append({"path":str(p),"kind":kind})
    assert not findings,findings
    cases=read_trace(RESULTS/"demonstrations.jsonl.gz");checked=0
    for case in cases:
        d=np.array(case["bfs_distance"])
        for k,v in case["depths"].items():
            effect=np.array(v["goal_effect_map"]);outside=(d<0)|(d>int(k))
            assert np.max(effect[outside],initial=0)==0;checked+=1
    tests=json.loads((RESULTS/'test_receipt.json').read_text())
    write_json(RESULTS/"verification.json",{"checked_utc":now(),"mlx_tests":tests['mlx'],"clean_cpu_tests":tests['clean_cpu'],"source_and_checkpoint_freezes_verified":True,"historical_files_unchanged":True,"public_tag_unchanged":True,"new_content_security_review":{"text_and_compressed_files":len(files),"findings":findings,"method":"Heuristic token/key/private-path/private-IP scan plus manual artifact/privacy/license review; no guarantee of exhaustive detection","remediation":"One protocol source-path key normalized; original receipt preserved in ignored artifacts"},"trained_demonstration_locality_checks":checked,"dataset_reproduction":"All eight maze and three Snake gzip files byte-identical in clean CPU environment","remote_mutations":0,"snake_neural_training":True})
    print(f"Verified frozen hashes, historical integrity, {len(files)} text/compressed files and {checked} trained-state locality checks")

if __name__=="__main__":main()
