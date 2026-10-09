"""Prespecified small Qwen rollout and alternative-text diagnostics."""
import json
import time
import numpy as np
import mlx.core as mx
from latent_loop.backbone import environment,get_model
from latent_loop.environments.maze import Maze
from latent_loop.planning.data import ROOT,RESULTS,ARTIFACTS,load_rows,write_json,save_rows,digest
from latent_loop.planning.qwen import build,batch,predict
from latent_loop.planning.evaluate import correct_and_metrics,paired_interval,read_trace
from latent_loop.planning.rollout import navigate

def run():
    config=json.loads((ROOT/"config.json").read_text()); c=config["qwen"]; environment(config["memory_gib"])
    frozen=json.loads((ROOT/"qwen_frozen_protocol.json").read_text())
    for p,h in frozen["hashes"].items():
        if digest(p)!=h: raise RuntimeError("Qwen frozen artifact changed")
    base,tokenizer,_=get_model(); summary={"rollouts":{},"alternative_serialization":{}}
    for split in ("fresh_id","larger"):
        rows=load_rows(split,True)[:12]; summary["rollouts"][split]={}
        # Fixed first 12 episodes, three seeds, deterministic actions. Small and
        # descriptive: no model/depth choices made from these episodes.
        for condition,depth in (("qwen_onepass",1),("qwen_recurrent",1),("qwen_recurrent",2),("qwen_recurrent",4),("qwen_recurrent",8)):
            seed_records=[]
            for seed in c["seeds"]:
                path=RESULTS/"rollouts"/f"{condition}_d{depth}_seed{seed}_{split}.jsonl.gz"
                if path.exists(): records=read_trace(path)
                else:
                    model=build(base,tokenizer,config,condition,seed)
                    model.load_weights(str(ARTIFACTS/"checkpoints"/f"{condition}_seed{seed}"/"best.safetensors"),strict=False)
                    def policy(maze):
                        ids,lengths=batch(tokenizer,[maze.to_dict()]); trace=model(ids,lengths,(depth,)); mx.eval(trace)
                        return int(mx.argmax(trace[depth]["logits"],axis=-1)[0])
                    records=[]
                    for row in rows:
                        budget=max(24,row["labels"]["distance"]*4)
                        records.append({"id":row["id"],**navigate(Maze.from_dict(row),policy,budget)})
                    save_rows(path,records)
                seed_records.append(records)
            flat=[r for values in seed_records for r in values]
            summary["rollouts"][split][f"{condition}@{depth}"]={"completion":paired_interval([[r["success"] for r in values] for values in seed_records]),"steps_mean":float(np.mean([r["steps"] for r in flat])),"invalid_moves_mean":float(np.mean([r["invalid_moves"] for r in flat])),"inference_seconds_mean":float(np.mean([r["inference_seconds"] for r in flat]))}
            write_json(RESULTS/"qwen_diagnostics.json",summary)
            print(f"Qwen rollout {split} {condition}@{depth}: "+str(summary["rollouts"][split][f"{condition}@{depth}"]["completion"]["mean"]),flush=True)
    rows=load_rows("fresh_id",True)[:120]
    for condition,depth in (("qwen_onepass",1),("qwen_recurrent",4)):
        metrics=[]
        for seed in c["seeds"]:
            model=build(base,tokenizer,config,condition,seed); model.load_weights(str(ARTIFACTS/"checkpoints"/f"{condition}_seed{seed}"/"best.safetensors"),strict=False)
            trace=predict(model,tokenizer,rows,(depth,),alternative=True)
            save_rows(RESULTS/"trajectories"/f"{condition}_seed{seed}_alternative_text.jsonl.gz",[{"id":r["id"],"depths":{str(depth):trace[depth][i]}} for i,r in enumerate(rows)])
            metrics.append(correct_and_metrics(rows,[v["logits"] for v in trace[depth]])[1])
        summary["alternative_serialization"][condition]={"mean":{k:float(np.mean([v[k] for v in metrics])) for k in metrics[0]},"seed_metrics":metrics}
    write_json(RESULTS/"qwen_diagnostics.json",summary)

if __name__=="__main__": run()
