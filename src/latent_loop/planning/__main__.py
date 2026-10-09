"""Standalone CLI; historical commands and frozen sources remain unchanged."""
import argparse
import json
from latent_loop.planning.data import ROOT

def main():
    p=argparse.ArgumentParser(description="Experiment 5.1 deterministic planning")
    p.add_argument("command",choices=["generate","train","train-all","freeze","evaluate","rollout","report","qwen-train","snake"])
    p.add_argument("--condition",default="rec_final4"); p.add_argument("--seed",type=int,default=11)
    p.add_argument("--epochs",type=int); p.add_argument("--pilot",action="store_true")
    args=p.parse_args(); config=json.loads((ROOT/"config.json").read_text())
    if args.command=="generate":
        from latent_loop.planning.data import generate
        generate(config)
    elif args.command in ("train","train-all"):
        from latent_loop.planning.train import train
        conditions=config["conditions"] if args.command=="train-all" else [args.condition]
        seeds=config["seeds"] if args.command=="train-all" else [args.seed]
        for condition in conditions:
            for seed in seeds: train(condition,seed,args.epochs,args.pilot)
    elif args.command in ("freeze","evaluate","rollout"):
        from latent_loop.planning.evaluate import freeze,evaluate,rollouts
        {"freeze":freeze,"evaluate":evaluate,"rollout":rollouts}[args.command]()
    elif args.command=="qwen-train":
        from latent_loop.planning.qwen import train
        train(args.condition,args.seed,args.epochs)
    elif args.command=="snake":
        from latent_loop.planning.snake_experiment import run
        run()
    elif args.command=="report":
        from latent_loop.planning.report import report
        report()

if __name__=="__main__": main()
