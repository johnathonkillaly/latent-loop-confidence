"""CLI for independently reproducible stages without changing old commands."""
import argparse

def main():
    p=argparse.ArgumentParser(description='Experiment5.2 scaling/falsification; local research')
    p.add_argument('command',choices=['generate','validate','train','freeze','evaluate','rollout','latency','propagation','extrapolation','audit','report']);p.add_argument('--condition');p.add_argument('--seed',type=int);a=p.parse_args()
    if a.command=='generate':
        from .data import generate
        generate()
    elif a.command=='validate':
        from .validate import run
        run()
    elif a.command=='train':
        from .train import train
        from .data import config
        for c in [a.condition] if a.condition else config()['conditions']:
            for s in [a.seed] if a.seed else config()['seeds']:train(c,s)
    elif a.command in ('freeze','evaluate','rollout','latency'):
        from .evaluate import freeze,evaluate,rollouts,latency
        {'freeze':freeze,'evaluate':evaluate,'rollout':rollouts,'latency':latency}[a.command]()
    else:
        from importlib import import_module
        import_module('latent_loop.scaling.'+{'propagation':'diagnostic_export'}.get(a.command,a.command)).run()
if __name__=='__main__':main()
