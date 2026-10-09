"""Small explicit experiment entry points; heavy MLX imports are lazy."""
import argparse

def main():
    parser=argparse.ArgumentParser(description='Experiment 5.3 — Decision Bridge')
    parser.add_argument('command',choices=['generate','audit','train','freeze','evaluate','rollout','latency','diagnostics','analyze','report','example'])
    parser.add_argument('--condition');parser.add_argument('--seed',type=int);parser.add_argument('--objective',choices=['correctness','ranking','direct'])
    args=parser.parse_args()
    if args.command in ('generate','audit'):
        from . import data
        print(getattr(data,args.command)())
    elif args.command=='train':
        from . import train
        if args.condition and args.seed:
            print(train.fit_scorer(args.condition,args.seed,args.objective) if args.objective else train.fit_base(args.condition,args.seed))
        else:train.main()
    elif args.command in ('freeze','evaluate','rollout','latency'):
        from . import evaluate
        getattr(evaluate,args.command)()
    elif args.command=='diagnostics':
        from .diagnostics import run
        run()
    elif args.command=='analyze':
        from .analysis import main
        main()
    elif args.command=='report':
        from .report import main
        main()
    else:
        from .interface import example
        print(example())

if __name__=='__main__':main()
