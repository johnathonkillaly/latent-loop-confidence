import argparse
import json
from pathlib import Path

from .data import generate


def main():
    p = argparse.ArgumentParser(description="Experiment 5E: frozen-Qwen decision intervention")
    p.add_argument("command", choices=["generate", "pilot", "extract", "train", "freeze", "evaluate", "timing", "report", "audit"])
    p.add_argument("--config", default="experiments/experiment_05e/config.json")
    p.add_argument("--splits", nargs="*")
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.command in ("generate", "pilot"):
        print(json.dumps(generate(config, pilot=args.command == "pilot"), indent=2))
    elif args.command == "extract":
        from .features import extract
        extract(config, args.splits or ["train", "validation"])
    else:
        from .study import run
        run(args.command, config)


if __name__ == "__main__":
    main()
