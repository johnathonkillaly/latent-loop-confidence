"""Commands run relative to an explicit experiment directory; public reports need no GPU."""

import argparse
import json
import os
import signal
import time
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=[
            "data",
            "download",
            "extract",
            "train",
            "evaluate",
            "run",
            "report",
            "middle",
            "smoke",
        ],
    )
    parser.add_argument("--config")
    parser.add_argument(
        "--run-dir", default=".", help="Independent working directory for this run"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--variant",
        choices=[
            "confidence",
            "recurrence",
            "memory",
            "feedforward",
            "wide_feedforward",
            "recurrence_continuation",
            "memory_continuation",
        ],
        help="Fit one auxiliary stage (train only)",
    )
    args = parser.parse_args()
    if args.variant and args.command != "train":
        parser.error("--variant applies to train")
    config_path = Path(
        args.config
        or ("configs/smoke.yaml" if args.command == "smoke" else "configs/pilot.yaml")
    ).resolve()
    config = yaml.safe_load(config_path.read_text())
    from latent_loop.backbone import MODEL_ID, REVISION

    if config["model"] != MODEL_ID or config["model_revision"] != REVISION:
        raise ValueError(
            "This validated loader supports only the pinned Qwen3-0.6B checkpoint"
        )
    if args.command == "smoke" and args.run_dir == ".":
        args.run_dir = "artifacts/smoke"
    root = Path(args.run_dir)
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    for p in [
        "data",
        "artifacts",
        "experiments",
        "results/metrics",
        "results/figures",
        "reports",
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + config["max_hours"] * 3600
    interrupted = False

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def guard():
        if interrupted:
            raise InterruptedError("Safely stopped at a checkpoint boundary")
        if time.monotonic() > deadline - 60:
            raise TimeoutError(
                "Time budget reached; reserving one minute for reporting"
            )

    from latent_loop.training import atomic_json

    signature = Path("artifacts/features/signature.json")
    if signature.exists():
        previous = json.loads(signature.read_text())
        if (
            previous["assistant_prefix"] != config.get("assistant_prefix", "Answer: (")
            or previous["dataset"]["seed"] != config["seed"]
            or {s: d["size"] for s, d in previous["dataset"]["splits"].items()}
            != config["sizes"]
        ):
            raise ValueError(
                "Run directory contains features for another dataset/prompt; use a fresh --run-dir"
            )
    atomic_json("experiments/active_config.json", config)
    start = time.time()
    command = args.command
    try:
        if command in ("data", "run", "smoke"):
            from latent_loop.datasets import generate

            print(generate("data", config["sizes"], config["seed"]), flush=True)
        if command == "download":
            from latent_loop.backbone import ensure_model_cached

            ensure_model_cached()
        if command in ("extract", "run", "smoke"):
            from latent_loop.backbone import extract

            extract(config, guard)
        if command in ("train", "evaluate", "run", "smoke"):
            from latent_loop.training import run_seed, run_stage

            if command == "evaluate":
                for seed in [args.seed] if args.seed is not None else config["seeds"]:
                    for name in [
                        "confidence",
                        "recurrence",
                        "memory",
                        "feedforward",
                        "wide_feedforward",
                        "recurrence_continuation",
                        "memory_continuation",
                    ]:
                        p = Path(f"artifacts/checkpoints/{name}_seed{seed}/state.json")
                        if not p.exists() or not json.loads(p.read_text()).get(
                            "complete"
                        ):
                            raise RuntimeError(
                                "Evaluation requires completed checkpoints"
                            )
            for seed in [args.seed] if args.seed is not None else config["seeds"]:
                try:
                    if args.variant:
                        run_stage(config, seed, args.variant, guard)
                    else:
                        run_seed(config, seed, guard)
                except (InterruptedError, TimeoutError):
                    raise
                except Exception as e:
                    atomic_json(
                        f"experiments/failure_seed{seed}.json",
                        {
                            "type": type(e).__name__,
                            "message": str(e).replace(str(Path.cwd()), "<run-dir>"),
                        },
                    )
                    raise
        if command == "middle":
            from latent_loop.benchmark import middle_benchmark

            middle_benchmark(config, guard)
        if command in ("report", "run", "smoke"):
            from latent_loop.reporting import report

            report()
    finally:
        from datetime import datetime, timezone

        atomic_json(
            f"experiments/last_{command}.json",
            {
                "command": command,
                "seconds": time.time() - start,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "interrupted": interrupted,
            },
        )


if __name__ == "__main__":
    main()
