"""Reproducible staged 4B execution with prospective sealing and safe resumption."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

from latent_loop.experiment04b.common import (
    EXP,
    OUT,
    freeze,
    guard,
    open_lockbox,
    preservation,
    setup,
    utc,
    verify_freeze,
)
from latent_loop.training import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "development",
            "freeze",
            "final",
            "timing",
            "report",
            "verify",
            "all",
        ],
    )
    parser.add_argument("--config", default=str(EXP / "config.json"))
    parser.add_argument(
        "--restart-budget",
        action="store_true",
        help="Start a fresh wall-clock budget for reproduction/resumption without changing the frozen scientific protocol.",
    )
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.restart_budget:
        os.environ["LATENT_LOOP_4B_STARTED_UTC"] = utc()
    setup()
    started = time.perf_counter()
    if args.stage in ("prepare", "all"):
        from latent_loop.experiment04b.data import prepare

        if not (EXP / "dataset_manifest.json").exists():
            prepare(config)
        preservation()
    if args.stage in ("development", "all"):
        if not (EXP / "frozen_protocol.json").exists():
            from latent_loop.experiment04b.train import development

            development(config)
            from latent_loop.experiment04b.evaluate import evaluate

            evaluate(config, ["controller_val"])
        else:
            verify_freeze(config)
    if args.stage in ("freeze", "all"):
        freeze(config)
    if args.stage in ("final", "all"):
        guard(config, 35)
        open_lockbox(config)
        from latent_loop.experiment04b.data import final_trajectories

        final_trajectories(config)
        from latent_loop.experiment04b.evaluate import evaluate

        evaluate(config)
    if args.stage in ("timing", "all"):
        from latent_loop.experiment04b.timing import measure

        if not (OUT / "latency.json").exists():
            measure(config)
    if args.stage in ("report", "all"):
        from latent_loop.experiment04b.report import report

        report(config)
    if args.stage in ("verify", "all"):
        from latent_loop.experiment04b.report import audit

        audit(config)
    info = {
        "stage": args.stage,
        "completed_utc": utc(),
        "seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "budget_started_utc": os.environ.get(
            "LATENT_LOOP_4B_STARTED_UTC", config["started_utc"]
        ),
        "preservation": preservation(),
    }
    try:
        import mlx.core as mx

        info["peak_mlx_bytes"] = mx.get_peak_memory()
    except ImportError:
        pass
    atomic_json(OUT / "runs" / f"{args.stage}.json", info)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
