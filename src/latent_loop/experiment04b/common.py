"""4B run identity, preservation, budgets and prospective protocol sealing."""

import json
import os
import signal
from datetime import datetime, timedelta, timezone
from pathlib import Path

from latent_loop.experiment04a.common import checkpoint_paths as old_checkpoints
from latent_loop.experiment04a.data import sha
from latent_loop.training import atomic_json

EXP = Path("experiments/experiment_04b")
OUT = Path("results/experiment_04b")
CACHE = Path("artifacts/experiment_04b")
FINAL = (
    "fresh_id",
    "new_template",
    "structural_shift",
    "harder_depth",
    "numeric_range",
)
INTERRUPTED = False


def utc():
    return datetime.now(timezone.utc).isoformat()


def setup():
    for p in (EXP, OUT, CACHE):
        p.mkdir(parents=True, exist_ok=True)

    def stop(signum, frame):
        global INTERRUPTED
        INTERRUPTED = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def guard(config, reserve_minutes=0):
    if INTERRUPTED:
        raise InterruptedError(
            "Safe boundary reached; resume cached batches/epoch checkpoints"
        )
    deadline = datetime.fromisoformat(
        os.environ.get("LATENT_LOOP_4B_STARTED_UTC", config["started_utc"])
    ) + timedelta(hours=config["max_hours"])
    if datetime.now(timezone.utc) + timedelta(minutes=reserve_minutes) > deadline:
        raise TimeoutError(
            "4B resource budget reached; preserve caches and reporting reserve"
        )


def preservation():
    previous = json.loads((EXP / "historical_inventory.json").read_text())
    altered = [p for p, h in previous.items() if p != "README.md" and sha(p) != h]
    assert not altered, altered
    return {
        "historical_files": len(previous),
        "all_unchanged_except_readme_link": True,
        "checked_utc": utc(),
    }


def checkpoint_paths(seed):
    paths = old_checkpoints(seed)
    paths["regression"] = Path(
        f"artifacts/experiment_03/checkpoints/regression_seed{seed}/best.safetensors"
    )
    return paths


def frozen_hashes(config):
    files = [
        EXP / n
        for n in (
            "config.json",
            "plan.md",
            "dataset_manifest.json",
            "validation_partitions.json",
            "compatibility_audit.json",
        )
    ]
    files += sorted(
        p
        for p in Path("src/latent_loop/experiment04b").glob("*.py")
        if p.name != "report.py"
    )
    files += [EXP / f"selection_seed{s}.json" for s in config["parent_seeds"]]
    files += sorted((OUT / "training").glob("*.json"))
    files += sorted(Path("tests").glob("test_experiment04b*.py"))
    models = {
        str(s): {k: sha(p) for k, p in checkpoint_paths(s).items()}
        for s in config["parent_seeds"]
    }
    fitted = {}
    for s in config["parent_seeds"]:
        chosen = json.loads((EXP / f"selection_seed{s}.json").read_text())
        for name in chosen["deployment_checkpoints"]:
            p = CACHE / "checkpoints" / f"{name}_seed{s}" / "best.safetensors"
            fitted[p.as_posix()] = sha(p)
    return {
        "files": {p.as_posix(): sha(p) for p in files},
        "frozen_models": models,
        "fitted_models": fitted,
    }


def freeze(config):
    if (EXP / "lockbox_opened.json").exists():
        raise RuntimeError("Consumed final set: no further model/policy selection")
    value = {
        "frozen_utc": utc(),
        "hashes": frozen_hashes(config),
        "final_splits": list(FINAL),
        "selection_data": "4A controller_train plus partitioned controller_val only",
        "primary": {
            k: config[k]
            for k in ("primary_policy", "primary_selector", "primary_split")
        },
        "status": "4B final labels/scores remain unopened; 4A final tests excluded from selection",
    }
    path = EXP / "frozen_protocol.json"
    if path.exists():
        prior = json.loads(path.read_text())
        assert prior["hashes"] == value["hashes"]
        return prior
    atomic_json(path, value)
    return value


def verify_freeze(config):
    value = json.loads((EXP / "frozen_protocol.json").read_text())
    assert value["hashes"] == frozen_hashes(config), (
        "Frozen 4B protocol/checkpoints changed"
    )
    preservation()
    return value


def open_lockbox(config):
    value = verify_freeze(config)
    path = EXP / "lockbox_opened.json"
    if not path.exists():
        atomic_json(
            path,
            {
                "opened_utc": utc(),
                "frozen_utc": value["frozen_utc"],
                "protocol_sha256": sha(EXP / "frozen_protocol.json"),
                "status": "4B fixed-protocol final evaluation opened; consumed for future selection",
                "splits": list(FINAL),
            },
        )
