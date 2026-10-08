"""Run identity, immutable historical audit, resource guards and lockbox checks."""

import gzip
import json
import signal
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from latent_loop.experiment03.science import targets
from latent_loop.experiment04a.data import CACHE, EXP, FINAL, OUT, sha
from latent_loop.training import atomic_json

INTERRUPTED = False


def utc():
    return datetime.now(timezone.utc).isoformat()


def guard(config):
    if INTERRUPTED:
        raise InterruptedError(
            "Stopped safely at a batch/stage boundary; cached work preserved"
        )
    if datetime.now(timezone.utc) > datetime.fromisoformat(
        config["started_utc"]
    ) + timedelta(hours=config["max_hours"]):
        raise TimeoutError("Experiment 4A wall-clock budget reached")


def setup():
    for p in (EXP, OUT, CACHE):
        p.mkdir(parents=True, exist_ok=True)

    def stop(signum, frame):
        global INTERRUPTED
        INTERRUPTED = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)


def checkpoint_paths(seed):
    paths = {
        "refiner": Path(
            f"artifacts/checkpoints/recurrence_seed{seed}/best.safetensors"
        ),
        "one_pass": Path(
            f"artifacts/checkpoints/feedforward_seed{seed}/best.safetensors"
        ),
    }
    for name in ("confidence", "improvement", "joint"):
        paths[name] = Path(
            f"artifacts/experiment_03/checkpoints/{name}_seed{seed}/best.safetensors"
        )
    return paths


def preservation():
    inventory = json.loads((EXP / "historical_inventory.json").read_text())
    changes = [p for p, v in inventory.items() if p != "README.md" and sha(p) != v]
    assert not changes, changes
    return {
        "historical_files": len(inventory),
        "all_unchanged_except_readme_link": True,
        "checked_utc": utc(),
    }


def audit(config):
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", config["start_commit"]], text=True
    ).splitlines()
    path = EXP / "historical_inventory.json"
    if not path.exists():
        atomic_json(path, {p: sha(p) for p in files})
    assert (
        subprocess.check_output(
            ["git", "merge-base", "HEAD", config["start_commit"]], text=True
        ).strip()
        == config["start_commit"]
    )
    checks = 0
    means = {}
    model_hashes = {}
    for seed in config["parent_seeds"]:
        raw = json.load(
            gzip.open(f"results/experiment_03/predictions_seed{seed}.json.gz", "rt")
        )
        recorded = json.loads(
            Path(f"results/experiment_03/seed{seed}.json").read_text()
        )
        for split, r in raw.items():
            loss, correct, gain, _ = targets(r["logits"], r["labels"])
            assert np.allclose(gain, r["gain"], atol=1e-8)
            for name, depth in r["depths"].items():
                d = np.asarray(depth)
                chosen = correct[d - 1, np.arange(len(d))]
                assert (
                    abs(
                        chosen.mean()
                        - recorded["splits"][split]["policies"][name]["accuracy"]
                    )
                    < 1e-12
                )
                assert (
                    abs(
                        loss[d - 1, np.arange(len(d))].mean()
                        - recorded["splits"][split]["policies"][name]["exact_nll"]
                    )
                    < 1e-8
                )
                checks += 1
        means[str(seed)] = {
            name: recorded["splits"]["fresh_id"]["policies"][name]["accuracy"]
            for name in ["frozen", "one_pass", "improvement", "oracle_best"]
        }
        model_hashes[str(seed)] = {k: sha(p) for k, p in checkpoint_paths(seed).items()}
    for name in model_hashes["11"]:
        assert len({model_hashes[str(s)][name] for s in config["parent_seeds"]}) == 3
    atomic_json(
        EXP / "existing_audit.json",
        {
            "completed_utc": utc(),
            "starting_commit": config["start_commit"],
            "historical_policy_accuracy_and_exact_nll_checks": checks,
            "historical_fresh_id_accuracy": means,
            "checkpoint_hashes": model_hashes,
            "preservation": preservation(),
            "findings": [
                "No invalidating label leakage found in the inspected live APIs; existing neural inputs are current state, option probabilities, entropy, margin, depth, completed change.",
                "Experiment 3 reused abstract task structures; it did not reject alpha-renamed instances or provide a prospective sealed final evaluation.",
                "Historical candidate confidence and current-decision confidence use different supervision; comparisons here hold current argmax decisions fixed.",
                "Scalar controls fitted to in-sample head predictions may favor calibration; heads are frozen in 4A, independent controller data avoids new stacking cross-fit leakage.",
                "Candidate NLL is four-choice conditional; original metrics cap probabilities at 1e-12. New trajectory NLL is stable and unclipped.",
                "Oracle best-depth uses labels/future losses and is never a deployed controller. Some oracle gains may be rank changes between wrong answers.",
                "Host .item() calls synchronize. Full and auxiliary timing scopes require explicit transfers, warmups and executed-row counts.",
            ],
        },
    )


def freeze(config):
    if (EXP / "lockbox_opened.json").exists():
        raise RuntimeError(
            "Lockbox already opened; no further protocol selection is allowed"
        )
    files = [
        "experiments/experiment_04a/config.json",
        "experiments/experiment_04a/dataset_manifest.json",
        "experiments/experiment_04a/historical_structure_exclusions.json",
        "experiments/experiment_04a/plan.md",
    ]
    files += [
        p.as_posix()
        for p in sorted(Path("src/latent_loop/experiment04a").glob("*.py"))
        if p.name != "report.py"
    ]
    files += [
        f"experiments/experiment_04a/selection_seed{s}.json"
        for s in config["parent_seeds"]
    ]
    model_hashes = {
        str(s): {k: sha(p) for k, p in checkpoint_paths(s).items()}
        for s in config["parent_seeds"]
    }
    value = {
        "frozen_utc": utc(),
        "source_config_data_selection_hashes": {p: sha(p) for p in files},
        "model_hashes": model_hashes,
        "final_splits": list(FINAL),
        "selection_data": ["controller_train", "controller_val"],
        "final_evaluation_status": "Unopened; no final scores extracted during development",
        "4b_protocol": "These 4A final sets become consumed evaluation sets when opened. 4B may use controller_train/val trajectories for development but must create a new independently seeded prospective final set.",
    }
    path = EXP / "frozen_protocol.json"
    if path.exists():
        previous = json.loads(path.read_text())
        assert (
            previous["source_config_data_selection_hashes"]
            == value["source_config_data_selection_hashes"]
        )
        return previous
    atomic_json(path, value)
    return value


def verify_freeze():
    value = json.loads((EXP / "frozen_protocol.json").read_text())
    for p, v in value["source_config_data_selection_hashes"].items():
        assert sha(p) == v, f"Frozen protocol changed: {p}"
    for seed, hashes in value["model_hashes"].items():
        for name, path in checkpoint_paths(int(seed)).items():
            assert sha(path) == hashes[name]
    return value


def open_lockbox():
    frozen = verify_freeze()
    path = EXP / "lockbox_opened.json"
    if not path.exists():
        atomic_json(
            path,
            {
                "opened_utc": utc(),
                "protocol_frozen_utc": frozen["frozen_utc"],
                "protocol_sha256": sha(EXP / "frozen_protocol.json"),
                "final_splits": frozen["final_splits"],
                "status": "Opened once for fixed-protocol evaluation; final sets are now consumed and not suitable for further architecture selection.",
            },
        )
