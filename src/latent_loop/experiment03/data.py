"""Deterministic fresh data and historical artifact audit."""

import gzip
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from latent_loop.datasets import TASKS, generate_one, load, reference
from latent_loop.training import atomic_json

EXP = Path("experiments/experiment_03")
OUT = Path("results/experiment_03")
CACHE = Path("artifacts/experiment_03")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def historical_audit():
    old = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", "cd0a326"], text=True
    ).splitlines()
    inventory = {p: sha(p) for p in old}
    checks, recomputed = [], {}
    for seed in [11, 22, 33]:
        raw = json.load(
            gzip.open(f"results/metrics/predictions_seed{seed}.json.gz", "rt")
        )
        recorded = json.loads(Path(f"results/metrics/seed{seed}.json").read_text())
        for row in recorded["results"]:
            split, variant = row["split"], row["variant"]
            logits = np.asarray(raw[split][variant]["logits"], float)
            y = np.asarray(raw[split]["labels"])
            acc = float((logits.argmax(-1) == y).mean())
            lp = logits - logsumexp(logits, axis=-1, keepdims=True)
            capped_nll = float(
                -np.maximum(lp[np.arange(len(y)), y], np.log(1e-12)).mean()
            )
            checks.append(
                abs(acc - row["accuracy"]) < 1e-12
                and abs(capped_nll - row["nll"]) < 1e-6
            )
        recomputed[str(seed)] = {
            v: float(
                (
                    np.asarray(raw["test"][v]["logits"]).argmax(-1)
                    == raw["test"]["labels"]
                ).mean()
            )
            for v in [
                "baseline",
                "confidence_only",
                "feedforward",
                "recurrence_selected",
                "memory_selected",
                "recurrence_adaptive",
            ]
        }
    if not all(checks):
        raise AssertionError("Historical metric recomputation failed")
    ds = json.loads(Path("data/manifest.json").read_text())
    cache_signature = json.loads(Path("artifacts/features/signature.json").read_text())
    assert cache_signature["dataset"] == ds
    for split, info in ds["splits"].items():
        assert sha(f"data/{split}.jsonl") == info["sha256"]
        a = np.load(f"artifacts/features/{split}.npz")
        assert a["h"].shape == (info["size"], 1024)
        assert a["base"].shape == (info["size"], 4)
        assert a["h"].dtype == np.float32
    checkpoint_hashes = {
        f"{name}_seed{seed}": sha(
            f"artifacts/checkpoints/{name}_seed{seed}/best.safetensors"
        )
        for name in [
            "recurrence",
            "feedforward",
            "recurrence_continuation",
            "confidence",
        ]
        for seed in [11, 22, 33]
    }
    issues = [
        "Historical continuation target is any later allowed loss improvement, not immediate gain; final-loop labels are zero and omitted during fitting.",
        "Historical continuation targets came from the same examples used to fit the refiner. This is not final-test leakage but can bias controller supervision; Experiment 3 uses fresh examples.",
        "Historical metric NLL clips true-label probabilities at 1e-12. Experiment 3 target loss uses stable logsumexp without clipping; policy report retains historical metric convention and additionally records exact NLL.",
        "Old confidence predicts candidate correctness at backbone depth zero. It is not a current recurrent-decision confidence head. An adapted candidate architecture is fitted separately here.",
        "Train/validation/test templates share semantics and may share alpha-renamed structures; exact uniqueness does not prove structural disjointness.",
        "Historical development adjusted answer formatting on training examples and is exploratory. The final prefix is frozen here.",
        "Original head timings exclude backbone. They cannot establish full inference speedup; Experiment 3 measures synchronized full live inference.",
    ]
    audit = {
        "timestamp": utc(),
        "historical_commit": "cd0a326db2203cdfec902affbc1f611641794ca3",
        "recomputed_rows": len(checks),
        "all_accuracy_and_nll_checks_passed": all(checks),
        "recomputed_test_accuracies": recomputed,
        "means": {
            v: float(np.mean([r[v] for r in recomputed.values()]))
            for v in next(iter(recomputed.values()))
        },
        "dataset_hashes_verified": True,
        "feature_signature_and_shapes_verified": True,
        "checkpoint_hashes": checkpoint_hashes,
        "issues_and_limits": issues,
        "source_review": {
            "live_inputs": "Current state, current option probabilities, depth; historical live API takes no labels or future states.",
            "causal_cache": "Final real prompt position after right padding; FP32 parity and prefix tests recorded in historical backbone metadata.",
            "stopping": "At most four paid loops; compulsory final stop. Future-min labels are training supervision only.",
            "cross_seed": "Separate checkpoint directories, explicit initialization seed offsets, distinct SHA256 values verified.",
            "normalization": "Four candidate conditional softmax throughout primary comparison; vocabulary statistics explicitly separate.",
            "test_isolation": "Historical selections from val; fresh controller selections from controller_val only.",
        },
    }
    assert len({checkpoint_hashes[f"recurrence_seed{s}"] for s in [11, 22, 33]}) == 3
    atomic_json(EXP / "historical_inventory.json", inventory)
    atomic_json(EXP / "audit.json", audit)
    return audit


def template(row):
    s, task = row["spec"], row["task"]
    if task == "comparison":
        q = (
            "Free-memory inventory in MB:\n"
            + "\n".join(f"{n} has {v}" for n, v in zip(s["names"], s["values"]))
            + "\nReturn the name with the largest free-memory value."
        )
    elif task == "arithmetic":
        q = (
            f"A counter begins at {s['start']}. Execute these numbered instructions sequentially (not as an expression with precedence):\n"
            + "\n".join(f"{i + 1}: {op} {v}" for i, (op, v) in enumerate(s["ops"]))
            + "\nReturn the resulting integer."
        )
    elif task == "state":
        q = (
            "Initial register table: "
            + ", ".join(f"{k}={v}" for k, v in s["initial"].items())
            + ". Perform each instruction in sequence:\n"
        )
        q += "\n".join(
            f"exchange values in {x} and {y}"
            if op == "swap"
            else f"assign {x} the current value of {y}"
            if op == "copy"
            else f"set {x} to ({x} + {y}) modulo 10"
            for op, x, y in s["ops"]
        )
        q += f"\nReturn the value in register {s['query']}."
    elif task == "pointer":
        q = (
            "Successor table: "
            + ", ".join(f"successor({k})={v}" for k, v in s["mapping"].items())
            + f". Begin at {s['start']}. Take {s['steps']} successor steps and return the endpoint."
        )
    elif task == "routing":
        q = (
            "Dispatch table: "
            + ", ".join(f"{verb} => {tool}" for verb, tool in s["registry"].items())
            + f". A {s['verb']} job arrives. Which tool should receive it?"
        )
    else:
        q = (
            "Eligibility requires BOTH readiness and authorization.\n"
            + "\n".join(
                f"{name}: readiness {'yes' if a else 'no'}, authorization {'yes' if b else 'no'}"
                for name, (a, b) in s["rows"]
            )
            + "\nWhich item meets both requirements?"
        )
    return (
        q
        + "\n"
        + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", row["choices"]))
        + "\nAnswer with only the correct option letter."
    )


def fresh_data(config):
    directory = CACHE / "data"
    directory.mkdir(parents=True, exist_ok=True)
    historical = [
        r for split in ["train", "val", "test", "ood"] for r in load("data", split)
    ]
    seen, prompts = {r["id"] for r in historical}, {r["prompt"] for r in historical}
    splits, manifest = (
        {},
        {
            "version": "experiment03-fresh-v1",
            "seed": config["data_seed"],
            "splits": {},
            "historical_overlap": 0,
            "cross_split_instance_overlap": 0,
            "cross_split_prompt_overlap": 0,
            "structural_disjointness": "Not claimed: shared task semantics and alpha-renamed structures are possible.",
            "template_shift": "One new paraphrase per task, same generator and difficulty as fresh_id; tests wording transfer, not new algorithms.",
            "depth_shift": "Existing OOD depths 5-8 and numeric range; easy tasks remain depth1.",
        },
    )
    for i, (split, size) in enumerate(config["sizes"].items()):
        rng = random.Random(config["data_seed"] + i * 100000)
        source = (
            "train"
            if split == "controller_train"
            else "val"
            if split == "controller_val"
            else "ood"
            if split == "depth_shift"
            else "test"
        )
        rows = []
        while len(rows) < size:
            row = generate_one(rng, TASKS[len(rows) % len(TASKS)], source, len(rows))
            if row["id"] in seen:
                continue
            if split == "template_shift":
                row["prompt"] = template(row)
            if row["prompt"] in prompts:
                continue
            assert row["choices"][row["label"]] == reference(row["task"], row["spec"])
            row["split"] = split
            seen.add(row["id"])
            prompts.add(row["prompt"])
            rows.append(row)
        content = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        path = directory / f"{split}.jsonl"
        if path.exists() and path.read_text() != content:
            raise ValueError("Fresh data changed: use a new experiment directory")
        path.write_text(content)
        manifest["splits"][split] = {
            "size": size,
            "sha256": sha(path),
            "source_distribution": source,
            "label_counts": np.bincount(
                [r["label"] for r in rows], minlength=4
            ).tolist(),
            "tasks": {t: sum(r["task"] == t for r in rows) for t in TASKS},
        }
        splits[split] = rows
    for split in ["historical_test", "historical_ood"]:
        splits[split] = load("data", split.removeprefix("historical_"))
    atomic_json(EXP / "dataset_manifest.json", manifest)
    with gzip.GzipFile(filename=str(OUT / "examples.json.gz"), mode="wb", mtime=0) as f:
        f.write(json.dumps(splits, sort_keys=True).encode())
    return splits


def verify_preservation():
    inventory = json.loads((EXP / "historical_inventory.json").read_text())
    # README alone may receive a validated link after numerical work is complete.
    changed = [
        p for p, digest in inventory.items() if p != "README.md" and sha(p) != digest
    ]
    if changed:
        raise AssertionError(f"Historical files changed: {changed}")
    return {
        "checked": len(inventory),
        "unchanged_except_documented_readme_link": True,
        "timestamp": utc(),
    }
