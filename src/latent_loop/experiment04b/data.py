"""Reuse compatible development trajectories and generate fresh sealed final data."""

import copy
import gzip
import itertools
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from latent_loop.experiment04a.data import (
    TASKS,
    canonical,
    digest,
    reference,
    render,
    row,
    sha,
    structure,
)
from latent_loop.experiment04b.common import (
    CACHE,
    EXP,
    FINAL,
    OUT,
    guard,
    preservation,
    utc,
    verify_freeze,
)
from latent_loop.training import atomic_json


def public_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def load(split, allow_final=False):
    if split.startswith("controller_"):
        return public_rows(
            Path("results/experiment_04a/datasets") / f"{split}.jsonl.gz"
        )
    if not allow_final or not (EXP / "lockbox_opened.json").exists():
        raise PermissionError(
            "Prospective 4B data are sealed until protocol freeze/open"
        )
    manifest = json.loads((EXP / "dataset_manifest.json").read_text())
    p = CACHE / "datasets" / "sealed" / f"{split}.jsonl.gz"
    assert sha(p) == manifest["splits"][split]["sha256"]
    return public_rows(p)


def historical_key(item):
    if item["generator_id"] == "mixed_units":
        return digest(
            [
                "normalized_memory_bytes",
                sorted(
                    v * 1024 ** (["KB", "MB", "GB"].index(u) + 1)
                    for v, u in item["spec"]["items"]
                ),
            ]
        )
    return item["structural_hash"]


def numerical_transform(item, rng, factor):
    """Scale real operands, not entity/query indices; preserve reference-solver labels."""
    r = copy.deepcopy(item)
    s = r["spec"]
    family = r["generator_id"]
    old_answer = r["choices"][r["label"]]
    if family == "accumulator":
        s["initial"] *= factor
        s["ops"] = [[op, v if op == "multiply" else v * factor] for op, v in s["ops"]]
        simulated = str(int(old_answer) * factor)
    elif family == "expression_tree":

        def scale(node):
            return (
                node * factor
                if isinstance(node, int)
                else [node[0], scale(node[1]), scale(node[2])]
            )

        s["tree"] = scale(s["tree"])

        # Independent explicit postorder simulator; reference uses a constructed expression.
        def execute(node):
            if isinstance(node, int):
                return node
            a, b = execute(node[1]), execute(node[2])
            return (
                a + b if node[0] == "add" else a - b if node[0] == "subtract" else a * b
            )

        simulated = str(execute(s["tree"]))
    elif family in ("register_program", "inventory_transfers"):
        s["initial"] = [v * factor for v in s["initial"]]
        s["ops"] = [[op, i, v if op == "swap" else v * factor] for op, i, v in s["ops"]]
        simulated = str(int(old_answer) * factor)
    elif family in ("mixed_units", "resource_filter"):
        s["items"] = [
            [v * factor, u if isinstance(u, str) else u * factor] for v, u in s["items"]
        ]
        if "minimum_cpu" in s:
            s["minimum_cpu"] *= factor
        simulated = r["entities"].index(old_answer)
    elif family in ("numeric_conjunction", "disjunction_negation"):
        s["items"] = [[a * factor, b * factor, flag] for a, b, flag in s["items"]]
        s["thresholds"] = [v * factor for v in s["thresholds"]]
        simulated = r["entities"].index(old_answer)
    elif family in ("capability_constraints", "incomplete_request"):
        if s["request"]["jobs"] is not None:
            s["request"]["jobs"] *= factor
        for tool in s["tools"]:
            tool["capacity"] *= factor
        simulated = (
            r["entities"].index(old_answer)
            if old_answer in r["entities"]
            else old_answer
        )
    else:
        simulated = r["entities"].index(
            old_answer
        )  # Pointer task is an unshifted range control.
    answer = reference(family, s)
    assert answer == simulated, (family, simulated, answer)
    answer_text = r["entities"][answer] if isinstance(answer, int) else answer
    if family in (
        "accumulator",
        "expression_tree",
        "register_program",
        "inventory_transfers",
    ):
        number = int(answer_text)
        distractors = rng.sample([v for v in range(-150, 151) if v], 3)
        r["choices"] = [answer_text] + [str(number + v * factor) for v in distractors]
        rng.shuffle(r["choices"])
    r["label"] = r["choices"].index(answer_text)
    r["prompt"] = (
        render(family, s, r["entities"], r["template_id"])
        + "\n"
        + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"]))
        + "\nAnswer with only the correct option letter."
    )
    numbers = []

    def collect(v):
        if isinstance(v, (int, float)):
            numbers.append(abs(v))
        elif isinstance(v, list):
            for item in v:
                collect(item)
        elif isinstance(v, dict):
            for item in v.values():
                collect(item)

    collect(s)
    r.update(
        id=digest([family, s]),
        structural_hash=structure(family, s),
        numeric_max=max(numbers, default=0),
        prompt_characters=len(r["prompt"]),
        numeric_shift=family not in ("functional_graph", "alternating_maps"),
    )
    return r


def prepare(config):
    if (EXP / "lockbox_opened.json").exists():
        raise RuntimeError("Cannot regenerate or select on consumed 4B data")
    # Compatibility checks are limited to hashes and development supervision, not old final scores.
    registry = json.loads(
        Path(
            "experiments/experiment_04a/development_trajectory_registry.json"
        ).read_text()
    )
    checks = []
    for reg in registry:
        assert sha(reg["public_trajectory"]) == reg["public_sha256"]
        assert sha(reg["hidden_artifact"]) == reg["hidden_sha256"]
        hidden = np.load(reg["hidden_artifact"])
        assert hidden["states"].shape == (4, reg["instances"], 1024)
        t = np.load(
            Path("artifacts/experiment_04a/trajectories")
            / f"{reg['split']}_seed{reg['seed']}.npz"
        )
        assert np.array_equal(t["correct"], t["logits"].argmax(-1) == t["labels"])
        selection = json.loads(
            Path(
                f"experiments/experiment_04a/selection_seed{reg['seed']}.json"
            ).read_text()
        )
        key = "training" if reg["split"] == "controller_train" else "validation"
        tp = (
            Path("artifacts/experiment_04a/trajectories")
            / f"{reg['split']}_seed{reg['seed']}.npz"
        )
        assert sha(tp) == selection["signature"][f"{key}_trajectory_sha256"]
        checks.append(
            {
                "split": reg["split"],
                "seed": reg["seed"],
                "trajectory_sha256": sha(tp),
                "hidden_sha256": reg["hidden_sha256"],
                "states": int(t["correct"].size),
            }
        )
    atomic_json(
        EXP / "compatibility_audit.json",
        {
            "checked_utc": utc(),
            "compatible_development_caches": checks,
            "schema": "latent-loop-trajectory-v1",
            "preservation": preservation(),
            "final_4a_data_role": "Identity/structure exclusion only; no selection on their labels or scores",
        },
    )
    rows = load("controller_val")
    rng = np.random.default_rng(config["validation_partition_seed"])
    parts = {name: [] for name in ("checkpoint", "calibration", "policy")}
    for task in TASKS:
        ix = np.array([i for i, r in enumerate(rows) if r["task"] == task])
        assert len(ix) % 3 == 0
        for name, values in zip(parts, np.array_split(rng.permutation(ix), 3)):
            parts[name].extend(values.tolist())
    parts = {k: sorted(v) for k, v in parts.items()}
    atomic_json(
        EXP / "validation_partitions.json",
        {
            "seed": config["validation_partition_seed"],
            "source_sha256": sha(
                "results/experiment_04a/datasets/controller_val.jsonl.gz"
            ),
            "indices": parts,
            "sizes": {k: len(v) for k, v in parts.items()},
        },
    )
    old = [
        r
        for p in Path("results/experiment_04a/datasets").glob("*.gz")
        for r in public_rows(p)
    ]
    shapes = {r["structural_hash"] for r in old}
    ids = {r["id"] for r in old}
    prompts = {r["prompt"] for r in old}
    historical = set(
        json.loads(
            Path(
                "experiments/experiment_04a/historical_structure_exclusions.json"
            ).read_text()
        )["fingerprints"]
    )
    manifest = {
        "version": "decision-value-v1",
        "development_source": "4A controller_train/val (unchanged)",
        "split_seeds": config["final_seeds"],
        "splits": {},
        "excluded_prior_instances": len(old),
        "prior_manifest_sha256": sha(
            "experiments/experiment_04a/dataset_manifest.json"
        ),
        "numeric_shift_definition": "Scale real arithmetic/state operands, numeric predicates, resource capacities/requests by 17; no scaling of entity/query indices. Pointer remains a fresh nominal-range control.",
        "integrity": {},
    }
    rejection = {
        "duplicate_instance_or_shape_or_prompt": 0,
        "historical_compatible_shape": 0,
        "invalid_or_nonunique": 0,
    }
    initial = len(ids)
    for split in FINAL:
        rng = random.Random(config["final_seeds"][split])
        approaches = [2] if split == "structural_shift" else [0, 1]
        strata = (
            ["extrapolation"]
            if split == "harder_depth"
            else ["easy", "moderate", "hard"]
        )
        cells = list(itertools.product(TASKS, approaches, strata))
        count = config["final_sizes"][split]
        assert count % len(cells) == 0
        generated = []
        while len(generated) < count:
            guard(config, 30)
            task, approach, stratum = cells[len(generated) % len(cells)]
            item = row(
                rng,
                task,
                approach,
                stratum,
                rng.choice([2, 3] if split == "new_template" else [0, 1]),
                split,
            )
            if item is None:
                rejection["invalid_or_nonunique"] += 1
                continue
            if split == "numeric_range":
                item = numerical_transform(item, rng, config["numeric_scale"])
            if (
                item["id"] in ids
                or item["structural_hash"] in shapes
                or item["prompt"] in prompts
            ):
                rejection["duplicate_instance_or_shape_or_prompt"] += 1
                continue
            if historical_key(item) in historical:
                rejection["historical_compatible_shape"] += 1
                continue
            assert len(set(item["choices"])) == 4
            label = reference(item["generator_id"], item["spec"])
            assert item["choices"][item["label"]] == (
                item["entities"][label] if isinstance(label, int) else label
            )
            ids.add(item["id"])
            shapes.add(item["structural_hash"])
            prompts.add(item["prompt"])
            generated.append(item)
        p = CACHE / "datasets" / "sealed" / f"{split}.jsonl.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(canonical(r) + "\n" for r in generated).encode()
        if p.exists():
            with gzip.open(p, "rb") as f:
                assert f.read() == content, "Recipe changed; incompatible sealed draft"
        else:
            with gzip.GzipFile(filename=str(p), mode="wb", mtime=0) as f:
                f.write(content)
        import hashlib

        manifest["splits"][split] = {
            "n": count,
            "seed": config["final_seeds"][split],
            "sha256": sha(p),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "balanced_task_count": count // 6,
            "path": p.as_posix(),
            "sealed": True,
            "generator_ids": sorted({r["generator_id"] for r in generated}),
        }
    manifest["integrity"] = {
        "new_unique_instances": len(ids) - initial,
        "overlap_with_prior_or_other_splits": 0,
        "generation_rejections": rejection,
        "independent_solvers_verified": True,
        "abstract_algorithm_overlap": "Allowed within ID/lexical/range shifts; constants/query-preserving alpha duplicates rejected",
    }
    atomic_json(EXP / "dataset_manifest.json", manifest)
    print(
        json.dumps(
            {"sizes": config["final_sizes"], "integrity": manifest["integrity"]}
        ),
        flush=True,
    )


def trajectory(split, seed):
    directory = (
        Path("artifacts/experiment_04a") if split.startswith("controller_") else CACHE
    )
    return dict(np.load(directory / "trajectories" / f"{split}_seed{seed}.npz"))


def features(split):
    directory = (
        Path("artifacts/experiment_04a") if split.startswith("controller_") else CACHE
    )
    return dict(np.load(directory / "features" / f"{split}.npz"))


@contextmanager
def foundation_workspace():
    """Scoped adapter reuses 4A extraction/export implementation, never its destinations."""
    from latent_loop.experiment04a import pipeline

    changes = {
        "EXP": EXP,
        "OUT": OUT,
        "CACHE": CACHE,
        "FINAL": FINAL,
        "load": load,
        "guard": guard,
    }
    prior = {k: getattr(pipeline, k) for k in changes}
    for k, v in changes.items():
        setattr(pipeline, k, v)
    try:
        yield pipeline
    finally:
        for k, v in prior.items():
            setattr(pipeline, k, v)


def final_trajectories(config):
    verify_freeze(config)
    assert (EXP / "lockbox_opened.json").exists()
    with foundation_workspace() as pipeline:
        pipeline.extract(config, list(FINAL), final=True)
        pipeline.collect(config, list(FINAL), final=True)
    preservation()
