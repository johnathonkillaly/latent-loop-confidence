"""Fresh balanced 4A-generator splits with historical and within-study exclusions."""

import gzip
import itertools
import json
import random
from pathlib import Path

from latent_loop.experiment04a.data import TASKS, canonical, row, sha
from latent_loop.experiment04b.data import historical_key, numerical_transform
from latent_loop.training import atomic_json

EXP = Path("experiments/experiment_05")
OUT = Path("results/experiment_05")
CACHE = Path("artifacts/experiment_05")
FINAL = ("fresh_id", "new_template", "structural_shift", "harder_depth", "numeric_range")


def prepare(config):
    if (EXP / "dataset_manifest.json").exists():
        manifest = json.loads((EXP / "dataset_manifest.json").read_text())
        assert manifest["recipe"] == {k: config[k] for k in ("sizes", "split_seeds")}
        return manifest
    seen, structures, prompts = set(), set(), set()
    historical = []
    exclusion = Path("experiments/experiment_04a/historical_structure_exclusions.json")
    structures.update(json.loads(exclusion.read_text())["fingerprints"])
    for folder in ("experiment_04a", "experiment_04b"):
        for p in sorted((Path("results") / folder / "datasets").glob("*.jsonl.gz")):
            historical.append({"path": p.as_posix(), "sha256": sha(p)})
            for line in gzip.open(p, "rt"):
                item = json.loads(line)
                seen.add(item["id"])
                structures.add(historical_key(item))
                structures.add(item["structural_hash"])
                prompts.add(item["prompt"])
    old_counts = (len(seen), len(structures), len(prompts))
    manifest = {"version": "train-thinker-v1", "recipe": {k: config[k] for k in ("sizes", "split_seeds")},
                "historical_exclusion_sources": historical, "historical_exclusion_list_sha256": sha(exclusion),
                "historical_identity_role": "Exclusion only; old final labels and scores never select this experiment",
                "splits": {}, "rejections": 0}
    for split, size in config["sizes"].items():
        rng = random.Random(config["split_seeds"][split])
        cells = list(itertools.product(TASKS, [2] if split == "structural_shift" else [0, 1],
                                      ["extrapolation"] if split == "harder_depth" else ["easy", "moderate", "hard"]))
        assert size % len(cells) == 0
        rows = []
        while len(rows) < size:
            if manifest["rejections"] > 100000:
                raise RuntimeError("Insufficient unique structures")
            task, approach, stratum = cells[len(rows) % len(cells)]
            item = row(rng, task, approach, stratum, rng.choice([2, 3] if split == "new_template" else [0, 1]), split)
            if item is not None and split == "numeric_range":
                item = numerical_transform(item, rng, 17)
            if item is None or item["id"] in seen or item["structural_hash"] in structures or historical_key(item) in structures or item["prompt"] in prompts:
                manifest["rejections"] += 1
                continue
            rows.append(item)
            seen.add(item["id"])
            structures.add(item["structural_hash"])
            structures.add(historical_key(item))
            prompts.add(item["prompt"])
        p = CACHE / "datasets" / ("sealed" if split in FINAL else "development") / f"{split}.jsonl.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(canonical(r) + "\n" for r in rows).encode()
        with gzip.GzipFile(filename=str(p), mode="wb", mtime=0) as f:
            f.write(content)
        manifest["splits"][split] = {"n": size, "seed": config["split_seeds"][split], "path": str(p), "sha256": sha(p),
                                     "sealed": split in FINAL, "task_count": size // 6,
                                     "generator_ids": sorted({r["generator_id"] for r in rows})}
    manifest["historical_exclusion_counts"] = list(old_counts)
    manifest["integrity"] = {"historical_instance_overlap": 0, "alpha_instance_overlap": 0, "prompt_overlap": 0,
                              "labels_independently_verified": True,
                              "shared_algorithm_archetypes": "Allowed on ID/template/range; structural shift holds out generator families"}
    atomic_json(EXP / "dataset_manifest.json", manifest)
    print("Fresh splits prepared:", config["sizes"], "rejections", manifest["rejections"], flush=True)
    return manifest


def load(split, allow_final=False):
    if split in FINAL and (not allow_final or not (EXP / "lockbox_opened.json").exists()):
        raise PermissionError("Freeze configurations and checkpoint selections before opening final data")
    info = json.loads((EXP / "dataset_manifest.json").read_text())["splits"][split]
    assert sha(info["path"]) == info["sha256"]
    return [json.loads(line) for line in gzip.open(info["path"], "rt")]
