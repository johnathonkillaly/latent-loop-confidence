"""Frozen-model development, protocol freeze, and a single prospective evaluation."""

import argparse
import gzip
import json
import os
import resource
import shutil
import time
from pathlib import Path

import numpy as np
from scipy.special import expit

from latent_loop.experiment03.science import observables, scalar_predict, targets
from latent_loop.experiment04a.common import (
    audit,
    checkpoint_paths,
    freeze,
    guard,
    open_lockbox,
    preservation,
    setup,
    utc,
    verify_freeze,
)
from latent_loop.experiment04a.data import CACHE, EXP, FINAL, OUT, build, load, sha
from latent_loop.experiment04a.science import (
    all_confidence,
    binary,
    confidence_calibrators,
    fit_controls,
    scalar_inputs,
    stopping,
    strata,
    transition_stats,
)
from latent_loop.experiment04a.trajectory import SCHEMA_VERSION, record
from latent_loop.training import atomic_json


def compressed(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with gzip.GzipFile(filename=str(temp), mode="wb", mtime=0) as f:
        f.write(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())
    temp.replace(path)


def components(seed):
    import mlx.core as mx

    from latent_loop.experiment03.heads import Probe
    from latent_loop.models import Refiner

    paths = checkpoint_paths(seed)
    models = {}
    for name in ("refiner", "one_pass"):
        m = Refiner(1024, 64, False)
        m.load_weights(str(paths[name]))
        m.freeze()
        models[name] = m
    for i, name in enumerate(("confidence", "improvement", "joint")):
        mx.random.seed(seed + 5000 + i)
        m = Probe(1032, 64, 3 if name == "joint" else 1)
        m.load_weights(str(paths[name]))
        m.freeze()
        models[name] = m
    return models


def extract(config, splits, final=False):
    import mlx.core as mx

    from latent_loop.backbone import environment, get_model, hidden_forward, tokenize

    if final:
        verify_freeze()
        assert (EXP / "lockbox_opened.json").exists()
    env = environment(config["memory_gib"])
    model, tokenizer, manifest = get_model()
    features = CACHE / "features"
    features.mkdir(exist_ok=True)
    label_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in "ABCD"]
    assert label_ids == [32, 33, 34, 35]
    decoder = model.model.embed_tokens.weight[mx.array(label_ids)]
    expected = np.load("artifacts/features/decoder.npz")["decoder"]
    assert np.array_equal(np.array(decoder), expected)
    signature = {
        "model": manifest,
        "prefix": "Answer: (",
        "precision": "FP32",
        "feature_source_sha256": sha("src/latent_loop/backbone.py"),
    }
    sigpath = features / "model_signature.json"
    if sigpath.exists():
        assert json.loads(sigpath.read_text()) == signature
    else:
        atomic_json(sigpath, signature)
    timing = []
    start = time.perf_counter()
    ds = json.loads((EXP / "dataset_manifest.json").read_text())
    for split in splits:
        rows = load(split, allow_final=final)
        target = features / f"{split}.npz"
        sig = features / f"{split}.json"
        stamp = {
            "model": signature,
            "dataset_content_sha256": ds["splits"][split]["content_sha256"],
        }
        if sig.exists():
            assert json.loads(sig.read_text()) == stamp
        else:
            atomic_json(sig, stamp)
        if target.exists():
            continue
        folder = features / split
        folder.mkdir(exist_ok=True)
        for begin in range(0, len(rows), config["extract_batch"]):
            guard(config)
            path = folder / f"{begin:06d}.npz"
            if path.exists():
                continue
            encoded = [
                tokenize(tokenizer, r["prompt"])
                for r in rows[begin : begin + config["extract_batch"]]
            ]
            lengths = [len(v) for v in encoded]
            ids = mx.array(
                [
                    v
                    + [tokenizer.pad_token_id or tokenizer.eos_token_id]
                    * (max(lengths) - len(v))
                    for v in encoded
                ]
            )
            mx.synchronize()
            tic = time.perf_counter()
            h = hidden_forward(model, ids)[mx.arange(len(ids)), mx.array(lengths) - 1]
            vocabulary = model.model.embed_tokens.as_linear(h)
            logits = vocabulary[:, mx.array(label_ids)]
            lp = logits - mx.logsumexp(vocabulary, axis=-1, keepdims=True)
            greedy = mx.argmax(vocabulary, axis=-1)
            mx.eval(h, logits, lp, greedy)
            mx.synchronize()
            seconds = time.perf_counter() - tic
            np.savez_compressed(
                path,
                h=np.array(h),
                base=np.array(logits),
                full_logp=np.array(lp),
                greedy=np.array(greedy),
                lengths=np.array(lengths),
                seconds=seconds,
            )
            timing.append(
                {"split": split, "begin": begin, "n": len(ids), "seconds": seconds}
            )
            if begin % 504 == 0:
                print(f"extract {split}: {begin + len(ids)}/{len(rows)}", flush=True)
        shards = [dict(np.load(p)) for p in sorted(folder.glob("*.npz"))]
        np.savez_compressed(
            target,
            **{
                key: np.concatenate([r[key] for r in shards])
                for key in ("h", "base", "full_logp", "greedy", "lengths")
            },
        )
    # Audit native candidate projection on development only.
    sample = load("controller_val")[0]
    ids = mx.array([tokenize(tokenizer, sample["prompt"])])
    h = hidden_forward(model, ids)[:, -1]
    native = model(ids)[:, -1, mx.array(label_ids)]
    direct = h @ decoder.T
    error = float(mx.max(mx.abs(native - direct)).item())
    assert error < 0.001
    meta = {
        "completed_utc": utc(),
        "final": final,
        "environment": env,
        "model": manifest,
        "batches": timing,
        "command_seconds": time.perf_counter() - start,
        "native_candidate_parity_max_abs": error,
        "peak_mlx_bytes": mx.get_peak_memory(),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "load_average": list(os.getloadavg()),
    }
    path = OUT / ("final_extraction.json" if final else "development_extraction.json")
    if not path.exists():
        atomic_json(path, meta)


def features(split):
    return dict(np.load(CACHE / "features" / f"{split}.npz"))


def collect(config, splits, final=False):
    import mlx.core as mx

    from latent_loop.backbone import MODEL_ID, REVISION, environment

    environment(config["memory_gib"])
    if final:
        verify_freeze()
    directory = CACHE / "trajectories"
    directory.mkdir(exist_ok=True)
    decoder = mx.array(np.load("artifacts/features/decoder.npz")["decoder"])
    registry = []
    for seed in config["parent_seeds"]:
        models = components(seed)
        hashes = {
            "backbone_id": MODEL_ID,
            "backbone_revision": REVISION,
            "refiner_sha256": sha(checkpoint_paths(seed)["refiner"]),
        }
        for split in splits:
            guard(config)
            rows = load(split, allow_final=final)
            a = features(split)
            path = directory / f"{split}_seed{seed}.npz"
            hidden_path = directory / f"{split}_seed{seed}_hidden.npz"
            stamp = {
                "feature_sha256": sha(CACHE / "features" / f"{split}.npz"),
                "checkpoint_hashes": {
                    k: sha(v) for k, v in checkpoint_paths(seed).items()
                },
                "schema": SCHEMA_VERSION,
                "gain_delta": config["gain_delta_nats"],
            }
            sigpath = path.with_suffix(".json")
            if path.exists():
                assert json.loads(sigpath.read_text()) == stamp
                t = dict(np.load(path))
            else:
                logits, states = models["refiner"](
                    mx.array(a["h"]), mx.array(a["base"]), decoder, 4, True
                )
                logits, states = np.array(logits), np.array(states)
                x, scalar = observables(states, logits, a["h"])
                labels = np.array([r["label"] for r in rows], np.int32)
                loss, correct, gain, useful = targets(
                    logits, labels, config["gain_delta_nats"]
                )
                ff = np.array(
                    models["one_pass"](
                        mx.array(a["h"]), mx.array(a["base"]), decoder, 1
                    )
                )
                scores = {}
                for name in ("confidence", "improvement", "joint"):
                    out = np.array(models[name](mx.array(x.reshape(-1, 1032)))).reshape(
                        4, len(rows), -1
                    )
                    if name == "joint":
                        scores.update(
                            joint_confidence=expit(out[..., 0]),
                            joint_gain=expit(out[..., 1]),
                        )
                    else:
                        scores[name] = expit(out[..., 0])
                np.savez_compressed(hidden_path, states=states, backbone=a["h"])
                np.savez_compressed(
                    path,
                    logits=logits,
                    x=x,
                    scalar=scalar,
                    labels=labels,
                    loss=loss,
                    correct=correct,
                    gain=gain,
                    useful=useful,
                    one_pass=ff,
                    **scores,
                )
                atomic_json(sigpath, stamp)
                t = dict(np.load(path))
            hidden = dict(np.load(hidden_path))
            states = hidden["states"]
            pub = OUT / "trajectories" / f"{split}_seed{seed}.jsonl.gz"
            pub.parent.mkdir(exist_ok=True)
            hidden_hash = sha(hidden_path)
            if not pub.exists():
                with gzip.GzipFile(filename=str(pub), mode="wb", mtime=0) as f:
                    for i, row in enumerate(rows):
                        for k in range(4):
                            previous = a["h"][i] if k == 0 else states[k - 1, i]
                            ref = {
                                "artifact_path": hidden_path.as_posix(),
                                "sha256": hidden_hash,
                                "array": "states",
                                "depth_index": k,
                                "instance_index": i,
                                "dtype": "float32",
                                "dimensions": [1024],
                            }
                            item = record(
                                row,
                                seed,
                                k + 1,
                                t["logits"][:, i],
                                states[:, i],
                                previous,
                                t["loss"][:, i],
                                t["correct"][:, i],
                                ref,
                                hashes,
                            )
                            f.write(
                                (
                                    json.dumps(
                                        item, separators=(",", ":"), allow_nan=False
                                    )
                                    + "\n"
                                ).encode()
                            )
            registry.append(
                {
                    "split": split,
                    "seed": seed,
                    "schema": SCHEMA_VERSION,
                    "instances": len(rows),
                    "state_rows": 4 * len(rows),
                    "public_trajectory": pub.as_posix(),
                    "public_sha256": sha(pub),
                    "hidden_artifact": hidden_path.as_posix(),
                    "hidden_sha256": hidden_hash,
                    "model_hashes": hashes,
                }
            )
            target = OUT / "datasets" / f"{split}.jsonl.gz"
            target.parent.mkdir(exist_ok=True)
            source = (
                CACHE
                / "datasets"
                / ("sealed" if split in FINAL else "development")
                / f"{split}.jsonl.gz"
            )
            if not target.exists():
                shutil.copyfile(source, target)
        print(f"trajectory schema export seed {seed}: complete", flush=True)
    atomic_json(
        EXP
        / (
            "final_trajectory_registry.json"
            if final
            else "development_trajectory_registry.json"
        ),
        registry,
    )


def trajectory(split, seed):
    return dict(np.load(CACHE / "trajectories" / f"{split}_seed{seed}.npz"))


def neural_scores(t):
    return {
        name: t[name]
        for name in ("confidence", "improvement", "joint_confidence", "joint_gain")
    }


def choose_settings(config, t, scores):
    selections = {}
    sweep = {}
    for name, grid in (
        ("confidence", config["confidence_thresholds"]),
        ("improvement", config["gain_thresholds"]),
    ):
        options = []
        for threshold in grid:
            setting = {"threshold": threshold}
            d = stopping(name, scores, setting)
            ix = np.arange(len(d))
            options.append(
                {
                    "setting": setting,
                    "accuracy": float(t["correct"][d - 1, ix].mean()),
                    "nll": float(t["loss"][d - 1, ix].mean()),
                    "mean_loops": float(d.mean()),
                }
            )
        best = min(options, key=lambda r: (-r["accuracy"], r["mean_loops"], r["nll"]))
        selections[name] = best["setting"]
        sweep[name] = options
    selections["fixed_selected"] = {
        "depth": min(
            range(4), key=lambda k: (-t["correct"][k].mean(), t["loss"][k].mean(), k)
        )
        + 1
    }
    return selections, sweep


def develop(config):
    if (EXP / "lockbox_opened.json").exists():
        raise RuntimeError("Final set consumed: development selection is closed")
    for seed in config["parent_seeds"]:
        guard(config)
        tr, va = (
            trajectory("controller_train", seed),
            trajectory("controller_val", seed),
        )
        controls = fit_controls(tr, neural_scores(tr), load("controller_train"))
        calibrators = confidence_calibrators(va, neural_scores(va))
        selections, sweep = choose_settings(config, va, neural_scores(va))
        path = EXP / f"selection_seed{seed}.json"
        signature = {
            "training_trajectory_sha256": sha(
                CACHE / "trajectories" / f"controller_train_seed{seed}.npz"
            ),
            "validation_trajectory_sha256": sha(
                CACHE / "trajectories" / f"controller_val_seed{seed}.npz"
            ),
            "config_sha256": sha(EXP / "config.json"),
        }
        value = {
            "selected_utc": utc(),
            "seed": seed,
            "signature": signature,
            "scalar_controls": controls,
            "calibrators": calibrators,
            "policies": selections,
            "validation_sweep": sweep,
            "selection_splits": ["controller_train", "controller_val"],
            "no_new_neural_training": True,
        }
        if path.exists():
            assert json.loads(path.read_text())["signature"] == signature
        else:
            atomic_json(path, value)
    evaluate(config, ["controller_train", "controller_val"], False)


def evaluate(config, splits, final=False):
    if final:
        verify_freeze()
    for seed in config["parent_seeds"]:
        selected = json.loads((EXP / f"selection_seed{seed}.json").read_text())
        for split in splits:
            guard(config)
            rows = load(split, allow_final=final)
            t = trajectory(split, seed)
            a = features(split)
            scores = neural_scores(t)
            confidence = all_confidence(t, scores, selected["calibrators"])
            inputs = scalar_inputs(t, scores, rows)
            gains = {
                "existing_gain": scores["improvement"],
                "joint_gain": scores["joint_gain"],
            }
            gains.update(
                {
                    name: scalar_predict(model, inputs[name])
                    for name, model in selected["scalar_controls"].items()
                    if name != "train_prior"
                }
            )
            gains["train_prior"] = np.full(
                (4, len(rows)),
                selected["scalar_controls"]["train_prior"]["probability"],
            )
            report = {
                "seed": seed,
                "split": split,
                "n": len(rows),
                "confidence": {
                    name: binary(p, t["correct"]) for name, p in confidence.items()
                },
                "gain": {name: binary(p[:3], t["useful"]) for name, p in gains.items()},
                "gain_by_depth": {
                    str(k + 1): {
                        name: binary(p[k], t["useful"][k]) for name, p in gains.items()
                    }
                    for k in range(3)
                },
                "transitions": transition_stats(t, config["gain_delta_nats"]),
                "policies": {},
                "strata": {},
                "unrestricted_backbone": {
                    "accuracy": float(
                        (a["greedy"] == np.array([32, 33, 34, 35])[t["labels"]]).mean()
                    ),
                    "option_token_rate": float(
                        np.isin(a["greedy"], [32, 33, 34, 35]).mean()
                    ),
                    "full_vocabulary_label_nll": float(
                        -a["full_logp"][np.arange(len(rows)), t["labels"]].mean()
                    ),
                },
            }
            depths = {f"fixed_{d}": np.full(len(rows), d, int) for d in range(1, 5)}
            for name in ("confidence", "improvement"):
                depths[name] = stopping(name, scores, selected["policies"][name])
            depths["fixed_selected"] = np.full(
                len(rows), selected["policies"]["fixed_selected"]["depth"], int
            )
            depths["oracle_best_loss"] = t["loss"].argmin(0) + 1
            # Privileged decision ceiling: first correct depth, or depth1 if none; pays to reach it.
            depths["oracle_first_correct"] = np.where(
                t["correct"].any(0), t["correct"].argmax(0) + 1, 1
            )
            decisions = {
                name: t["logits"][d - 1, np.arange(len(rows))]
                for name, d in depths.items()
            }
            decisions.update(frozen=a["base"], one_pass=t["one_pass"])
            masks = strata(rows, a["lengths"])
            for name, logits in decisions.items():
                predicted = logits.argmax(-1)
                correct = predicted == t["labels"]
                loss = (
                    np.logaddexp.reduce(logits.astype(np.float64), axis=-1)
                    - logits[np.arange(len(rows)), t["labels"]]
                )
                initial = t["correct"][0]
                depth = depths.get(
                    name, np.full(len(rows), 0 if name == "frozen" else 1, int)
                )
                tasks = {
                    task: float(correct[mask].mean())
                    for task, mask in masks["task"].items()
                }
                report["policies"][name] = {
                    "accuracy": float(correct.mean()),
                    "macro_task_accuracy": float(np.mean(list(tasks.values()))),
                    "nll": float(loss.mean()),
                    "mean_loops": float(depth.mean()),
                    "wrong_to_right": float((~initial & correct).mean()),
                    "right_to_wrong": float((initial & ~correct).mean()),
                    "stopping_counts": np.bincount(depth, minlength=5).tolist(),
                    "per_task": tasks,
                }
            for field, categories in masks.items():
                report["strata"][field] = {}
                for category, mask in categories.items():
                    report["strata"][field][category] = {
                        "n": int(mask.sum()),
                        "accuracy": {
                            name: float(
                                (logits[mask].argmax(-1) == t["labels"][mask]).mean()
                            )
                            for name, logits in decisions.items()
                        },
                        "confidence_auroc": binary(
                            scores["confidence"][:, mask], t["correct"][:, mask]
                        )["auroc"],
                        "gain_auroc": binary(
                            scores["improvement"][:3, mask], t["useful"][:, mask]
                        )["auroc"],
                        "gain_prevalence": float(t["useful"][:, mask].mean()),
                    }
            atomic_json(OUT / "metrics" / f"{split}_seed{seed}.json", report)
            compressed(
                OUT / "predictions" / f"{split}_seed{seed}.json.gz",
                {
                    "ids": [r["id"] for r in rows],
                    "labels": t["labels"].tolist(),
                    "base_logits": a["base"].tolist(),
                    "one_pass_logits": t["one_pass"].tolist(),
                    "logits": t["logits"].tolist(),
                    "correct": t["correct"].tolist(),
                    "loss": t["loss"].tolist(),
                    "gain": t["gain"].tolist(),
                    "confidence": {name: p.tolist() for name, p in confidence.items()},
                    "gain_predictions": {name: p.tolist() for name, p in gains.items()},
                    "depths": {name: d.tolist() for name, d in depths.items()},
                    "lengths": a["lengths"].tolist(),
                },
            )
    print("evaluation complete: " + ", ".join(splits), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "audit",
            "data",
            "development",
            "freeze",
            "final",
            "timing",
            "report",
            "all",
        ],
    )
    parser.add_argument("--config", default="experiments/experiment_04a/config.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    setup()
    assert config["max_recurrent_depth"] == 4
    start = time.perf_counter()
    if args.stage in ("audit", "all"):
        audit(config)
    if args.stage in ("data", "all"):
        if (EXP / "lockbox_opened.json").exists():
            raise RuntimeError("Cannot regenerate an opened lockbox")
        build(config)
    if args.stage in ("development", "all"):
        extract(config, ["controller_train", "controller_val"])
        collect(config, ["controller_train", "controller_val"])
        develop(config)
    if args.stage in ("freeze", "all"):
        freeze(config)
    if args.stage in ("final", "all"):
        open_lockbox()
        extract(config, list(FINAL), True)
        collect(config, list(FINAL), True)
        evaluate(config, list(FINAL), True)
    if args.stage in ("timing", "all"):
        from latent_loop.experiment04a.timing import measure

        measure(config)
    if args.stage in ("report", "all"):
        from latent_loop.experiment04a.report import report

        report(config)
    atomic_json(
        EXP / f"stage_{args.stage}.json",
        {
            "stage": args.stage,
            "completed_utc": utc(),
            "command_seconds": time.perf_counter() - start,
            "preservation": preservation(),
        },
    )


if __name__ == "__main__":
    main()
