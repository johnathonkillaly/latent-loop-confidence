"""Frozen model inference and complete, problem-grouped public measurements."""

import gzip
import json

import numpy as np
from scipy.special import expit, softmax

from latent_loop.experiment03.science import scalar_predict
from latent_loop.experiment04a.data import TASKS, sha
from latent_loop.experiment04b.common import (
    CACHE,
    EXP,
    FINAL,
    OUT,
    guard,
    utc,
    verify_freeze,
)
from latent_loop.experiment04b.data import features, load, trajectory
from latent_loop.experiment04b.science import (
    RANKERS,
    calibrated_confidence,
    calibrated_softmax,
    deterministic_mixture,
    outcomes,
    pair_completion,
    policy_events,
    ranker_inputs,
    signal_inputs,
    stop_depth,
    transition_metrics,
    visited_selection,
)
from latent_loop.training import atomic_json


def settings(seed):
    return json.loads((EXP / f"selection_seed{seed}.json").read_text())


def infer(config, seed, t, rows):
    """Labels are never read by model or deployment scores; rows feed diagnostic controls only."""
    from latent_loop.experiment04b.models import make_model
    from latent_loop.experiment04b.train import predict, regression, scalar_features

    chosen = settings(seed)
    models = {}
    raw = {}
    for name in chosen["deployment_checkpoints"]:
        model = make_model(name, config, seed)
        model.load_weights(
            str(CACHE / "checkpoints" / f"{name}_seed{seed}" / "best.safetensors")
        )
        models[name] = model
        if not name.startswith("ranker_"):
            raw[name] = predict(model, t["x"])
    variants = {
        name: calibrated_softmax(raw[name], chosen["calibrators"][name])
        for name in ("d4_plain", "d4_weighted", "matched")
    }
    variants.update(
        {
            name + "_raw": softmax(raw[name], axis=-1)
            for name in ("d4_plain", "d4_weighted", "matched", "random")
        }
    )
    event = [
        scalar_predict(chosen["calibrators"]["events"][ev], raw[f"event_{ev}"])
        for ev in ("WR", "RW")
    ]
    variants["binary_pair"] = pair_completion(*event, chosen["class_prior"])
    variants["prevalence"] = np.broadcast_to(
        chosen["class_prior"], variants["matched"].shape
    )
    variants["no_transition"] = np.broadcast_to(
        [0.5, 0, 0, 0.5], variants["matched"].shape
    )
    c = calibrated_confidence(t["confidence"], chosen["confidence_calibration"])
    v = regression(seed, t)
    d = variants[chosen["D_model"]]
    scores = {"C": c, "V": v, "D": d, "legacy_gain": t["improvement"]}
    for name, setting in chosen["stacks"].items():
        scores[f"stack_{name}"] = policy_events(setting, signal_inputs(name, c, v, d))
    for name, value in scalar_features(t, c, v, rows).items():
        events = policy_events(chosen["scalar_controls"][name], value)
        variants[name] = pair_completion(
            events[..., 0], events[..., 1], chosen["class_prior"]
        )
    for name in chosen["stacks"]:
        ev = scores[f"stack_{name}"]
        variants[f"stack_{name}"] = pair_completion(
            ev[..., 0], ev[..., 1], chosen["class_prior"]
        )
    quality = {"max_C": c}
    quality.update(
        {
            f"ranker_{name}": expit(
                predict(models[f"ranker_{name}"], ranker_inputs(t["x"], c, v, d, name))[
                    ..., 0
                ]
            )
            for name in RANKERS
        }
    )
    return scores, variants, quality


def paths(scores, quality, chosen, ids):
    """Deployable path simulation uses only predictions of visited states."""
    rules = {
        **chosen["rules"],
        **{f"fixed_{k}": {"kind": "fixed", "depth": k} for k in range(1, 5)},
    }
    result = {}
    for name, rule in rules.items():
        paid = stop_depth(scores, rule)
        result[name] = {
            "paid": paid,
            "final": paid,
            **{selector: visited_selection(q, paid) for selector, q in quality.items()},
        }
    # Matched means depend on paid work only, never on test labels or task identity.
    for name in ("d_direct", "d_utility", "stack_CV", "stack_CVD", "validation_best"):
        source = (
            chosen["validation_best_combination"]["policy"]
            if name == "validation_best"
            else name
        )
        paid = deterministic_mixture(ids, float(result[source]["paid"].mean()))
        result[f"matched_{name}"] = {
            "paid": paid,
            "final": paid,
            **{s: visited_selection(q, paid) for s, q in quality.items()},
        }
    return result


def decision_log(paid):
    return [
        [
            "continue" if k < int(depth) else "stop" if k == int(depth) else "unvisited"
            for k in range(1, 5)
        ]
        for depth in paid
    ]


def measure_selection(t, paid, selected, ix):
    n = len(paid)
    j = np.arange(n)
    correct = t["correct"][selected - 1, j]
    final = t["correct"][paid - 1, j]
    visible = np.arange(4)[:, None] < paid
    oracle = np.any(t["correct"] & visible, axis=0)
    return {
        "n": int(len(ix)),
        "accuracy": float(correct[ix].mean()),
        "nll": float(t["loss"][selected[ix] - 1, ix].mean()),
        "mean_paid_loops": float(paid[ix].mean()),
        "mean_selected_depth": float(selected[ix].mean()),
        "earlier_selected_fraction": float((selected[ix] < paid[ix]).mean()),
        "recovered_fraction": float((correct[ix] & ~final[ix]).mean()),
        "damaged_fraction": float((~correct[ix] & final[ix]).mean()),
        "best_visited_oracle_accuracy": float(oracle[ix].mean()),
        "oracle_gap": float((oracle[ix] & ~correct[ix]).mean()),
    }


def failures(t, scores, path, rows, seed, split):
    """All event labels are offline diagnostics; representatives chosen by stable problem hash."""
    paid, selected = path["paid"], path["ranker_CVD"]
    n, j = len(paid), np.arange(len(paid))
    y = outcomes(t["correct"])
    actual = (y == 1).astype(int) - (y == 2)
    v = scores["D"][:3, :, 1] - scores["D"][:3, :, 2]
    pcorrect = t["correct"][paid - 1, j]
    scorrect = t["correct"][selected - 1, j]
    definitions = {
        "A": (y == 1, "all eligible adjacent transitions"),
        "B": (y == 2, "all eligible adjacent transitions"),
        "C": ((y == 0) & (t["gain"] > 0.01), "all eligible adjacent transitions"),
        "D": ((y == 0) & (t["gain"] < -0.01), "all eligible adjacent transitions"),
        "E": (
            (y == 2) & (np.arange(1, 4)[:, None] == paid),
            "all problems; avoided damage immediately after a stopped correct state",
        ),
        "F": (
            (y == 2) & (np.arange(1, 4)[:, None] < paid),
            "all problems; damage on a visited continuation",
        ),
        "G": (
            (scorrect & ~pcorrect & (selected < paid))[None],
            "all problems; recovered earlier correct answer",
        ),
        "H": (
            (~scorrect & pcorrect & (selected < paid))[None],
            "all problems; wrong earlier answer replaces correct final visited answer",
        ),
    }
    summary, examples = {}, []
    for category, (mask, denominator) in definitions.items():
        depth, index = np.where(mask)
        if category in ("G", "H"):
            depth = selected[index] - 1
        continuation_depth = np.minimum(depth, 2)
        summary[category] = {
            "count": int(len(index)),
            "denominator": int(3 * n if category in "ABCD" else n),
            "frequency": float(len(index) / (3 * n if category in "ABCD" else n)),
            "definition": denominator,
            "loss_delta_tolerance": 0.01,
            "mean_C": float(scores["C"][depth, index].mean()) if len(index) else None,
            "mean_predicted_D": float(v[continuation_depth, index].mean())
            if len(index)
            else None,
            "mean_actual_D": float(actual[continuation_depth, index].mean())
            if len(index)
            else None,
            "mean_selected_depth": float(selected[index].mean())
            if len(index)
            else None,
            "mean_paid_loops": float(paid[index].mean()) if len(index) else None,
            "event_loop_not_necessarily_visited": category in "ABCD",
        }
        ordered = sorted(
            range(len(index)), key=lambda z: (rows[index[z]]["id"], int(depth[z]))
        )[:3]
        for z in ordered:
            i, k = int(index[z]), int(depth[z])
            examples.append(
                {
                    "category": category,
                    "seed": seed,
                    "split": split,
                    "instance_id": rows[i]["id"],
                    "prompt": rows[i]["prompt"],
                    "choices": rows[i]["choices"],
                    "gold_label": int(t["labels"][i]),
                    "event_depth": k + 1,
                    "predictions_by_depth": t["logits"][:, i].argmax(-1).tolist(),
                    "correct_by_depth": t["correct"][:, i].tolist(),
                    "loss_by_depth": t["loss"][:, i].tolist(),
                    "C_by_depth": scores["C"][:, i].tolist(),
                    "predicted_D_by_depth": (
                        scores["D"][:3, i, 1] - scores["D"][:3, i, 2]
                    ).tolist(),
                    "actual_D_by_depth": actual[:, i].tolist(),
                    "selected_depth": int(selected[i]),
                    "paid_loops": int(paid[i]),
                    "decisions": decision_log(paid[i : i + 1])[0],
                    "selection_rule": "stable lexicographic instance ID; no performance-based example choice",
                }
            )
    return summary, examples


def evaluate(config, splits=None):
    if splits is None:
        splits = FINAL
    if any(s in FINAL for s in splits):
        verify_freeze(config)
    partitions = json.loads((EXP / "validation_partitions.json").read_text())["indices"]
    for split in splits:
        rows = load(split, allow_final=split in FINAL)
        ids = [r["id"] for r in rows]
        f = features(split)
        for seed in config["parent_seeds"]:
            guard(config, 25)
            t = trajectory(split, seed)
            chosen = settings(seed)
            scores, variants, quality = infer(config, seed, t, rows)
            pp = paths(scores, quality, chosen, ids)
            y = outcomes(t["correct"])
            groups = {
                "all": np.arange(len(rows)),
                **{
                    task: np.flatnonzero([r["task"] == task for r in rows])
                    for task in TASKS
                },
            }
            if split == "controller_val":
                groups.update(
                    {
                        "validation_" + name: np.asarray(ix)
                        for name, ix in partitions.items()
                    }
                )
            groups.update(
                {
                    "stratum_" + str(s): np.flatnonzero(
                        [r["difficulty_stratum"] == s for r in rows]
                    )
                    for s in sorted({r["difficulty_stratum"] for r in rows})
                }
            )
            transitions = {
                group: {
                    name: transition_metrics(p[:3, ix], y[:, ix])
                    for name, p in variants.items()
                }
                for group, ix in groups.items()
            }
            depth_transitions = {
                str(k + 1): {
                    name: transition_metrics(p[k], y[k]) for name, p in variants.items()
                }
                for k in range(3)
            }
            selection = {
                group: {
                    name: {
                        sel: measure_selection(t, p["paid"], p[sel], ix)
                        for sel in p
                        if sel != "paid"
                    }
                    for name, p in pp.items()
                }
                for group, ix in groups.items()
            }
            baselines = {
                group: {
                    "frozen_accuracy": float(
                        (f["base"][ix].argmax(-1) == t["labels"][ix]).mean()
                    ),
                    "one_pass_accuracy": float(
                        (t["one_pass"][ix].argmax(-1) == t["labels"][ix]).mean()
                    ),
                    "best_allowed_oracle_accuracy": float(
                        t["correct"][:, ix].any(0).mean()
                    ),
                    "best_allowed_oracle_paid_loops": 4,
                }
                for group, ix in groups.items()
            }
            failure, examples = failures(t, scores, pp["d_direct"], rows, seed, split)
            outfile = OUT / "predictions" / f"{split}_seed{seed}.jsonl.gz"
            outfile.parent.mkdir(parents=True, exist_ok=True)
            with gzip.GzipFile(filename=str(outfile), mode="wb", mtime=0) as stream:
                for i, row in enumerate(rows):
                    record = {
                        "instance_id": row["id"],
                        "task": row["task"],
                        "stratum": row["difficulty_stratum"],
                        "label": int(t["labels"][i]),
                        "logits": t["logits"][:, i].tolist(),
                        "correct": t["correct"][:, i].tolist(),
                        "loss": t["loss"][:, i].tolist(),
                        "frozen_correct": bool(f["base"][i].argmax() == t["labels"][i]),
                        "one_pass_correct": bool(
                            t["one_pass"][i].argmax() == t["labels"][i]
                        ),
                        "C": scores["C"][:, i].tolist(),
                        "V": scores["V"][:, i].tolist(),
                        "legacy_gain": scores["legacy_gain"][:, i].tolist(),
                        "transition_probabilities": {
                            name: value[:, i].tolist()
                            for name, value in variants.items()
                        },
                        "stack_events": {
                            name: value[:, i].tolist()
                            for name, value in scores.items()
                            if name.startswith("stack_")
                        },
                        "quality": {
                            name: value[:, i].tolist()
                            for name, value in quality.items()
                        },
                        "paths": {
                            name: {sel: int(value[i]) for sel, value in p.items()}
                            for name, p in pp.items()
                        },
                        "decisions": {
                            name: decision_log(p["paid"][i : i + 1])[0]
                            for name, p in pp.items()
                        },
                    }
                    stream.write(
                        (json.dumps(record, separators=(",", ":")) + "\n").encode()
                    )
            atomic_json(
                OUT / "metrics" / f"{split}_seed{seed}.json",
                {
                    "seed": seed,
                    "split": split,
                    "completed_utc": utc(),
                    "instances": len(rows),
                    "D_model": chosen["D_model"],
                    "transitions": transitions,
                    "transitions_by_depth": depth_transitions,
                    "selection": selection,
                    "baselines": baselines,
                    "failures_D_ranker_CVD": failure,
                    "predictions_sha256": sha(outfile),
                },
            )
            atomic_json(OUT / "examples" / f"{split}_seed{seed}.json", examples)
            print(f"evaluated {split} parent {seed}: {len(rows)} problems", flush=True)
