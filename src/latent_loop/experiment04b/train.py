"""Resumable lightweight fits, grouped OOF stacking, and development-only selection."""

import json
import time
from pathlib import Path

import numpy as np

from latent_loop.experiment03.heads import Probe
from latent_loop.experiment03.science import fit_scalar
from latent_loop.experiment04a.data import TASKS, sha
from latent_loop.experiment04b.common import CACHE, EXP, OUT, guard, utc
from latent_loop.experiment04b.data import load, trajectory
from latent_loop.experiment04b.models import make_model
from latent_loop.experiment04b.science import (
    COMBINATIONS,
    RANKERS,
    calibrate_multiclass,
    calibrated_confidence,
    calibrated_softmax,
    choose_policy,
    confidence_calibration,
    fit_event_policy,
    outcomes,
    policy_events,
    ranker_inputs,
    signal_inputs,
)
from latent_loop.training import atomic_json, parameter_count


def fit_model(
    name,
    seed,
    config,
    x,
    y,
    vx,
    vy,
    *,
    objective="class4",
    weighted=False,
    fixed_epochs=None,
    extra_signature=None,
):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    guard(config, 45)
    model = make_model(name, config, seed)
    folder = CACHE / "checkpoints" / f"{name}_seed{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    signature = {
        "config_sha256": sha(EXP / "config.json"),
        "source": {
            p: sha(p)
            for p in (
                "src/latent_loop/experiment04b/models.py",
                "src/latent_loop/experiment04b/train.py",
            )
        },
        "objective": objective,
        "weighted": weighted,
        "fixed_epochs": fixed_epochs,
        "extra": extra_signature,
        "input_arrays": [
            {
                "shape": list(a.shape),
                "dtype": str(a.dtype),
                "sha256": sha_bytes(np.ascontiguousarray(a).tobytes()),
            }
            for a in (x, y, vx, vy)
        ],
    }
    epochs = fixed_epochs if fixed_epochs is not None else config["epochs"]
    opt = optim.AdamW(
        learning_rate=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    info = {
        "name": name,
        "seed": seed,
        "parameters": parameter_count(model),
        "signature": signature,
        "started_utc": utc(),
        "next_epoch": 0,
        "best_epoch": -1,
        "best_validation_nll": 1e30,
        "history": [],
        "training_seconds": 0.0,
        "training_items": int(x.shape[1] if objective == "ranker" else len(x)),
        "validation_items": int(vx.shape[1] if objective == "ranker" else len(vx)),
    }
    state = folder / "state.json"
    if state.exists():
        info = json.loads(state.read_text())
        assert info["signature"] == signature, "Incompatible checkpoint signature"
        model.load_weights(str(folder / "last.safetensors"))
        opt.state = tree_unflatten(
            list(mx.load(str(folder / "optimizer.safetensors")).items())
        )
    if info.get("complete"):
        model.load_weights(str(folder / "best.safetensors"))
        return model, info
    weights = np.ones(4, np.float32)
    if weighted:
        counts = np.bincount(np.asarray(y, int), minlength=4).astype(float)
        weights = 1 / np.sqrt(np.maximum(counts, 1))
        weights /= np.dot(weights, counts / counts.sum())
    wx = mx.array(weights.astype(np.float32))
    xx, yy, vxx, vyy = [mx.array(a) for a in (x, y, vx, vy)]
    mx.eval(xx, yy, vxx, vyy)

    def loss(m, xb, yb, lengths=None, training=True):
        if objective == "class4":
            out = m(xb)
            ce = nn.losses.cross_entropy(out, yb, reduction="none")
            return mx.mean(ce * wx[yb]) if weighted and training else mx.mean(ce)
        if objective == "binary":
            return nn.losses.binary_cross_entropy(m(xb)[:, 0], yb, reduction="mean")
        shape = xb.shape[:2]
        out = m(xb.reshape(-1, 1036)).reshape(shape)
        point = nn.losses.binary_cross_entropy(out, yb, reduction="none")
        if lengths is None:
            lengths = mx.full((shape[1],), 4, mx.int32)
        visible = mx.arange(4)[:, None] < lengths[None]
        point = mx.sum(point * visible) / mx.maximum(mx.sum(visible), 1)
        # Pairwise supervision only; model scoring itself remains per-current-state.
        pair = mx.array(0.0)
        denominator = mx.array(0.0)
        for a in range(4):
            for b in range(a + 1, 4):
                valid = visible[a] & visible[b] & (yb[a] != yb[b])
                direction = yb[a] - yb[b]
                pair += mx.sum(mx.logaddexp(0, -direction * (out[a] - out[b])) * valid)
                denominator += mx.sum(valid)
        return point + 0.5 * pair / mx.maximum(denominator, 1)

    gradient = nn.value_and_grad(model, loss)
    n = x.shape[1] if objective == "ranker" else len(x)
    for epoch in range(info["next_epoch"], epochs):
        guard(config, 30)
        started = time.perf_counter()
        rng = np.random.default_rng(seed * 10000 + epoch + sum(map(ord, name)))
        order = rng.permutation(n)
        values = []
        for begin in range(0, n, config["batch_size"]):
            ix = mx.array(order[begin : begin + config["batch_size"]])
            if objective == "ranker":
                lengths = mx.array(rng.integers(1, 5, len(ix)), mx.int32)
                value, grads = gradient(model, xx[:, ix], yy[:, ix], lengths)
            else:
                value, grads = gradient(model, xx[ix], yy[ix])
            grads, norm = optim.clip_grad_norm(grads, 1.0)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, value, norm)
            values.append(float(value.item()))
        vl = float(loss(model, vxx, vyy, training=False).item())
        assert np.isfinite(vl + np.mean(values))
        if fixed_epochs is not None or vl < info["best_validation_nll"]:
            info["best_validation_nll"], info["best_epoch"] = vl, epoch
            model.save_weights(str(folder / "best.safetensors"))
        model.save_weights(str(folder / "last.safetensors"))
        mx.save_safetensors(
            str(folder / "optimizer.safetensors"), dict(tree_flatten(opt.state))
        )
        info["history"].append(
            {
                "epoch": epoch,
                "train_objective": float(np.mean(values)),
                "validation_original_distribution_objective": vl,
            }
        )
        info.update(
            next_epoch=epoch + 1,
            training_seconds=info["training_seconds"] + time.perf_counter() - started,
            peak_mlx_bytes=mx.get_peak_memory(),
        )
        atomic_json(state, info)
    info.update(
        complete=True,
        completed_utc=utc(),
        checkpoint_sha256=sha(folder / "best.safetensors"),
        class_weights=weights.tolist(),
    )
    atomic_json(state, info)
    atomic_json(OUT / "training" / f"{name}_seed{seed}.json", info)
    model.load_weights(str(folder / "best.safetensors"))
    print(
        f"fit {name} parent {seed}: epoch {info['best_epoch']}, objective {info['best_validation_nll']:.4f}",
        flush=True,
    )
    return model, info


def predict(model, x):
    import mlx.core as mx

    shape = x.shape[:-1]
    return np.array(model(mx.array(x.reshape(-1, x.shape[-1])))).reshape(*shape, -1)


def regression(seed, t):
    model = Probe(1032, 64, 1)
    model.load_weights(
        f"artifacts/experiment_03/checkpoints/regression_seed{seed}/best.safetensors"
    )
    model.freeze()
    info = json.loads(
        Path(
            f"artifacts/experiment_03/checkpoints/regression_seed{seed}/state.json"
        ).read_text()
    )
    return predict(model, t["x"])[..., 0] * info["scale"]


def scalar_features(t, c, v, rows):
    simple = np.stack(
        [
            c,
            v,
            t["scalar"][..., 1],
            t["scalar"][..., 2],
            t["scalar"][..., 3],
            t["scalar"][..., 4],
        ],
        axis=-1,
    )
    metadata = np.array(
        [
            [
                *np.eye(6)[TASKS.index(r["task"])],
                r["operations"],
                r["reasoning_depth"],
                np.log1p(r["numeric_max"]),
            ]
            for r in rows
        ]
    )
    return {
        "scalar_CV": simple[..., :2],
        "entropy_control": simple[..., 2:3],
        "margin_control": simple[..., 3:4],
        "scalar_all": simple,
        "task_diagnostic": np.concatenate(
            [simple, np.broadcast_to(metadata[None], (4, *metadata.shape))], axis=-1
        ),
    }


def development(config):
    if (EXP / "lockbox_opened.json").exists():
        raise RuntimeError("4B consumed lockbox: selection closed")
    from latent_loop.backbone import environment

    environment(config["memory_gib"])
    parts = json.loads((EXP / "validation_partitions.json").read_text())["indices"]
    ck, cal, pol = [
        np.array(parts[name]) for name in ("checkpoint", "calibration", "policy")
    ]
    for seed in config["parent_seeds"]:
        tr, va = (
            trajectory("controller_train", seed),
            trajectory("controller_val", seed),
        )
        ty, vy = outcomes(tr["correct"]), outcomes(va["correct"])
        prior = np.bincount(ty.ravel(), minlength=4) / ty.size
        prevalence = {
            "train": prior.tolist(),
            "validation": (np.bincount(vy.ravel(), minlength=4) / vy.size).tolist(),
            "by_depth": [
                (np.bincount(ty[k], minlength=4) / len(ty[k])).tolist()
                for k in range(3)
            ],
            "by_task": {
                task: (
                    np.bincount(
                        ty[
                            :, [r["task"] == task for r in load("controller_train")]
                        ].ravel(),
                        minlength=4,
                    )
                    / ty[:, [r["task"] == task for r in load("controller_train")]].size
                ).tolist()
                for task in TASKS
            },
        }
        atomic_json(OUT / "prevalence" / f"development_seed{seed}.json", prevalence)
        full_signature = {
            "train_sha256": sha(
                f"artifacts/experiment_04a/trajectories/controller_train_seed{seed}.npz"
            ),
            "val_sha256": sha(
                f"artifacts/experiment_04a/trajectories/controller_val_seed{seed}.npz"
            ),
            "partition_sha256": sha(EXP / "validation_partitions.json"),
        }
        x, y = tr["x"][:3].reshape(-1, 1032), ty.ravel()
        vx, vyy = va["x"][:3, ck].reshape(-1, 1032), vy[:, ck].ravel()
        models, infos = {}, {}
        for name in ("d4_plain", "d4_weighted", "matched"):
            models[name], infos[name] = fit_model(
                name,
                seed,
                config,
                x,
                y,
                vx,
                vyy,
                weighted=name == "d4_weighted",
                extra_signature=full_signature,
            )
        for name, index in (("event_WR", 1), ("event_RW", 2)):
            models[name], infos[name] = fit_model(
                name,
                seed,
                config,
                x,
                (y == index).astype(np.float32),
                vx,
                (vyy == index).astype(np.float32),
                objective="binary",
                extra_signature=full_signature,
            )
        selected_d = min(
            ("d4_plain", "d4_weighted"), key=lambda n: infos[n]["best_validation_nll"]
        )
        raw_val = {name: predict(model, va["x"]) for name, model in models.items()}
        raw_train = {name: predict(model, tr["x"]) for name, model in models.items()}
        random_model = make_model("random", config, seed)
        folder = CACHE / "checkpoints" / f"random_seed{seed}"
        folder.mkdir(parents=True, exist_ok=True)
        random_model.save_weights(str(folder / "best.safetensors"))
        raw_val["random"] = predict(random_model, va["x"])
        raw_train["random"] = predict(random_model, tr["x"])
        calibrators = {
            name: calibrate_multiclass(raw_val[name][:3, cal], vy[:, cal])
            for name in ("d4_plain", "d4_weighted", "matched")
        }
        calibrators["events"] = {
            ev: fit_scalar(
                raw_val[f"event_{ev}"][:3, cal].reshape(-1, 1),
                (vy[:, cal] == index).ravel(),
            )
            for ev, index in (("WR", 1), ("RW", 2))
        }
        ccal = confidence_calibration(va["confidence"], va["correct"], cal)
        cv, ct = (
            calibrated_confidence(va["confidence"], ccal),
            calibrated_confidence(tr["confidence"], ccal),
        )
        vv, vt = regression(seed, va), regression(seed, tr)
        dv = calibrated_softmax(raw_val[selected_d], calibrators[selected_d])
        # Grouped OOF D predictions for ranker training: all depths of a problem stay together.
        assignments = np.empty(len(tr["labels"]), int)
        order = np.random.default_rng(450701 + seed).permutation(len(assignments))
        for fold, ix in enumerate(np.array_split(order, config["oof_folds"])):
            assignments[ix] = fold
        oof = np.zeros((4, len(assignments), 4), np.float32)
        for fold in range(config["oof_folds"]):
            keep = np.flatnonzero(assignments != fold)
            held = np.flatnonzero(assignments == fold)
            name = f"oof{fold}"
            model, info = fit_model(
                name,
                seed,
                config,
                tr["x"][:3, keep].reshape(-1, 1032),
                ty[:, keep].ravel(),
                vx,
                vyy,
                weighted=selected_d == "d4_weighted",
                fixed_epochs=infos[selected_d]["best_epoch"] + 1,
                extra_signature={
                    **full_signature,
                    "fold": fold,
                    "held_problem_hash": sha_bytes(held.tobytes()),
                },
            )
            oof[:, held] = predict(model, tr["x"][:, held])
        dt = calibrated_softmax(oof, calibrators[selected_d])
        # Cross-fit statistical transforms within stack-fitting validation subset.
        stack_c = np.empty((4, len(cal)))
        stack_d = np.empty((4, len(cal), 4))
        cal_assign = np.arange(len(cal)) % config["oof_folds"]
        cal_assign = np.random.default_rng(450801 + seed).permutation(cal_assign)
        for fold in range(config["oof_folds"]):
            keep = cal[cal_assign != fold]
            held = cal[cal_assign == fold]
            positions = np.flatnonzero(cal_assign == fold)
            setting = calibrate_multiclass(raw_val[selected_d][:3, keep], vy[:, keep])
            stack_d[:, positions] = calibrated_softmax(
                raw_val[selected_d][:, held], setting
            )
            setting_c = confidence_calibration(va["confidence"], va["correct"], keep)
            stack_c[:, positions] = calibrated_confidence(
                va["confidence"][:, held], setting_c
            )
        stacks = {
            name: fit_event_policy(
                signal_inputs(name, stack_c[:3], vv[:3, cal], stack_d[:3]), vy[:, cal]
            )
            for name in COMBINATIONS
        }
        controls = {
            name: fit_event_policy(values[:3], ty)
            for name, values in scalar_features(
                tr, ct, vt, load("controller_train")
            ).items()
        }
        ranker_models = {}
        for combination in RANKERS:
            name = f"ranker_{combination}"
            tx = ranker_inputs(tr["x"], ct, vt, dt, combination)
            valx = ranker_inputs(
                va["x"][:, ck], cv[:, ck], vv[:, ck], dv[:, ck], combination
            )
            ranker_models[name], infos[name] = fit_model(
                name,
                seed,
                config,
                tx,
                tr["correct"].astype(np.float32),
                valx,
                va["correct"][:, ck].astype(np.float32),
                objective="ranker",
                extra_signature={
                    **full_signature,
                    "D": selected_d,
                    "OOF_folds": config["oof_folds"],
                    "signal_mask": combination,
                },
            )
        scores = {"C": cv, "V": vv, "D": dv, "legacy_gain": va["improvement"]}
        scores.update(
            {
                f"stack_{name}": policy_events(setting, signal_inputs(name, cv, vv, dv))
                for name, setting in stacks.items()
            }
        )
        rules, sweeps = {}, {}
        for name, kind, grid in [
            ("confidence", "C", [0, 0.25, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]),
            ("value", "V", [-1, -0.5, -0.1, 0, 0.01, 0.05, 0.1, 0.5, 1]),
            ("legacy_gain", "legacy_gain", [0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1]),
            ("d_direct", "D", config["decision_thresholds"]),
            ("d_utility", "utility", config["cost_lambdas_accuracy_per_loop"]),
        ] + [
            (f"stack_{name}", f"stack_{name}", config["decision_thresholds"])
            for name in COMBINATIONS
        ]:
            rules[name], sweeps[name] = choose_policy(
                config, scores, va, pol, kind, grid
            )
        fixed = (
            min(
                range(4),
                key=lambda k: (
                    -va["correct"][k, pol].mean(),
                    va["loss"][k, pol].mean(),
                    k,
                ),
            )
            + 1
        )
        rules["fixed_selected"] = {"kind": "fixed", "depth": fixed}
        rank_quality = {
            name: predict(
                model, ranker_inputs(va["x"], cv, vv, dv, name.removeprefix("ranker_"))
            )[..., 0]
            for name, model in ranker_models.items()
        }
        from latent_loop.experiment04b.science import stop_depth, visited_selection

        choices = []
        for policy, rule in rules.items():
            paid = stop_depth(scores, rule)
            for selector, q in [("final", None), ("max_C", cv), *rank_quality.items()]:
                depth = paid if q is None else visited_selection(q, paid)
                acc = float(va["correct"][depth[pol] - 1, pol].mean())
                choices.append(
                    {
                        "policy": policy,
                        "selector": selector,
                        "accuracy": acc,
                        "mean_paid_loops": float(paid[pol].mean()),
                    }
                )
        best = min(
            choices,
            key=lambda r: (
                -r["accuracy"],
                r["mean_paid_loops"],
                r["policy"],
                r["selector"],
            ),
        )
        value = {
            "seed": seed,
            "selected_utc": utc(),
            "D_model": selected_d,
            "D_selection_metric": "Original-distribution multiclass NLL on checkpoint partition",
            "class_prior": prior.tolist(),
            "calibrators": calibrators,
            "confidence_calibration": ccal,
            "stacks": stacks,
            "scalar_controls": controls,
            "rules": rules,
            "validation_sweeps": sweeps,
            "validation_combo_results": choices,
            "validation_best_combination": best,
            "deployment_checkpoints": [
                "d4_plain",
                "d4_weighted",
                "matched",
                "event_WR",
                "event_RW",
                "random",
                *[f"ranker_{name}" for name in RANKERS],
            ],
            "OOF": {
                "folds": config["oof_folds"],
                "problem_fold_assignment": assignments.tolist(),
                "fixed_epochs": infos[selected_d]["best_epoch"] + 1,
                "training_example_never_in_its_D_fold_fit": True,
            },
            "regression_scale": json.loads(
                Path(
                    f"artifacts/experiment_03/checkpoints/regression_seed{seed}/state.json"
                ).read_text()
            )["scale"],
            "signature": full_signature,
            "training_completed": True,
        }
        atomic_json(EXP / f"selection_seed{seed}.json", value)
        print(
            f"development parent {seed}: selected {selected_d}; best {best}", flush=True
        )


def sha_bytes(value):
    import hashlib

    return hashlib.sha256(value).hexdigest()
