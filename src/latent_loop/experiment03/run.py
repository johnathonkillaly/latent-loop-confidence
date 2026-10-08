"""Run Experiment 3 using preserved checkpoints. Run from the repository root."""

import argparse
import gzip
import hashlib
import json
import resource
import signal
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.special import expit

from latent_loop.experiment03.data import (
    CACHE,
    EXP,
    OUT,
    fresh_data,
    historical_audit,
    sha,
    utc,
    verify_preservation,
)
from latent_loop.experiment03.science import (
    binary_metrics,
    fit_scalar,
    observables,
    opportunity,
    policy_depth,
    policy_metrics,
    regression_metrics,
    scalar_predict,
    targets,
)
from latent_loop.training import atomic_json, parameter_count

_INTERRUPTED = False


def guard(config):
    if _INTERRUPTED:
        raise InterruptedError(
            "Stopped at an extraction-batch or training-epoch boundary; resume with the same configuration"
        )
    if datetime.now().astimezone() > datetime.fromisoformat(
        config["started_utc"]
    ) + timedelta(hours=config["max_hours"]):
        raise TimeoutError("Experiment budget reached; checkpoints preserved")


def load_features(split):
    path = (
        Path("artifacts/features") / (split.removeprefix("historical_") + ".npz")
        if split.startswith("historical_")
        else CACHE / "features" / f"{split}.npz"
    )
    return dict(np.load(path))


def extract(config, rows):
    import mlx.core as mx

    from latent_loop.backbone import environment, get_model, hidden_forward, tokenize

    env = environment(config["memory_gb"])
    model, tokenizer, manifest = get_model()
    directory = CACHE / "features"
    directory.mkdir(parents=True, exist_ok=True)
    signature = {
        "model": manifest,
        "datasets": json.loads((EXP / "dataset_manifest.json").read_text()),
        "prefix": "Answer: (",
    }
    sigpath = directory / "signature.json"
    if sigpath.exists() and json.loads(sigpath.read_text()) != signature:
        raise ValueError("Feature signature changed")
    atomic_json(sigpath, signature)
    decoder = mx.array(np.load("artifacts/features/decoder.npz")["decoder"])
    recording = {"environment": env, "model": manifest, "batches": []}
    tic_all = time.perf_counter()
    for split in config["sizes"]:
        if (directory / f"{split}.npz").exists():
            continue
        shards = directory / split
        shards.mkdir(exist_ok=True)
        for start in range(0, len(rows[split]), config["extract_batch"]):
            guard(config)
            path = shards / f"{start:06d}.npz"
            if path.exists():
                continue
            batch = rows[split][start : start + config["extract_batch"]]
            encoded = [tokenize(tokenizer, r["prompt"]) for r in batch]
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
            h = hidden_forward(model, ids)[mx.arange(len(batch)), mx.array(lengths) - 1]
            scores = model.model.embed_tokens.as_linear(h)[
                :, mx.array([32, 33, 34, 35])
            ]
            mx.eval(h, scores)
            mx.synchronize()
            elapsed = time.perf_counter() - tic
            np.savez_compressed(
                path,
                h=np.array(h),
                base=np.array(scores),
                lengths=np.asarray(lengths),
                seconds=elapsed,
            )
            recording["batches"].append(
                {"split": split, "start": start, "seconds": elapsed, "n": len(batch)}
            )
            if start % 240 == 0:
                print(
                    f"extract {split}: {start + len(batch)}/{len(rows[split])}",
                    flush=True,
                )
        loaded = [dict(np.load(p)) for p in sorted(shards.glob("*.npz"))]
        np.savez_compressed(
            directory / f"{split}.npz",
            **{
                k: np.concatenate([s[k] for s in loaded])
                for k in ["h", "base", "lengths"]
            },
        )
    # Check live candidate projection and interception against native logits.
    ids = mx.array([tokenize(tokenizer, rows["controller_val"][0]["prompt"])])
    h = hidden_forward(model, ids)[:, -1]
    native = model(ids)[:, -1, mx.array([32, 33, 34, 35])]
    candidate = h @ decoder.T
    error = float(mx.max(mx.abs(native - candidate)).item())
    if error > 0.001:
        raise AssertionError(f"Live decoder mismatch {error}")
    recording.update(
        {
            "live_candidate_native_max_abs": error,
            "command_seconds": time.perf_counter() - tic_all,
            "peak_mlx_bytes": mx.get_peak_memory(),
            "peak_process_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        }
    )
    if not (OUT / "extraction.json").exists():
        atomic_json(OUT / "extraction.json", recording)


def refiner(seed, kind="recurrence"):
    from latent_loop.models import Refiner

    m = Refiner(1024, 64, False)
    m.load_weights(f"artifacts/checkpoints/{kind}_seed{seed}/best.safetensors")
    m.freeze()
    return m


def trajectories(config, rows):
    import mlx.core as mx

    from latent_loop.backbone import environment

    environment(config["memory_gb"])
    directory = CACHE / "trajectories"
    directory.mkdir(exist_ok=True)
    decoder = mx.array(np.load("artifacts/features/decoder.npz")["decoder"])
    opportunities, parity = {}, []
    for seed in config["seeds"]:
        m = refiner(seed)
        opportunities[str(seed)] = {}
        old = json.load(
            gzip.open(f"results/metrics/predictions_seed{seed}.json.gz", "rt")
        )
        for split in rows:
            guard(config)
            a = load_features(split)
            signature = {
                "parent": sha(
                    f"artifacts/checkpoints/recurrence_seed{seed}/best.safetensors"
                ),
                "ids": hashlib.sha256(
                    json.dumps([r["id"] for r in rows[split]]).encode()
                ).hexdigest(),
                "features": sha(
                    (
                        Path("artifacts/features")
                        / (split.removeprefix("historical_") + ".npz")
                    )
                    if split.startswith("historical_")
                    else CACHE / "features" / f"{split}.npz"
                ),
            }
            path = directory / f"{split}_seed{seed}.npz"
            sigpath = path.with_suffix(".json")
            if path.exists():
                if json.loads(sigpath.read_text()) != signature:
                    raise ValueError("Trajectory signature changed")
                logits = np.load(path)["logits"]
            else:
                logits, states = m(
                    mx.array(a["h"]), mx.array(a["base"]), decoder, 4, True
                )
                logits, states = np.array(logits), np.array(states)
                x, scalar = observables(states, logits, a["h"])
                y = np.asarray([r["label"] for r in rows[split]])
                loss, correct, gain, useful = targets(logits, y, config["delta"])
                np.savez_compressed(
                    path,
                    logits=logits,
                    states=states,
                    x=x,
                    scalar=scalar,
                    labels=y,
                    loss=loss,
                    correct=correct,
                    gain=gain,
                    useful=useful,
                )
                atomic_json(sigpath, signature)
            if split.startswith("historical_"):
                archived = np.stack(
                    [
                        old[split.removeprefix("historical_")][f"recurrence_d{d}"][
                            "logits"
                        ]
                        for d in range(1, 5)
                    ]
                )
                error = float(np.max(np.abs(logits - archived)))
                assert error < 1e-5, error
                parity.append({"seed": seed, "split": split, "max_abs": error})
            # Final fresh targets are cached, but excluded from the prefit analysis and selection.
            if split in [
                "controller_train",
                "controller_val",
                "historical_test",
                "historical_ood",
            ]:
                opportunities[str(seed)][split] = opportunity(
                    logits, [r["label"] for r in rows[split]], config["delta"]
                )
        print(f"frozen trajectories complete: seed {seed}", flush=True)
    atomic_json(EXP / "trajectory_parity.json", parity)
    path = OUT / "opportunity_before_training.json"
    if not path.exists():
        atomic_json(
            path,
            {
                "completed_utc": utc(),
                "delta": config["delta"],
                "new_heads_fitted": False,
                "analysis_splits": list(next(iter(opportunities.values()))),
                "results": opportunities,
            },
        )
    verify_preservation()


def trajectory(split, seed):
    return dict(np.load(CACHE / "trajectories" / f"{split}_seed{seed}.npz"))


def training_arrays(t, scale):
    # Equal representation of all four depths; gain loss has a terminal mask.
    n = t["x"].shape[1]
    gain = np.concatenate([t["gain"], np.zeros((1, n))])
    useful = np.concatenate([t["useful"], np.zeros((1, n))])
    mask = np.broadcast_to(np.array([1, 1, 1, 0])[:, None], (4, n))
    labels = (
        np.stack([t["correct"], useful, gain / scale, mask], axis=-1)
        .reshape(-1, 4)
        .astype(np.float32)
    )
    return t["x"].reshape(-1, t["x"].shape[-1]), labels


def make_head(name, config, seed, decoder):
    import mlx.core as mx

    from latent_loop.experiment03.heads import CandidateProbe, Probe

    mx.random.seed(
        seed
        + 3000
        + [
            "confidence",
            "improvement",
            "regression",
            "joint",
            "candidate_confidence",
        ].index(name)
    )
    return (
        CandidateProbe(1024, decoder, config["width"])
        if name == "candidate_confidence"
        else Probe(1032, config["width"], 3 if name == "joint" else 1)
    )


def fit_head(name, seed, config, decoder):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    parent = sha(f"artifacts/checkpoints/recurrence_seed{seed}/best.safetensors")
    tr, va = trajectory("controller_train", seed), trajectory("controller_val", seed)
    for cached in [tr, va]:
        if not np.array_equal(
            cached["useful"], cached["gain"] > config.get("delta", 0.01)
        ):
            raise ValueError(
                "Cached target threshold changed; prepare in a fresh experiment cache"
            )
    scale = max(float(tr["gain"].std()), 0.01)
    x, y = training_arrays(tr, scale)
    vx, vy = [mx.array(v) for v in training_arrays(va, scale)]
    head = make_head(name, config, seed, decoder)
    directory = CACHE / "checkpoints" / f"{name}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    signature = {
        "config": config,
        "parent": parent,
        "datasets": json.loads((EXP / "dataset_manifest.json").read_text()),
    }
    info = {
        "name": name,
        "seed": seed,
        "signature": signature,
        "parameters": parameter_count(head),
        "next_epoch": 0,
        "best_val_loss": 1e30,
        "training_seconds": 0.0,
        "scale": scale,
        "history": [],
        "started_utc": utc(),
    }
    opt = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=0.01)
    statepath = directory / "state.json"
    if statepath.exists():
        info = json.loads(statepath.read_text())
        assert info["signature"] == signature, "Checkpoint configuration changed"
        head.load_weights(str(directory / "last.safetensors"))
        opt.state = tree_unflatten(
            list(mx.load(str(directory / "optimizer.safetensors")).items())
        )
    if info.get("complete"):
        head.load_weights(str(directory / "best.safetensors"))
        return head, info

    def objective(model, xb, yb):
        output = model(xb)
        mask = yb[:, 3]
        den = mx.maximum(mx.sum(mask), 1)
        if name in ["confidence", "candidate_confidence"]:
            return nn.losses.binary_cross_entropy(
                output[:, 0], yb[:, 0], reduction="mean"
            )
        if name == "improvement":
            return (
                mx.sum(
                    nn.losses.binary_cross_entropy(
                        output[:, 0], yb[:, 1], reduction="none"
                    )
                    * mask
                )
                / den
            )
        if name == "regression":
            return mx.sum((output[:, 0] - yb[:, 2]) ** 2 * mask) / den
        return (
            nn.losses.binary_cross_entropy(output[:, 0], yb[:, 0], reduction="mean")
            + mx.sum(
                nn.losses.binary_cross_entropy(output[:, 1], yb[:, 1], reduction="none")
                * mask
            )
            / den
            + mx.sum((output[:, 2] - yb[:, 2]) ** 2 * mask) / den
        )

    grad = nn.value_and_grad(head, objective)
    for epoch in range(info["next_epoch"], config["epochs"]):
        guard(config)
        start = time.perf_counter()
        order = np.random.default_rng(seed * 10000 + epoch).permutation(len(x))
        losses = []
        for begin in range(0, len(x), config["batch_size"]):
            idx = order[begin : begin + config["batch_size"]]
            value, g = grad(head, mx.array(x[idx]), mx.array(y[idx]))
            g, norm = optim.clip_grad_norm(g, 1.0)
            opt.update(head, g)
            mx.eval(head.parameters(), opt.state, value, norm)
            losses.append(float(value.item()))
        vl = float(objective(head, vx, vy).item())
        assert np.isfinite(vl + np.mean(losses))
        info["history"].append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": vl}
        )
        if vl < info["best_val_loss"]:
            info["best_val_loss"], info["best_epoch"] = vl, epoch
            head.save_weights(str(directory / "best.safetensors"))
        head.save_weights(str(directory / "last.safetensors"))
        mx.save_safetensors(
            str(directory / "optimizer.safetensors"), dict(tree_flatten(opt.state))
        )
        info.update(
            {
                "next_epoch": epoch + 1,
                "training_seconds": info["training_seconds"]
                + time.perf_counter()
                - start,
                "peak_mlx_bytes": mx.get_peak_memory(),
            }
        )
        atomic_json(statepath, info)
    info.update({"complete": True, "completed_utc": utc()})
    atomic_json(statepath, info)
    atomic_json(OUT / "training" / f"{name}_seed{seed}.json", info)
    head.load_weights(str(directory / "best.safetensors"))
    print(
        f"fit {name} seed {seed}: best epoch {info['best_epoch']}, val {info['best_val_loss']:.4f}",
        flush=True,
    )
    return head, info


def infer_heads(heads, infos, t):
    import mlx.core as mx

    x = mx.array(t["x"].reshape(-1, 1032))
    shape = t["x"].shape[:2]
    output = {name: np.array(m(x)).reshape(*shape, -1) for name, m in heads.items()}
    return {
        "confidence": expit(output["confidence"][..., 0]),
        "candidate_confidence": expit(output["candidate_confidence"][..., 0]),
        "improvement": expit(output["improvement"][..., 0]),
        "regression": output["regression"][..., 0] * infos["regression"]["scale"],
        "joint_confidence": expit(output["joint"][..., 0]),
        "joint_gain": expit(output["joint"][..., 1]),
        "joint_regression": output["joint"][..., 2] * infos["joint"]["scale"],
    }


def scalar_models(train, scores, rows):
    scalar = train["scalar"][:3].reshape(-1, 5)
    confidence = scores["confidence"][:3].reshape(-1, 1)
    task = np.tile(
        np.eye(6)[
            [
                [
                    "comparison",
                    "arithmetic",
                    "state",
                    "pointer",
                    "routing",
                    "classification",
                ].index(r["task"])
                for r in rows
            ]
        ],
        (3, 1),
    )
    difficulty = np.tile([r["difficulty"] for r in rows], 3)[:, None]
    controls = np.concatenate([confidence, scalar, task, difficulty], axis=-1)
    useful, gain = train["useful"].ravel(), train["gain"].ravel()
    models = {}
    for name, x in {
        "entropy": scalar[:, 1:2],
        "margin": scalar[:, 2:3],
        "change": scalar[:, 3:4],
        "confidence_gain": confidence,
        "scalar_all": np.column_stack([confidence, scalar]),
        "task_depth_control": controls,
        "control_plus_gain": np.column_stack(
            [controls, scores["improvement"][:3].ravel()]
        ),
    }.items():
        models[name] = fit_scalar(x, useful)
        models[name + "_reg"] = fit_scalar(x, gain, "regression")
    models["correctness_scalar"] = fit_scalar(
        train["scalar"].reshape(-1, 5), train["correct"].ravel()
    )
    return models


def add_scalar_scores(models, t, scores, rows):
    scalar, c = t["scalar"], scores["confidence"][..., None]
    taskidx = [
        [
            "comparison",
            "arithmetic",
            "state",
            "pointer",
            "routing",
            "classification",
        ].index(r["task"])
        for r in rows
    ]
    task = np.broadcast_to(np.eye(6)[taskidx], (*scalar.shape[:2], 6))
    difficulty = np.broadcast_to(
        np.array([r["difficulty"] for r in rows])[None, :, None], (*scalar.shape[:2], 1)
    )
    controls = np.concatenate([c, scalar, task, difficulty], axis=-1)
    features = {
        "entropy": scalar[..., 1:2],
        "margin": scalar[..., 2:3],
        "change": scalar[..., 3:4],
        "confidence_gain": c,
        "scalar_all": np.concatenate([c, scalar], axis=-1),
        "task_depth_control": controls,
        "control_plus_gain": np.concatenate(
            [controls, scores["improvement"][..., None]], axis=-1
        ),
    }
    for name, x in features.items():
        scores[name] = scalar_predict(models[name], x)
        scores[name + "_reg"] = scalar_predict(models[name + "_reg"], x)
    scores["correctness_scalar"] = scalar_predict(models["correctness_scalar"], scalar)
    return scores


def select_policies(config, t, scores):
    candidates = {f"fixed_{d}": [{}] for d in range(1, 5)}
    candidates["confidence"] = [
        {"threshold": v} for v in config["confidence_thresholds"]
    ]
    for name in [
        "improvement",
        "old_continuation",
        "entropy",
        "margin",
        "change",
        "confidence_gain",
    ]:
        candidates[name] = [
            {"threshold": v} for v in config["classification_thresholds"]
        ]
    candidates["voc"] = [{"cost": v} for v in config["costs_nats_per_loop"]]
    candidates["joint"] = [
        {"confidence_threshold": c, "gain_threshold": g}
        for c in config["confidence_thresholds"]
        for g in config["classification_thresholds"]
    ]
    selections, sweep = {}, {}
    for name, settings in candidates.items():
        items = []
        for setting in settings:
            depth = policy_depth(name, scores, setting)
            m = policy_metrics(t["logits"], t["labels"], depth)
            items.append(
                {
                    "setting": setting,
                    "accuracy": m["accuracy"],
                    "nll": m["nll"],
                    "mean_loops": m["mean_loops"],
                }
            )
        if name == "voc":
            selected = min(
                items,
                key=lambda r: (r["nll"] + 0.01 * r["mean_loops"], r["mean_loops"]),
            )
        else:
            selected = min(
                items, key=lambda r: (-r["accuracy"], r["mean_loops"], r["nll"])
            )
        selections[name], sweep[name] = selected["setting"], items
    # Match confidence-only compute to the selected improvement policy on validation.
    target = policy_metrics(
        t["logits"],
        t["labels"],
        policy_depth("improvement", scores, selections["improvement"]),
    )["mean_loops"]
    matched = []
    for threshold in np.linspace(0, 1, 101):
        setting = {"threshold": float(threshold)}
        depth = policy_depth("confidence_matched", scores, setting)
        m = policy_metrics(t["logits"], t["labels"], depth)
        matched.append(
            {
                "setting": setting,
                "accuracy": m["accuracy"],
                "mean_loops": m["mean_loops"],
                "nll": m["nll"],
            }
        )
    chosen = min(
        matched, key=lambda r: (abs(r["mean_loops"] - target), -r["accuracy"], r["nll"])
    )
    selections["confidence_matched"], sweep["confidence_matched"] = (
        chosen["setting"],
        matched,
    )
    selections.update({"oracle_next": {"delta": config["delta"]}, "oracle_best": {}})
    return selections, sweep


def predictor_metrics(t, scores, train):
    report = {
        "confidence": {},
        "classification": {},
        "regression": {},
        "per_depth": {},
        "observed_correlations": {},
    }
    for name in [
        "confidence",
        "candidate_confidence",
        "joint_confidence",
        "correctness_scalar",
    ]:
        report["confidence"][name] = binary_metrics(scores[name], t["correct"])
    report["confidence"]["raw_probability"] = binary_metrics(
        t["scalar"][..., 0], t["correct"]
    )
    prior_c = float(train["correct"].mean())
    report["confidence"]["train_prior"] = binary_metrics(
        np.full(t["correct"].shape, prior_c), t["correct"]
    )
    for name in [
        "improvement",
        "joint_gain",
        "old_continuation",
        "entropy",
        "margin",
        "change",
        "confidence_gain",
        "scalar_all",
        "task_depth_control",
        "control_plus_gain",
    ]:
        report["classification"][name] = binary_metrics(scores[name][:3], t["useful"])
    prior = float(train["useful"].mean())
    report["classification"]["train_prior"] = binary_metrics(
        np.full(t["useful"].shape, prior), t["useful"]
    )
    for name in [
        "regression",
        "joint_regression",
        "confidence_gain_reg",
        "scalar_all_reg",
        "task_depth_control_reg",
        "control_plus_gain_reg",
    ]:
        report["regression"][name] = regression_metrics(scores[name][:3], t["gain"])
    report["regression"]["train_mean"] = regression_metrics(
        np.full(t["gain"].shape, float(train["gain"].mean())), t["gain"]
    )
    for k in range(3):
        report["per_depth"][str(k + 1)] = {
            name: binary_metrics(scores[name][k], t["useful"][k])
            for name in [
                "improvement",
                "joint_gain",
                "confidence_gain",
                "task_depth_control",
                "control_plus_gain",
            ]
        }
    for name, value in {
        "confidence": scores["confidence"][:3],
        "raw_probability": t["scalar"][:3, :, 0],
        "entropy": t["scalar"][:3, :, 1],
        "margin": t["scalar"][:3, :, 2],
        "previous_change": t["scalar"][:3, :, 3],
    }.items():
        report["observed_correlations"][name] = regression_metrics(value, t["gain"])[
            "spearman"
        ]
    return report


def behavior(t, scores, rows, threshold):
    c, actual = scores["confidence"][:3], t["correct"][:3]
    confident = c >= threshold
    categories, examples = {}, []
    n = len(rows)
    best = t["loss"].argmin(0) + 1
    for name, mask in {
        "correct_confident": actual & confident,
        "wrong_uncertain": ~actual & ~confident,
        "wrong_confident": ~actual & confident,
        "correct_uncertain": actual & ~confident,
    }.items():
        selected = np.argwhere(mask)
        categories[name] = {"n": int(mask.sum()), "threshold": threshold}
        if not mask.any():
            continue
        harm = actual & ~t["correct"][1:]
        correctable = ~actual & t["correct"][1:]
        categories[name].update(
            {
                "meaningful_improvement_probability": float(t["useful"][mask].mean()),
                "harmful_loss_probability": float((t["gain"][mask] < -0.01).mean()),
                "correction_probability": float(correctable[mask].mean()),
                "degradation_probability": float(harm[mask].mean()),
                "mean_predicted_gain": float(scores["regression"][:3][mask].mean()),
                "mean_actual_gain": float(t["gain"][mask].mean()),
                "oracle_best_depth_counts": np.bincount(
                    np.broadcast_to(best, (3, n))[mask], minlength=5
                )[1:].tolist(),
                "oracle_remaining_best_depth_counts": np.bincount(
                    np.stack([t["loss"][k:].argmin(0) + k + 1 for k in range(3)])[mask],
                    minlength=5,
                )[1:].tolist(),
            }
        )
        # Deterministic representative cases: strongest correction, harm, uncorrectable uncertainty, plus first.
        order = sorted(
            selected.tolist(), key=lambda pair: -abs(float(t["gain"][tuple(pair)]))
        )
        event_cases = []
        for event in [correctable, harm, ~actual & ~t["correct"].any(0)[None, :]]:
            event_cases.extend([v for v in order if event[tuple(v)]][:1])
        for k, i in event_cases + order[:2]:
            examples.append(
                {
                    "category": name,
                    "depth": k + 1,
                    "row": rows[i],
                    "confidence": float(c[k, i]),
                    "predicted_gain": float(scores["regression"][k, i]),
                    "actual_gain": float(t["gain"][k, i]),
                    "losses": t["loss"][:, i].tolist(),
                    "predictions": t["logits"][:, i].argmax(-1).tolist(),
                    "probabilities": np.exp(
                        t["logits"][:, i]
                        - np.logaddexp.reduce(t["logits"][:, i], axis=-1)[:, None]
                    ).tolist(),
                    "oracle_best_depth": int(best[i]),
                }
            )
    return categories, examples


def train_and_evaluate(config, rows):
    import mlx.core as mx

    from latent_loop.backbone import environment
    from latent_loop.models import Continuation

    environment(config["memory_gb"])
    assert (OUT / "opportunity_before_training.json").exists(), (
        "Opportunity analysis must precede training"
    )
    decoder = mx.array(np.load("artifacts/features/decoder.npz")["decoder"])
    for seed in config["seeds"]:
        guard(config)
        heads, infos = {}, {}
        for name in [
            "confidence",
            "improvement",
            "regression",
            "joint",
            "candidate_confidence",
        ]:
            heads[name], infos[name] = fit_head(name, seed, config, decoder)
        train = trajectory("controller_train", seed)
        tr_scores = infer_heads(heads, infos, train)
        controls = scalar_models(train, tr_scores, rows["controller_train"])
        old = Continuation(1024)
        old.load_weights(
            f"artifacts/checkpoints/recurrence_continuation_seed{seed}/best.safetensors"
        )
        # Freeze all policy choices on the new controller validation set before final evaluation.
        val = trajectory("controller_val", seed)
        scores = add_scalar_scores(
            controls, val, infer_heads(heads, infos, val), rows["controller_val"]
        )
        scores["old_continuation"] = np.stack(
            [
                np.array(
                    mx.sigmoid(
                        old(
                            mx.array(val["states"][k]),
                            mx.array(val["logits"][k]),
                            k + 1,
                        )
                    )
                )
                for k in range(4)
            ]
        )
        selections, sweep = select_policies(config, val, scores)
        atomic_json(
            EXP / f"selection_seed{seed}.json",
            {
                "selected_utc": utc(),
                "selection_split": "controller_val",
                "settings": selections,
                "validation_sweep": sweep,
                "controls": controls,
            },
        )
        result, public, cases = {}, {}, []
        ff = refiner(seed, "feedforward")
        for split in [
            "controller_val",
            "historical_test",
            "historical_ood",
            "fresh_id",
            "template_shift",
            "depth_shift",
        ]:
            guard(config)
            t = trajectory(split, seed)
            scores = add_scalar_scores(
                controls, t, infer_heads(heads, infos, t), rows[split]
            )
            scores["old_continuation"] = np.stack(
                [
                    np.array(
                        mx.sigmoid(
                            old(
                                mx.array(t["states"][k]),
                                mx.array(t["logits"][k]),
                                k + 1,
                            )
                        )
                    )
                    for k in range(4)
                ]
            )
            predictors = predictor_metrics(t, scores, train)
            policies, depths = {}, {}
            for name, setting in selections.items():
                depth = policy_depth(name, scores, setting, t["logits"], t["labels"])
                depths[name] = depth
                policies[name] = policy_metrics(
                    t["logits"], t["labels"], depth, scores["confidence"]
                )
                chosen_loss = t["loss"][depth - 1, np.arange(len(depth))]
                policies[name]["exact_nll"] = float(chosen_loss.mean())
                # Label-informed task/difficulty controls are diagnostics only, not policies.
                policies[name]["per_task"] = {}
                for task in sorted({r["task"] for r in rows[split]}):
                    ix = np.array([r["task"] == task for r in rows[split]])
                    policies[name]["per_task"][task] = policy_metrics(
                        t["logits"][:, ix],
                        t["labels"][ix],
                        depth[ix],
                        scores["confidence"][:, ix],
                    )
            a = load_features(split)
            ff_logits = np.array(ff(mx.array(a["h"]), mx.array(a["base"]), decoder, 1))
            for name, logits in [("frozen", a["base"]), ("one_pass", ff_logits)]:
                policies[name] = policy_metrics(
                    np.broadcast_to(logits, (4, *logits.shape)),
                    t["labels"],
                    np.ones(len(t["labels"]), int),
                )
                policies[name]["mean_loops"] = 0.0 if name == "frozen" else 1.0
                policies[name]["per_task"] = {
                    task: policy_metrics(
                        np.broadcast_to(
                            logits[
                                ix := np.array([r["task"] == task for r in rows[split]])
                            ],
                            (4, int(ix.sum()), 4),
                        ),
                        t["labels"][ix],
                        np.ones(int(ix.sum()), int),
                    )
                    for task in sorted({r["task"] for r in rows[split]})
                }
            categories, examples = behavior(
                t, scores, rows[split], selections["confidence"]["threshold"]
            )
            cases.extend([{**case, "split": split, "seed": seed} for case in examples])
            task_predictors = {}
            for task in sorted({r["task"] for r in rows[split]}):
                ix = np.array([r["task"] == task for r in rows[split]])
                task_predictors[task] = {
                    "improvement": binary_metrics(
                        scores["improvement"][:3, ix], t["useful"][:, ix]
                    ),
                    "confidence_gain": binary_metrics(
                        scores["confidence_gain"][:3, ix], t["useful"][:, ix]
                    ),
                    "regression": regression_metrics(
                        scores["regression"][:3, ix], t["gain"][:, ix]
                    ),
                }
            matched = {}
            for name in ["confidence", "improvement", "joint", "voc"]:
                avg = policies[name]["mean_loops"]
                lo, hi = int(np.floor(avg)), min(4, int(np.ceil(avg)))
                w = avg - lo
                mixture = {
                    key: (1 - w) * policies[f"fixed_{lo}"][key]
                    + w * policies[f"fixed_{hi}"][key]
                    for key in ["accuracy", "nll", "wrong_to_right", "right_to_wrong"]
                }
                mixture.update(
                    {
                        "mean_loops": avg,
                        "floor": lo,
                        "ceiling": hi,
                        "ceiling_probability": w,
                        "scope": "Expected outcome of label-independent random floor/ceiling mixture; not an executed controller or measured latency.",
                    }
                )
                matched[name] = mixture
            result[split] = {
                "predictors": predictors,
                "task_predictors": task_predictors,
                "policies": policies,
                "matched_fixed_mixtures": matched,
                "behavior": categories,
                "opportunity": opportunity(t["logits"], t["labels"], config["delta"]),
            }
            public[split] = {
                "ids": [r["id"] for r in rows[split]],
                "labels": t["labels"].tolist(),
                "logits": t["logits"].tolist(),
                "base_logits": a["base"].tolist(),
                "one_pass_logits": ff_logits.tolist(),
                "loss": t["loss"].tolist(),
                "gain": t["gain"].tolist(),
                "scalar_observables": t["scalar"].tolist(),
                "scores": {n: s.tolist() for n, s in scores.items()},
                "depths": {n: d.tolist() for n, d in depths.items()},
            }
        atomic_json(
            OUT / f"seed{seed}.json",
            {"seed": seed, "settings": selections, "splits": result},
        )
        atomic_json(OUT / f"cases_seed{seed}.json", cases)
        with gzip.GzipFile(
            filename=str(OUT / f"predictions_seed{seed}.json.gz"), mode="wb", mtime=0
        ) as f:
            f.write(json.dumps(public).encode())
        print(f"final evaluation complete: seed {seed}", flush=True)
    atomic_json(EXP / "preservation.json", verify_preservation())


def main():
    global _INTERRUPTED
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["audit", "prepare", "run", "timing", "report", "all"]
    )
    parser.add_argument("--config", default="experiments/experiment_03/config.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    _INTERRUPTED = False

    def request_stop(signum, frame):
        global _INTERRUPTED
        _INTERRUPTED = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if config["max_depth"] != 4:
        raise ValueError(
            "This protocol uses exactly four trajectory depths; maximum deployed depth is four"
        )
    for p in [EXP, OUT, CACHE]:
        p.mkdir(parents=True, exist_ok=True)
    if args.stage in ["audit", "all"]:
        print(json.dumps(historical_audit()["means"], indent=2))
    if args.stage == "audit":
        return
    if args.stage in ["prepare", "run", "all"]:
        rows = fresh_data(config)
        if args.stage in ["prepare", "all"]:
            extract(config, rows)
            trajectories(config, rows)
        if args.stage in ["run", "all"]:
            train_and_evaluate(config, rows)
    if args.stage in ["timing", "all"]:
        from latent_loop.experiment03.timing import measure

        measure(config)
    if args.stage in ["report", "all"]:
        from latent_loop.experiment03.report import report

        report(config)


if __name__ == "__main__":
    main()
