"""CPU statistical controls and problem-clustered evaluation."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logit, logsumexp, softmax

from latent_loop.experiment03.science import (
    binary_metrics,
    fit_scalar,
    scalar_predict,
)
from latent_loop.experiment04a.data import TASKS


def binary(p, y):
    m = binary_metrics(p, y)
    p = np.asarray(p, dtype=np.float64).ravel().clip(1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=np.float64).ravel()
    m["log_loss"] = float(np.mean(-y * np.log(p) - (1 - y) * np.log1p(-p)))
    return m


def scalar_inputs(t, scores, rows):
    s = t["scalar"]
    shape = s.shape[:2]
    task = np.broadcast_to(
        np.eye(6)[[TASKS.index(r["task"]) for r in rows]], (*shape, 6)
    )
    depth = np.broadcast_to(np.arange(1, 5)[:, None, None] / 4, (*shape, 1))
    metadata = np.broadcast_to(
        np.array(
            [
                [
                    r["operations"],
                    r["reasoning_depth"],
                    np.log1p(r["numeric_max"]),
                    np.log1p(r["prompt_characters"]),
                ]
                for r in rows
            ]
        )[None, :, :],
        (*shape, 4),
    )
    c = scores["confidence"][..., None]
    return {
        "confidence_only": c,
        "entropy": s[..., 1:2],
        "margin": s[..., 2:3],
        "current_depth": depth,
        "scalar_logistic": np.concatenate([c, s], axis=-1),
        "task_difficulty_diagnostic": np.concatenate([c, s, task, metadata], axis=-1),
        "diagnostic_plus_hidden_gain": np.concatenate(
            [c, s, task, metadata, scores["improvement"][..., None]], axis=-1
        ),
    }


def fit_controls(train, scores, rows):
    result = {}
    for name, x in scalar_inputs(train, scores, rows).items():
        result[name] = fit_scalar(
            x[:3].reshape(-1, x.shape[-1]), train["useful"].ravel()
        )
    result["train_prior"] = {"probability": float(train["useful"].mean())}
    return result


def confidence_calibrators(val, scores):
    logits, y = val["logits"], val["labels"]

    def objective(logtemp):
        scaled = logits / np.exp(logtemp)
        return float(
            (logsumexp(scaled, axis=-1) - scaled[:, np.arange(len(y)), y]).mean()
        )

    fit = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
    models = {"temperature": float(np.exp(fit.x))}
    for name, x in {
        "entropy_calibration": val["scalar"][..., 1:2],
        "margin_calibration": val["scalar"][..., 2:3],
        "confidence_recalibrated": logit(scores["confidence"].clip(1e-6, 1 - 1e-6))[
            ..., None
        ],
        "joint_recalibrated": logit(scores["joint_confidence"].clip(1e-6, 1 - 1e-6))[
            ..., None
        ],
    }.items():
        models[name] = fit_scalar(x.reshape(-1, 1), val["correct"].ravel())
    return models


def all_confidence(t, scores, calibrators):
    result = {
        "raw_probability": t["scalar"][..., 0],
        "temperature": softmax(t["logits"] / calibrators["temperature"], axis=-1).max(
            -1
        ),
        "existing_confidence": scores["confidence"],
        "joint_confidence": scores["joint_confidence"],
    }
    inputs = {
        "entropy_calibration": t["scalar"][..., 1:2],
        "margin_calibration": t["scalar"][..., 2:3],
        "confidence_recalibrated": logit(scores["confidence"].clip(1e-6, 1 - 1e-6))[
            ..., None
        ],
        "joint_recalibrated": logit(scores["joint_confidence"].clip(1e-6, 1 - 1e-6))[
            ..., None
        ],
    }
    for name, x in inputs.items():
        result[name] = scalar_predict(calibrators[name], x)
    return result


def confidence_states(scores, calibrators):
    return scalar_predict(
        calibrators["confidence_recalibrated"],
        logit(scores["confidence"].clip(1e-6, 1 - 1e-6))[..., None],
    )


def stopping(name, scores, setting):
    n = scores["confidence"].shape[1]
    if name.startswith("fixed_"):
        return np.full(n, int(name[-1]), int)
    if name == "confidence":
        go = scores["confidence"][:3] < setting["threshold"]
    elif name == "improvement":
        go = scores["improvement"][:3] > setting["threshold"]
    else:
        raise ValueError(name)
    depth = np.full(n, 4, int)
    alive = np.ones(n, bool)
    for k in range(3):
        stop = alive & ~go[k]
        depth[stop] = k + 1
        alive[stop] = False
    return depth


def bootstrap(values, repeats=1000, seed=440001):
    # [parent seed, depth (optional), independent problem] preserves within-problem clusters.
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values.mean(1)
    rng = np.random.default_rng(seed)
    res = [
        values[rng.integers(0, len(values), len(values))][
            :, rng.integers(0, values.shape[1], values.shape[1])
        ].mean()
        for _ in range(repeats)
    ]
    return {
        "estimate": float(values.mean()),
        "cluster_seed_bootstrap95": np.quantile(res, [0.025, 0.975]).tolist(),
        "unique_problems": values.shape[1],
        "parent_seeds": len(values),
        "repeats": repeats,
    }


def auc_interval(probabilities, labels, repeats=400, seed=440001):
    p, y = np.asarray(probabilities), np.asarray(labels)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        seeds = rng.integers(0, len(p), len(p))
        items = rng.integers(0, p.shape[-1], p.shape[-1])
        values = [binary(p[s][..., items], y[s][..., items])["auroc"] for s in seeds]
        estimates.append(np.mean([v for v in values if v is not None]))
    return {
        "estimate": float(np.mean([binary(a, b)["auroc"] for a, b in zip(p, y)])),
        "cluster_seed_bootstrap95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "unique_problems": p.shape[-1],
        "parent_seeds": len(p),
        "repeats": repeats,
    }


def transition_stats(t, delta=0.01):
    correct = t["correct"]
    logits = t["logits"]
    gain = t["gain"]
    pred = logits.argmax(-1)
    p = softmax(logits, axis=-1).max(-1)
    result = []
    for k in range(3):
        up = ~correct[k] & correct[k + 1]
        down = correct[k] & ~correct[k + 1]
        useful = gain[k] > delta
        changed = pred[k] != pred[k + 1]
        result.append(
            {
                "from_depth": k + 1,
                "n": len(up),
                "positive_loss_gain": float((gain[k] > 0).mean()),
                "meaningful_loss_gain": float(useful.mean()),
                "loss_increase": float((gain[k] < 0).mean()),
                "meaningful_loss_harm": float((gain[k] < -delta).mean()),
                "wrong_to_right": float(up.mean()),
                "right_to_wrong": float(down.mean()),
                "wrong_to_wrong": float((~correct[k] & ~correct[k + 1]).mean()),
                "right_to_right": float((correct[k] & correct[k + 1]).mean()),
                "unchanged_prediction": float((~changed).mean()),
                "confidence_increase_without_accuracy_improvement": float(
                    ((p[k + 1] > p[k]) & ~up).mean()
                ),
                "correction_given_useful_gain": float(up[useful].mean())
                if useful.any()
                else None,
                "useful_gain_unchanged_or_still_wrong": float((useful & ~up).mean()),
                "mean_signed_loss_gain": float(gain[k].mean()),
            }
        )
    return result


def strata(rows, lengths):
    fields = {
        "task": [r["task"] for r in rows],
        "generator": [r["generator_id"] for r in rows],
        "difficulty": [r["difficulty_stratum"] for r in rows],
        "template": [str(r["template_id"]) for r in rows],
        "reasoning_depth": [str(r["reasoning_depth"]) for r in rows],
        "operations": [str(r["operations"]) for r in rows],
        "numeric_range": [
            "0-31"
            if r["numeric_max"] <= 31
            else "32-255"
            if r["numeric_max"] <= 255
            else "256+"
            for r in rows
        ],
        "token_length": [
            "<=160" if n <= 160 else "161-320" if n <= 320 else "321+" for n in lengths
        ],
    }
    return {
        name: {
            value: np.array([v == value for v in values])
            for value in sorted(set(values))
        }
        for name, values in fields.items()
    }
