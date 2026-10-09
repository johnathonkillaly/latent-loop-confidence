"""CPU targets, calibration, causal policy simulation, selection and metrics."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, logsumexp, softmax

from latent_loop.experiment03.science import fit_scalar, scalar_predict
from latent_loop.experiment04a.science import binary

CLASS_NAMES = ("WW", "WR", "RW", "RR")
COMBINATIONS = ("C", "V", "D", "CV", "CD", "VD", "CVD")
RANKERS = ("C", "CV", "CD", "CVD")


def outcomes(correct):
    correct = np.asarray(correct, bool)
    return (2 * correct[:-1].astype(np.int32) + correct[1:].astype(np.int32)).astype(
        np.int32
    )


def calibrate_multiclass(logits, labels):
    x, y = np.asarray(logits, float).reshape(-1, 4), np.asarray(labels, int).ravel()

    def objective(v):
        z = x / np.exp(v[0]) + v[1:] - np.mean(v[1:])
        return float(
            (logsumexp(z, -1) - z[np.arange(len(y)), y]).mean()
            + 0.001 * np.mean(v[1:] ** 2)
        )

    fit = minimize(
        objective, np.zeros(5), method="L-BFGS-B", bounds=[(-2, 2)] + [(-5, 5)] * 4
    )
    assert fit.success and np.isfinite(fit.fun), fit.message
    return {
        "temperature": float(np.exp(fit.x[0])),
        "bias": (fit.x[1:] - fit.x[1:].mean()).tolist(),
        "calibration_nll": float(fit.fun),
    }


def calibrated_softmax(logits, setting):
    return softmax(
        np.asarray(logits) / setting["temperature"] + np.asarray(setting["bias"]),
        axis=-1,
    )


def confidence_calibration(raw_c, correct, ix):
    return {
        str(k + 1): fit_scalar(
            logit(np.asarray(raw_c[k, ix]).clip(1e-6, 1 - 1e-6))[:, None],
            correct[k, ix],
        )
        for k in range(4)
    }


def calibrated_confidence(raw_c, setting):
    return np.stack(
        [
            scalar_predict(
                setting[str(k + 1)],
                logit(np.asarray(raw_c[k]).clip(1e-6, 1 - 1e-6))[:, None],
            )
            for k in range(4)
        ]
    )


def pair_completion(wr, rw, prior):
    wr, rw = np.asarray(wr), np.asarray(rw)
    den = np.maximum(wr + rw, 1)
    wr, rw = wr / den, rw / den
    remaining = np.maximum(1 - wr - rw, 0)
    share_rr = prior[3] / max(prior[0] + prior[3], 1e-12)
    return np.stack([remaining * (1 - share_rr), wr, rw, remaining * share_rr], axis=-1)


def signal_inputs(name, c, v, d):
    values = []
    if "C" in name:
        values.append(c[..., None])
    if "V" in name:
        values.append(v[..., None])
    if "D" in name:
        values.append(d[..., 1:3])
    return np.concatenate(values, axis=-1)


def fit_event_policy(x, labels):
    return {
        event: fit_scalar(x.reshape(-1, x.shape[-1]), (labels == index).ravel())
        for event, index in (("WR", 1), ("RW", 2))
    }


def policy_events(setting, x):
    return np.stack([scalar_predict(setting[ev], x) for ev in ("WR", "RW")], axis=-1)


def ranker_inputs(x, c, v, d, combination):
    """Shared current-state scoring with constant masks; terminal continuation is zero."""
    c, v, d = np.array(c), np.array(v), np.array(d)
    v[-1] = 0
    d[-1, :, 1:3] = 0
    extra = np.stack(
        [
            c if "C" in combination else np.zeros_like(c),
            v if "V" in combination else np.zeros_like(v),
            d[..., 1] if "D" in combination else np.zeros_like(c),
            d[..., 2] if "D" in combination else np.zeros_like(c),
        ],
        axis=-1,
    )
    return np.concatenate([x, extra], axis=-1).astype(np.float32)


def stop_depth(scores, rule):
    """Only the score of each reached current depth influences the next decision."""
    n = scores["C"].shape[1]
    if rule["kind"] == "fixed":
        return np.full(n, rule["depth"], int)
    kind = rule["kind"]
    if kind == "C":
        go = scores["C"][:3] < rule["threshold"]
    elif kind == "V":
        go = scores["V"][:3] > rule["threshold"]
    elif kind == "legacy_gain":
        go = scores["legacy_gain"][:3] > rule["threshold"]
    elif kind == "D":
        go = (scores["D"][:3, :, 1] - scores["D"][:3, :, 2]) > rule["threshold"]
    elif kind == "utility":
        go = (scores["D"][:3, :, 1] - scores["D"][:3, :, 2] - rule["lambda"]) > 0
    elif kind.startswith("stack_"):
        go = np.diff(scores[kind][:3, :, ::-1], axis=-1)[..., 0] > rule["threshold"]
    else:
        raise ValueError(kind)
    depths = np.full(n, 4, int)
    alive = np.ones(n, bool)
    for k in range(3):
        stop = alive & ~go[k]
        depths[stop] = k + 1
        alive[stop] = False
    return depths


def visited_selection(quality, paid_depth):
    """Selection cannot see unvisited scores; ties choose the earliest visited depth."""
    q, d = np.asarray(quality), np.asarray(paid_depth, int)
    assert q.shape == (4, len(d)) and np.all((d >= 1) & (d <= 4))
    visible = np.arange(4)[:, None] < d
    return np.where(visible, q, -np.inf).argmax(0) + 1


def choose_policy(config, scores, t, ix, kind, grid):
    sweep = []
    for value in grid:
        rule = {"kind": kind, "lambda" if kind == "utility" else "threshold": value}
        paid = stop_depth(scores, rule)
        correct = t["correct"][paid - 1, np.arange(len(paid))]
        loss = t["loss"][paid - 1, np.arange(len(paid))]
        accuracy, loops = float(correct[ix].mean()), float(paid[ix].mean())
        utility = (
            accuracy - config["utility_selection_penalty_accuracy_per_loop"] * loops
        )
        sweep.append(
            {
                "rule": rule,
                "accuracy": accuracy,
                "mean_paid_loops": loops,
                "nll": float(loss[ix].mean()),
                "utility": utility,
            }
        )
    best = min(
        sweep,
        key=lambda r: (
            -(r["utility"] if kind == "utility" else r["accuracy"]),
            r["mean_paid_loops"],
            r["nll"],
        ),
    )
    return best["rule"], sweep


def transition_metrics(probabilities, labels):
    p = np.asarray(probabilities, float).reshape(-1, 4).clip(1e-12, 1)
    y = np.asarray(labels, int).ravel()
    truth = np.eye(4)[y]
    predicted = p.argmax(-1)
    classes = {name: binary(p[:, k], y == k) for k, name in enumerate(CLASS_NAMES)}
    for k, name in enumerate(CLASS_NAMES):
        selected = np.argsort(-p[:, k], kind="stable")[: max(1, len(y) // 10)]
        classes[name]["precision_at_top_10pct"] = float((y[selected] == k).mean())
        classes[name]["recall_at_top_10pct"] = float(
            (y[selected] == k).sum() / max(1, (y == k).sum())
        )
    value = p[:, 1] - p[:, 2]
    actual = (y == 1).astype(float) - (y == 2)
    groups = np.array_split(np.argsort(value, kind="stable"), 10)
    return {
        "n": len(y),
        "class_prevalence": truth.mean(0).tolist(),
        "four_class_accuracy": float((predicted == y).mean()),
        "four_class_log_loss": float(-np.log(p[np.arange(len(y)), y]).mean()),
        "multiclass_brier": float(np.mean(np.sum((p - truth) ** 2, axis=-1))),
        "classes": classes,
        "conditional_event_ranking": {
            "WR_given_current_wrong": binary(p[y < 2, 1], y[y < 2] == 1)
            if np.any(y < 2)
            else None,
            "RW_given_current_right": binary(p[y >= 2, 2], y[y >= 2] == 2)
            if np.any(y >= 2)
            else None,
        },
        "decision_value": {
            "mean_predicted": float(value.mean()),
            "mean_actual": float(actual.mean()),
            "mse": float(np.mean((value - actual) ** 2)),
            "bins": [
                {
                    "n": len(g),
                    "predicted": float(value[g].mean()),
                    "observed": float(actual[g].mean()),
                }
                for g in groups
                if len(g)
            ],
        },
    }


def deterministic_mixture(ids, mean_depth):
    """No labels, task metadata or controller scores; randomized adjacent-depth control."""
    import hashlib

    lower = int(np.floor(mean_depth))
    lower = max(1, min(4, lower))
    fraction = mean_depth - lower
    u = np.array(
        [
            int(
                hashlib.sha256(("4B matched compute:" + v).encode()).hexdigest()[:16],
                16,
            )
            / 2**64
            for v in ids
        ]
    )
    return np.where(u < fraction, min(4, lower + 1), lower).astype(int)
