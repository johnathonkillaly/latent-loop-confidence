"""CPU-only immediate-gain targets, metrics, controls, and stopping rules."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax
from scipy.stats import rankdata, spearmanr

from latent_loop.evaluation import calibration, metrics


def targets(logits, labels, delta=0.01):
    logits = np.asarray(logits, dtype=np.float64)
    loss = logsumexp(logits, axis=-1) - logits[:, np.arange(len(labels)), labels]
    correct = logits.argmax(-1) == np.asarray(labels)[None, :]
    gain = loss[:-1] - loss[1:]
    return loss, correct, gain, gain > delta


def observables(states, logits, original):
    """No labels/future states: each row is observable immediately at its depth."""
    p = softmax(logits, axis=-1)
    ent = -(p * np.log(p.clip(1e-30))).sum(-1)
    ordered = np.sort(p, axis=-1)
    margin = ordered[..., -1] - ordered[..., -2]
    previous = np.concatenate([original[None], states[:-1]], axis=0)
    change = np.linalg.norm(states - previous, axis=-1) / np.maximum(
        np.linalg.norm(previous, axis=-1), 1e-6
    )
    depth = np.broadcast_to(np.arange(1, len(states) + 1)[:, None] / 4, ent.shape)
    scalar = np.stack([p.max(-1), ent, margin, change, depth], axis=-1)
    norm = states / np.sqrt(np.mean(states * states, axis=-1, keepdims=True) + 1e-6)
    x = np.concatenate([norm, p, scalar[..., 1:]], axis=-1).astype(np.float32)
    return x, scalar.astype(np.float32)


def binary_metrics(p, y):
    p, y = np.asarray(p, dtype=float).ravel(), np.asarray(y, dtype=bool).ravel()
    pos, neg = int(y.sum()), int((~y).sum())
    auc = (
        float((rankdata(p)[y].sum() - pos * (pos + 1) / 2) / (pos * neg))
        if pos and neg
        else None
    )
    # Group tied scores so a constant predictor has AP exactly equal to prevalence.
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    tp = np.cumsum(y[order])[ends]
    precision = tp / (ends + 1)
    recall = tp / max(pos, 1)
    ap = float(np.sum(precision * np.diff(np.r_[0, recall]))) if pos else None
    ece, bins = calibration(p, y)
    pred = p >= 0.5
    selected = np.argsort(-p, kind="stable")[: max(1, len(p) // 2)]
    risk = 1 - np.cumsum(y[order]) / np.arange(1, len(p) + 1)
    return {
        "n": len(p),
        "prevalence": float(y.mean()),
        "auroc": auc,
        "average_precision": ap,
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece,
        "calibration": bins,
        "precision_at_0.5": float(y[pred].mean()) if pred.any() else None,
        "recall_at_0.5": float(pred[y].mean()) if pos else None,
        "selected_fraction_at_0.5": float(pred.mean()),
        "accuracy_at_50pct_coverage": float(y[selected].mean()),
        "aurc": float(risk.mean()),
        "risk_coverage": [
            {"coverage": float(c), "risk": float(risk[max(0, int(c * len(p)) - 1)])}
            for c in np.linspace(0.05, 1, 20)
        ],
    }


def regression_metrics(pred, actual):
    pred, actual = np.asarray(pred).ravel(), np.asarray(actual).ravel()
    corr = (
        spearmanr(pred, actual).statistic
        if np.std(pred) > 1e-12 and np.std(actual) > 1e-12
        else None
    )
    groups = np.array_split(np.argsort(pred), min(10, len(pred)))
    return {
        "n": len(pred),
        "mae": float(np.mean(np.abs(pred - actual))),
        "rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
        "spearman": float(corr) if corr is not None and np.isfinite(corr) else None,
        "calibration": [
            {
                "count": len(g),
                "predicted": float(pred[g].mean()),
                "observed": float(actual[g].mean()),
            }
            for g in groups
            if len(g)
        ],
    }


def opportunity(logits, labels, delta=0.01):
    loss, correct, gain, useful = targets(logits, labels, delta)
    p = softmax(logits, axis=-1)
    prediction = logits.argmax(-1)
    output = []
    for k in range(len(gain)):
        changed = prediction[k] != prediction[k + 1]
        up = ~correct[k] & correct[k + 1]
        down = correct[k] & ~correct[k + 1]
        output.append(
            {
                "transition": f"{k + 1}->{k + 2}",
                "n": len(labels),
                "positive_loss_gain": float((gain[k] > 0).mean()),
                "meaningful_gain": float(useful[k].mean()),
                "harmful_loss": float((gain[k] < -delta).mean()),
                "mean_gain": float(gain[k].mean()),
                "gain_quantiles": np.quantile(
                    gain[k], [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
                ).tolist(),
                "changed": int(changed.sum()),
                "wrong_to_right": int(up.sum()),
                "right_to_wrong": int(down.sum()),
                "beneficial_among_changed": float(up.sum() / changed.sum())
                if changed.any()
                else None,
                "confidence_increased_while_becoming_wrong": int(
                    (down & (p[k + 1].max(-1) > p[k].max(-1))).sum()
                ),
                "wrong_and_more_confident": int(
                    (~correct[k + 1] & (p[k + 1].max(-1) > p[k].max(-1))).sum()
                ),
            }
        )
    return {
        "depth_accuracy": correct.mean(1).tolist(),
        "depth_nll": loss.mean(1).tolist(),
        "transitions": output,
    }


def fit_scalar(x, y, kind="binary"):
    x, y = np.asarray(x, float), np.asarray(y, float)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-6)
    z = np.column_stack([np.ones(len(x)), (x - mean) / std])
    if kind == "binary":

        def objective(w):
            a = z @ w
            reg = np.r_[0, w[1:]] * 0.001
            return (
                np.mean(np.logaddexp(0, a) - y * a) + 0.0005 * np.sum(w[1:] ** 2),
                z.T @ (expit(a) - y) / len(y) + reg,
            )

        solved = minimize(objective, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B")
        if not solved.success:
            raise RuntimeError(solved.message)
        weight = solved.x
    else:
        penalty = np.eye(z.shape[1]) * 0.001 * len(z)
        penalty[0, 0] = 0
        weight = np.linalg.solve(z.T @ z + penalty, z.T @ y)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weight": weight.tolist(),
        "kind": kind,
    }


def scalar_predict(model, x):
    z = (np.asarray(x) - model["mean"]) / model["std"]
    a = model["weight"][0] + z @ np.asarray(model["weight"])[1:]
    return expit(a) if model["kind"] == "binary" else a


def depths_from_continue(continuation):
    """Only completed decisions may stop; depth four always stops."""
    continuation = np.asarray(continuation, bool)
    n = continuation.shape[1]
    depths, alive = np.full(n, 4, dtype=int), np.ones(n, bool)
    for k in range(3):
        halt = alive & ~continuation[k]
        depths[halt], alive[halt] = k + 1, False
    return depths


def policy_depth(name, scores, setting, logits=None, labels=None):
    n = scores["confidence"].shape[1]
    if name.startswith("fixed_"):
        return np.full(n, int(name[-1]), dtype=int)
    if name in ("confidence", "confidence_matched"):
        go = scores["confidence"][:3] < setting["threshold"]
    elif name in (
        "improvement",
        "old_continuation",
        "entropy",
        "margin",
        "change",
        "confidence_gain",
    ):
        go = scores[name][:3] > setting["threshold"]
    elif name == "voc":
        go = scores["regression"][:3] > setting["cost"]
    elif name == "joint":
        go = (scores["joint_gain"][:3] > setting["gain_threshold"]) & (
            scores["joint_confidence"][:3] < setting["confidence_threshold"]
        )
    elif name == "oracle_next":
        if logits is None or labels is None:
            raise ValueError("Oracle requires labels and must never be deployed")
        go = targets(logits, labels)[2] > setting["delta"]
    elif name == "oracle_best":
        if logits is None or labels is None:
            raise ValueError("Oracle requires labels and must never be deployed")
        return targets(logits, labels)[0].argmin(0) + 1
    else:
        raise ValueError(name)
    return depths_from_continue(go)


def policy_metrics(logits, labels, depth, confidence=None):
    n = len(labels)
    chosen = logits[depth - 1, np.arange(n)]
    c = confidence[depth - 1, np.arange(n)] if confidence is not None else None
    out = metrics(chosen, labels, c)
    start, finish = logits[0].argmax(-1) == labels, chosen.argmax(-1) == labels
    out.update(
        {
            "mean_loops": float(depth.mean()),
            "wrong_to_right": int((~start & finish).sum()),
            "right_to_wrong": int((start & ~finish).sum()),
            "changed": int((logits[0].argmax(-1) != chosen.argmax(-1)).sum()),
            "stopping_counts": [int((depth == d).sum()) for d in range(1, 5)],
        }
    )
    return out


def paired_bootstrap(differences, seed=830017, repeats=1500):
    """Resample parent seeds and unique items, preserving all paired comparisons."""
    d = np.asarray(differences)
    rng = np.random.default_rng(seed)
    values = [
        d[rng.integers(0, d.shape[0], d.shape[0])][
            :, rng.integers(0, d.shape[1], d.shape[1])
        ].mean()
        for _ in range(repeats)
    ]
    return {
        "mean_delta": float(d.mean()),
        "hierarchical_seed_item_bootstrap95": np.quantile(
            values, [0.025, 0.975]
        ).tolist(),
        "seeds": d.shape[0],
        "unique_items": d.shape[1],
        "repeats": repeats,
    }
