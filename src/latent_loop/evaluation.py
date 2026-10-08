"""CPU-only metrics and paired diagnostics. NLL is conditional on the options."""

import numpy as np
from scipy.special import softmax
from scipy.stats import binomtest, rankdata


def calibration(confidence, correct, bins=10):
    c, y = np.asarray(confidence), np.asarray(correct)
    rows = []
    for i in range(bins):
        mask = (
            (c >= i / bins) & (c < (i + 1) / bins)
            if i < bins - 1
            else (c >= i / bins) & (c <= 1)
        )
        if mask.any():
            rows.append(
                {
                    "count": int(mask.sum()),
                    "confidence": float(c[mask].mean()),
                    "accuracy": float(y[mask].mean()),
                }
            )
    ece = sum(r["count"] * abs(r["confidence"] - r["accuracy"]) for r in rows) / len(c)
    return ece, rows


def metrics(logits, labels, confidence=None):
    p = softmax(np.asarray(logits, dtype=np.float64), axis=-1)
    y = np.asarray(labels)
    pred = p.argmax(-1)
    correct = pred == y
    c = p.max(-1) if confidence is None else np.asarray(confidence)
    ece, bins = calibration(c, correct)
    order = np.argsort(-c)
    risk = 1 - np.cumsum(correct[order]) / np.arange(1, len(y) + 1)
    onehot = np.eye(p.shape[-1])[y]
    n = len(y)
    acc = float(correct.mean())
    z = 1.96
    center = (acc + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    positives = int(correct.sum())
    negatives = n - positives
    auc = (
        float(
            (rankdata(c)[correct].sum() - positives * (positives + 1) / 2)
            / (positives * negatives)
        )
        if positives and negatives
        else None
    )
    return {
        "n": n,
        "accuracy": acc,
        "accuracy_wilson95": [center - half, center + half],
        "nll": float(-np.log(p[np.arange(n), y].clip(1e-12)).mean()),
        "brier_multiclass": float(((p - onehot) ** 2).sum(-1).mean()),
        "brier_correctness": float(((c - correct) ** 2).mean()),
        "ece": float(ece),
        "correctness_auroc": auc,
        "aurc": float(risk.mean()),
        "selective_accuracy_50": float(correct[order[: max(1, n // 2)]].mean()),
        "calibration": bins,
    }


def transitions(base, other, y, seed=99):
    a = softmax(base, axis=-1)
    b = softmax(other, axis=-1)
    ca = a.argmax(-1) == y
    cb = b.argmax(-1) == y
    up = int((~ca & cb).sum())
    down = int((ca & ~cb).sum())
    d = cb.astype(float) - ca
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), (3000, len(d)))].mean(-1)
    return {
        "wrong_to_right": up,
        "right_to_wrong": down,
        "changed": int((a.argmax(-1) != b.argmax(-1)).sum()),
        "accuracy_delta": float(d.mean()),
        "delta_bootstrap95": np.quantile(boot, [0.025, 0.975]).tolist(),
        "mcnemar_exact_p": float(binomtest(up, up + down, 0.5).pvalue)
        if up + down
        else 1.0,
        "correct_less_confident": int((ca & (b.max(-1) < a.max(-1))).sum()),
        "wrong_more_confident": int((~ca & (b.max(-1) > a.max(-1))).sum()),
    }


def oracle_labels(logits, labels, delta=0.01):
    p = softmax(logits, axis=-1)
    loss = -np.log(p[:, np.arange(len(labels)), labels].clip(1e-12))
    targets = np.zeros_like(loss, dtype=np.float32)
    for k in range(len(loss) - 1):
        targets[k] = loss[k + 1 :].min(0) < loss[k] - delta
    return targets, loss


def stopping_depth(probabilities, threshold, minimum=1):
    # arrays are [depth, example]; max depth is a compulsory stop.
    depth = np.full(probabilities.shape[1], probabilities.shape[0], dtype=int)
    alive = np.ones(len(depth), bool)
    for k in range(minimum - 1, len(probabilities) - 1):
        halt = alive & (probabilities[k] < threshold)
        depth[halt] = k + 1
        alive[halt] = False
    return depth
