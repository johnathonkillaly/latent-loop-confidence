"""Native MLX stopping and visited-only incremental selection, without target inputs."""

import numpy as np


def components(config, seed):
    from latent_loop.experiment03.heads import Probe
    from latent_loop.experiment04a.pipeline import components as foundation_components
    from latent_loop.experiment04b.common import CACHE, checkpoint_paths
    from latent_loop.experiment04b.evaluate import settings
    from latent_loop.experiment04b.models import make_model

    chosen = settings(seed)
    models = foundation_components(seed)
    models["regression"] = Probe(1032, 64)
    models["regression"].load_weights(str(checkpoint_paths(seed)["regression"]))
    for name in chosen["deployment_checkpoints"]:
        models[name] = make_model(name, config, seed)
        models[name].load_weights(
            str(CACHE / "checkpoints" / f"{name}_seed{seed}" / "best.safetensors")
        )
    return models, chosen


def scalar(setting, x):
    import mlx.core as mx

    z = (x - mx.array(setting["mean"])) / mx.array(setting["std"])
    w = mx.array(setting["weight"])
    return mx.sigmoid(z @ w[1:] + w[0])


def runtime_spec(variant, chosen):
    if variant == "validation_best":
        best = chosen["validation_best_combination"]
        return best["policy"], best["selector"]
    name, _, selector = variant.partition(":")
    return name, selector or "final"


def live(variant, h, base, decoder, models, chosen):
    """All visited states are paid; only current features can determine continuation."""
    import mlx.core as mx

    from latent_loop.experiment03.heads import live_features

    n = len(h)
    name, selector = runtime_spec(variant, chosen)
    work = {
        "refiner_rows": 0,
        "C_rows": 0,
        "V_rows": 0,
        "D_rows": 0,
        "ranker_rows": 0,
        "policy_rows": 0,
        "midloop_host_barriers": 0,
    }
    if name in ("frozen", "one_pass"):
        out = base if name == "frozen" else models["one_pass"](h, base, decoder, 1)
        depth = mx.full((n,), int(name == "one_pass"), mx.int32)
        work["refiner_rows"] = n * int(name == "one_pass")
        return out, depth, depth, work
    rule = (
        {"kind": "fixed", "depth": int(name[-1])}
        if name.startswith("fixed_")
        else chosen["rules"][name]
    )
    kind = rule["kind"]
    signals = kind.removeprefix("stack_") if kind.startswith("stack_") else ""
    rank_signals = (
        selector.removeprefix("ranker_") if selector.startswith("ranker_") else ""
    )
    need_c = kind == "C" or "C" in signals or selector == "max_C" or "C" in rank_signals
    need_v = kind == "V" or "V" in signals or "V" in rank_signals
    need_d = kind in ("D", "utility") or "D" in signals or "D" in rank_signals
    active = np.arange(n)
    z, hi, bi = h, h, base
    best_q = mx.full((n,), -float("inf"))
    best_scores = base
    best_depth = mx.zeros((n,), mx.int32)
    outputs, paid, selected = [None] * n, np.zeros(n, int), [None] * n
    for depth in range(1, 5):
        previous = z
        z = models["refiner"].step(z, hi, [])
        scores = bi + (z - hi) @ decoder.T
        size = len(active)
        work["refiner_rows"] += size
        x = live_features(z, scores, previous, depth)
        c, v, d = mx.zeros((size,)), mx.zeros((size,)), mx.zeros((size, 4))
        if need_c:
            raw = mx.clip(mx.sigmoid(models["confidence"](x)[:, 0]), 1e-6, 1 - 1e-6)
            c = scalar(
                chosen["confidence_calibration"][str(depth)],
                (mx.log(raw) - mx.log1p(-raw))[:, None],
            )
            work["C_rows"] += size
        if need_v and depth < 4:
            v = models["regression"](x)[:, 0] * chosen["regression_scale"]
            work["V_rows"] += size
        if need_d and depth < 4:
            model_name = chosen["D_model"]
            setting = chosen["calibrators"][model_name]
            d = mx.softmax(
                models[model_name](x) / setting["temperature"]
                + mx.array(setting["bias"]),
                axis=-1,
            )
            work["D_rows"] += size
        if selector == "final":
            best_scores, best_depth = scores, mx.full((size,), depth, mx.int32)
        else:
            if selector == "max_C":
                quality = c
            else:
                extra = mx.stack(
                    [
                        c if "C" in rank_signals else mx.zeros_like(c),
                        v if "V" in rank_signals else mx.zeros_like(v),
                        d[:, 1] if "D" in rank_signals else mx.zeros_like(c),
                        d[:, 2] if "D" in rank_signals else mx.zeros_like(c),
                    ],
                    axis=-1,
                )
                quality = models[selector](mx.concatenate([x, extra], axis=-1))[:, 0]
                work["ranker_rows"] += size
            better = quality > best_q
            best_q = mx.where(better, quality, best_q)
            best_scores = mx.where(better[:, None], scores, best_scores)
            best_depth = mx.where(better, depth, best_depth)
        if kind == "fixed":
            keep = np.full(size, depth < rule["depth"], bool)
        elif depth == 4:
            keep = np.zeros(size, bool)
        else:
            if kind == "C":
                go = c < rule["threshold"]
            elif kind == "V":
                go = v > rule["threshold"]
            elif kind == "legacy_gain":
                go = mx.sigmoid(models["improvement"](x)[:, 0]) > rule["threshold"]
            elif kind in ("D", "utility"):
                value = d[:, 1] - d[:, 2]
                go = value > rule.get("threshold", rule.get("lambda", 0))
            elif kind.startswith("stack_"):
                fields = []
                if "C" in signals:
                    fields.append(c[:, None])
                if "V" in signals:
                    fields.append(v[:, None])
                if "D" in signals:
                    fields.append(d[:, 1:3])
                inputs = mx.concatenate(fields, axis=-1)
                setting = chosen["stacks"][signals]
                go = (
                    scalar(setting["WR"], inputs) - scalar(setting["RW"], inputs)
                    > rule["threshold"]
                )
                work["policy_rows"] += size
            else:
                raise ValueError(kind)
            keep = np.asarray(go).astype(bool)
            work["midloop_host_barriers"] += 1
        for local in np.flatnonzero(~keep):
            i = active[local]
            outputs[i], paid[i], selected[i] = (
                best_scores[local],
                depth,
                best_depth[local],
            )
        active = active[keep]
        if not len(active):
            break
        ix = mx.array(np.flatnonzero(keep))
        z, hi, bi = z[ix], hi[ix], bi[ix]
        best_q, best_scores, best_depth = best_q[ix], best_scores[ix], best_depth[ix]
    assert work["refiner_rows"] == int(paid.sum())
    return mx.stack(outputs), mx.array(paid), mx.stack(selected), work
