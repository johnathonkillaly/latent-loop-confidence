"""Small staged MLX experiments with epoch checkpoints and held-out selection."""

import gzip
import json
import resource
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logsumexp, softmax

from latent_loop.datasets import load
from latent_loop.evaluation import metrics, oracle_labels, stopping_depth, transitions


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def features(split):
    arrays = dict(np.load(Path("artifacts/features") / f"{split}.npz"))
    rows = load("data", split)
    arrays["labels"] = np.array([r["label"] for r in rows], dtype=np.int32)
    p = softmax(arrays["base"], axis=-1)
    sortedp = np.sort(p, axis=-1)
    margin = sortedp[:, -1] - sortedp[:, -2]
    ent = -np.sum(p * np.log(p.clip(1e-12)), axis=-1)
    n = len(rows)
    arrays["stats"] = np.stack(
        [
            arrays["full_logp"],
            np.broadcast_to(arrays["entropy"][:, None], (n, 4)),
            np.broadcast_to(margin[:, None], (n, 4)),
            np.broadcast_to(ent[:, None], (n, 4)),
        ],
        axis=-1,
    ).astype(np.float32)
    return arrays, rows


def parameter_count(model):
    from mlx.utils import tree_flatten

    return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))


def fit(model, name, seed, config, train_arrays, val_arrays, loss_fn, guard):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten

    path = Path("artifacts/checkpoints") / f"{name}_seed{seed}"
    path.mkdir(parents=True, exist_ok=True)
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=0.01)
    grad_fn = nn.value_and_grad(model, loss_fn)
    info = {
        "name": name,
        "seed": seed,
        "next_epoch": 0,
        "best_validation_loss": 1e30,
        "history": [],
        "trainable_parameters": parameter_count(model),
        "optimizer": "AdamW",
        "learning_rate": config["learning_rate"],
        "gradient_clip": 1.0,
        "batch_size": config["train_batch"],
        "training_seconds": 0.0,
    }
    if (path / "state.json").exists():
        info = json.loads((path / "state.json").read_text())
        if (
            info["learning_rate"] != config["learning_rate"]
            or info["batch_size"] != config["train_batch"]
            or info["trainable_parameters"] != parameter_count(model)
            or (info.get("complete") and info["next_epoch"] != config["epochs"])
        ):
            raise ValueError(
                "Checkpoint hyperparameters changed; use a fresh --run-dir"
            )
        model.load_weights(str(path / "last.safetensors"))
        optimizer.state = tree_unflatten(
            list(mx.load(str(path / "optimizer.safetensors")).items())
        )
    if info.get("complete"):
        model.load_weights(str(path / "best.safetensors"))
        return info
    val = [mx.array(v) for v in val_arrays]
    start = time.perf_counter()
    for epoch in range(info["next_epoch"], config["epochs"]):
        guard()
        rng = np.random.default_rng(seed * 10000 + epoch)
        order = rng.permutation(len(train_arrays[0]))
        losses = []
        gradnorm = []
        for begin in range(0, len(order), config["train_batch"]):
            ix = order[begin : begin + config["train_batch"]]
            batch = [mx.array(v[ix]) for v in train_arrays]
            depth = (
                int(rng.integers(1, config["train_depth"] + 1))
                if name in ("recurrence", "memory")
                else 1
            )
            value, grads = grad_fn(model, *batch, depth=depth)
            grads, norm = optim.clip_grad_norm(grads, 1.0)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, value, norm)
            v = float(value.item())
            g = float(norm.item())
            if not np.isfinite(v + g):
                raise FloatingPointError(f"nonfinite {name} loss/gradient")
            losses.append(v)
            gradnorm.append(g)
        depths = (
            range(1, config["train_depth"] + 1)
            if name in ("recurrence", "memory")
            else [1]
        )
        vl = float(np.mean([loss_fn(model, *val, depth=d).item() for d in depths]))
        info["history"].append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": vl,
                "gradient_norm_mean": float(np.mean(gradnorm)),
                "gradient_norm_max": float(np.max(gradnorm)),
            }
        )
        if vl < info["best_validation_loss"]:
            info["best_validation_loss"] = vl
            info["best_epoch"] = epoch
            model.save_weights(str(path / "best.safetensors"))
        model.save_weights(str(path / "last.safetensors"))
        mx.save_safetensors(
            str(path / "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
        )
        info["next_epoch"] = epoch + 1
        info["training_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        info["peak_mlx_bytes"] = mx.get_peak_memory()
        info["peak_process_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        atomic_json(path / "state.json", info)
        atomic_json(Path("results/metrics/training") / f"{name}_seed{seed}.json", info)
        if epoch % 5 == 0:
            print(
                f"train {name} seed={seed} epoch={epoch} loss={np.mean(losses):.4f} val={vl:.4f}",
                flush=True,
            )
    info["complete"] = True
    atomic_json(path / "state.json", info)
    atomic_json(Path("results/metrics/training") / f"{name}_seed{seed}.json", info)
    model.load_weights(str(path / "best.safetensors"))
    return info


def time_head(fn, repeats):
    import mlx.core as mx

    for _ in range(3):
        mx.eval(fn())
        mx.synchronize()
    times = []
    for _ in range(repeats):
        mx.synchronize()
        start = time.perf_counter()
        mx.eval(fn())
        mx.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": float(np.median(times)),
        "p10_ms": float(np.quantile(times, 0.1)),
        "p90_ms": float(np.quantile(times, 0.9)),
        "raw_ms": times,
        "batch_size": 1,
        "warmup": 3,
        "repeats": repeats,
        "scope": "auxiliary head only on cached hidden states",
    }


def adaptive_infer(
    model, head, h, base, decoder, threshold, max_depth, confidence=False
):
    """Real early-stop path, used for timing. It accepts no outcome labels."""
    import mlx.core as mx

    z = h
    history = []
    for k in range(max_depth):
        next_z = model.step(z, h, history)
        history = (history + [z])[-model.window :]
        z = next_z
        logits = base + (z - h) @ decoder.T
        probability = (
            float(mx.max(mx.softmax(logits, axis=-1)).item())
            if confidence
            else float(mx.sigmoid(head(z, logits, k + 1)).item())
        )
        if probability >= threshold if confidence else probability < threshold:
            break
    return logits


def run_stage(config, seed, name, guard):
    """Fit one stage; continuation requires its trained parent checkpoint."""
    import mlx.core as mx
    import mlx.nn as nn

    from latent_loop.backbone import environment
    from latent_loop.models import Confidence, Continuation, Refiner, rms

    environment(config["memory_gb"])
    tr, _ = features("train")
    va, _ = features("val")
    decoder = mx.array(np.load("artifacts/features/decoder.npz")["decoder"])
    dim = tr["h"].shape[-1]
    offsets = {
        "confidence": 0,
        "recurrence": 100,
        "memory": 200,
        "feedforward": 300,
        "wide_feedforward": 400,
        "recurrence_continuation": 700,
        "memory_continuation": 701,
    }
    mx.random.seed(seed + offsets[name])
    if name == "confidence":
        model = Confidence(dim, config["width"])

        def objective(m, h, base, stats, y, depth=1):
            logits = m(h, decoder, base, stats)
            target = (mx.arange(4)[None, :] == y[:, None]).astype(mx.float32)
            return nn.losses.binary_cross_entropy(logits, target, reduction="mean")

        keys = ["h", "base", "stats", "labels"]
        train = [tr[k] for k in keys]
        val = [va[k] for k in keys]
    elif name.endswith("_continuation"):
        parent = name.removesuffix("_continuation")
        checkpoint = Path(f"artifacts/checkpoints/{parent}_seed{seed}")
        if not (checkpoint / "state.json").exists() or not json.loads(
            (checkpoint / "state.json").read_text()
        ).get("complete"):
            raise ValueError(
                f"Train {parent} first, using the same run directory and seed"
            )
        recurrent = Refiner(
            dim, config["width"], parent == "memory", config["memory_window"]
        )
        recurrent.load_weights(str(checkpoint / "best.safetensors"))
        mx.random.seed(seed + offsets[name])
        model = Continuation(dim)

        def prepare(a):
            logits, states = recurrent(
                mx.array(a["h"]),
                mx.array(a["base"]),
                decoder,
                config["train_depth"],
                True,
            )
            logits, states = np.array(logits), np.array(states)
            target, _ = oracle_labels(logits, a["labels"], config["delta"])
            horizon = config["train_depth"] - 1
            return [
                states[:horizon].reshape(-1, dim),
                logits[:horizon].reshape(-1, 4),
                np.broadcast_to(
                    np.arange(1, horizon + 1)[:, None], (horizon, len(a["h"]))
                )
                .reshape(-1)
                .astype(np.float32),
                target[:horizon].reshape(-1),
            ]

        train, val = prepare(tr), prepare(va)

        def objective(m, z, logits, depths, y, depth=1):
            x = mx.concatenate(
                [rms(z), mx.softmax(logits, axis=-1), depths[:, None] / 4], axis=-1
            )
            score = m.output(nn.silu(m.hidden(x))).squeeze(-1)
            return nn.losses.binary_cross_entropy(score, y, reduction="mean")
    else:
        width = config["width"] * (4 if name == "wide_feedforward" else 1)
        model = Refiner(dim, width, name == "memory", config["memory_window"])

        def objective(m, h, base, y, depth=1):
            return nn.losses.cross_entropy(
                m(h, base, decoder, depth), y, reduction="mean"
            )

        keys = ["h", "base", "labels"]
        train = [tr[k] for k in keys]
        val = [va[k] for k in keys]
    return fit(model, name, seed, config, train, val, objective, guard)


def run_seed(config, seed, guard):
    import mlx.core as mx
    import mlx.nn as nn

    from latent_loop.backbone import environment
    from latent_loop.models import Confidence, Continuation, Refiner

    environment(config["memory_gb"])
    mx.random.seed(seed)
    backbone = json.loads(Path("results/metrics/backbone.json").read_text())
    if backbone.get("assistant_prefix", "") != config.get(
        "assistant_prefix", "Answer: ("
    ):
        raise ValueError("Training prompt does not match cached backbone features")
    if "dataset_manifest" in backbone and backbone["dataset_manifest"] != json.loads(
        Path("data/manifest.json").read_text()
    ):
        raise ValueError("Training data changed after feature extraction")
    datasets = {s: features(s) for s in config["sizes"]}
    tr = datasets["train"][0]
    va = datasets["val"][0]
    decoder = mx.array(np.load("artifacts/features/decoder.npz")["decoder"])
    emb = decoder
    dim = tr["h"].shape[1]
    models = {}
    infos = {}
    evaluated = []
    recorded = {}
    selections = {}
    # Baseline selection is conditioned on the same four options for all variants.
    logt = minimize_scalar(
        lambda t: metrics(va["base"] / np.exp(t), va["labels"])["nll"],
        bounds=(-3, 3),
        method="bounded",
    ).x
    temperature = float(np.exp(logt))
    selections["temperature"] = temperature
    # Simple margin and entropy logistic calibrators, fitted on validation only.
    pval = softmax(va["base"], axis=-1)
    cv = (pval.argmax(-1) == va["labels"]).astype(float)
    calibrators = {}
    for key, values in {
        "margin": np.sort(pval, axis=-1)[:, -1] - np.sort(pval, axis=-1)[:, -2],
        "entropy": -(pval * np.log(pval.clip(1e-12))).sum(-1),
    }.items():
        obj = lambda w: (
            np.mean(np.logaddexp(0, w[0] * values + w[1]) - cv * (w[0] * values + w[1]))
            + 0.001 * w[0] ** 2
        )
        calibrators[key] = minimize(obj, [0.0, 0.0], method="BFGS").x.tolist()
    selections["simple_calibrators"] = calibrators
    confidence = Confidence(dim, config["width"])

    def confidence_loss(m, h, base, stats, y, depth=1):
        logits = m(h, emb, base, stats)
        targets = (mx.arange(4)[None, :] == y[:, None]).astype(mx.float32)
        return nn.losses.binary_cross_entropy(logits, targets, reduction="mean")

    infos["confidence"] = fit(
        confidence,
        "confidence",
        seed,
        config,
        [tr[k] for k in ["h", "base", "stats", "labels"]],
        [va[k] for k in ["h", "base", "stats", "labels"]],
        confidence_loss,
        guard,
    )
    models["confidence"] = confidence
    confval = np.array(
        confidence(
            *[mx.array(va[k]) for k in ["h"]],
            emb,
            mx.array(va["base"]),
            mx.array(va["stats"]),
        )
    )
    logconfval = -np.logaddexp(0, -confval)
    baseval = va["base"] - logsumexp(va["base"], axis=-1, keepdims=True)
    betas = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    beta = min(
        betas,
        key=lambda b: (
            -metrics(baseval + b * logconfval, va["labels"])["accuracy"],
            metrics(baseval + b * logconfval, va["labels"])["nll"],
            b,
        ),
    )
    selections["fusion_beta"] = beta

    def recurrence_loss(m, h, base, y, depth=1):
        return nn.losses.cross_entropy(m(h, base, decoder, depth), y, reduction="mean")

    for name, width, memory in [
        ("recurrence", config["width"], False),
        ("memory", config["width"], True),
        ("feedforward", config["width"], False),
        ("wide_feedforward", config["width"] * 4, False),
    ]:
        # Each stage has its own seed so skipping/resuming cannot change initialization.
        mx.random.seed(
            seed
            + {
                "recurrence": 100,
                "memory": 200,
                "feedforward": 300,
                "wide_feedforward": 400,
            }[name]
        )
        m = Refiner(dim, width, memory, config["memory_window"])
        parity = float(
            mx.max(
                mx.abs(
                    m(mx.array(va["h"][:2]), mx.array(va["base"][:2]), decoder, 4)
                    - mx.array(va["base"][:2])
                )
            ).item()
        )
        if parity != 0:
            raise AssertionError("closed residual gate must preserve logits exactly")
        infos[name] = fit(
            m,
            name,
            seed,
            config,
            [tr[k] for k in ["h", "base", "labels"]],
            [va[k] for k in ["h", "base", "labels"]],
            recurrence_loss,
            guard,
        )
        infos[name]["initial_parity_max_abs"] = parity
        models[name] = m
    # Frozen/untrained closed gate should exactly reproduce baseline; randomized open gate is a negative control.
    mx.random.seed(seed + 500)
    untrained = Refiner(dim, config["width"], False)
    models["untrained"] = untrained
    mx.random.seed(seed + 600)
    random_update = Refiner(dim, config["width"], False)
    random_update.gate = mx.array(0.1)
    models["random_update"] = random_update
    trajectories = {}
    for name in ["recurrence", "memory"]:
        trajectories[name] = {}
        for split, (arrays, _) in datasets.items():
            output, states = models[name](
                mx.array(arrays["h"]),
                mx.array(arrays["base"]),
                decoder,
                config["eval_depth"],
                True,
            )
            trajectories[name][split] = (np.array(output), np.array(states))
        vl = trajectories[name]["val"][0]
        selected = min(
            range(config["train_depth"]),
            key=lambda k: (
                -metrics(vl[k], va["labels"])["accuracy"],
                metrics(vl[k], va["labels"])["nll"],
                k,
            ),
        )
        selections[name + "_fixed_depth"] = selected + 1
        # Continue oracle uses only the training-range horizon, not unseen eval depths.
        trainlog, trainstates = trajectories[name]["train"]
        vallog, valstates = trajectories[name]["val"]
        trainoracle, _ = oracle_labels(
            trainlog[: config["train_depth"]], tr["labels"], config["delta"]
        )
        valoracle, _ = oracle_labels(
            vallog[: config["train_depth"]], va["labels"], config["delta"]
        )

        def flatten(z, l, y):
            horizon = config["train_depth"] - 1
            return [
                z[:horizon].reshape(-1, dim),
                l[:horizon].reshape(-1, 4),
                np.broadcast_to(
                    np.arange(1, horizon + 1)[:, None], (horizon, z.shape[1])
                )
                .reshape(-1)
                .astype(np.float32),
                y[:horizon].reshape(-1),
            ]

        # Oracle training labels are created here and never passed to adaptive_infer.
        mx.random.seed(seed + 700 + (name == "memory"))
        head = Continuation(dim)

        def continuation_loss(m, z, l, k, y, depth=1):
            # Same depth-normalized feature used during live inference.
            p = mx.softmax(l, axis=-1)
            from latent_loop.models import rms

            x = mx.concatenate([rms(z), p, k[:, None] / 4], axis=-1)
            logits = m.output(nn.silu(m.hidden(x))).squeeze(-1)
            return nn.losses.binary_cross_entropy(logits, y, reduction="mean")

        cname = name + "_continuation"
        infos[cname] = fit(
            head,
            cname,
            seed,
            config,
            flatten(trainstates, trainlog, trainoracle),
            flatten(valstates, vallog, valoracle),
            continuation_loss,
            guard,
        )
        models[cname] = head
        cp = np.stack(
            [
                np.array(
                    mx.sigmoid(head(mx.array(valstates[k]), mx.array(vallog[k]), k + 1))
                )
                for k in range(config["train_depth"])
            ]
        )
        fixed_nll = metrics(vallog[selected], va["labels"])["nll"]
        options = []
        for threshold in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
            depths = stopping_depth(cp, threshold)
            result = metrics(vallog[depths - 1, np.arange(len(depths))], va["labels"])
            options.append((depths.mean(), result["nll"], threshold))
        eligible = [o for o in options if o[1] <= fixed_nll + 0.01]
        threshold = (
            min(eligible, key=lambda o: (o[0], o[1]))
            if eligible
            else min(options, key=lambda o: (o[1], o[0]))
        )[2]
        selections[cname + "_threshold"] = threshold
        prob = softmax(vallog[: config["train_depth"]], axis=-1).max(-1)
        confidence_options = []
        for t in [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]:
            ds = stopping_depth(1 - prob, 1 - t)
            result = metrics(vallog[ds - 1, np.arange(len(ds))], va["labels"])
            confidence_options.append((ds.mean(), result["nll"], t))
        eligible = [o for o in confidence_options if o[1] <= fixed_nll + 0.01]
        selections[name + "_confidence_threshold"] = (
            min(eligible, key=lambda o: (o[0], o[1]))
            if eligible
            else min(confidence_options, key=lambda o: (o[1], o[0]))
        )[2]
    sample = va
    h, b, st = [mx.array(sample[k][:1]) for k in ["h", "base", "stats"]]
    timings = {
        "baseline_head": time_head(lambda: b, config["latency_repeats"]),
        "confidence": time_head(
            lambda: confidence(h, emb, b, st), config["latency_repeats"]
        ),
    }
    for name in ["recurrence", "memory", "feedforward", "wide_feedforward"]:
        for d in [1, 2, 3, 4, 8] if name in ("recurrence", "memory") else [1]:
            timings[f"{name}_d{d}"] = time_head(
                lambda m=models[name], depth=d: m(h, b, decoder, depth),
                config["latency_repeats"],
            )
    for name in ["recurrence", "memory"]:
        samples = []
        for i in range(12):
            hi, bi = mx.array(va["h"][i : i + 1]), mx.array(va["base"][i : i + 1])
            samples.append(
                time_head(
                    lambda n=name, hidden=hi, base=bi: adaptive_infer(
                        models[n],
                        models[n + "_continuation"],
                        hidden,
                        base,
                        decoder,
                        selections[n + "_continuation_threshold"],
                        config["train_depth"],
                    ),
                    config["latency_repeats"],
                )
            )
        timings[name + "_adaptive"] = {
            "median_ms": float(np.mean([s["median_ms"] for s in samples])),
            "samples": samples,
            "batch_size": 1,
            "scope": "live auxiliary stopping over 12 validation examples; mean of per-example medians",
        }

    def add(split, name, logits, depth=0, confidence_values=None, extra=None):
        arrays, rows = datasets[split]
        m = metrics(logits, arrays["labels"], confidence_values)
        m.update(
            {
                "seed": seed,
                "split": split,
                "variant": name,
                "average_depth": float(np.mean(depth)),
                "transitions_vs_baseline": transitions(
                    arrays["base"], logits, arrays["labels"]
                ),
                "per_task": {},
                "by_difficulty": {},
            }
        )
        for task in sorted(set(r["task"] for r in rows)):
            ix = np.array([r["task"] == task for r in rows])
            m["per_task"][task] = metrics(logits[ix], arrays["labels"][ix])
        for d in sorted(set(r["difficulty"] for r in rows)):
            ix = np.array([r["difficulty"] == d for r in rows])
            m["by_difficulty"][str(d)] = metrics(logits[ix], arrays["labels"][ix])
        if extra:
            m.update(extra)
        evaluated.append(m)
        recorded.setdefault(split, {})[name] = {
            "logits": logits.tolist(),
            "depth": np.broadcast_to(depth, (len(rows),)).tolist(),
        }
        atomic_json(
            f"results/metrics/seed{seed}.json",
            {
                "results": evaluated,
                "selections": selections,
                "training": infos,
                "timings": timings,
            },
        )

    for split in ["val", "test", "ood"]:
        arrays, rows = datasets[split]
        x = arrays
        labels = x["labels"]
        base = x["base"]
        add(split, "baseline", base)
        add(split, "temperature", base / temperature)
        add(
            split,
            "constant_calibration",
            base,
            confidence_values=np.full(
                len(labels), float((va["base"].argmax(-1) == va["labels"]).mean())
            ),
        )
        ps = softmax(base, axis=-1)
        for key, values in {
            "margin": np.sort(ps, axis=-1)[:, -1] - np.sort(ps, axis=-1)[:, -2],
            "entropy": -(ps * np.log(ps.clip(1e-12))).sum(-1),
        }.items():
            w = calibrators[key]
            add(
                split,
                key + "_calibration",
                base,
                confidence_values=expit(w[0] * values + w[1]),
            )
        c = np.array(
            confidence(mx.array(x["h"]), emb, mx.array(base), mx.array(x["stats"]))
        )
        cs = expit(c)
        idx = base.argmax(-1)
        add(
            split,
            "confidence_calibration",
            base,
            confidence_values=cs[np.arange(len(idx)), idx],
        )
        lc = -np.logaddexp(0, -c)
        lp = base - logsumexp(base, axis=-1, keepdims=True)
        add(split, "confidence_only", lc)
        add(
            split,
            "reranking",
            lp + beta * lc,
            extra={"fusion_beta": beta, "shortlist_coverage": 1.0},
        )
        top2 = np.argsort(-base, axis=-1)[:, :2]
        mask = np.zeros_like(base, bool)
        np.put_along_axis(mask, top2, True, axis=-1)
        add(
            split,
            "reranking_top2",
            np.where(mask, lp + beta * lc, -1e9),
            extra={
                "shortlist_coverage": float(mask[np.arange(len(labels)), labels].mean())
            },
        )
        for name in ["feedforward", "wide_feedforward", "untrained", "random_update"]:
            depth = 4 if name in ("untrained", "random_update") else 1
            logits = np.array(
                models[name](mx.array(x["h"]), mx.array(base), decoder, depth)
            )
            add(split, name, logits, depth)
        for name in ["recurrence", "memory"]:
            logits, states = trajectories[name][split]
            for k in range(config["eval_depth"]):
                original = x["h"]
                cos = (states[k] * original).sum(-1) / (
                    np.linalg.norm(states[k], axis=-1)
                    * np.linalg.norm(original, axis=-1)
                    + 1e-12
                )
                add(
                    split,
                    f"{name}_d{k + 1}",
                    logits[k],
                    k + 1,
                    extra={
                        "state_cosine_mean": float(cos.mean()),
                        "state_relative_change": float(
                            (
                                np.linalg.norm(states[k] - original, axis=-1)
                                / (np.linalg.norm(original, axis=-1) + 1e-12)
                            ).mean()
                        ),
                    },
                )
            selected = selections[name + "_fixed_depth"]
            add(split, name + "_selected", logits[selected - 1], selected)
            # Memory-disabled ablation uses the trained memory weights, rather than retraining.
            if name == "memory":
                models[name].memory = False
                ablation = np.array(
                    models[name](mx.array(x["h"]), mx.array(base), decoder, selected)
                )
                models[name].memory = True
                add(split, "memory_disabled", ablation, selected)
            head = models[name + "_continuation"]
            cp = np.stack(
                [
                    np.array(
                        mx.sigmoid(
                            head(mx.array(states[k]), mx.array(logits[k]), k + 1)
                        )
                    )
                    for k in range(config["train_depth"])
                ]
            )
            ds = stopping_depth(cp, selections[name + "_continuation_threshold"])
            oracle, loss = oracle_labels(
                logits[: config["train_depth"]], labels, config["delta"]
            )
            truth = oracle[:-1].reshape(-1)
            pred = cp[:-1].reshape(-1)
            from latent_loop.evaluation import calibration

            ce, bins = calibration(pred, truth)
            add(
                split,
                name + "_adaptive",
                logits[ds - 1, np.arange(len(ds))],
                ds,
                extra={
                    "continuation_brier": float(((pred - truth) ** 2).mean()),
                    "continuation_ece": ce,
                    "continuation_calibration": bins,
                    "continue_base_rate": float(truth.mean()),
                    "stopping_label_error": float(
                        (
                            (cp[:-1] >= selections[name + "_continuation_threshold"])
                            != oracle[:-1]
                        ).mean()
                    ),
                    "depth_histogram": np.bincount(
                        ds, minlength=config["train_depth"] + 1
                    ).tolist(),
                },
            )
            confidence_probs = softmax(logits[: config["train_depth"]], axis=-1).max(-1)
            cds = stopping_depth(
                1 - confidence_probs, 1 - selections[name + "_confidence_threshold"]
            )
            add(
                split,
                name + "_confidence_stop",
                logits[cds - 1, np.arange(len(cds))],
                cds,
            )
            ods = stopping_depth(oracle, 0.5)
            add(
                split,
                name + "_oracle_loss",
                logits[ods - 1, np.arange(len(ods))],
                ods,
                extra={"diagnostic_only": True},
            )
            # Correctness oracle is a separate optimistic upper-bound diagnostic.
            correct = logits[: config["train_depth"]].argmax(-1) == labels
            best = np.where(correct.any(0), correct.argmax(0), loss.argmin(0))
            add(
                split,
                name + "_oracle_accuracy",
                logits[best, np.arange(len(best))],
                best + 1,
                extra={"diagnostic_only": True},
            )
        recorded[split]["ids"] = [r["id"] for r in rows]
        recorded[split]["labels"] = labels.tolist()
    with gzip.open(f"results/metrics/predictions_seed{seed}.json.gz", "wt") as f:
        json.dump(recorded, f, separators=(",", ":"))
    print(f"seed {seed} completed: {len(evaluated)} evaluation rows", flush=True)
    return evaluated
