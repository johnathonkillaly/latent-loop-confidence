"""Live timing with paid row counts: scalar, active compaction, and masked budget."""

import json
import os
import resource
import time

import numpy as np

from latent_loop.experiment04a.common import guard, utc, verify_freeze
from latent_loop.experiment04a.data import EXP, OUT, TASKS, load
from latent_loop.training import atomic_json


def live_policy(name, h, base, decoder, models, setting):
    """Runtime inputs are current states only. Returns outputs, paid work and barriers."""
    import mlx.core as mx

    from latent_loop.experiment03.heads import live_features

    n = len(h)
    if name == "frozen":
        return (
            base,
            mx.zeros((n,), mx.int32),
            {"refiner_rows": 0, "controller_rows": 0, "midloop_host_barriers": 0},
        )
    if name == "one_pass":
        return (
            models["one_pass"](h, base, decoder, 1),
            mx.ones((n,), mx.int32),
            {"refiner_rows": n, "controller_rows": 0, "midloop_host_barriers": 0},
        )
    if name.startswith("fixed_"):
        d = int(name[-1])
        return (
            models["refiner"](h, base, decoder, d),
            mx.full((n,), d, mx.int32),
            {"refiner_rows": n * d, "controller_rows": 0, "midloop_host_barriers": 0},
        )
    policy, mechanism = name.split(":")
    mechanism, _, diagnostic_threshold = mechanism.partition("@")
    head = models[policy]
    threshold = (
        float(diagnostic_threshold) if diagnostic_threshold else setting["threshold"]
    )
    counts = {"refiner_rows": 0, "controller_rows": 0, "midloop_host_barriers": 0}

    def go(z, scores, previous, depth):
        probability = mx.sigmoid(head(live_features(z, scores, previous, depth))[:, 0])
        return (
            probability < threshold
            if policy == "confidence"
            else probability > threshold
        )

    if mechanism == "scalar":
        outputs = []
        depths = []
        for i in range(n):
            z = h[i : i + 1]
            hi = h[i : i + 1]
            bi = base[i : i + 1]
            for depth in range(1, 5):
                previous = z
                z = models["refiner"].step(z, hi, [])
                scores = bi + (z - hi) @ decoder.T
                counts["refiner_rows"] += 1
                if depth == 4:
                    break
                counts["controller_rows"] += 1
                counts["midloop_host_barriers"] += 1
                if not bool(go(z, scores, previous, depth).item()):
                    break
            outputs.append(scores[0])
            depths.append(depth)
        return mx.stack(outputs), mx.array(depths), counts
    if mechanism == "compact":
        active = np.arange(n)
        z = h
        hi = h
        bi = base
        outputs = [None] * n
        depths = np.zeros(n, int)
        for depth in range(1, 5):
            previous = z
            z = models["refiner"].step(z, hi, [])
            scores = bi + (z - hi) @ decoder.T
            counts["refiner_rows"] += len(active)
            if depth == 4:
                keep = np.zeros(len(active), bool)
            else:
                counts["controller_rows"] += len(active)
                counts["midloop_host_barriers"] += 1
                keep = np.array(go(z, scores, previous, depth)).astype(bool)
            for local in np.flatnonzero(~keep):
                global_index = active[local]
                outputs[global_index] = scores[local]
                depths[global_index] = depth
            active = active[keep]
            if not len(active):
                break
            ix = mx.array(np.flatnonzero(keep))
            z, hi, bi = z[ix], hi[ix], bi[ix]
        return mx.stack(outputs), mx.array(depths), counts
    if mechanism == "masked":
        # Device-resident selection, but all rows execute all four refinements.
        z = h
        alive = mx.ones((n,), mx.bool_)
        chosen = base
        depths = mx.full((n,), 4, mx.int32)
        for depth in range(1, 5):
            previous = z
            z = models["refiner"].step(z, h, [])
            scores = base + (z - h) @ decoder.T
            counts["refiner_rows"] += n
            if depth == 4:
                stop = alive
            else:
                counts["controller_rows"] += n
                stop = alive & ~go(z, scores, previous, depth)
            chosen = mx.where(stop[:, None], scores, chosen)
            depths = mx.where(stop, depth, depths)
            alive = alive & ~stop
        return chosen, depths, counts
    raise ValueError(name)


def timing_indices(rows, count):
    if count % 6:
        raise ValueError("Timing subset must balance all six tasks")
    groups = {
        task: [i for i, r in enumerate(rows) if r["task"] == task] for task in TASKS
    }
    return [groups[task][rank] for rank in range(count // 6) for task in TASKS]


def measure(config):
    import mlx.core as mx

    from latent_loop.backbone import environment, get_model, hidden_forward, tokenize
    from latent_loop.experiment04a.pipeline import components, features, trajectory
    from latent_loop.experiment04a.science import stopping

    verify_freeze()
    env = environment(config["memory_gib"])
    model, tokenizer, manifest = get_model()
    rows = load("controller_val")
    indices = timing_indices(rows, config["timing_examples"])
    decoder = mx.array(np.load("artifacts/features/decoder.npz")["decoder"])
    variants = ["frozen", "one_pass", "fixed_1", "fixed_2", "fixed_3", "fixed_4"] + [
        f"{p}:{m}"
        for p in ("confidence", "improvement")
        for m in ("scalar", "compact", "masked")
    ]
    # A fixed diagnostic threshold exercises nontrivial halting even when validation
    # selects stopping after one loop. It is not an accuracy-selected deployed policy.
    variants += [
        f"{p}:{m}@0.5"
        for p in ("confidence", "improvement")
        for m in ("scalar", "compact", "masked")
    ]
    records = []
    component_records = []
    controller_only = []
    started = time.perf_counter()
    load_start = list(os.getloadavg())
    cached = features("controller_val")
    for seed in config["parent_seeds"]:
        models = components(seed)
        settings = json.loads((EXP / f"selection_seed{seed}.json").read_text())[
            "policies"
        ]
        t = trajectory("controller_val", seed)
        expected = {
            name: stopping(
                name.split(":")[0],
                {"confidence": t["confidence"], "improvement": t["improvement"]},
                {"threshold": 0.5} if "@" in name else settings[name.split(":")[0]],
            )
            for name in variants
            if ":" in name
        }
        for name in ("confidence", "improvement", "joint"):
            x = mx.array(t["x"][0, :1])
            samples = []
            for _ in range(3):
                mx.eval(models[name](x))
                mx.synchronize()
            for _ in range(30):
                mx.synchronize()
                tic = time.perf_counter()
                mx.eval(models[name](x))
                mx.synchronize()
                samples.append((time.perf_counter() - tic) * 1000)
            controller_only.append(
                {
                    "seed": seed,
                    "name": name,
                    "raw_ms": samples,
                    "median_ms": float(np.median(samples)),
                }
            )
        for batch_size in config["timing_batch_sizes"]:
            batches = [
                indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
            ]
            for name in variants:
                for _ in range(3):
                    warm = batches[0]
                    encoded = [tokenize(tokenizer, rows[i]["prompt"]) for i in warm]
                    lens = [len(x) for x in encoded]
                    ids = mx.array(
                        [
                            x
                            + [tokenizer.pad_token_id or tokenizer.eos_token_id]
                            * (max(lens) - len(x))
                            for x in encoded
                        ]
                    )
                    h = hidden_forward(model, ids)[
                        mx.arange(len(ids)), mx.array(lens) - 1
                    ]
                    base = h @ decoder.T
                    output, depth, _ = live_policy(
                        name,
                        h,
                        base,
                        decoder,
                        models,
                        settings.get(name.split(":")[0], {}),
                    )
                    mx.eval(output, depth)
                    mx.synchronize()
            for batch_index, batch in enumerate(batches):
                guard(config)
                items = [rows[i] for i in batch]
                hcached = mx.array(cached["h"][batch])
                bcached = mx.array(cached["base"][batch])
                mx.eval(hcached, bcached)
                collected = {
                    name: {"full_ms": [], "aux_ms": [], "depths": [], "work": []}
                    for name in variants
                }
                for repeat in range(config["timing_repeats"]):
                    order = np.random.default_rng(
                        seed * 10000 + batch_size * 1000 + batch_index * 10 + repeat
                    ).permutation(variants)
                    for name in order:
                        setting = settings.get(name.split(":")[0], {})
                        mx.synchronize()
                        tic = time.perf_counter()
                        encoded = [tokenize(tokenizer, r["prompt"]) for r in items]
                        lens = [len(x) for x in encoded]
                        ids = mx.array(
                            [
                                x
                                + [tokenizer.pad_token_id or tokenizer.eos_token_id]
                                * (max(lens) - len(x))
                                for x in encoded
                            ]
                        )
                        h = hidden_forward(model, ids)[
                            mx.arange(len(items)), mx.array(lens) - 1
                        ]
                        base = h @ decoder.T
                        output, depth, work = live_policy(
                            name, h, base, decoder, models, setting
                        )
                        mx.eval(output, depth)
                        mx.synchronize()
                        host_output = np.array(output)
                        host_depth = np.array(depth)
                        collected[name]["full_ms"].append(
                            (time.perf_counter() - tic) * 1000
                        )
                        collected[name]["depths"].append(host_depth.tolist())
                        collected[name]["work"].append(work)
                        if name == "frozen":
                            wanted = np.zeros(len(batch), int)
                            scores = cached["base"][batch]
                        elif name == "one_pass":
                            wanted = np.ones(len(batch), int)
                            scores = t["one_pass"][batch]
                        elif name.startswith("fixed_"):
                            wanted = np.full(len(batch), int(name[-1]), int)
                            scores = t["logits"][wanted - 1, batch]
                        else:
                            wanted = expected[name][batch]
                            scores = t["logits"][wanted - 1, batch]
                        assert np.array_equal(wanted, host_depth), (
                            seed,
                            name,
                            batch,
                            host_depth,
                            wanted,
                        )
                        assert np.max(np.abs(scores - host_output)) < 0.002, (
                            seed,
                            name,
                            batch,
                        )
                        if ":compact" in name or ":scalar" in name:
                            assert work["refiner_rows"] == int(wanted.sum())
                        if ":masked" in name:
                            assert work["refiner_rows"] == 4 * len(batch)
                        mx.synchronize()
                        tic = time.perf_counter()
                        output, depth, _ = live_policy(
                            name, hcached, bcached, decoder, models, setting
                        )
                        mx.eval(output, depth)
                        mx.synchronize()
                        np.array(output)
                        np.array(depth)
                        collected[name]["aux_ms"].append(
                            (time.perf_counter() - tic) * 1000
                        )
                records.extend(
                    {
                        "seed": seed,
                        "variant": name,
                        "batch_size": len(batch),
                        "batch_index": batch_index,
                        "instance_ids": [r["id"] for r in items],
                        "tasks": [r["task"] for r in items],
                        "prompt_lengths": cached["lengths"][batch].tolist(),
                        **values,
                    }
                    for name, values in collected.items()
                )
                if batch_index == 0:
                    # Extra component synchronization is diagnostic, not normal-path attribution.
                    start = time.perf_counter()
                    encoded = [tokenize(tokenizer, r["prompt"]) for r in items]
                    lens = [len(v) for v in encoded]
                    tok_ms = (time.perf_counter() - start) * 1000
                    mx.synchronize()
                    start = time.perf_counter()
                    ids = mx.array(
                        [
                            v
                            + [tokenizer.pad_token_id or tokenizer.eos_token_id]
                            * (max(lens) - len(v))
                            for v in encoded
                        ]
                    )
                    mx.eval(ids)
                    mx.synchronize()
                    input_ms = (time.perf_counter() - start) * 1000
                    start = time.perf_counter()
                    h = hidden_forward(model, ids)[
                        mx.arange(len(items)), mx.array(lens) - 1
                    ]
                    base = h @ decoder.T
                    mx.eval(h, base)
                    mx.synchronize()
                    backbone_ms = (time.perf_counter() - start) * 1000
                    for mechanism in ("scalar", "compact", "masked"):
                        start = time.perf_counter()
                        out, d, work = live_policy(
                            "improvement:" + mechanism,
                            h,
                            base,
                            decoder,
                            models,
                            settings["improvement"],
                        )
                        mx.eval(out, d)
                        mx.synchronize()
                        aux_ms = (time.perf_counter() - start) * 1000
                        start = time.perf_counter()
                        np.array(out)
                        np.array(d)
                        output_ms = (time.perf_counter() - start) * 1000
                        component_records.append(
                            {
                                "seed": seed,
                                "batch_size": len(batch),
                                "mechanism": mechanism,
                                "tokenization_ms": tok_ms,
                                "input_array_materialization_ms": input_ms,
                                "backbone_decode_ms": backbone_ms,
                                "auxiliary_ms": aux_ms,
                                "output_array_conversion_ms": output_ms,
                                "executed_work": work,
                            }
                        )
            print(f"timing seed {seed} batch {batch_size}: complete", flush=True)
    summary = {}
    for batch_size in config["timing_batch_sizes"]:
        summary[str(batch_size)] = {}
        for name in variants:
            items = [
                r
                for r in records
                if r["batch_size"] == batch_size and r["variant"] == name
            ]
            values = [np.median(r["full_ms"]) for r in items]
            aux = [np.median(r["aux_ms"]) for r in items]
            summary[str(batch_size)][name] = {
                "mean_batch_median_ms": float(np.mean(values)),
                "amortized_ms_per_item": float(np.mean(values) / batch_size),
                "p10_batch_ms": float(np.quantile(values, 0.1)),
                "p90_batch_ms": float(np.quantile(values, 0.9)),
                "aux_mean_batch_median_ms": float(np.mean(aux)),
                "mean_returned_depth": float(
                    np.mean([d for r in items for d in r["depths"][0]])
                ),
                "mean_executed_refiner_rows_per_item": float(
                    np.mean([r["work"][0]["refiner_rows"] / batch_size for r in items])
                ),
                "mean_midloop_host_barriers_per_batch": float(
                    np.mean([r["work"][0]["midloop_host_barriers"] for r in items])
                ),
            }
    atomic_json(
        OUT / "latency.json",
        {
            "completed_utc": utc(),
            "scope": "Full input tokenization, array materialization, right-padded frozen backbone, label decoding, live auxiliary inference and host stopping barriers, final Metal sync and output conversion. Model loading excluded. Auxiliary-only uses h/base already on device and includes output conversion.",
            "warmups": 3,
            "repeats": config["timing_repeats"],
            "timed_validation_ids": [rows[i]["id"] for i in indices],
            "balanced_task_count": len(indices) // 6,
            "summary": summary,
            "records": records,
            "component_diagnostics": component_records,
            "controller_only": controller_only,
            "environment": env,
            "model_revision": manifest["revision"],
            "start_load_average": load_start,
            "end_load_average": list(os.getloadavg()),
            "peak_mlx_bytes": mx.get_peak_memory(),
            "peak_process_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "command_seconds": time.perf_counter() - started,
            "work_accounting": "Compact computes only alive rows; scalar computes rows serially; masked executes 4 loops for every row despite earlier returned decisions. Masked execution is not compute saving.",
        },
    )
