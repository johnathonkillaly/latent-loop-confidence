"""Synchronized live inference including backbone and actual stopping decisions."""

import gzip
import json
import resource
import time

import numpy as np

from latent_loop.experiment03.data import EXP, OUT, utc
from latent_loop.training import atomic_json


def live_aux(name, setting, h, base, decoder, refiner, onepass, heads, infos, old):
    import mlx.core as mx

    from latent_loop.experiment03.heads import live_features

    if name == "frozen":
        return base, 0
    if name == "one_pass":
        return onepass(h, base, decoder, 1), 1
    if name.startswith("fixed_"):
        depth = int(name[-1])
        return refiner(h, base, decoder, depth), depth
    z = h
    for depth in range(1, 5):
        previous = z
        z = refiner.step(z, h, [])
        scores = base + (z - h) @ decoder.T
        if depth == 4:
            return scores, depth
        if name == "old_continuation":
            go = float(mx.sigmoid(old(z, scores, depth)).item()) > setting["threshold"]
        else:
            x = live_features(z, scores, previous, depth)
            if name in ("confidence", "confidence_matched"):
                go = (
                    float(mx.sigmoid(heads["confidence"](x)[0, 0]).item())
                    < setting["threshold"]
                )
            elif name == "improvement":
                go = (
                    float(mx.sigmoid(heads[name](x)[0, 0]).item())
                    > setting["threshold"]
                )
            elif name == "voc":
                gain = (
                    float(heads["regression"](x)[0, 0].item())
                    * infos["regression"]["scale"]
                )
                go = gain > setting["cost"]
            elif name == "joint":
                output = heads[name](x)[0]
                mx.eval(output)
                go = (
                    float(mx.sigmoid(output[1]).item()) > setting["gain_threshold"]
                    and float(mx.sigmoid(output[0]).item())
                    < setting["confidence_threshold"]
                )
            else:
                raise ValueError(name)
        if not go:
            return scores, depth
    raise AssertionError("Unreachable")


def measure(config):
    import mlx.core as mx

    from latent_loop.backbone import environment, get_model, hidden_forward, tokenize
    from latent_loop.experiment03.data import CACHE
    from latent_loop.experiment03.run import (
        guard,
        load_features,
        make_head,
        refiner,
        trajectory,
    )
    from latent_loop.models import Continuation

    env = environment(config["memory_gb"])
    model, tokenizer, manifest = get_model()
    decoder = mx.array(np.load("artifacts/features/decoder.npz")["decoder"])
    all_rows = json.load(gzip.open(OUT / "examples.json.gz", "rt"))
    rows = all_rows["controller_val"][: config["latency_examples"]]
    variants = [
        "frozen",
        "one_pass",
        "fixed_1",
        "fixed_2",
        "fixed_3",
        "fixed_4",
        "confidence",
        "confidence_matched",
        "improvement",
        "voc",
        "joint",
        "old_continuation",
    ]
    records, parity, head_only = [], [], []
    started = time.perf_counter()
    for seed in config["seeds"]:
        heads, infos = {}, {}
        for name in [
            "confidence",
            "improvement",
            "regression",
            "joint",
            "candidate_confidence",
        ]:
            info = json.loads(
                (
                    CACHE / "checkpoints" / f"{name}_seed{seed}" / "state.json"
                ).read_text()
            )
            head = make_head(name, config, seed, decoder)
            head.load_weights(
                str(CACHE / "checkpoints" / f"{name}_seed{seed}" / "best.safetensors")
            )
            heads[name], infos[name] = head, info
        refinement, onepass = refiner(seed), refiner(seed, "feedforward")
        old = Continuation(1024)
        old.load_weights(
            f"artifacts/checkpoints/recurrence_continuation_seed{seed}/best.safetensors"
        )
        settings = json.loads((EXP / f"selection_seed{seed}.json").read_text())[
            "settings"
        ]
        raw = json.load(gzip.open(OUT / f"predictions_seed{seed}.json.gz", "rt"))[
            "controller_val"
        ]
        cached = load_features("controller_val")
        traj = trajectory("controller_val", seed)
        x = mx.array(traj["x"][0, :1])
        for name, head in heads.items():
            for _ in range(3):
                mx.eval(head(x))
                mx.synchronize()
            measurements = []
            for _ in range(50):
                mx.synchronize()
                tic = time.perf_counter()
                mx.eval(head(x))
                mx.synchronize()
                measurements.append((time.perf_counter() - tic) * 1000)
            head_only.append(
                {
                    "seed": seed,
                    "head": name,
                    "raw_ms": measurements,
                    "median_ms": float(np.median(measurements)),
                    "scope": "Controller MLP only, cached observable features already on device; no feature construction/refiner/backbone.",
                }
            )

        def full(row, name):
            ids = mx.array([tokenize(tokenizer, row["prompt"])])
            h = hidden_forward(model, ids)[:, -1]
            base = h @ decoder.T
            return live_aux(
                name,
                settings.get(name, {}),
                h,
                base,
                decoder,
                refinement,
                onepass,
                heads,
                infos,
                old,
            )

        for name in variants:
            for _ in range(3):
                mx.eval(full(rows[0], name)[0])
                mx.synchronize()
        for i, row in enumerate(rows):
            guard(config)
            h, base = (
                mx.array(cached["h"][i : i + 1]),
                mx.array(cached["base"][i : i + 1]),
            )
            mx.eval(h, base)
            # Test streaming observable equality to the independently cached feature construction.
            from latent_loop.experiment03.heads import live_features

            previous = h
            errors = []
            for k in range(4):
                state = mx.array(traj["states"][k, i : i + 1])
                logit = mx.array(traj["logits"][k, i : i + 1])
                x = np.array(live_features(state, logit, previous, k + 1))
                errors.append(float(np.max(np.abs(x - traj["x"][k, i : i + 1]))))
                previous = state
            assert max(errors) < 0.001, errors
            parity.append(
                {"seed": seed, "id": row["id"], "features_max_abs": max(errors)}
            )
            # Interleave policy order to reduce drift from thermals/background load.
            samples = {v: {"full_ms": [], "aux_ms": [], "depth": []} for v in variants}
            for repeat in range(config["latency_repeats"]):
                order = np.random.default_rng(
                    seed * 10000 + i * 100 + repeat
                ).permutation(variants)
                for name in order:
                    mx.synchronize()
                    tic = time.perf_counter()
                    scores, depth = full(row, name)
                    mx.eval(scores)
                    mx.synchronize()
                    samples[name]["full_ms"].append((time.perf_counter() - tic) * 1000)
                    samples[name]["depth"].append(depth)
                    observed = np.array(scores)[0]
                    if name in raw["depths"]:
                        wanted = int(raw["depths"][name][i])
                        expected = np.asarray(raw["logits"])[wanted - 1, i]
                        assert depth == wanted, (seed, i, name, depth, wanted)
                    else:
                        expected = np.asarray(
                            raw[
                                "base_logits" if name == "frozen" else "one_pass_logits"
                            ]
                        )[i]
                    assert np.max(np.abs(observed - expected)) < 0.001, (seed, i, name)
                    mx.synchronize()
                    tic = time.perf_counter()
                    scores, auxdepth = live_aux(
                        name,
                        settings.get(name, {}),
                        h,
                        base,
                        decoder,
                        refinement,
                        onepass,
                        heads,
                        infos,
                        old,
                    )
                    mx.eval(scores)
                    mx.synchronize()
                    samples[name]["aux_ms"].append((time.perf_counter() - tic) * 1000)
                    assert auxdepth == depth
            records.extend(
                {
                    "seed": seed,
                    "id": row["id"],
                    "task": row["task"],
                    "length": int(cached["lengths"][i]),
                    "variant": name,
                    **samples[name],
                }
                for name in variants
            )
            if i % 6 == 0:
                print(f"full latency seed {seed}: {i + 1}/{len(rows)}", flush=True)
    summary = {}
    for name in variants:
        items = [r for r in records if r["variant"] == name]
        full = [np.median(r["full_ms"]) for r in items]
        aux = [np.median(r["aux_ms"]) for r in items]
        summary[name] = {
            "full_mean_of_item_medians_ms": float(np.mean(full)),
            "full_median_ms": float(np.median(full)),
            "full_p10_ms": float(np.quantile(full, 0.1)),
            "full_p90_ms": float(np.quantile(full, 0.9)),
            "aux_mean_of_item_medians_ms": float(np.mean(aux)),
            "mean_loops_timed_subset": float(np.mean([r["depth"][0] for r in items])),
        }
    # A measured reference increment expresses VOC's nat/loop cost as lambda * milliseconds.
    reference_ms = (
        summary["fixed_4"]["aux_mean_of_item_medians_ms"]
        - summary["fixed_1"]["aux_mean_of_item_medians_ms"]
    )
    reference_ms /= 3
    voc_mapping = {
        str(seed): {
            "selected_nats_per_loop": json.loads(
                (EXP / f"selection_seed{seed}.json").read_text()
            )["settings"]["voc"]["cost"],
            "reference_incremental_ms": reference_ms,
        }
        for seed in config["seeds"]
    }
    for v in voc_mapping.values():
        v["lambda_nats_per_ms"] = (
            v["selected_nats_per_loop"] / reference_ms if reference_ms > 0 else None
        )
    atomic_json(
        OUT / "latency.json",
        {
            "completed_utc": utc(),
            "environment": env,
            "model_revision": manifest["revision"],
            "scope": "Full: tokenization, ID transfer, frozen backbone, four-label projection, live refiner/probe/stopping, final Metal synchronization. Model load excluded. Auxiliary: cached h/base already on device, live refinement/stopping and final sync.",
            "selection": "First 24 balanced controller_val examples, prescribed before fitting; shared across seeds; no labels used for timing selection.",
            "repeats": config["latency_repeats"],
            "warmup_calls_per_variant_and_seed": 3,
            "summary": summary,
            "head_only": head_only,
            "records": records,
            "streaming_parity": parity,
            "voc_cost_conversion": voc_mapping,
            "command_seconds": time.perf_counter() - started,
            "peak_mlx_bytes": mx.get_peak_memory(),
            "peak_process_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)
