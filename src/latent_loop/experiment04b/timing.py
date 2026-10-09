"""4A timing mechanics extended to native D, calibration, stacking and selectors."""

import json
import os
import resource
import time

import numpy as np

from latent_loop.experiment04a.data import sha
from latent_loop.experiment04a.timing import timing_indices
from latent_loop.experiment04b.common import OUT, guard, utc, verify_freeze
from latent_loop.experiment04b.data import features, load, trajectory
from latent_loop.experiment04b.evaluate import infer, paths
from latent_loop.experiment04b.runtime import components, live, runtime_spec
from latent_loop.training import atomic_json

VARIANTS = [
    "frozen",
    "one_pass",
    "fixed_1",
    "fixed_2",
    "fixed_3",
    "fixed_4",
    "confidence",
    "value",
    "legacy_gain",
    "d_direct",
    "d_utility",
    "stack_CV",
    "stack_CD",
    "stack_CVD",
    "fixed_4:max_C",
    "fixed_4:ranker_C",
    "fixed_4:ranker_CV",
    "fixed_4:ranker_CD",
    "fixed_4:ranker_CVD",
    "d_direct:max_C",
    "d_direct:ranker_CVD",
    "stack_CVD:ranker_CVD",
    "validation_best",
]


def measure(config, *, check_protocol=True):
    import mlx.core as mx

    from latent_loop.backbone import environment, get_model, hidden_forward, tokenize

    if check_protocol:
        verify_freeze(config)
    env = environment(config["memory_gib"])
    model, tokenizer, manifest = get_model()
    rows = load("controller_val")
    indices = timing_indices(rows, config["timing_examples"])
    decoder = mx.array(np.load("artifacts/features/decoder.npz")["decoder"])
    cached = features("controller_val")
    signature = {
        "protocol": sha("experiments/experiment_04b/frozen_protocol.json"),
        "source": sha(__file__),
        "examples": config["timing_examples"],
        "repeats": config["timing_repeats"],
        "batch_sizes": config["timing_batch_sizes"],
    }
    partial = OUT / "latency_partial.json"
    records, heads = [], []
    if partial.exists():
        previous = json.loads(partial.read_text())
        assert previous["signature"] == signature, "Incompatible latency resume"
        records = previous["records"]
    started = time.perf_counter()
    load_start = list(os.getloadavg())

    def encode(items):
        encoded = [tokenize(tokenizer, r["prompt"]) for r in items]
        lengths = [len(x) for x in encoded]
        ids = mx.array(
            [
                x
                + [tokenizer.pad_token_id or tokenizer.eos_token_id]
                * (max(lengths) - len(x))
                for x in encoded
            ]
        )
        h = hidden_forward(model, ids)[mx.arange(len(items)), mx.array(lengths) - 1]
        return h, h @ decoder.T

    for seed in config["parent_seeds"]:
        models, chosen = components(config, seed)
        t = trajectory("controller_val", seed)
        scores, variants, quality = infer(config, seed, t, rows)
        expected = paths(scores, quality, chosen, [r["id"] for r in rows])
        for name in ["confidence", "regression", chosen["D_model"], "ranker_CVD"]:
            x = mx.array(t["x"][0, :1])
            if name.startswith("ranker_"):
                x = mx.concatenate([x, mx.zeros((1, 4))], axis=-1)
            for _ in range(3):
                mx.eval(models[name](x))
                mx.synchronize()
            samples = []
            for _ in range(30):
                mx.synchronize()
                tic = time.perf_counter()
                mx.eval(models[name](x))
                mx.synchronize()
                samples.append((time.perf_counter() - tic) * 1000)
            heads.append(
                {
                    "seed": seed,
                    "model": name,
                    "raw_ms": samples,
                    "median_ms": float(np.median(samples)),
                }
            )
        for batch_size in config["timing_batch_sizes"]:
            batches = [
                indices[i : i + batch_size] for i in range(0, len(indices), batch_size)
            ]
            for name in VARIANTS:
                for _ in range(3):
                    h, b = encode([rows[i] for i in batches[0]])
                    out, paid, selected, _ = live(name, h, b, decoder, models, chosen)
                    mx.eval(out, paid, selected)
                    mx.synchronize()
            for bi, batch in enumerate(batches):
                done = [
                    r
                    for r in records
                    if r["seed"] == seed
                    and r["batch_size"] == batch_size
                    and r["batch_index"] == bi
                ]
                if len(done) == len(VARIANTS):
                    continue
                guard(config, 25)
                items = [rows[i] for i in batch]
                hc, bc = mx.array(cached["h"][batch]), mx.array(cached["base"][batch])
                mx.eval(hc, bc)
                collected = {
                    name: {
                        "full_ms": [],
                        "aux_ms": [],
                        "paid": [],
                        "selected": [],
                        "work": [],
                    }
                    for name in VARIANTS
                }
                for repeat in range(config["timing_repeats"]):
                    for name in np.random.default_rng(
                        seed * 10000 + batch_size * 1000 + bi * 10 + repeat
                    ).permutation(VARIANTS):
                        mx.synchronize()
                        tic = time.perf_counter()
                        h, b = encode(items)
                        out, paid, selected, work = live(
                            name, h, b, decoder, models, chosen
                        )
                        mx.eval(out, paid, selected)
                        mx.synchronize()
                        hostout, hp, hs = (
                            np.array(out),
                            np.array(paid),
                            np.array(selected),
                        )
                        elapsed = (time.perf_counter() - tic) * 1000
                        if name in ("frozen", "one_pass"):
                            wantpaid = np.full(len(batch), int(name == "one_pass"))
                            wantsel = wantpaid
                            wantout = (
                                cached["base"][batch]
                                if name == "frozen"
                                else t["one_pass"][batch]
                            )
                        else:
                            pol, sel = runtime_spec(name, chosen)
                            wantpaid = expected[pol]["paid"][batch]
                            wantsel = expected[pol][sel][batch]
                            wantout = t["logits"][wantsel - 1, batch]
                        assert np.array_equal(hp, wantpaid), (
                            seed,
                            name,
                            "paid",
                            hp,
                            wantpaid,
                        )
                        assert np.array_equal(hs, wantsel), (
                            seed,
                            name,
                            "selected",
                            hs,
                            wantsel,
                        )
                        assert np.max(np.abs(hostout - wantout)) < 0.002, (
                            seed,
                            name,
                            "logits",
                        )
                        assert work["refiner_rows"] == int(hp.sum())
                        collected[name]["full_ms"].append(elapsed)
                        collected[name]["paid"].append(hp.tolist())
                        collected[name]["selected"].append(hs.tolist())
                        collected[name]["work"].append(work)
                        mx.synchronize()
                        tic = time.perf_counter()
                        out, paid, selected, _ = live(
                            name, hc, bc, decoder, models, chosen
                        )
                        mx.eval(out, paid, selected)
                        mx.synchronize()
                        np.array(out)
                        np.array(paid)
                        np.array(selected)
                        collected[name]["aux_ms"].append(
                            (time.perf_counter() - tic) * 1000
                        )
                records.extend(
                    {
                        "seed": seed,
                        "variant": name,
                        "batch_size": len(batch),
                        "batch_index": bi,
                        "instance_ids": [r["id"] for r in items],
                        "tasks": [r["task"] for r in items],
                        "prompt_lengths": cached["lengths"][batch].tolist(),
                        **v,
                    }
                    for name, v in collected.items()
                )
                # Preserve completed batches if an interrupt or budget boundary is reached.
                atomic_json(
                    OUT / "latency_partial.json",
                    {
                        "records": records,
                        "completed_utc": utc(),
                        "signature": signature,
                    },
                )
            print(f"4B timing parent {seed} batch {batch_size} complete", flush=True)
    summary = {}
    for size in config["timing_batch_sizes"]:
        summary[str(size)] = {}
        for name in VARIANTS:
            rr = [
                r for r in records if r["batch_size"] == size and r["variant"] == name
            ]
            med = np.array([np.median(r["full_ms"]) for r in rr])
            aux = np.array([np.median(r["aux_ms"]) for r in rr])
            summary[str(size)][name] = {
                "mean_batch_median_ms": float(med.mean()),
                "amortized_ms_per_item": float(med.mean() / size),
                "p10_batch_ms": float(np.quantile(med, 0.1)),
                "p90_batch_ms": float(np.quantile(med, 0.9)),
                "aux_mean_batch_median_ms": float(aux.mean()),
                "mean_paid_loops": float(
                    np.mean([d for r in rr for d in r["paid"][0]])
                ),
                "mean_selected_depth": float(
                    np.mean([d for r in rr for d in r["selected"][0]])
                ),
                "mean_host_barriers_per_batch": float(
                    np.mean([r["work"][0]["midloop_host_barriers"] for r in rr])
                ),
                "work_counts_per_item": {
                    k: float(np.mean([r["work"][0][k] / size for r in rr]))
                    for k in (
                        "refiner_rows",
                        "C_rows",
                        "V_rows",
                        "D_rows",
                        "ranker_rows",
                        "policy_rows",
                    )
                },
            }
    atomic_json(
        OUT / "latency.json",
        {
            "completed_utc": utc(),
            "summary": summary,
            "records": records,
            "controller_only": heads,
            "scope": "Tokenization, input arrays, right-padded frozen backbone, candidate decoding, native heads/calibration/policies/selection, active compaction and host stopping barriers, final Metal synchronization and output conversion. Model loading excluded. Auxiliary-only uses h/base on device.",
            "validation_ids": [rows[i]["id"] for i in indices],
            "warmups": 3,
            "repeats": config["timing_repeats"],
            "parity_verified_every_repeat": True,
            "budget": "Paid loops equal executed refiner rows per item; selecting earlier never refunds work.",
            "environment": env,
            "model_revision": manifest["revision"],
            "peak_mlx_bytes": mx.get_peak_memory(),
            "peak_process_rss_bytes": resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            "start_load_average": load_start,
            "end_load_average": list(os.getloadavg()),
            "command_seconds": time.perf_counter() - started,
        },
    )
    (OUT / "latency_partial.json").unlink(missing_ok=True)
