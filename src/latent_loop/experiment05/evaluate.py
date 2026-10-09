"""Prospective configuration freeze, once-only final evaluation, and measured timing."""

import argparse
import gzip
import json
import resource
import shutil
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from scipy.special import logsumexp, softmax

from latent_loop.backbone import environment, get_model
from latent_loop.experiment04a.data import TASKS, canonical, sha
from latent_loop.experiment05.data import CACHE, EXP, FINAL, OUT, load
from latent_loop.experiment05.model import Thinker
from latent_loop.experiment05.train import baseline, guard, make_model, predict, utc
from latent_loop.experiment05.validate import batch
from latent_loop.training import atomic_json


def checkpoint(condition, seed):
    return CACHE / "checkpoints" / f"main_{condition}_seed{seed}" / "best.safetensors"


def freeze(config):
    if (EXP / "frozen_protocol.json").exists():
        verify(config)
        return
    if (EXP / "lockbox_opened.json").exists():
        raise RuntimeError("Cannot select after final consumption")
    fits = {}
    scores = {}
    for condition in config["conditions"]:
        fits[condition] = {}
        for seed in config["seeds"]:
            path = OUT / "training" / f"main_{condition}_seed{seed}.json"
            fit = json.loads(path.read_text())
            assert fit["complete"] and fit["signature"]["config_sha256"] == sha(EXP / "config.json")
            record = next(r for r in fit["epochs"] if r["epoch"] == fit["best_epoch"])
            chosen_depth = "1" if condition.startswith("one_pass") else str(config["primary_inference_depth"])
            fits[condition][str(seed)] = {"checkpoint_sha256": sha(checkpoint(condition, seed)),
                                          "training_record_sha256": sha(path), "best_epoch": fit["best_epoch"],
                                          "validation": record["validation_by_depth"],
                                          "selection_nll": record["validation_by_depth"][chosen_depth]["nll"]}
        scores[condition] = float(np.mean([r["selection_nll"] for r in fits[condition].values()]))
    recurrent = min(("recurrent", "reinjection", "multi_depth"), key=lambda c: (scores[c], c))
    onepass = min(("one_pass", "one_pass_compute"), key=lambda c: (scores[c], c))
    sources = [EXP / "config.json", EXP / "dataset_manifest.json", EXP / "plan.md", EXP / "class_prior_diagnostic.json", *[p for p in Path("src/latent_loop/experiment05").glob("*.py") if p.name != "report.py"]]
    value = {"frozen_utc": utc(), "selected_recurrent_family": recurrent, "selected_one_pass_family": onepass,
             "family_selection_rule": "Mean validation NLL at fixed primary inference depth across seeds; no final data used",
             "primary_inference_depth": config["primary_inference_depth"], "validation_selection_scores": scores,
             "fits": fits, "source_hashes": {str(p): sha(p) for p in sources},
             "stress_rule": "24 fresh-ID problems, four per task, lexicographic instance IDs; depths 1..8 diagnostic only",
             "reinjection_ablation_rule": "Fixed/no gate at inference on all F/G final trajectories; same trained weights, no refitting"}
    atomic_json(EXP / "frozen_protocol.json", value)
    print("Protocol frozen", recurrent, onepass, scores, flush=True)


def verify(config):
    frozen = json.loads((EXP / "frozen_protocol.json").read_text())
    for p, expected in frozen["source_hashes"].items():
        assert sha(p) == expected, f"Frozen source changed: {p}"
    for condition, seeds in frozen["fits"].items():
        for seed, record in seeds.items():
            assert sha(checkpoint(condition, int(seed))) == record["checkpoint_sha256"]
    return frozen


def write_trace(path, rows, trace, frozen_logits):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.GzipFile(filename=str(temporary), mode="wb", mtime=0) as stream:
        labels = np.array([r["label"] for r in rows])
        p0 = softmax(frozen_logits, axis=-1)
        for i, row in enumerate(rows):
            record = {"id": row["id"], "task": row["task"], "generator_id": row["generator_id"],
                      "difficulty_stratum": row["difficulty_stratum"], "reasoning_depth": row["reasoning_depth"],
                      "label": int(labels[i]), "initial_prediction": int(frozen_logits[i].argmax()),
                      "initial_logits": frozen_logits[i].tolist(), "initial_correct_answer_probability": float(p0[i, labels[i]]),
                      "depths": {}}
            for k, values in trace.items():
                logits = values["logits"][i]
                p = softmax(logits)
                order = np.sort(p)
                record["depths"][str(k)] = {"prediction": int(logits.argmax()), "logits": logits.tolist(),
                                             "conditional_loss": float(logsumexp(logits) - logits[labels[i]]),
                                             "correct_answer_probability": float(p[labels[i]]), "confidence": float(p.max()),
                                             "margin": float(order[-1] - order[-2]), "entropy": float(-np.sum(p * np.log(np.maximum(p, 1e-12)))),
                                             "state_norm": float(values["state_norm"][i]), "state_distance": float(values["state_distance"][i]),
                                             "state_drift": float(values["state_drift"][i])}
            stream.write((canonical(record) + "\n").encode())
    temporary.replace(path)


def final(config):
    frozen = verify(config)
    receipt = EXP / "lockbox_opened.json"
    if not receipt.exists():
        atomic_json(receipt, {"opened_utc": utc(), "protocol_sha256": sha(EXP / "frozen_protocol.json"),
                              "final_selection_allowed": False})
    else:
        assert json.loads(receipt.read_text())["protocol_sha256"] == sha(EXP / "frozen_protocol.json")
    environment(config["memory_gib"])
    base, tokenizer, _ = get_model()
    # Reference labels and frozen decisions are evaluated only after protocol lock.
    baseline(config, base, tokenizer, FINAL, final=True)
    for split in FINAL:
        rows = load(split, allow_final=True)
        ref = np.load(CACHE / "reference" / f"{split}.npz")["logits"]
        public_dataset = OUT / "datasets" / f"{split}.jsonl.gz"
        public_dataset.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(json.loads((EXP / "dataset_manifest.json").read_text())["splits"][split]["path"], public_dataset)
        # Both plain recurrence and a bounded, initialized gate control are frozen.
        for condition, raw, injection in (("untrained_plain", True, "none"), ("untrained_bounded", False, "fixed")):
            target = OUT / "trajectories" / f"{condition}_{split}.jsonl.gz"
            if not target.exists():
                model = Thinker(base, [tokenizer.encode(x, add_special_tokens=False)[0] for x in "ABCD"], config["middle_start"], config["middle_count"], 0, injection,
                                config["update_initial"], config["norm_cap"])
                trace = predict(model, tokenizer, rows, config, raw_recurrence=raw)
                write_trace(target, rows, trace, ref)
        for seed in config["seeds"]:
            for condition in config["conditions"]:
                target = OUT / "trajectories" / f"{condition}_{split}_seed{seed}.jsonl.gz"
                if not target.exists():
                    model = make_model(base, tokenizer, config, condition, seed)
                    model.load_weights(str(checkpoint(condition, seed)), strict=False)
                    model.eval()
                    depths = (1,) if condition.startswith("one_pass") else (1, 2, 3, 4)
                    trace = predict(model, tokenizer, rows, config, depths)
                    write_trace(target, rows, trace, ref)
                    print("Final", split, condition, seed, flush=True)
                if condition in ("reinjection", "multi_depth"):
                    for mode in ("none", "fixed"):
                        ablation = OUT / "ablations" / f"{condition}_{mode}_{split}_seed{seed}.jsonl.gz"
                        if not ablation.exists():
                            model = make_model(base, tokenizer, config, condition, seed)
                            model.load_weights(str(checkpoint(condition, seed)), strict=False)
                            trace = predict(model, tokenizer, rows, config, reinjection_override=mode)
                            write_trace(ablation, rows, trace, ref)
        if split == "fresh_id":
            chosen = sorted(i for task in TASKS for i in sorted([i for i, r in enumerate(rows) if r["task"] == task], key=lambda j: rows[j]["id"])[:4])
            for seed in config["seeds"]:
                for condition in ("recurrent", "reinjection", "multi_depth"):
                    target = OUT / "stress" / f"{condition}_seed{seed}.jsonl.gz"
                    if not target.exists():
                        model = make_model(base, tokenizer, config, condition, seed)
                        model.load_weights(str(checkpoint(condition, seed)), strict=False)
                        trace = predict(model, tokenizer, [rows[i] for i in chosen], config, tuple(range(1, 9)))
                        write_trace(target, [rows[i] for i in chosen], trace, ref[chosen])
    atomic_json(OUT / "final_execution.json", {"completed_utc": utc(), "peak_mlx_bytes": mx.get_peak_memory(),
                                              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                              "protocol_sha256": sha(EXP / "frozen_protocol.json"),
                                              "final_evaluation_selection_changes": False})


def timing(config):
    frozen = verify(config)
    environment(config["memory_gib"])
    base, tokenizer, _ = get_model()
    rows = load("val")
    selected = [i for task in TASKS for i in [i for i, r in enumerate(rows) if r["task"] == task][:4]]
    records = []
    conditions = ("one_pass", "one_pass_compute", "recurrent", "reinjection", "multi_depth", "frozen")
    start = time.perf_counter()
    # Order is randomized per seed to reduce order/thermal confounding.
    for seed in config["seeds"]:
        for condition in np.random.default_rng(seed + 500).permutation(conditions):
            if condition == "frozen":
                model = Thinker(base, [tokenizer.encode(x, add_special_tokens=False)[0] for x in "ABCD"], config["middle_start"], config["middle_count"], rank=0)
            else:
                model = make_model(base, tokenizer, config, condition, seed)
                model.load_weights(str(checkpoint(condition, seed)), strict=False)
            model.eval()
            for depth in ((1,) if condition.startswith("one_pass") or condition == "frozen" else (1, 2, 3, 4)):
                for bs in (1, 8):
                    for j in range(0, len(selected), bs):
                        group = selected[j:j + bs]
                        items = [rows[i] for i in group]
                        def infer():
                            ids, lengths, _ = batch(tokenizer, items)
                            result = model(ids, lengths, (depth,))[depth]["logits"]
                            mx.eval(result)
                            mx.synchronize()
                            return np.array(result), int(ids.shape[1])
                        for _ in range(config["timing_warmups"]):
                            infer()
                        times = []
                        reference_logits, _ = infer()
                        for _ in range(config["timing_repeats"]):
                            guard(config)
                            mx.synchronize()
                            tic = time.perf_counter()
                            actual, max_length = infer()
                            elapsed = 1000 * (time.perf_counter() - tic)
                            assert np.array_equal(actual, reference_logits)
                            times.append(elapsed)
                        records.append({"condition": str(condition), "seed": seed, "depth": depth, "batch_size": bs,
                                        "ids": [r["id"] for r in items], "max_prompt_tokens": max_length,
                                        "raw_batch_ms": times, "median_batch_ms": float(np.median(times)),
                                        "amortized_ms_per_item": float(np.median(times) / len(items)),
                                        "subset_accuracy": float(np.mean(actual.argmax(-1) == [r["label"] for r in items]))})
    atomic_json(OUT / "latency.json", {"scope": "Full tokenization, padded input arrays, encoder, shared recurrent middle block, late decoder, four-label projection, Metal synchronization and host output conversion; model loading excluded",
                                       "dataset": "24 balanced validation items, same prompts for all conditions and seeds",
                                       "timing_warning": "Batch 8 amortization is throughput, not single-request latency; final accuracy uses a different independent population",
                                       "warmups": config["timing_warmups"], "repeats": config["timing_repeats"], "records": records,
                                       "command_seconds": time.perf_counter() - start, "peak_mlx_bytes": mx.get_peak_memory(),
                                       "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print("Full synchronized timing complete", len(records), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "final", "timing"))
    args = parser.parse_args()
    config = json.loads((EXP / "config.json").read_text())
    globals()[args.stage](config)


if __name__ == "__main__":
    main()
