"""Resumable parameter-efficient fits and public validation measurements."""

import datetime
import gzip
import hashlib
import json
import math
import os
import resource
import signal
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from latent_loop.backbone import environment, get_model
from latent_loop.experiment04a.data import sha
from latent_loop.experiment05.data import CACHE, EXP, OUT, load
from latent_loop.experiment05.model import Thinker, objective
from latent_loop.experiment05.validate import batch
from latent_loop.training import atomic_json

INTERRUPTED = False


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def guard(config):
    if INTERRUPTED:
        raise InterruptedError("Stopped after preserving the last completed batch")
    elapsed = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(config["started_utc"])).total_seconds()
    if elapsed > config["max_hours"] * 3600 - config["report_reserve_minutes"] * 60:
        raise TimeoutError("Training budget reached; evaluation/report reserve retained")


def make_model(base, tokenizer, config, condition, seed):
    mx.random.seed(seed)
    labels = [tokenizer.encode(x, add_special_tokens=False)[0] for x in "ABCD"]
    model = Thinker(base, labels, config["middle_start"], config["middle_count"], config["rank"],
                    "learned" if condition in ("reinjection", "multi_depth") else "none",
                    config["update_initial"], config["norm_cap"])
    if condition.startswith("one_pass"):
        model.freeze(keys=["update_logit"], recurse=False)
    return model


def compress(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as f:
        f.write(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())


def reference(base, tokenizer, rows, split, config):
    target = CACHE / "reference" / f"{split}.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        value = np.load(target)
        assert len(value["logits"]) == len(rows)
        return value["logits"], value["hidden"]
    # Strip any previous condition's adapters before extracting the native state.
    Thinker(base, [0, 1, 2, 3], config["middle_start"], config["middle_count"], rank=0)
    labels = [tokenizer.encode(x, add_special_tokens=False)[0] for x in "ABCD"]
    decoder = base.model.embed_tokens.weight[mx.array(labels)]
    logits, hidden = [], []
    for begin in range(0, len(rows), config["eval_batch_size"]):
        guard(config)
        ids, lengths, _ = batch(tokenizer, rows[begin:begin + config["eval_batch_size"]])
        h = base.model(ids)[mx.arange(len(lengths)), lengths - 1]
        l = h @ decoder.T
        mx.eval(h, l)
        hidden.append(np.array(h))
        logits.append(np.array(l))
    value = {"logits": np.concatenate(logits), "hidden": np.concatenate(hidden)}
    np.savez_compressed(target, **value)
    return value["logits"], value["hidden"]


def metrics(logits, labels):
    from scipy.special import logsumexp, softmax
    logits = np.asarray(logits)
    p = softmax(logits, axis=-1)
    prediction = logits.argmax(-1)
    correct = prediction == labels
    conf = p.max(-1)
    ece = 0.0
    for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        sel = (conf >= a) & (conf < b if b < 1 else conf <= b)
        if sel.any():
            ece += sel.mean() * abs(correct[sel].mean() - conf[sel].mean())
    return {"accuracy": float(correct.mean()), "nll": float(np.mean(logsumexp(logits, axis=-1) - logits[np.arange(len(labels)), labels])),
            "ece": float(ece), "multiclass_brier": float(np.mean(np.sum((p - np.eye(4)[labels])**2, axis=-1)))}


def predict(model, tokenizer, rows, config, depths=(1, 2, 3, 4), **kwargs):
    trace = {k: {f: [] for f in ("logits", "state_norm", "state_distance", "state_drift")} for k in depths}
    for begin in range(0, len(rows), config["eval_batch_size"]):
        guard(config)
        ids, lengths, _ = batch(tokenizer, rows[begin:begin + config["eval_batch_size"]])
        result = model(ids, lengths, depths, **kwargs)
        mx.eval(result)
        for k in depths:
            for field in trace[k]:
                trace[k][field].append(np.array(result[k][field]))
    return {k: {f: np.concatenate(v) for f, v in values.items()} for k, values in trace.items()}


def baseline(config, base, tokenizer, splits=("train", "val"), final=False):
    from latent_loop.models import Refiner
    label_ids = [tokenizer.encode(x, add_special_tokens=False)[0] for x in "ABCD"]
    decoder = base.model.embed_tokens.weight[mx.array(label_ids)]
    summary = {}
    for split in splits:
        rows = load(split, allow_final=final)
        labels = np.array([r["label"] for r in rows])
        logits, hidden = reference(base, tokenizer, rows, split, config)
        old = {}
        hashes = {}
        for seed in config["seeds"]:
            path = Path(f"artifacts/checkpoints/feedforward_seed{seed}/best.safetensors")
            if not path.exists():
                raise FileNotFoundError("Historical one-pass checkpoint missing; regenerate the pilot first")
            head = Refiner(hidden.shape[-1], 64, False)
            head.load_weights(str(path))
            head.freeze()
            old[str(seed)] = np.array(head(mx.array(hidden), mx.array(logits), decoder, depth=1)).tolist()
            hashes[str(seed)] = sha(path)
        record = {"ids": [r["id"] for r in rows], "labels": labels.tolist(), "frozen_logits": logits.tolist(),
                  "historical_one_pass_logits": old, "historical_checkpoint_sha256": hashes}
        compress(OUT / "baselines" / f"{split}.json.gz", record)
        result = metrics(logits, labels)
        result["tasks"] = {task: metrics(logits[[i for i, r in enumerate(rows) if r["task"] == task]], labels[[i for i, r in enumerate(rows) if r["task"] == task]]) for task in sorted({r["task"] for r in rows})}
        summary[split] = result
    atomic_json(OUT / ("final_baseline_competence.json" if final else "baseline_competence.json"), summary)
    print("Frozen competence", {s: v["accuracy"] for s, v in summary.items()}, flush=True)
    return summary


def save_checkpoint(model, optimizer, folder, state):
    folder.mkdir(parents=True, exist_ok=True)
    for filename, arrays in (("last.safetensors", model.trainable_parameters()), ("optimizer.safetensors", optimizer.state)):
        temporary = folder / (filename + ".tmp.safetensors")
        mx.save_safetensors(str(temporary), dict(tree_flatten(arrays)))
        temporary.replace(folder / filename)
    atomic_json(folder / "state.json", state)


def fit(config, base, tokenizer, condition, seed, pilot=False):
    guard(config)
    if not json.loads((OUT / "implementation_validation.json").read_text())["all_passed"]:
        raise RuntimeError("Implementation validation must pass before fitting")
    if (EXP / "lockbox_opened.json").exists():
        raise RuntimeError("No fitting after final lockbox consumption")
    train = load("train")[:config["pilot_train"]] if pilot else load("train")
    val = load("val")[:config["pilot_val"]] if pilot else load("val")
    refs = np.load(CACHE / "reference/train.npz")["logits"][:len(train)]
    model = make_model(base, tokenizer, config, condition, seed)
    model.train()
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    epochs = 1 if pilot else config["compute_control_epochs"] if condition == "one_pass_compute" else config["epochs"]
    mode = "pilot" if pilot else "main"
    folder = CACHE / "checkpoints" / f"{mode}_{condition}_seed{seed}"
    log = OUT / "training" / f"{mode}_{condition}_seed{seed}.json"
    stamp = {"config_sha256": sha(EXP / "config.json"), "dataset_sha256": sha(EXP / "dataset_manifest.json"),
             "source_sha256": {str(p): sha(p) for p in [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("data.py")]},
             "condition": condition, "seed": seed, "mode": mode}
    # Public signatures contain portable source names, never absolute paths.
    stamp["source_sha256"] = {"src/latent_loop/experiment05/" + Path(k).name: v for k, v in stamp["source_sha256"].items()}
    batches = math.ceil(len(train) / config["batch_size"])
    state = {"signature": stamp, "next_epoch": 0, "next_batch": 0, "best_validation_nll": None,
             "best_epoch": None, "complete": False, "epochs": [], "batch_records": [], "trainable_parameters": model.parameter_counts(),
             "training_seconds": 0.0, "examples_processed": 0}
    if (folder / "state.json").exists():
        state = json.loads((folder / "state.json").read_text())
        assert state["signature"] == stamp, "Changed configuration/data/source cannot resume the same checkpoint"
        if state["complete"]:
            print("Completed fit reused:", mode, condition, seed, flush=True)
            return state
        model.load_weights(str(folder / "last.safetensors"), strict=False)
        optimizer.state = tree_unflatten(list(mx.load(str(folder / "optimizer.safetensors")).items()))
    fn = nn.value_and_grad(model, objective)
    print("Fit", mode, condition, seed, "parameters", state["trainable_parameters"]["trainable"], flush=True)
    try:
        for epoch in range(state["next_epoch"], epochs):
            order = np.random.default_rng(seed + epoch * 173).permutation(len(train))
            first_batch = state["next_batch"] if epoch == state["next_epoch"] else 0
            for bi in range(first_batch, batches):
                guard(config)
                tic = time.perf_counter()
                ix = order[bi * config["batch_size"]:(bi + 1) * config["batch_size"]]
                ids, lengths, labels = batch(tokenizer, [train[i] for i in ix])
                if condition.startswith("one_pass"):
                    depths, weights = (1,), (1,)
                elif condition == "multi_depth":
                    depths, weights = (1, 2, 3, 4), tuple(config["depth_weights"])
                else:
                    depth = int(np.random.default_rng(seed * 100000 + epoch * 1000 + bi).integers(1, 5))
                    depths, weights = (depth,), (1,)
                value, gradients = fn(model, ids, lengths, labels, mx.array(refs[ix]), depths, weights, config["preservation_alpha"])
                gradients, norm = optim.clip_grad_norm(gradients, 1.0)
                mx.eval(value, norm)
                if not np.isfinite(float(value.item())) or not np.isfinite(float(norm.item())):
                    raise FloatingPointError("Nonfinite loss/gradient; measurements preserved")
                optimizer.update(model, gradients)
                mx.eval(model.trainable_parameters(), optimizer.state)
                mx.synchronize()
                seconds = time.perf_counter() - tic
                state["training_seconds"] += seconds
                state["examples_processed"] += len(ix)
                state["batch_records"].append({"epoch": epoch, "batch": bi, "loss": float(value.item()), "gradient_norm": float(norm.item()),
                                               "seconds": seconds, "depths": list(depths), "peak_mlx_bytes": mx.get_peak_memory()})
                state["next_epoch"], state["next_batch"] = epoch, bi + 1
                if (bi + 1) % config["checkpoint_every_batches"] == 0:
                    save_checkpoint(model, optimizer, folder, state)
                    atomic_json(log, state)
            guard(config)
            depths = (1,) if condition.startswith("one_pass") else (1, 2, 3, 4)
            model.eval()
            trace = predict(model, tokenizer, val, config, depths)
            labels = np.array([r["label"] for r in val])
            measured = {str(k): metrics(trace[k]["logits"], labels) for k in depths}
            score = float(np.mean([r["nll"] for r in measured.values()]))
            epoch_batches = [r for r in state["batch_records"] if r["epoch"] == epoch]
            state["epochs"].append({"epoch": epoch, "train_loss": float(np.mean([r["loss"] for r in epoch_batches])),
                                    "validation_mean_nll": score, "validation_by_depth": measured,
                                    "state_norm_by_depth": {str(k): float(np.mean(trace[k]["state_norm"])) for k in depths},
                                    "state_drift_by_depth": {str(k): float(np.mean(trace[k]["state_drift"])) for k in depths}})
            if state["best_validation_nll"] is None or score < state["best_validation_nll"]:
                state["best_validation_nll"], state["best_epoch"] = score, epoch
                folder.mkdir(parents=True, exist_ok=True)
                mx.save_safetensors(str(folder / "best.safetensors"), dict(tree_flatten(model.trainable_parameters())))
            state["next_epoch"], state["next_batch"] = epoch + 1, 0
            state["complete"] = epoch + 1 == epochs
            state["peak_mlx_bytes"] = mx.get_peak_memory()
            state["peak_process_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            save_checkpoint(model, optimizer, folder, state)
            atomic_json(log, state)
            print(f"{mode} {condition} seed={seed} epoch={epoch + 1}/{epochs} train={state['epochs'][-1]['train_loss']:.4f} val={score:.4f} accuracy={measured} elapsed_fit={state['training_seconds']:.1f}s", flush=True)
            model.train()
    finally:
        save_checkpoint(model, optimizer, folder, state)
        atomic_json(log, state)
    return state


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("baseline", "pilot", "main"))
    parser.add_argument("--config", default=str(EXP / "config.json"))
    parser.add_argument("--condition")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    def interrupted(signum, frame):
        global INTERRUPTED
        INTERRUPTED = True
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    environment(config["memory_gib"])
    base, tokenizer, _ = get_model()
    if args.stage == "baseline":
        baseline(config, base, tokenizer)
    else:
        if not (CACHE / "reference/train.npz").exists():
            baseline(config, base, tokenizer)
        conditions = [args.condition] if args.condition else ["one_pass", "multi_depth"] if args.stage == "pilot" else config["conditions"]
        seeds = [args.seed] if args.seed is not None else [11] if args.stage == "pilot" else config["seeds"]
        for seed in seeds:
            for condition in conditions:
                fit(config, base, tokenizer, condition, seed, args.stage == "pilot")


if __name__ == "__main__":
    main()
