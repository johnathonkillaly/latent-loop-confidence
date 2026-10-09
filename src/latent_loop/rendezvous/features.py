"""Frozen Qwen extraction, with resumable, fingerprinted private feature caches."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from latent_loop.backbone import MODEL_ID, REVISION, environment, get_model, tokenize
from .data import ROOT, read_rows

CACHE = Path("artifacts/experiment_05e/features")


def normalize(a):
    return a / np.maximum(np.sqrt(np.mean(a * a, axis=-1, keepdims=True)), 1e-6)


def design(h, candidate, label, logits):
    centered = logits - logits.mean(axis=-1, keepdims=True)
    lp = centered - np.log(np.exp(centered - centered.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)) - centered.max(axis=-1, keepdims=True)
    p = np.exp(lp)
    entropy = -(p * lp).sum(axis=-1)
    sorted_z = np.sort(logits, axis=-1)
    margin = sorted_z[:, -1] - sorted_z[:, -2]
    scalars = np.stack([np.clip(centered / 5, -5, 5), p, np.clip(lp / 5, -5, 0), np.repeat(entropy[:, None] / 2, 4, axis=1), np.repeat(margin[:, None] / 5, 4, axis=1)], axis=-1)
    return np.concatenate([np.repeat(normalize(h)[:, None, :], 4, axis=1), normalize(candidate), normalize(label), scalars], axis=-1).astype(np.float32)


def load_features(split):
    rows = read_rows(ROOT / "datasets" / f"{split}.jsonl.gz")
    files = sorted((CACHE / split).glob("*.npz"))
    chunks = [dict(np.load(p)) for p in files]
    if sum(len(c["h"]) for c in chunks) != len(rows):
        raise RuntimeError(f"Incomplete extraction: {split}")
    combined = {k: np.concatenate([c[k] for c in chunks]) for k in ("h", "candidate", "label", "logits", "greedy", "lengths")}
    combined["x"] = design(combined["h"], combined["candidate"], combined["label"], combined["logits"])
    combined["y"] = np.array([[i in r["optimal"] for i in range(4)] for r in rows], dtype=np.float32)
    return rows, combined


def extract(config, splits):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    final = [s for s in splits if s not in ("pilot", "train", "validation")]
    if final:
        if not (ROOT / "protocol_freeze.json").exists():
            raise RuntimeError("Freeze the prospective protocol before final extraction")
        opening_path = ROOT / "final_openings.json"
        openings = json.loads(opening_path.read_text()) if opening_path.exists() else []
        openings.extend({"split":s,"opened_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"purpose":"frozen-backbone final feature extraction"} for s in final)
        opening_path.write_text(json.dumps(openings,indent=2)+"\n")
    env = environment(config["memory_gb"])
    model, tokenizer, manifest = get_model()
    label_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in "ABCD"]
    manifest.update({"environment": env, "label_tokens": dict(zip("ABCD", label_ids)), "prefix": config["assistant_prefix"], "parameters": sum(p.size for _, p in tree_flatten(model.parameters())), "trainable_parameters": sum(p.size for _, p in tree_flatten(model.trainable_parameters())), "candidate_representation": "mean frozen input embeddings of observed candidate description; separate frozen label-token embedding", "input_features": "RMS-normalized final prompt hidden, mean description embedding, label embedding; centered logits/5, conditional probability, log probability/5, entropy/2, margin/5"})
    if manifest["trainable_parameters"] != 0:
        raise AssertionError("Backbone must be frozen")
    CACHE.mkdir(parents=True, exist_ok=True)
    embedding = model.model.embed_tokens.weight
    # Interception, determinism, padding, and causal-prefix checks on an actual task.
    first = read_rows(ROOT / "datasets" / f"{splits[0]}.jsonl.gz")[0]
    ids = tokenize(tokenizer, first["prompt"], config["assistant_prefix"])
    a = mx.array([ids])
    h = model.model(a)
    native = model(a)[:, -1]
    projected = model.model.embed_tokens.as_linear(h[:, -1])
    repeat = model.model(a)[:, -1]
    cut = min(30, len(ids) - 1)
    prefix = model.model(a[:, :cut])
    padded = model.model(mx.array([ids + [tokenizer.pad_token_id] * 3]))[:, len(ids)-1]
    errors = {"native_projection": float(mx.max(mx.abs(native-projected)).item()), "repeat": float(mx.max(mx.abs(h[:, -1]-repeat)).item()), "causal_prefix": float(mx.max(mx.abs(h[:, :cut]-prefix)).item()), "right_padding": float(mx.max(mx.abs(h[:, -1]-padded)).item())}
    if max(errors.values()) > 1e-3:
        raise AssertionError(errors)
    manifest["integration_checks"] = errors
    del h, native, projected, repeat, prefix, padded
    manifest["extractions"] = {}
    start = time.perf_counter()
    for split in splits:
        rows = read_rows(ROOT / "datasets" / f"{split}.jsonl.gz")
        path = CACHE / split; path.mkdir(exist_ok=True)
        signature = {"model": MODEL_ID, "revision": REVISION, "dataset": hashlib.sha256((ROOT / "datasets" / f"{split}.jsonl.gz").read_bytes()).hexdigest(), "prefix": config["assistant_prefix"], "feature_version": 1}
        sigpath = path / "signature.json"
        if sigpath.exists() and json.loads(sigpath.read_text()) != signature:
            raise RuntimeError("Feature signature mismatch")
        sigpath.write_text(json.dumps(signature, indent=2) + "\n")
        seconds = 0
        for begin in range(0, len(rows), config["extract_batch"]):
            target = path / f"{begin:06d}.npz"
            if target.exists():
                seconds += float(np.load(target)["seconds"])
                continue
            items = rows[begin:begin+config["extract_batch"]]
            encoded = [tokenize(tokenizer, r["prompt"], config["assistant_prefix"]) for r in items]
            lengths = [len(t) for t in encoded]
            batch = mx.array([t + [tokenizer.pad_token_id] * (max(lengths)-len(t)) for t in encoded])
            mx.synchronize(); tic = time.perf_counter()
            h = model.model(batch)[mx.arange(len(items)), mx.array(lengths)-1]
            vocab = model.model.embed_tokens.as_linear(h).astype(mx.float32)
            all_logits = vocab[:, mx.array(label_ids)]
            label_order = mx.array([["ABCD".index(l) for l in r["labels"]] for r in items])
            logits = mx.take_along_axis(all_logits, label_order, axis=1)
            label_embedding = embedding[mx.array(label_ids)][label_order]
            descriptions = []
            for r in items:
                descriptions.append(mx.stack([mx.mean(embedding[mx.array(tokenizer.encode(d, add_special_tokens=False))], axis=0) for d in r["candidate_descriptions"]]))
            candidate = mx.stack(descriptions)
            greedy = mx.argmax(vocab, axis=-1)
            mx.eval(h, logits, candidate, label_embedding, greedy)
            mx.synchronize(); elapsed = time.perf_counter()-tic
            np.savez_compressed(target, h=np.array(h), candidate=np.array(candidate), label=np.array(label_embedding), logits=np.array(logits), greedy=np.array(greedy), lengths=np.array(lengths), seconds=np.array(elapsed))
            seconds += elapsed
            del batch, h, vocab, all_logits, logits, candidate, label_embedding, greedy
            if begin % 160 == 0:
                print(f"extract {split} {begin}/{len(rows)} {seconds:.1f}s peak={mx.get_peak_memory()/1024**3:.2f}GiB", flush=True)
        manifest["extractions"][split] = {"count": len(rows), "forward_seconds": seconds}
    manifest["this_invocation_seconds"] = time.perf_counter()-start
    manifest["peak_mlx_gib"] = mx.get_peak_memory()/1024**3
    target = ROOT / ("pilot_backbone.json" if splits == ["pilot"] else "backbone_" + "_".join(splits) + ".json")
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(target), "seconds": manifest["this_invocation_seconds"], "peak_mlx_gib": manifest["peak_mlx_gib"]}), flush=True)
