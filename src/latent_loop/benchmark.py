"""Minimal middle-layer training, causal control, and synchronized prompt latency."""

import time
from pathlib import Path

import numpy as np

from latent_loop.datasets import load
from latent_loop.evaluation import metrics, transitions
from latent_loop.training import atomic_json


def middle_benchmark(config, guard):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_lm.models.base import create_attention_mask

    from latent_loop.backbone import environment, get_model, middle_forward, tokenize

    environment(config["memory_gb"])
    model, tokenizer, manifest = get_model()
    labels = mx.array(
        [tokenizer.encode(l, add_special_tokens=False)[0] for l in "ABCD"]
    )
    decoder = model.model.embed_tokens.weight[labels]
    prefix = config.get("assistant_prefix", "Answer: (")

    def batch(rows):
        encoded = [tokenize(tokenizer, r["prompt"], prefix) for r in rows]
        lengths = list(map(len, encoded))
        return (
            mx.array(
                [
                    x + [tokenizer.pad_token_id] * (max(lengths) - len(x))
                    for x in encoded
                ]
            ),
            mx.array(lengths) - 1,
            mx.array([r["label"] for r in rows]),
        )

    class MiddleInjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = model
            self.base.freeze()
            self.alpha = mx.zeros((3,))

        def __call__(self, ids, last, depth=2):
            h = self.base.model.embed_tokens(ids)
            mask = create_attention_mask(h, None)
            for layer in self.base.layers[:12]:
                h = layer(h, mask=mask)
            anchor = h
            for k in range(depth):
                for j, layer in enumerate(self.base.layers[12:15]):
                    if k:
                        h = h + self.alpha[j] * anchor * mx.rsqrt(
                            mx.mean(anchor * anchor, axis=-1, keepdims=True) + 1e-6
                        )
                    h = layer(h, mask=mask)
            for layer in self.base.layers[15:]:
                h = layer(h, mask=mask)
            h = self.base.model.norm(h)[mx.arange(ids.shape[0]), last]
            return h @ decoder.T

    module = MiddleInjection()
    optimizer = optim.Adam(learning_rate=0.01)

    def loss(m, ids, last, y):
        return nn.losses.cross_entropy(m(ids, last), y, reduction="mean")

    gradient = nn.value_and_grad(module, loss)
    train = load("data", "train")[:240]
    val = load("data", "val")[:60]
    history = []
    best = 1e30
    bestalpha = mx.zeros((3,))
    start = time.perf_counter()
    checkpoint = Path("artifacts/middle")
    checkpoint.mkdir(exist_ok=True)
    for epoch in range(5):
        guard()
        order = np.random.default_rng(801 + epoch).permutation(len(train))
        losses = []
        for begin in range(0, len(order), 4):
            ids, last, y = batch([train[i] for i in order[begin : begin + 4]])
            value, grads = gradient(module, ids, last, y)
            grads, norm = optim.clip_grad_norm(grads, 1.0)
            optimizer.update(module, grads)
            mx.eval(module.alpha, optimizer.state, value)
            losses.append(float(value.item()))
        vals = []
        for begin in range(0, len(val), 8):
            ids, last, y = batch(val[begin : begin + 8])
            vals.append(float(loss(module, ids, last, y).item()))
        vl = float(np.mean(vals))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_loss": vl,
                "alpha": np.array(module.alpha).tolist(),
            }
        )
        if vl < best:
            best = vl
            bestalpha = mx.array(module.alpha)
        mx.save_safetensors(
            str(checkpoint / "last.safetensors"), {"alpha": module.alpha}
        )
        mx.save_safetensors(str(checkpoint / "best.safetensors"), {"alpha": bestalpha})
        atomic_json(
            "results/metrics/middle_training.json",
            {
                "trainable_parameters": 3,
                "epochs": history,
                "train_examples": 240,
                "val_examples": 60,
                "batch_size": 4,
                "optimizer": "Adam",
                "learning_rate": 0.01,
                "training_depth": 2,
                "seed": 801,
                "training_seconds": time.perf_counter() - start,
                "peak_mlx_bytes": mx.get_peak_memory(),
            },
        )
        print(
            f"middle epoch={epoch} train={np.mean(losses):.4f} val={vl:.4f}", flush=True
        )
    module.alpha = bestalpha
    results = []
    for split in ["test", "ood"]:
        rows = load("data", split)
        y = np.array([r["label"] for r in rows])
        outputs = {}
        for depth in [1, 2, 3, 4]:
            for kind in ["plain", "injection"] if depth > 1 else ["plain"]:
                pieces = []
                latencies = []
                for begin in range(0, len(rows), 8):
                    guard()
                    ids, last, _ = batch(rows[begin : begin + 8])
                    mx.synchronize()
                    tic = time.perf_counter()
                    if kind == "injection":
                        logits = module(ids, last, depth)
                    else:
                        h = middle_forward(model, ids, depth)[
                            mx.arange(ids.shape[0]), last
                        ]
                        logits = h @ decoder.T
                    mx.eval(logits)
                    mx.synchronize()
                    latencies.append(time.perf_counter() - tic)
                    pieces.append(np.array(logits))
                outputs[f"middle_{kind}_d{depth}"] = np.concatenate(pieces)
                base = outputs["middle_plain_d1"]
                result = metrics(outputs[f"middle_{kind}_d{depth}"], y)
                result.update(
                    {
                        "split": split,
                        "variant": f"middle_{kind}_d{depth}",
                        "depth": depth,
                        "transitions_vs_baseline": transitions(
                            base, outputs[f"middle_{kind}_d{depth}"], y
                        ),
                        "batch_size": 8,
                        "raw_batch_seconds": latencies,
                        "seconds": sum(latencies),
                        "trained_parameters": 3 if kind == "injection" else 0,
                    }
                )
                results.append(result)
                atomic_json("results/metrics/middle.json", results)
        np.savez_compressed(
            checkpoint / f"{split}_predictions.npz", **outputs, labels=y
        )
    # One-loop parity must hold even after injection training.
    ids, last, _ = batch(load("data", "val")[:1])
    original = model.model(ids)[0, last[0]] @ decoder.T
    parity = float(mx.max(mx.abs(module(ids, last, 1)[0] - original)).item())
    timings = []
    timing_rows = load("data", "val")[:12]
    for r in timing_rows:
        ids, last, _ = batch([r])

        def call():
            return model.model.embed_tokens.as_linear(model.model(ids)[0, last[0]])

        for _ in range(3):
            mx.eval(call())
            mx.synchronize()
        times = []
        for _ in range(config["latency_repeats"]):
            guard()
            mx.synchronize()
            tic = time.perf_counter()
            mx.eval(call())
            mx.synchronize()
            times.append((time.perf_counter() - tic) * 1000)
        timings.append(
            {
                "id": r["id"],
                "task": r["task"],
                "tokens": ids.shape[1],
                "median_ms": float(np.median(times)),
                "raw_ms": times,
            }
        )
    atomic_json(
        "results/metrics/latency_backbone.json",
        {
            "samples": timings,
            "batch_size": 1,
            "warmup": 3,
            "repeats": config["latency_repeats"],
            "scope": "prompt through frozen FP32 backbone and full-vocabulary final-position projection",
            "median_ms": float(np.median([r["median_ms"] for r in timings])),
            "one_loop_parity_after_training": parity,
            "peak_mlx_bytes": mx.get_peak_memory(),
            "middle_block_zero_based": [12, 13, 14],
            "note": "Middle experiment is a three-scalar input-injection pilot; no LMA or halting in middle layers.",
        },
    )
