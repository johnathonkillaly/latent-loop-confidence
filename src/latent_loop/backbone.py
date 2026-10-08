"""Read-only Qwen3 access through mlx-lm; no generated reasoning tokens."""

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np

MODEL_ID = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def environment(memory_gb=48):
    import mlx.core as mx

    if not mx.metal.is_available():
        raise RuntimeError(
            "Metal unavailable: large experiments cannot silently use CPU"
        )
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(int(memory_gb * 1024**3))
    mx.set_cache_limit(512 * 1024**2)
    return {
        "device": dict(mx.device_info()),
        "backend": "MLX Metal",
        "machine": platform.machine(),
        "os": platform.mac_ver()[0],
        "versions": {
            p: importlib.metadata.version(p)
            for p in ["mlx", "mlx-lm", "numpy", "transformers"]
        },
        "memory_budget_gb": memory_gb,
    }


def get_model():
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    path = snapshot_download(
        MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
        local_files_only=True,
    )
    model, tokenizer = load(path)
    import mlx.core as mx

    # BF16 kernels change accumulation with tensor shape; FP32 makes the
    # intercepted final-position and causal-prefix checks well conditioned.
    model.set_dtype(mx.float32)
    model.freeze()
    config = json.loads((Path(path) / "config.json").read_text())
    manifest = {
        "id": MODEL_ID,
        "revision": REVISION,
        "config": config,
        "license": "Apache-2.0",
        "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
        "weight_precision": str(model.model.embed_tokens.weight.dtype),
        "tokenizer_sha256": hashlib.sha256(
            (Path(path) / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "chat_template": "apply_chat_template(enable_thinking=False)",
        "frozen": True,
    }
    return model, tokenizer, manifest


def tokenize(tokenizer, prompt, prefix="Answer: ("):
    text = (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        + prefix
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    for letter in "ABCD":
        label = tokenizer.encode(letter, add_special_tokens=False)
        if (
            len(label) != 1
            or tokenizer.encode(text + letter, add_special_tokens=False) != ids + label
        ):
            raise ValueError("Option letter is not an aligned single next token")
    return ids


def hidden_forward(model, ids):
    return model.model(ids)


def middle_forward(model, ids, depth=1, start=12, count=3):
    """Untrained middle-layer control. Independent same-depth causal passes, no KV cache."""
    from mlx_lm.models.base import create_attention_mask

    h = model.model.embed_tokens(ids)
    mask = create_attention_mask(h, None)
    for layer in model.layers[:start]:
        h = layer(h, mask=mask)
    for _ in range(depth):
        for layer in model.layers[start : start + count]:
            h = layer(h, mask=mask)
    for layer in model.layers[start + count :]:
        h = layer(h, mask=mask)
    return model.model.norm(h)


def extract(config, guard):
    import mlx.core as mx

    from latent_loop.datasets import load

    out = Path("artifacts/features")
    out.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads(Path("data/manifest.json").read_text())
    signature = {
        "dataset": dataset_manifest,
        "model": MODEL_ID,
        "revision": REVISION,
        "assistant_prefix": config.get("assistant_prefix", "Answer: ("),
        "precision": "float32",
    }
    signature_path = out / "signature.json"
    if signature_path.exists() and json.loads(signature_path.read_text()) != signature:
        raise ValueError(
            "Feature cache belongs to another dataset/model/prompt. Use a fresh --run-dir."
        )
    signature_path.write_text(json.dumps(signature, indent=2) + "\n")
    if (
        all((out / f"{s}.npz").exists() for s in config["sizes"])
        and Path("results/metrics/backbone.json").exists()
    ):
        print(
            "Reusing verified feature cache; retaining original extraction measurements",
            flush=True,
        )
        return
    env = environment(config["memory_gb"])
    model, tokenizer, manifest = get_model()
    manifest["assistant_prefix"] = signature["assistant_prefix"]
    manifest["dataset_manifest"] = dataset_manifest
    metadata = Path("results/metrics/backbone.json")
    label_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in "ABCD"]
    decoder = model.model.embed_tokens.weight[mx.array(label_ids)].astype(mx.float32)
    np.savez(out / "decoder.npz", decoder=np.array(decoder))
    lengths_all = []
    times = []
    start_time = time.perf_counter()
    # Check intercepted implementation against the untouched public forward.
    sample = mx.array(
        [
            tokenize(
                tokenizer,
                load("data", "train")[0]["prompt"],
                manifest["assistant_prefix"],
            )
        ]
    )
    native = model(sample)[:, -1, :]
    final = hidden_forward(model, sample)[:, -1, :]
    intercepted = model.model.embed_tokens.as_linear(final)
    middle = middle_forward(model, sample, 1)
    middle_logits = model.model.embed_tokens.as_linear(middle[:, -1, :])
    parity = float(
        mx.max(
            mx.abs(native.astype(mx.float32) - intercepted.astype(mx.float32))
        ).item()
    )
    middle_parity = float(
        mx.max(
            mx.abs(native.astype(mx.float32) - middle_logits.astype(mx.float32))
        ).item()
    )
    # Causal prefix invariance includes a repeated middle block.
    cut = min(20, sample.shape[1] - 1)
    full = middle_forward(model, sample, 2)[:, :cut, :]
    prefix = middle_forward(model, sample[:, :cut], 2)
    causal = float(
        mx.max(mx.abs(full.astype(mx.float32) - prefix.astype(mx.float32))).item()
    )
    if parity > 1e-3 or middle_parity > 1e-3 or causal > 1e-3:
        raise RuntimeError(f"parity/causality failure {parity, middle_parity, causal}")
    del native, intercepted, middle, middle_logits, final, full, prefix
    for split in config["sizes"]:
        rows = load("data", split)
        directory = out / split
        directory.mkdir(exist_ok=True)
        batch = config["extract_batch"]
        for begin in range(0, len(rows), batch):
            guard()
            target = directory / f"{begin:06d}.npz"
            if target.exists():
                continue
            items = rows[begin : begin + batch]
            encoded = [
                tokenize(tokenizer, r["prompt"], manifest["assistant_prefix"])
                for r in items
            ]
            lengths = [len(t) for t in encoded]
            lengths_all.extend(lengths)
            ids = mx.array(
                [
                    t
                    + [tokenizer.pad_token_id or tokenizer.eos_token_id]
                    * (max(lengths) - len(t))
                    for t in encoded
                ]
            )
            mx.synchronize()
            tic = time.perf_counter()
            h = hidden_forward(model, ids)[mx.arange(len(items)), mx.array(lengths) - 1]
            vocabulary = model.model.embed_tokens.as_linear(h).astype(mx.float32)
            lp = vocabulary - mx.logsumexp(vocabulary, axis=-1, keepdims=True)
            logits = vocabulary[:, mx.array(label_ids)]
            full_logp = lp[:, mx.array(label_ids)]
            entropy = -mx.sum(mx.exp(lp) * lp, axis=-1)
            greedy = mx.argmax(vocabulary, axis=-1)
            mx.eval(h, logits, entropy, full_logp, greedy)
            mx.synchronize()
            elapsed = time.perf_counter() - tic
            times.append(elapsed)
            np.savez_compressed(
                target,
                h=np.array(h.astype(mx.float32)),
                base=np.array(logits),
                entropy=np.array(entropy),
                full_logp=np.array(full_logp),
                greedy=np.array(greedy),
                lengths=np.array(lengths),
                seconds=np.array(elapsed),
            )
            del h, vocabulary, lp, logits, full_logp, entropy, greedy, ids
            if begin % 128 == 0:
                print(
                    f"extract {split} {min(begin + batch, len(rows))}/{len(rows)} peak={mx.get_peak_memory() / 1024**3:.2f} GiB",
                    flush=True,
                )
        shards = [np.load(p) for p in sorted(directory.glob("*.npz"))]
        np.savez_compressed(
            out / f"{split}.npz",
            **{
                k: np.concatenate([s[k] for s in shards])
                for k in ["h", "base", "entropy", "full_logp", "greedy", "lengths"]
            },
        )
        for s in shards:
            s.close()
    # Persist resource results, without local paths or process names.
    manifest.update(env)
    manifest.update(
        {
            "hidden_parity_max_abs": parity,
            "middle_one_loop_parity_max_abs": middle_parity,
            "middle_prefix_causal_max_abs": causal,
            "label_token_ids": label_ids,
            "extraction_seconds": time.perf_counter() - start_time,
            "peak_mlx_bytes": mx.get_peak_memory(),
            "batch_size": config["extract_batch"],
            "timing": "synchronized feature extraction including full-vocabulary final projection; first batch included",
            "raw_batch_seconds": times,
        }
    )
    metadata.write_text(json.dumps(manifest, indent=2) + "\n")
    print("feature extraction complete", flush=True)


def ensure_model_cached():
    """Explicit small pinned download for machines without the local model."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        MODEL_ID,
        revision=REVISION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
    )
