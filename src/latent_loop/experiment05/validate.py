"""Real-checkpoint validation before any Experiment 5 training."""

import hashlib
import json
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from latent_loop.backbone import environment, get_model, tokenize
from latent_loop.experiment05.data import CACHE, EXP, OUT, load
from latent_loop.experiment05.model import Thinker, objective
from latent_loop.training import atomic_json


def batch(tokenizer, rows):
    tokens = [tokenize(tokenizer, r["prompt"]) for r in rows]
    lengths = [len(t) for t in tokens]
    ids = mx.array([t + [tokenizer.pad_token_id or tokenizer.eos_token_id] * (max(lengths) - len(t)) for t in tokens])
    return ids, mx.array(lengths), mx.array([r["label"] for r in rows])


def main():
    config = json.loads((EXP / "config.json").read_text())
    start = time.perf_counter()
    env = environment(config["memory_gib"])
    base, tokenizer, manifest = get_model()
    rows = load("train")[:2]
    ids, lengths, labels = batch(tokenizer, rows)
    native = base(ids)[mx.arange(len(rows)), lengths - 1][:, mx.array([tokenizer.encode(x, add_special_tokens=False)[0] for x in "ABCD"])]
    mx.eval(native)
    counts = {}
    for rank in (8, 16):
        mx.random.seed(11)
        model = Thinker(base, [tokenizer.encode(x, add_special_tokens=False)[0] for x in "ABCD"], config["middle_start"], config["middle_count"], rank, "learned")
        counts[str(rank)] = model.parameter_counts()
    mx.random.seed(11)
    model = Thinker(base, [tokenizer.encode(x, add_special_tokens=False)[0] for x in "ABCD"], config["middle_start"], config["middle_count"], 8, "learned")
    original_sample = hashlib.sha256(np.array(base.layers[config["middle_start"]].self_attn.q_proj.linear.weight).tobytes()).hexdigest()
    predicted = model(ids, lengths)[1]["logits"]
    parity = float(mx.max(mx.abs(predicted - native)).item())
    assert parity < 1e-3, parity
    fun = nn.value_and_grad(model, lambda m: objective(m, ids, lengths, labels, native, (1, 2, 3, 4), (1, 1, 1, 1), config["preservation_alpha"]))
    loss, gradients = fun(model)
    mx.eval(loss, gradients)
    grad_norms = {k: float(mx.linalg.norm(v).item()) for k, v in tree_flatten(gradients)}
    assert all(np.isfinite(v) for v in grad_norms.values())
    assert sum(v for k, v in grad_norms.items() if k.endswith("lora_b")) > 0
    assert grad_norms["update_logit"] > 0 and grad_norms["reinjection_logit"] > 0
    optimizer = optim.AdamW(learning_rate=config["learning_rate"])
    optimizer.update(model, gradients)
    mx.eval(model.trainable_parameters(), optimizer.state)
    cut = min(24, int(lengths[0].item()))
    full = model(ids[:1], depths=(1, 4), return_tokens=True)
    prefix = model(ids[:1, :cut], depths=(1, 4), return_tokens=True)
    errors = {str(k): float(mx.max(mx.abs(full[k]["logits"][:, :cut] - prefix[k]["logits"])).item()) for k in (1, 4)}
    assert max(errors.values()) < 0.002, errors
    expected = np.array(model(ids, lengths, depths=(4,))[4]["logits"])
    repeated = np.array(model(ids, lengths, depths=(4,))[4]["logits"])
    assert np.array_equal(expected, repeated)
    p = CACHE / "validation_adapter.safetensors"
    p.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(p), dict(tree_flatten(model.trainable_parameters())))
    model = Thinker(base, model.label_ids, config["middle_start"], config["middle_count"], 8, "learned")
    model.load_weights(str(p), strict=False)
    restored = np.array(model(ids, lengths, depths=(4,))[4]["logits"])
    assert np.array_equal(expected, restored)
    preserved = original_sample == hashlib.sha256(np.array(base.layers[config["middle_start"]].self_attn.q_proj.linear.weight).tobytes()).hexdigest()
    assert preserved
    record = {"all_passed": True, "depth_one_max_abs_error": parity, "trained_causal_prefix_max_abs_error": errors,
              "gradient_norm_by_parameter": grad_norms, "nonzero_lora_b_gradients": True,
              "initial_lora_a_gradient": "Zero is expected when zero-output B is initialized; B and open recurrent gates receive gradients",
              "deterministic_evaluation": True, "exact_adapter_restoration": True,
              "optimizer_resume": "Exact two-step resume verified in the tiny-Qwen numerical integration test",
              "frozen_backbone_sample_unchanged": preserved, "rank_parameter_counts": counts,
              "model_manifest": manifest, "environment": env, "seconds": time.perf_counter() - start,
              "peak_mlx_bytes": mx.get_peak_memory(), "cache_semantics": "No KV caches; full-prefix attention with identical RoPE positions each loop"}
    atomic_json(OUT / "implementation_validation.json", record)
    print(json.dumps({k: record[k] for k in ("all_passed", "depth_one_max_abs_error", "trained_causal_prefix_max_abs_error", "seconds", "peak_mlx_bytes")}), flush=True)


if __name__ == "__main__":
    main()
