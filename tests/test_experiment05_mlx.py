"""Architectural correctness checks for shared causal trainable recurrence."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("LATENT_LOOP_MLX_TESTS") != "1", reason="Opt-in MLX integration")


def tiny(rank=8, reinjection="learned"):
    import mlx.core as mx
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from latent_loop.experiment05.model import Thinker
    mx.set_default_device(mx.cpu)
    mx.random.seed(42)
    args = ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=6,
                     intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
                     rms_norm_eps=1e-6, vocab_size=64, max_position_embeddings=128,
                     head_dim=8, tie_word_embeddings=True, rope_theta=10000.0)
    base = Model(args)
    ids = mx.array([[1, 2, 3, 4, 5], [1, 2, 3, 9, 8]])
    native = np.array(base(ids)[..., :4])
    model = Thinker(base, [0, 1, 2, 3], 2, 2, rank, reinjection)
    return model, ids, native


def test_depth_one_parity_causality_positions_and_determinism():
    model, ids, native = tiny()
    result = model(ids, depths=(1, 2, 4, 8), return_tokens=True)
    np.testing.assert_allclose(np.array(result[1]["logits"]), native, atol=1e-5, rtol=1e-5)
    prefix = model(ids[:, :3], depths=(1, 2, 4, 8), return_tokens=True)
    for k in (1, 2, 4, 8):
        full = np.array(result[k]["logits"])
        np.testing.assert_allclose(full[:, :3], np.array(prefix[k]["logits"]), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(full[0, :3], full[1, :3], atol=2e-5, rtol=2e-5)
        np.testing.assert_array_equal(full, np.array(model(ids, depths=(k,), return_tokens=True)[k]["logits"]))


def test_shared_adapter_gradients_and_frozen_backbone():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from latent_loop.experiment05.model import objective
    model, ids, _ = tiny()
    ref = model(ids)[1]["logits"]
    args = (ids, mx.array([5, 5]), mx.array([0, 1]), ref)
    fun = nn.value_and_grad(model, lambda m: objective(m, *args, (1, 2, 3, 4), (1, 1, 1, 1)))
    value, gradients = fun(model)
    flat = dict(tree_flatten(gradients))
    assert np.isfinite(float(value.item()))
    assert all(np.isfinite(np.array(g)).all() for g in flat.values())
    assert sum(float(mx.sum(g * g).item()) for k, g in flat.items() if k.endswith("lora_b")) > 0
    assert abs(float(flat["update_logit"].item())) > 0
    assert abs(float(flat["reinjection_logit"].item())) > 0
    keys = model.parameter_counts()["trainable_keys"]
    assert len([k for k in keys if k.endswith("lora_b")]) == 4  # Not copied per depth.
    assert all(".linear.weight" not in k for k in keys)
    with pytest.raises(ValueError):
        model(ids, depths=(9,))


def test_optimizer_and_adapter_checkpoint_restoration(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as opt
    from mlx.utils import tree_flatten, tree_unflatten
    from latent_loop.experiment05.model import objective
    model, ids, _ = tiny()
    ref = mx.stop_gradient(model(ids)[1]["logits"])
    optimizer = opt.AdamW(learning_rate=1e-3)
    def step(m, o):
        value, grad = nn.value_and_grad(m, lambda q: objective(q, ids, mx.array([5, 5]), mx.array([0, 1]), ref, (1, 4), (1, 1)))(m)
        o.update(m, grad)
        mx.eval(m.trainable_parameters(), o.state, value)
    step(model, optimizer)
    mx.save_safetensors(str(tmp_path / "adapter.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(tmp_path / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    step(model, optimizer)
    expected = np.array(model(ids, depths=(4,))[4]["logits"])
    restored, ids2, _ = tiny()
    restored.load_weights(str(tmp_path / "adapter.safetensors"), strict=False)
    resumed = opt.AdamW(learning_rate=1e-3)
    resumed.state = tree_unflatten(list(mx.load(str(tmp_path / "optimizer.safetensors")).items()))
    step(restored, resumed)
    np.testing.assert_array_equal(expected, np.array(restored(ids2, depths=(4,))[4]["logits"]))


def test_rank16_and_bounded_state_norms():
    model, ids, _ = tiny(rank=16)
    result = model(ids, depths=(1, 4, 8))
    base = np.array(result[1]["state_norm"])
    assert model.parameter_counts()["trainable"] > 0
    assert np.all(np.array(result[8]["state_norm"]) <= model.norm_cap * base + 1e-4)
