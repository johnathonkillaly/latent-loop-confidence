"""Opt-in CPU MLX tests; ordinary CI never imports MLX or downloads a model."""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LATENT_LOOP_MLX_TESTS") != "1", reason="opt-in Apple MLX tests"
)


def test_closed_gate_parity_gradient_and_memory_batch_isolation():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from latent_loop.models import Refiner

    mx.set_default_device(mx.cpu)
    mx.random.seed(111)
    h = mx.random.normal((2, 16))
    base = mx.random.normal((2, 4))
    decoder = mx.random.normal((4, 16))
    m = Refiner(16, 8, True)
    assert float(mx.max(mx.abs(m(h, base, decoder, 4) - base)).item()) == 0

    def objective(model):
        return nn.losses.cross_entropy(
            model(h, base, decoder, 4), mx.array([0, 1]), reduction="mean"
        )

    value, grads = nn.value_and_grad(m, objective)(m)
    assert any(float(mx.max(mx.abs(g)).item()) > 0 for _, g in tree_flatten(grads))
    m.gate = mx.array(0.1)
    m.memory_gate = mx.array(0.2)
    full = m(h, base, decoder, 4)
    single = m(h[:1], base[:1], decoder, 4)
    assert float(mx.max(mx.abs(full[:1] - single)).item()) < 1e-5


def test_live_halting_matches_offline_path_without_ground_truth():
    import mlx.core as mx
    import numpy as np

    from latent_loop.models import Continuation, Refiner
    from latent_loop.training import adaptive_infer

    mx.set_default_device(mx.cpu)
    mx.random.seed(444)
    h = mx.random.normal((1, 16))
    base = mx.random.normal((1, 4))
    decoder = mx.random.normal((4, 16))
    m = Refiner(16, 8, True)
    m.gate = mx.array(0.2)
    head = Continuation(16)
    for threshold, expected_depth in [(0.0, 4), (1.0, 1)]:
        live = adaptive_infer(m, head, h, base, decoder, threshold, 4)
        fixed = m(h, base, decoder, expected_depth)
        assert np.allclose(np.array(live), np.array(fixed))


def test_epoch_checkpoint_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from mlx.utils import tree_flatten

    from latent_loop.models import Refiner
    from latent_loop.training import fit

    mx.set_default_device(mx.cpu)
    mx.random.seed(919)
    h = np.array(mx.random.normal((8, 8)))
    base = np.array(mx.random.normal((8, 4)))
    decoder = mx.random.normal((4, 8))
    y = np.arange(8, dtype=np.int32) % 4
    cfg = {"learning_rate": 0.001, "train_batch": 4, "epochs": 2, "train_depth": 4}
    arrays = [h, base, y]

    def loss(m, h, b, y, depth=1):
        return nn.losses.cross_entropy(m(h, b, decoder, depth), y, reduction="mean")

    def create():
        mx.random.seed(1001)
        return Refiner(8, 4, False)

    uninterrupted = tmp_path / "full"
    uninterrupted.mkdir()
    monkeypatch.chdir(uninterrupted)
    full = create()
    fit(full, "feedforward", 17, cfg, arrays, arrays, loss, lambda: None)
    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    monkeypatch.chdir(resumed_dir)
    count = 0

    def stop():
        nonlocal count
        count += 1
        if count == 2:
            raise InterruptedError("test interrupt after one saved epoch")

    with pytest.raises(InterruptedError):
        fit(create(), "feedforward", 17, cfg, arrays, arrays, loss, stop)
    resumed = create()
    info = fit(resumed, "feedforward", 17, cfg, arrays, arrays, loss, lambda: None)
    assert info["next_epoch"] == 2 and len(info["history"]) == 2
    for (_, a), (_, b) in zip(
        tree_flatten(full.parameters()), tree_flatten(resumed.parameters())
    ):
        assert np.allclose(np.array(a), np.array(b), atol=1e-7)
