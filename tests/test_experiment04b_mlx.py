import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LATENT_LOOP_MLX_TESTS") != "1",
    reason="Opt-in small MLX verification",
)


def test_gated_classifier_capacity_gradients_and_shared_ranker_permutation():
    import mlx.core as mx
    import mlx.nn as nn

    from latent_loop.experiment04b.models import GatedHead, MatchedHead
    from latent_loop.training import parameter_count

    mx.set_default_device(mx.cpu)
    mx.random.seed(450)
    d = GatedHead()
    matched = MatchedHead()
    wr = GatedHead(width=16, outputs=1)
    rw = GatedHead(width=16, outputs=1)
    assert abs(parameter_count(d) / parameter_count(matched) - 1) < 0.005
    assert (
        abs(parameter_count(d) / (parameter_count(wr) + parameter_count(rw)) - 1)
        < 0.005
    )
    x = mx.random.normal((8, 1032))
    y = mx.array([0, 1, 2, 3, 0, 1, 2, 3])
    value, grad = nn.value_and_grad(
        d, lambda m: nn.losses.cross_entropy(m(x), y, reduction="mean")
    )(d)
    assert np.isfinite(float(value.item())) and d(x).shape == (8, 4)
    from mlx.utils import tree_flatten

    assert sum(float(mx.sum(mx.abs(v)).item()) for _, v in tree_flatten(grad)) > 0
    ranker = GatedHead(1036, 16, 1)
    rx = mx.random.normal((4, 1036))
    perm = mx.array([3, 1, 0, 2])
    assert np.allclose(
        np.array(ranker(rx)[perm]), np.array(ranker(rx[perm])), atol=1e-5
    )


def test_native_statistical_transform_matches_cpu():
    mx = pytest.importorskip("mlx.core")
    from latent_loop.experiment03.science import fit_scalar, scalar_predict
    from latent_loop.experiment04b.runtime import scalar

    rng = np.random.default_rng(45)
    x = rng.normal(size=(30, 3))
    y = rng.integers(0, 2, 30)
    setting = fit_scalar(x, y)
    assert (
        np.max(
            np.abs(
                np.array(scalar(setting, mx.array(x.astype(np.float32))))
                - scalar_predict(setting, x)
            )
        )
        < 1e-6
    )
