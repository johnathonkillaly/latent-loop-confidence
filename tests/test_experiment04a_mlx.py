import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LATENT_LOOP_MLX_TESTS") != "1",
    reason="Opt-in small MLX verification",
)


def test_all_halting_mechanisms_match_and_count_actual_executed_rows():
    import mlx.core as mx

    from latent_loop.experiment03.heads import Probe
    from latent_loop.experiment04a.timing import live_policy
    from latent_loop.models import Refiner

    mx.set_default_device(mx.cpu)
    mx.random.seed(441)
    refiner = Refiner(1024, 4, False)
    refiner.gate = mx.array(0.05)
    refiner.injection = mx.array(0.01)
    models = {
        "refiner": refiner,
        "confidence": Probe(1032, 4),
        "improvement": Probe(1032, 4),
    }
    h = mx.random.normal((5, 1024))
    base = mx.random.normal((5, 4))
    decoder = mx.random.normal((4, 1024))
    for policy in ("confidence", "improvement"):
        for threshold in (0, 0.5, 1):
            outcomes = {}
            for mechanism in ("scalar", "compact", "masked"):
                out, depth, work = live_policy(
                    policy + ":" + mechanism,
                    h,
                    base,
                    decoder,
                    models,
                    {"threshold": threshold},
                )
                out, depth = np.array(out), np.array(depth)
                outcomes[mechanism] = (out, depth)
                assert 1 <= depth.min() <= depth.max() <= 4
                if mechanism == "masked":
                    assert work["refiner_rows"] == 20
                else:
                    assert work["refiner_rows"] == int(depth.sum())
            for mechanism in ("compact", "masked"):
                assert np.allclose(
                    outcomes[mechanism][0], outcomes["scalar"][0], atol=2e-4
                )
                assert np.array_equal(outcomes[mechanism][1], outcomes["scalar"][1])
