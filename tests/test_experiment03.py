import numpy as np
import pytest

from latent_loop.experiment03.science import (
    binary_metrics,
    depths_from_continue,
    observables,
    policy_depth,
    targets,
)


def test_immediate_gain_is_signed_and_not_any_future_gain():
    logits = np.array([[[2.0, 0]], [[0.0, 2]], [[4.0, 0]], [[1.0, 0]]])
    loss, _, gain, useful = targets(logits, [0], 0.01)
    assert gain[0, 0] < 0
    assert not useful[0, 0]
    assert loss[2, 0] < loss[0, 0]  # Any-future label would differ.
    assert gain[1, 0] > 0 and useful[1, 0]
    extreme = targets(np.array([[[0.0, 1000]], [[0.0, 999]]]), [0])[2]
    assert extreme[0, 0] == 1  # No probability clipping in target loss.


def test_current_observables_do_not_depend_on_future_or_other_examples():
    rng = np.random.default_rng(12)
    states = rng.normal(size=(4, 7, 5)).astype(np.float32)
    logits = rng.normal(size=(4, 7, 4)).astype(np.float32)
    original = rng.normal(size=(7, 5)).astype(np.float32)
    x, scalar = observables(states, logits, original)
    states[2:] += 100
    logits[2:] -= rng.normal(size=logits[2:].shape) * 100
    other, _ = observables(states, logits, original)
    assert np.array_equal(x[:2], other[:2])
    single, single_scalar = observables(states[:, :1], logits[:, :1], original[:1])
    assert np.allclose(single, other[:, :1])
    assert np.allclose(single_scalar[:2], scalar[:2, :1])


def test_tied_average_precision_constant_prior():
    y = np.array([1, 0, 1, 0, 0])
    m = binary_metrics(np.full(5, 0.4), y)
    assert m["average_precision"] == pytest.approx(0.4)
    assert m["auroc"] == pytest.approx(0.5)
    perfect = binary_metrics(y * 0.8 + 0.1, y)
    assert perfect["average_precision"] == 1
    assert perfect["auroc"] == 1


def test_stopping_budget_and_oracle_pay_intervening_loops():
    assert np.array_equal(depths_from_continue(np.ones((3, 2))), [4, 4])
    assert np.array_equal(depths_from_continue(np.zeros((3, 2))), [1, 1])
    assert np.array_equal(depths_from_continue([[1, 0], [0, 1], [1, 1]]), [2, 1])
    scores = {"confidence": np.zeros((4, 1))}
    logits = np.array([[[2.0, 0]], [[0.0, 2]], [[4.0, 0]], [[1.0, 0]]])
    assert policy_depth("oracle_best", scores, {}, logits, [0])[0] == 3
    assert policy_depth("oracle_next", scores, {"delta": 0.01}, logits, [0])[0] == 1
    with pytest.raises(ValueError):
        policy_depth("oracle_best", scores, {})
