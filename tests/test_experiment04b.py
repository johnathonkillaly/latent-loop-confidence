import random

import numpy as np
import pytest

from latent_loop.experiment04a.data import FAMILIES, reference, row
from latent_loop.experiment04b.data import foundation_workspace, numerical_transform
from latent_loop.experiment04b.science import (
    deterministic_mixture,
    outcomes,
    pair_completion,
    ranker_inputs,
    stop_depth,
    transition_metrics,
    visited_selection,
)


def test_four_transition_targets_and_proper_multiclass_metrics():
    correct = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0]], bool)
    assert outcomes(correct)[0].tolist() == [0, 1, 2, 3]
    p = np.eye(4)[outcomes(correct)] * 0.97 + 0.03 / 4
    m = transition_metrics(p, outcomes(correct))
    assert m["four_class_accuracy"] == 1 and m["multiclass_brier"] < 0.01
    assert m["classes"]["WR"]["auroc"] == 1
    assert m["classes"]["RW"]["average_precision"] == 1


def test_stopping_and_selection_ignore_unvisited_future_information():
    scores = {
        "C": np.zeros((4, 3)),
        "D": np.broadcast_to([0.7, 0.02, 0.05, 0.23], (4, 3, 4)).copy(),
    }
    rule = {"kind": "D", "threshold": 0}
    assert stop_depth(scores, rule).tolist() == [1, 1, 1]
    scores["D"][1:] = [0, 1, 0, 0]
    assert stop_depth(scores, rule).tolist() == [1, 1, 1]
    quality = np.array([[2, 0, 1], [1, 3, 2], [999, 2, 3], [1000, 1000, 4]])
    paid = np.array([2, 3, 4])
    assert visited_selection(quality, paid).tolist() == [1, 2, 4]
    assert paid.tolist() == [2, 3, 4]
    with pytest.raises(AssertionError):
        visited_selection(quality, np.array([0, 3, 4]))


def test_terminal_continuation_features_zero_and_parameter_masks_constant():
    x = np.zeros((4, 2, 1032))
    c = np.full((4, 2), 0.4)
    v = np.full((4, 2), 0.2)
    d = np.broadcast_to([0.6, 0.1, 0.05, 0.25], (4, 2, 4))
    a = ranker_inputs(x, c, v, d, "CVD")
    b = ranker_inputs(x, c, v, d, "C")
    assert a.shape == b.shape == (4, 2, 1036)
    assert np.array_equal(a[-1, :, -3:], np.zeros((2, 3)))
    assert np.all(b[..., -3:] == 0)
    perm = [3, 0, 2, 1]
    assert np.array_equal(
        a[perm, 0], a[:, 0][perm]
    )  # Per-state model receives explicit depth tags.


def test_binary_event_completion_and_deterministic_compute_control():
    p = pair_completion([0.1, 0.8], [0.2, 0.6], [0.6, 0.1, 0.1, 0.2])
    assert np.allclose(p.sum(-1), 1) and np.all(p >= 0)
    ids = [f"public synthetic {k}" for k in range(4000)]
    d = deterministic_mixture(ids, 2.4)
    assert set(d) == {2, 3} and abs(d.mean() - 2.4) < 0.03
    assert np.array_equal(d, deterministic_mixture(ids, 2.4))


def test_numeric_shift_uses_real_operands_and_independent_solvers():
    rng = random.Random(450999)
    for task, families in FAMILIES.items():
        for approach, family in enumerate(families[:2]):
            for stratum in ["easy", "moderate", "hard"]:
                source = None
                while source is None:
                    source = row(rng, task, approach, stratum, 0, "numeric_range")
                transformed = numerical_transform(source, rng, 17)
                answer = reference(family, transformed["spec"])
                text = (
                    transformed["entities"][answer]
                    if isinstance(answer, int)
                    else answer
                )
                assert transformed["choices"][transformed["label"]] == text
                assert len(set(transformed["choices"])) == 4
                if task == "pointer":
                    assert transformed["spec"] == source["spec"]


def test_foundation_reuse_adapter_restores_original_destinations_after_error():
    from latent_loop.experiment04a import pipeline

    old = pipeline.CACHE
    with pytest.raises(RuntimeError):
        with foundation_workspace():
            assert pipeline.CACHE.as_posix() == "artifacts/experiment_04b"
            raise RuntimeError("test safe restoration")
    assert pipeline.CACHE == old


def test_inference_paths_ignore_inaccessible_future_scores():
    from latent_loop.experiment04b.evaluate import decision_log, paths

    c = np.full((4, 3), 0.5)
    d = np.zeros((4, 3, 4))
    d[:, :, 0] = 1
    scores = {"C": c, "D": d}
    q = {"max_C": c.copy()}
    chosen = {
        "rules": {
            "d_direct": {"kind": "D", "threshold": 0},
            "d_utility": {"kind": "utility", "lambda": 0},
            "stack_CV": {"kind": "fixed", "depth": 1},
            "stack_CVD": {"kind": "fixed", "depth": 1},
        },
        "validation_best_combination": {"policy": "d_direct", "selector": "max_C"},
    }
    a = paths(scores, q, chosen, ["a", "b", "c"])
    scores["D"][1:, :, 1] = 100
    q["max_C"][1:] = 100
    b = paths(scores, q, chosen, ["a", "b", "c"])
    assert np.array_equal(a["d_direct"]["paid"], b["d_direct"]["paid"])
    assert np.array_equal(a["d_direct"]["max_C"], b["d_direct"]["max_C"])
    assert decision_log(np.array([1, 4])) == [
        ["stop", "unvisited", "unvisited", "unvisited"],
        ["continue", "continue", "continue", "stop"],
    ]
