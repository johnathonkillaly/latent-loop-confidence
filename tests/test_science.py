import numpy as np

from latent_loop.datasets import generate, load, reference
from latent_loop.evaluation import metrics, oracle_labels, stopping_depth, transitions


def test_datasets_deterministic_solvers_and_disjoint_instances(tmp_path):
    sizes = {"train": 60, "val": 30, "test": 30, "ood": 30}
    a = generate(tmp_path / "a", sizes, 101)
    b = generate(tmp_path / "b", sizes, 101)
    assert a == b
    ids = set()
    for split in sizes:
        for row in load(tmp_path / "a", split):
            assert row["id"] not in ids
            ids.add(row["id"])
            assert row["choices"][row["label"]] == reference(row["task"], row["spec"])
            assert len(set(row["choices"])) == 4
            if row["task"] in ("pointer", "arithmetic", "state"):
                assert (row["difficulty"] > 4) == (split == "ood")


def test_oracle_is_future_loss_improvement_not_current_error():
    # Example 0 is correct already but benefits; example 1 is wrong and never benefits.
    logits = np.array(
        [[[1.0, 0.0], [1.0, 0.0]], [[3.0, 0.0], [2.0, 0.0]], [[4.0, 0.0], [3.0, 0.0]]]
    )
    oracle, loss = oracle_labels(logits, np.array([0, 1]), 0.01)
    assert oracle[:, 0].tolist() == [1, 1, 0]
    assert oracle[:, 1].tolist() == [0, 0, 0]


def test_halting_does_not_consult_labels_and_forces_budget_stop():
    probabilities = np.array([[0.2, 0.8, 0.8], [0.9, 0.1, 0.8], [0.9, 0.9, 0.8]])
    assert stopping_depth(probabilities, 0.5).tolist() == [1, 2, 3]
    assert stopping_depth(probabilities, 0.5, minimum=2).tolist() == [3, 2, 3]


def test_metrics_and_paired_transitions_have_known_answers():
    base = np.array([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 2.0]])
    other = np.array([[0.0, 2.0], [0.0, 2.0], [0.0, 2.0], [0.0, 2.0]])
    y = np.array([1, 1, 0, 0])
    t = transitions(base, other, y)
    assert t["wrong_to_right"] == 1 and t["right_to_wrong"] == 1
    assert t["accuracy_delta"] == 0 and t["mcnemar_exact_p"] == 1
    m = metrics(np.zeros((4, 4)), np.arange(4))
    assert m["nll"] == np.log(4)
    assert m["brier_multiclass"] == 0.75


def test_temperature_changes_calibration_but_not_decision():
    x = np.array([[2.0, 1.0, 0.0, -1.0], [3.0, -2.0, 1.0, 0.0]])
    y = np.array([0, 2])
    assert metrics(x, y)["accuracy"] == metrics(x / 10, y)["accuracy"]
