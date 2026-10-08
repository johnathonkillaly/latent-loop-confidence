import json
import random
from dataclasses import fields

import numpy as np
import pytest

from latent_loop.experiment04a.data import (
    FAMILIES,
    graph_fingerprint,
    make_spec,
    reference,
    row,
    structure,
)
from latent_loop.experiment04a.science import binary
from latent_loop.experiment04a.trajectory import (
    RuntimeObservation,
    runtime_from_fields,
    runtime_observation,
    transition,
)


def test_independent_solvers_cover_every_semantic_generator_and_stratum():
    rng = random.Random(440999)
    for task, families in FAMILIES.items():
        for approach, family in enumerate(families):
            for difficulty, operations in [
                ("easy", 1),
                ("moderate", 3),
                ("hard", 5),
                ("extrapolation", 8),
            ]:
                count = 0
                while count < 15:
                    spec, answer = make_spec(
                        rng, task, approach, operations, difficulty
                    )
                    if spec is None:
                        continue
                    assert reference(family, spec) == answer, (family, spec)
                    count += 1
                example = None
                while example is None:
                    example = row(rng, task, approach, difficulty, 2, "test")
                assert len(set(example["choices"])) == 4
                answer = reference(family, example["spec"])
                expected = (
                    example["entities"][answer] if isinstance(answer, int) else answer
                )
                assert example["choices"][example["label"]] == expected


def test_alpha_renaming_register_and_graph_fingerprints():
    s = {"initial": [5, 9, 12], "ops": [["swap", 0, 2], ["add", 1, 4]], "query": 0}
    p = [2, 0, 1]
    initial = [0] * 3
    for i, v in enumerate(s["initial"]):
        initial[p[i]] = v
    renamed = {
        "initial": initial,
        "ops": [["swap", p[0], p[2]], ["add", p[1], 4]],
        "query": p[0],
    }
    assert structure("register_program", s) == structure("register_program", renamed)
    maps = [[1, 2, 0, 2], [2, 3, 1, 0]]
    perm = [2, 0, 3, 1]
    other = []
    for mapping in maps:
        transformed = [0] * 4
        for i, t in enumerate(mapping):
            transformed[perm[i]] = perm[t]
        other.append(transformed)
    assert graph_fingerprint(maps, 0, 3) == graph_fingerprint(other, perm[0], 3)
    assert graph_fingerprint(maps, 0, 3) != graph_fingerprint(maps, 0, 4)


def test_runtime_api_rejects_supervision_and_is_current_only():
    h = np.linspace(-1, 1, 1024, dtype=np.float32)
    obs = runtime_observation(h, [1, 2, 3, 4], h, 2)
    assert obs.vector().shape == (1032,)
    assert obs.depth == 2 and obs.previous_change == 0
    with pytest.raises(ValueError):
        obs.hidden[0] = 3
    values = {f.name: getattr(obs, f.name) for f in fields(RuntimeObservation)}
    for unsafe in [
        "label",
        "future_state",
        "future_loss",
        "oracle_depth",
        "next_prediction",
        "offline_supervision",
    ]:
        with pytest.raises(ValueError):
            runtime_from_fields({**values, unsafe: 0})
    with pytest.raises(TypeError):
        runtime_observation(h, [1, 2, 3, 4], h, 1, label=2)


def test_all_four_transition_labels_and_boolean_log_loss():
    assert [
        transition(a, b)
        for a, b in [(False, True), (True, False), (False, False), (True, True)]
    ] == ["W->R", "R->W", "W->W", "R->R"]
    assert binary([0.8, 0.2], np.array([True, False]))["log_loss"] == pytest.approx(
        -np.log(0.8)
    )


def test_lockbox_requires_frozen_hashes_and_logs_opening(tmp_path, monkeypatch):
    from latent_loop.experiment04a import common
    from latent_loop.experiment04a.data import EXP

    monkeypatch.chdir(tmp_path)
    EXP.mkdir(parents=True)
    for name in (
        "config.json",
        "dataset_manifest.json",
        "selection_seed11.json",
        "historical_structure_exclusions.json",
        "plan.md",
    ):
        (EXP / name).write_text("{}")
    for p in common.checkpoint_paths(11).values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"fixture checkpoint")
    common.freeze({"parent_seeds": [11]})
    common.open_lockbox()
    event = json.loads((EXP / "lockbox_opened.json").read_text())
    assert event["opened_utc"] >= event["protocol_frozen_utc"]
    with pytest.raises(RuntimeError):
        common.freeze({"parent_seeds": [11]})
    (EXP / "config.json").write_text('{"changed":true}')
    with pytest.raises(AssertionError):
        common.verify_freeze()
