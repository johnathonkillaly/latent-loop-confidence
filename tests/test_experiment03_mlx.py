import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LATENT_LOOP_MLX_TESTS") != "1", reason="Opt-in MLX verification"
)


def test_streaming_features_equal_cpu_cached_features():
    import mlx.core as mx

    from latent_loop.experiment03.heads import live_features
    from latent_loop.experiment03.science import observables

    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(10)
    states = rng.normal(size=(4, 3, 1024)).astype(np.float32)
    logits = rng.normal(size=(4, 3, 4)).astype(np.float32)
    h = rng.normal(size=(3, 1024)).astype(np.float32)
    expected, _ = observables(states, logits, h)
    previous = h
    for k in range(4):
        observed = np.array(
            live_features(
                mx.array(states[k]), mx.array(logits[k]), mx.array(previous), k + 1
            )
        )
        assert np.allclose(observed, expected[k], atol=1e-5)
        previous = states[k]


def test_experiment03_epoch_resume_exact(tmp_path, monkeypatch):
    import mlx.core as mx

    from latent_loop.experiment03 import run

    mx.set_default_device(mx.cpu)
    config = {"width": 4, "epochs": 3, "learning_rate": 0.001, "batch_size": 16}
    rng = np.random.default_rng(19)
    arrays = {
        "x": rng.normal(size=(4, 12, 1032)).astype(np.float32),
        "correct": rng.random((4, 12)) > 0.5,
        "gain": rng.normal(size=(3, 12)).astype(np.float32),
    }
    arrays["useful"] = arrays["gain"] > 0.01

    def setup(folder):
        folder.mkdir()
        monkeypatch.chdir(folder)
        Path("artifacts/checkpoints/recurrence_seed11").mkdir(parents=True)
        Path("artifacts/checkpoints/recurrence_seed11/best.safetensors").write_bytes(
            b"test frozen parent hash"
        )
        Path("experiments/experiment_03").mkdir(parents=True)
        Path("experiments/experiment_03/dataset_manifest.json").write_text("{}")
        Path("artifacts/experiment_03/trajectories").mkdir(parents=True)
        for split in ["controller_train", "controller_val"]:
            np.savez(
                f"artifacts/experiment_03/trajectories/{split}_seed11.npz", **arrays
            )

    decoder = mx.zeros((4, 1024))
    setup(tmp_path / "full")
    monkeypatch.setattr(run, "guard", lambda c: None)
    uninterrupted, _ = run.fit_head("joint", 11, config, decoder)
    reference_weights = dict(
        mx.load("artifacts/experiment_03/checkpoints/joint_seed11/last.safetensors")
    )
    setup(tmp_path / "resume")
    calls = [0]

    def interrupt(c):
        calls[0] += 1
        if calls[0] == 2:
            raise TimeoutError("test interruption")

    monkeypatch.setattr(run, "guard", interrupt)
    with pytest.raises(TimeoutError):
        run.fit_head("joint", 11, config, decoder)
    monkeypatch.setattr(run, "guard", lambda c: None)
    resumed, _ = run.fit_head("joint", 11, config, decoder)
    restored = dict(
        mx.load("artifacts/experiment_03/checkpoints/joint_seed11/last.safetensors")
    )
    assert reference_weights.keys() == restored.keys()
    for key in reference_weights:
        assert np.array_equal(np.array(reference_weights[key]), np.array(restored[key]))
    assert np.array_equal(
        np.array(uninterrupted(mx.array(arrays["x"][0]))),
        np.array(resumed(mx.array(arrays["x"][0]))),
    )
