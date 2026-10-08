# Reproduction

Create an isolated environment; do not update a global model-serving environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e '.[apple,test]'
```

For CPU-only CI or report regeneration, install `.[test]` without `apple`; the core imports do not require MLX. The lock file documents the actual macOS pilot environment, and marks Apple packages with platform conditions. CPU checks:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m latent_loop.cli data --config configs/pilot.yaml
.venv/bin/python -m latent_loop.cli report --config configs/pilot.yaml
```

Apple-only integration checks run on MLX's CPU device, without a pretrained model:

```sh
LATENT_LOOP_MLX_TESTS=1 .venv/bin/python -m pytest -q tests/test_mlx.py
```

These verify closed-gate parity, nonzero gradient flow, memory isolation between examples, label-free live stopping, and exact checkpoint resume compared with uninterrupted training.

Run the short real-model smoke test in a separate directory:

```sh
.venv/bin/python -m latent_loop.cli download --config configs/smoke.yaml
.venv/bin/python -m latent_loop.cli run --config configs/smoke.yaml --run-dir artifacts/smoke
```

Reproduce the full primary pilot and its eight-loop negative result:

```sh
.venv/bin/python -m latent_loop.cli download --config configs/pilot.yaml
.venv/bin/python -m latent_loop.cli run --config configs/pilot.yaml --run-dir artifacts/reproduction
.venv/bin/python -m latent_loop.cli middle --config configs/pilot.yaml --run-dir artifacts/reproduction
.venv/bin/python -m latent_loop.cli report --config configs/pilot.yaml --run-dir artifacts/reproduction
```

Stages can be run separately with `data`, `extract`, `train`, and `evaluate`. To fit just one module, use `train --variant confidence --seed 11`, `train --variant recurrence --seed 11`, `train --variant memory --seed 11`, or the `feedforward` / `wide_feedforward` variants. After its parent recurrence has trained, use `train --variant recurrence_continuation --seed 11` or `train --variant memory_continuation --seed 11`. `train --seed 11` fits one seed's seven modules; a completed checkpoint is reused on subsequent runs. `evaluate` requires all selected seed checkpoints to be complete and recomputes their metrics without refitting. The loader supports only the pinned, validated Qwen3-0.6B architecture. For a changed dataset, prefix, width, learning rate, batch, or completed epoch count, use a fresh run directory. Features validate their dataset/model/prefix signature, and completed checkpoints reject changed fitting parameters.

The backbone is pinned to a small public checkpoint; `download` is explicit. `extract` requires the pinned snapshot already in the Hugging Face cache. No model-scale job silently falls back to CPU. A six-hour command budget reserves a minute at the end. SIGINT/SIGTERM request a stop at the next checkpoint boundary; rerunning resumes auxiliary heads with saved optimizer state and deterministic per-epoch ordering. Feature shards resume at batch boundaries. The tiny middle-layer training stage restarts after interruption; partial measurements are retained.

Reproduce the initial answer-format negative result:

```sh
.venv/bin/python -m latent_loop.cli run --config configs/initial.yaml --run-dir artifacts/initial_reproduction
```

The original initial adaptive rows used a weaker fallback when no stopping threshold met the validation criterion. Current code selects minimum validation NLL in that case. The archived rows preserve the historical result; the frozen baseline and candidate/refinement fits reproduce the original format setting, while old adaptive-policy rows are not asserted to match current policy code exactly.

All synthetic dataset generators and reference solvers are public. Model weights, local hidden-state caches, and trainable checkpoints are ignored by git. Recorded decision logits are compressed into public JSON, so the main report and figures can be rebuilt without training. No credentials, private data, network workstation, remote Git repository, or publication service is required.
