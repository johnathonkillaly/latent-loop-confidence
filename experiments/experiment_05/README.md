# Experiment 5 local runbook

This study starts at public `v0.1.0` and remains on the local branch
`research/exp05-train-the-thinker`. It is not part of the GitHub release.
The [plan](plan.md) defines comparisons before final evaluation;
[config.json](config.json) records all training, depth, seed, and resource settings.

## Environment and inputs

Use the existing Python 3.12 environment with MLX 0.32.3 and mlx-lm 0.32.0 on
Apple Silicon. The pinned Qwen3 revision and inspected configuration are retained
from the historical project. No CUDA or remote workstation is used. Original
weights stay frozen; adapter and optimizer checkpoints are under ignored
`artifacts/experiment_05/checkpoints/`. The historical one-pass reference requires
the three ignored `artifacts/checkpoints/feedforward_seed*/best.safetensors`
files. Regenerate them with the original pilot recipe if absent; their identities
are recorded with the baseline measurements.

`--config`, `--condition`, and `--seed` are accepted by the training module.
For this completed study, keep the canonical config and manifests unchanged.
The six-hour clock includes validation, fitting and evaluation; the training
guard reserves twenty minutes. The MLX allocator is limited to 48 GiB. The
primary depth is four and depths five through eight are diagnostics only.

Training finished before a long session idle interval. On the user's subsequent
continuation, [resume_budget.json](resume_budget.json) recorded that interval
before final opening. Evaluation retains the original six-hour **active** budget
and reports total calendar elapsed time separately. No training configuration,
checkpoint or selection rule was changed by this clock accounting.

## Execution and resume

```sh
LATENT_LOOP_MLX_TESTS=1 .venv/bin/python -m pytest -q
.venv/bin/python -m latent_loop.experiment05.validate
.venv/bin/python -m latent_loop.experiment05.train pilot
.venv/bin/python -m latent_loop.experiment05.train main
.venv/bin/python -m latent_loop.experiment05.evaluate freeze
.venv/bin/python -m latent_loop.experiment05.evaluate final
.venv/bin/python -m latent_loop.experiment05.evaluate timing
.venv/bin/python -m latent_loop.experiment05.report
```

Main training starts independently of the pilot. SIGINT/SIGTERM finish the
current batch and save adapter, optimizer and batch position. Reissuing the same
training command resumes incomplete fits and reuses complete fits; source,
config and dataset hashes reject incompatible resumes. Safe resume was checked
numerically on the tiny Qwen integration fixture. Forced termination in the
middle of writing a multi-file checkpoint has not been validated.

Freeze selects families and checkpoints using validation only, then hashes
configuration, sources, manifests and checkpoints. Final evaluation requires
that freeze. Its receipt prevents subsequent fitting. An interrupted final
evaluation can resume the same fixed choices; it cannot choose a new model or
depth. CPU report regeneration uses saved labels, logits and state summaries
without weights or MLX. Reports and traces omit hidden-state arrays.

## Independent reruns

The consumed study is immutable evaluation evidence, rather than a reusable
test-selection workspace. For independent training, use a **separate checkout**
of the local research commit, an isolated environment, and a fresh experiment
namespace/clock/seeds. Do not delete or replace outputs in the original checkout.
The implementation currently uses `experiment_05` paths, so a separate checkout
is necessary for a second complete study. In that isolated checkout remove its
derived Experiment 5 receipts, manifests and outputs, update the canonical
configuration, and prepare new splits with
`latent_loop.experiment05.data.prepare(config)` before implementation validation.
Regenerate frozen baselines and historical one-pass checkpoints as required.
Re-establish pilot cost matching and freeze the new protocol before evaluating
new final examples. Exact refit hashes need not match this study.

The [research report](../../reports/experiment_05/train_the_thinker.md) and
[summary](../../results/experiment_05/summary.json) distinguish completed
conditions from deferred controllers, memory and trajectory losses.
