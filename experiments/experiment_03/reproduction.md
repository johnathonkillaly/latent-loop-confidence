# Reproduce Experiment 3

## Saved public results (CPU only)

Install the repository's locked environment and editable package as described in the main reproduction guide. From the repository root:

```sh
python -m latent_loop.experiment03.run report
python -m pytest -q
```

The report command reads public JSON/gzip predictions and recorded timings, verifies losses/targets/stops/accuracy/NLL, regenerates nine figures and Markdown/HTML, and imports no MLX. It requires neither ignored hidden states nor model weights. It updates only Experiment 3 outputs. Bootstrap seeds are fixed; timestamps change on regeneration.

## Fresh numerical run (Apple Silicon / Metal)

The local research checkout already has the original parent features/checkpoints. A clean clone must first obtain those ignored assets. Run the original workflow in its own ignored directory, preserving all public historical results:

```sh
python -m latent_loop.cli download --config configs/pilot.yaml --run-dir artifacts/parent_reproduction
python -m latent_loop.cli run --config configs/pilot.yaml --run-dir artifacts/parent_reproduction
```

Only in a clean clone where the following destinations do not exist, link the resulting ignored assets at the paths Experiment 3 expects:

```sh
ln -s artifacts/parent_reproduction/data data
ln -s parent_reproduction/artifacts/features artifacts/features
ln -s parent_reproduction/artifacts/checkpoints artifacts/checkpoints
```

Use the public pinned model revision and locked dependencies. The audit compares regenerated parent trajectories to the public archived FP32 predictions, allowing a 1e-5 tolerance. A mismatch should be investigated, never silently treated as an equivalent parent model.

Copy the Experiment 3 config with a new clock origin for the independent four-hour run:

```sh
python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
p = Path('experiments/experiment_03/config.json')
config = json.loads(p.read_text())
config['started_utc'] = datetime.now(timezone.utc).isoformat()
Path('artifacts/e3_reproduction_config.json').write_text(json.dumps(config, indent=2))
PY
python -m latent_loop.experiment03.run all --config artifacts/e3_reproduction_config.json
```

Use a separate checkout/cache for an independent run. A changed config is rejected by existing checkpoint signatures. An interrupted run reuses the same config clock and resumes at completed feature batches or training epochs; optimizer state, deterministic epoch ordering, last weights and best validation weights are preserved. To extend an expired budget, archive the original cache and start a separately documented run rather than silently changing the clock of an existing checkpoint.

The protocol fixes four trajectory/deployment depths. Width, target delta, data sizes/seed, optimization settings, timing sample/repetitions, memory limit and wall-clock budget are recorded in the config. Changed target thresholds require fresh trajectory/controller artifacts; they are checked before fitting. Defaults are 48 GiB and four hours. Final evaluation splits never select checkpoints or thresholds. No model-scale operation falls back to CPU; report rebuilding and CPU tests can run without an accelerator.

Neural fits: 15 (five heads × three seeds). Scalar statistical controls: 45 (15 × three seeds). Existing backbone/refiners/continuation heads/one-pass checkpoints are reused. The public results include six evaluated splits, per-task diagnostics, raw logits/scores/stopping depths, behavioral examples, bootstrap comparisons and synchronized timing samples. Hidden vectors, model weights and environments stay ignored.

Opt-in small MLX CPU verification, including exact interruption/resumption:

```sh
LATENT_LOOP_MLX_TESTS=1 python -m pytest -q
```
