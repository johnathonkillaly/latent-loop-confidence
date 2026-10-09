# Reproducing v0.3.0

This guide accompanies the completed v0.3.0 research release. Historical guidance remains in [v0.1](reproduction.md) and [v0.2](reproduction_v0.2.0.md). Use Python 3.12; package metadata permits ≥3.11. The existing environment pins MLX 0.32.3 and mlx-lm 0.32.0. No dependency upgrade was needed. CPU-only public analysis uses NumPy/SciPy/Matplotlib/Markdown; model extraction and neural fits require Apple Silicon/Metal.

## Public analyses without weights

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m latent_loop.rendezvous --help
.venv/bin/python experiments/experiment_05e/public_audit.py
.venv/bin/python experiments/experiment_05e/description_shift.py --audit
.venv/bin/python -m latent_loop.rendezvous report
.venv/bin/python experiments/experiment_05e/write_report.py
```

Run regeneration in an isolated checkout: derived figures/HTML/audits are rewritten. The CPU-only public audit verifies all solver labels, semantic split hashes, byte-identical dataset regeneration and paired estimates without checkpoints. The separate `python -m latent_loop.rendezvous audit` performs additional local checkpoint-hash checks and requires the ignored regenerated weights. `report` and `write_report.py` require only tracked measurements and verification receipts, not model weights.

For 5C/5D, follow reproduction commands in their detailed reports. Their scripts deliberately preserve consumed final lockboxes. Original checkpoints and cached states are not distributed; regenerating them requires fitting the documented predecessor model in a fresh run. Independent parameter counts and public decisions can be inspected without those caches.

## Model download and exact revision

```sh
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e '.[apple,test]'
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca',
                  allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja'])
PY
LATENT_LOOP_MLX_TESTS=1 RUN_MLX_TESTS=1 .venv/bin/python -m pytest -q
```

The public model is downloaded separately; it is not included in the original-code MIT license. The loader checks the exact cached revision, freezes all 596,049,920 parameters and uses FP32. Unfiltered offline Hub snapshot lookup can reject missing optional repository metadata; use the explicit allow-list above. Recorded required-file hashes are in `results/experiment_05e/model_checkpoint_verification.json`.

## Fresh 5E fitting and evaluation

The original run's tracked `final_openings.json` intentionally prevents fitting after final consumption. Do not delete its history or tune against its test results. Create a new isolated directory with the installed editable source accessible and a source link for provenance. These commands do not overwrite the original research results:

```sh
# From the original source checkout:
source_dir="$PWD"
run_python="$source_dir/.venv/bin/python"
mkdir -p artifacts/rendezvous-reproduction
cd artifacts/rendezvous-reproduction
ln -s "$source_dir/src" src
config_path="$source_dir/experiments/experiment_05e/config.json"

"$run_python" -m latent_loop.rendezvous generate --config "$config_path"
"$run_python" "$source_dir/experiments/experiment_05e/run_resumable.py" extract --config "$config_path" --splits train validation
"$run_python" "$source_dir/experiments/experiment_05e/run_resumable.py" train --config "$config_path"
"$run_python" -m latent_loop.rendezvous freeze --config "$config_path"
"$run_python" -m latent_loop.rendezvous extract --config "$config_path" --splits id wording format structure harder permutations
"$run_python" -m latent_loop.rendezvous evaluate --config "$config_path"
"$run_python" -m latent_loop.rendezvous timing --config "$config_path"
"$run_python" -m latent_loop.rendezvous audit --config "$config_path"
"$run_python" -m latent_loop.rendezvous report --config "$config_path"
```

`generate` recreates the recorded benchmark for numerical reproduction, not new prospective confirmation. A new scientific replication must use new data seeds and freeze a new protocol. The supplementary `run_resumable.py` wrapper atomically writes tensors and journals a complete best/latest/optimizer/summary epoch transaction; startup restores its last hash-verified transaction after an interrupted newer write. This post-freeze I/O wrapper leaves the frozen scientific implementation unchanged. Use it from the first epoch of a new run; no claim is made about hard power-loss durability. Training resumes completed epochs with latest model/optimizer state; feature extraction resumes fingerprinted chunks. A changed dataset/model/prefix signature is rejected. Do not change batch layout, dtype or dependencies while expecting byte-identical hashes. Memory allocation is capped at 48 GiB; no CUDA or external workstation is used.

The final configuration fixes seeds **511/622/733**, eight head conditions, width 64, 24 epochs, and a validation-only fusion grid including zero. Every condition uses the same 6,000 examples and update budget. The protocol selects correctness BCE and λ4 before final extraction. Deployment uses raw text and candidate descriptions only; solver outcomes are used exclusively in fitting and evaluation.

`verify_and_diagnose.py` requires the original directory layout, source config and local regenerated checkpoints. It performs post-freeze descriptive comparisons, live-versus-cache parity and memory checks without model selection. `write_report.py` builds the narrative from the recorded receipts; copy the experiment scripts/config and required original recorded receipts into a separate analysis checkout if reproducing the identical narrative there.

## Verified scope and limitations

* CPU and all-enabled MLX tests passed; CPU explicitly skips unavailable accelerators.
* 13,680 main labels pass independently expressed solvers; all eight main dataset files regenerate byte-identically; canonical cross-split duplicate count is zero.
* All 24 auxiliary fits were repeated from initialization in an isolated directory; every selected checkpoint is byte-identical **using the same verified cached frozen features**.
* Sixteen validation prompts verify live full-policy versus cached decisions; actual backbone native projection, causality, repeat and padding checks pass. Required checkpoint files are unchanged after inference.
* Full frozen-feature extraction was performed for the original run, but not independently repeated for the complete verification refit. Full historical end-to-end training was not rerun. Do not describe this as independent full-series reproducibility.
* Frozen-source and checkpoint hashes, final-opening logs, seed-specific decisions, negative controls and failed architectural results are retained. There was no final-test reselection.

Model weights, optimizer states, raw hidden features and large caches are ignored. Public scalar decisions/metrics and deterministic synthetic tasks suffice for analysis regeneration. No secret, private infrastructure, proprietary dataset or remote GPU is required.


## Supplementary option-description shift

`experiments/experiment_05e/description_shift.py` freezes and evaluates 600 new tasks in paired natural/novel description views using the original fixed models and lambda. This was added after the primary result and is exploratory; it does not alter the original protocol. Run `--audit` on CPU to validate recorded public views and decisions without checkpoints. Running without `--audit` requires local ignored checkpoints and performs full frozen-Qwen inference, with no fitting or reselection. It must never be used to choose a new lambda or head on these consumed results.
