# Experiment 5.3 — The Decision Bridge

Local research from `1f2e4947689984643bbadccb3327a8baec4cba81`, branch
`research/exp05d-decision-bridge`. No publication or Experiment 5.4 integration.

[Scientific report](../../reports/experiment_05d/decision_bridge.md) ·
[Self-contained HTML](../../reports/experiment_05d/decision_bridge.html) ·
[Rendezvous design handoff](../../reports/experiment_05d/rendezvous_handoff.md) ·
[Prospective protocol](protocol.md) · [Frozen selection](selection.json)

The main eight-loop base is already perfect on the 600 new small ID graphs.
Validation selects lambda zero; no primary scorer gain is established. A shallow
frozen recurrent readout can be repaired, predominantly in one seed, but a matched
direct candidate classifier is competitive. Recurrent distant-goal communication
works in controlled pairs; larger and long-path graph navigation still breaks.

36 main neural fits + 36 development pilot fits + two independent verification
refits. Seeds401/502/603, 3,000 disjoint graphs per base/scorer training stage,
600 validation, 2,240 ordinary final graphs and240 counterfactual pairs. BFS labels
are independently checked with Floyd–Warshall. Raw node flags/degree/adjacency
only at inference. All optimal first moves accepted; no route targets abstain.

The code is in `src/latent_loop/decision_bridge`. Use the existing `.venv` on Apple
Silicon; no new packages or pretrained model downloads are needed. CPU audits and
plotting require only the normal project dependencies. The separate CPU environment
under `artifacts/publication/cpu-env` is a local convenience, not a required public
artifact; install `.[test]` without the Apple extra on another machine.

```sh
LATENT_LOOP_MLX_TESTS=1 PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/python -m latent_loop.decision_bridge --help
PYTHONPATH=src .venv/bin/python -m latent_loop.decision_bridge example
PYTHONPATH=src .venv/bin/python -m latent_loop.decision_bridge.public_audit
PYTHONPATH=src .venv/bin/python -m latent_loop.decision_bridge.reproduce_data
PYTHONPATH=src .venv/bin/python -m latent_loop.decision_bridge analyze
PYTHONPATH=src .venv/bin/python -m latent_loop.decision_bridge report
```

The historical study is sealed. For live refitting, create a new reproduction
workspace with `python -m latent_loop.decision_bridge.retrain --destination
artifacts/experiment_05d/new_refit`; add `--all` for the complete36-fit study.
Existing destinations are protected. The default regenerates all data and fits
recurrent seed401 plus its correctness scorer. Full independent36-fit replication
has not been performed. Public scalar outcome audits, figures and datasets have
been regenerated, and the default two-fit verification reproduces exact hashes.

For a new study use staged commands `generate`, `train`, `freeze`, `evaluate`,
`rollout`, `latency`, `diagnostics`, `analyze`, `report` in a separate workspace.
Per-fit resume uses `train --condition recurrent --seed 401`;
add `--objective correctness` for its scorer. Every epoch persists model/optimizer
state and training receipts. No fitting is allowed after final opening. Seeds,
depths, count, memory and optimization budgets are configured in `config.json`.

The serial timing archive is retained; use the randomized interleaved supplement
for the report (`python -m latent_loop.decision_bridge.latency_interleaved`).
Normal desktop load and tiny timings limit speed comparisons. Scoring is still
executed in lambda-zero timing diagnostics; deployment can skip it.

The original easy-distribution pilot is preserved in
`results/experiment_05d_pilot_easy` and `pilot_easy_distribution/`. A transparent
post-freeze evaluation-only numerical-tie correction is recorded in
`evaluation_amendment.json`; no weights, data or selected hyperparameters changed.
Checkpoints/features remain ignored in `artifacts`, with public checkpoint hashes.
Original code and synthetic data use the inherited MIT license.
