# Initial public release audit

This audit applies to the completed historical research through Experiment 4B, followed by a release documentation commit. Experiment 5 is excluded.

## Historical snapshot and preservation

The final Experiment 4B scientific commit is **`66b530b7f1fe4f3f2f0b173b8e63a3dba008de00`**, on `research/exp04-decision-value`. Its parent, **`29140bd101e8a0c80ba5ef4cadf7000100e13327`**, contains Experiment 4A and is the reported scientific base, not the completed 4B result.

| Historical milestone | Commit |
|---|---|
| Initial implementation | `566705745b03500da8292250e96ad72dfc03e307` |
| Both pilots, negative controls and middle probe | `e7ac160` |
| Original reports | `47b661e` |
| Original publication audit / historical main | `cd0a326db2203cdfec902affbc1f611641794ca3` |
| Experiment 3 | `701bcd4f2169d302259266156ee1a9cd06a05dae` |
| Experiment 4A | `29140bd101e8a0c80ba5ef4cadf7000100e13327` |
| Experiment 4B | `66b530b7f1fe4f3f2f0b173b8e63a3dba008de00` |

All seven commits form a linear scientific history. No separately numbered Experiment 2 report exists. Original reports, results, configurations, frozen protocols, and failures are retained. Release changes add public orientation, this audit, verification helpers, a scientific-commit receipt, and a self-contained 4A HTML rendering; README presentation is updated. No completed scientific result is replaced. A complete all-ref Git bundle was created and verified locally before publication. No destructive history rewrite was needed.

## Security, privacy and licensing

A heuristic scan inspected **all 480 reachable historical blobs**, including **109 decompressed gzip artifacts**, across all historical refs. It found zero secret-pattern, private-path, internal-name, private-network, credential-assignment, or text-email hits. No historical file exceeded 10 MiB; no model weights, NPZ hidden-state caches, private keys, or environment files were committed. Commit identities use a generic project author and an intentionally invalid email domain. Commit subjects/messages and the seven historical snapshots were reviewed. No proprietary input source, private training checkpoint, or third-party implementation was identified.

Tracked source, compressed prompts/predictions, experiment manifests, timing logs, Markdown, HTML and SVG provenance were inspected. Dataset generators create synthetic public examples and check labels with independent solvers. PNG figures were regenerated from recorded measurements; a representative uncertainty/failure chart was visually inspected. `.gitignore` was checked against actual tracked files and history, rather than treated as proof of exclusion. Local model caches, environments and regeneration workspaces remain ignored.

The local repository initially had no remotes or tags. The CLI account label was stale: the authenticated GitHub `/user` endpoint identifies `johnathonkillaly`. Publication uses that verified current account and `johnathonkillaly/latent-loop-confidence`, after inspecting the destination. No unrelated repository is overwritten.

The [pinned Qwen3 license](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE) was checked at its recorded revision: Apache-2.0. Weights/tokenizer are downloaded separately. Original project code and generated artifacts retain MIT. Installed packages are not vendored and retain their own terms. [Qwen3's technical report](https://arxiv.org/abs/2505.09388), [RecurTrace v2](https://arxiv.org/html/2609.03379v2), and the additional research references are attributed. No RecurTrace code was copied or redistributed, and no official reproduction is claimed. See [third-party provenance](third_party_licenses.md).

The scanner is a review aid, not proof against unknown secret formats or arbitrary sensitive prose. Its own pattern definitions are manually reviewed rather than reported as literal internal-name findings. Public author identity and repository ownership are necessarily visible; personal local identifiers are not included in research artifacts.

## Scientific checks

Saved full-precision metrics agree with all supplied 4B reference values after rounding: WR AUROC/AP/prevalence **0.779/0.147/0.060**, RW **0.794/0.229/0.080**; frozen/one-pass/fixed-one/D/D-selector accuracies **33.80/33.31/34.70/34.85/34.88%**.

D minus one pass is +1.54 points [−0.95, +3.88]. Selection minus the final D-visited state is +0.03 [−1.13, +1.47]. Primary decision advantages are not statistically established. Event prediction over scalar C+V is supported within this synthetic setting, with stronger controls limiting architectural attribution. Secondary harder-depth and fixed-four selection findings remain qualified and are not promoted to primary success. Label/future-state oracles are privileged diagnostics.

Public verification recomputes independent solver labels, losses, probabilities, transitions, paid-prefix boundaries, stop/continue logs, metrics and grouped OOF integrity. Correctness discrimination, calibration, supervised loss gain, correction, degradation and reasoning capability remain separate concepts. Reports preserve negative transfer and recurrence damage. Public figures use recorded metrics, not fabricated or paper-derived values.

## Verified reproduction scope

- Fresh isolated core installation on Python 3.12.15, without MLX: **21 passed, 8 intentional integration skips**.
- Existing pinned Apple environment with `LATENT_LOOP_MLX_TESTS=1`: **29 passed**.
- CLI entry point, configuration loading and fresh dataset generation passed.
- Explicit pinned model-download command passed; the exact snapshot was already cached.
- Fresh tiny real-model extraction, seven auxiliary fits and 120 evaluation rows completed in an isolated directory.
- Original, Experiment 3, 4A and 4B public analyses regenerate from committed measurements without model weights or MLX. Each stage runs in a **separate clean checkout** because preservation hashes include earlier stages' generated manifests. Use `python scripts/verify_public_reports.py` from a core-only environment.

This release verifies analysis reproduction and a tiny execution path. **Full historical training and independent regeneration of every intermediate checkpoint were not rerun.** Exact Metal timing is device/load dependent. Initial adaptive results retain a historical fallback that differs from current policy code. Refit checkpoints can have different hashes, so frozen historical protocols cannot be silently reused.

For complete independent retraining, use isolated workspaces and the dependency order in [reproduction guidance](reproduction.md): original pilot checkpoints → E3 probes → 4A trajectories → 4B heads/rankers. Refresh runtime clocks before generating a new protocol, remove only that workspace's derived freeze/lockbox/selection/cache/output metadata, and use independently seeded final sets. Published consumed lockboxes remain audit-only. Saved public predictions suffice to inspect the historical conclusions.

Release URLs and the exact publication commit are recorded in the GitHub release and a subsequent local publication receipt, avoiding a self-referential commit hash in this file.
