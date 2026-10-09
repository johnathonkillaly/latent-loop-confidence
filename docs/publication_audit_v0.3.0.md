# v0.3.0 publication preparation audit

**Status: completed local research prepared for review; not published.** No v0.3.0 tag, branch push, release API mutation or publication action was performed. The user must review the consolidated snapshot before any future publication. No critical security/licensing blocker was found in this audit; scientific limitations are prominent release content.

## Snapshot and history

The verified starting branch was `research/exp05d-decision-bridge`, at `e1b02d3950bcd59b4de875549eb0da21dfc6304c`. The new local branch is **`research/exp05e-rendezvous`**. The verified predecessor commits include:

| Study | Scientific commit |
|---|---|
| 4B final | `66b530b7f1fe4f3f2f0b173b8e63a3dba008de00` |
| 5 | `17ae4e4acac70ef099a55522f0964381a01516d9` |
| 5B | `a6ccb384745c623f475e7b4b5fcc9699423b6e31` |
| 5C | `1f2e4947689984643bbadccb3327a8baec4cba81` |
| 5D | `e1b02d3950bcd59b4de875549eb0da21dfc6304c` |

The historical backup bundles remain outside the publication tree, including the verified 5D bundle. No historical commit was rewritten and no completed experiment artifact was overwritten. Current README, overview and limitations describe the consolidated draft; historical release audit documents and detailed experiment artifacts are preserved.

Read-only remote inspection confirms public `main` remains `1ebd435b94448d45e42141b49808839218c2b7b2`. Public v0.1.0 still peels to `0a637eb97d66eebbd740b612ab92e9095c496931`; v0.2.0 still peels to `1ebd435b94448d45e42141b49808839218c2b7b2`. The existing [v0.2.0 release](https://github.com/johnathonkillaly/latent-loop-confidence/releases/tag/v0.2.0) was verified with GitHub's release API. Remote refs contain no 5C/5D/5E branch or v0.3.0 tag. No unrelated remote work was changed.

## Scientific review

The draft preserves the distinctions between correctness discrimination, calibration, supervised loss improvement, actual correction/degradation, recurrent information propagation and reasoning capability. All ten new figures derive from recorded public decisions; charts and embedded HTML were regenerated and visually checked. Changed-answer examples include failures as well as corrections. The primary structural comparison and classifier strength check were frozen before extraction; final openings, selected checkpoints, dataset hashes and unchanged scientific-source hashes are recorded.

The release must retain these qualifications:

* Final-study fusion improves BaseLM **27.50% → 33.11%, +5.61 pp [2.89, 8.31]**, but **C2 reaches 42.58%** and a simple abstain prior **44.75%**. Score-only **42.25%** does not establish superiority over C2.
* All three primary head seeds improve over BaseLM; seed-instance decisions share 1,200 independent task instances. They are not 3,600 independent examples.
* Directed-graph fusion damages accuracy, novel option descriptions collapse both Q and C2, whole-policy permutation stability is poor, and fusion worsens confidence calibration. Candidate-description priors explain substantial performance; extra prompt-hidden information is not established by the controlled ablation.
* 5C demonstrates bounded spatial propagation without reliable long-path completion. 5D's one-loop gain is seed-dominated and eight-loop scoring is disabled by validation. Their failed and inconclusive controls are retained.
* Four-choice synthetic output adaptation does not establish general free-text generation improvement, underlying pretrained reasoning, or a special architecture beyond supervised candidate classification.

The closing recommendation is **PIVOT**, with no automatic continuation experiment.

## Reproducibility review

* **72 tests passed** with both MLX opt-ins enabled; **46 passed, 17 skipped** in the CPU-only environment. Skips are accelerator availability/opt-in checks, not silently counted as passes. The new experiment has seven targeted passing checks, including exact optimizer resume.
* Editable package installation, CLI help and installed dependency compatibility pass without upgrading dependencies.
* All **13,680 main observations** and **360 archived pilot observations** pass independent solvers. All eight main datasets regenerate byte-identically. An additional post-primary exploratory description-shift protocol contains 600 new independent tasks in 1,200 paired views; labels, hashes and unchanged semantic/label mappings pass its separate CPU audit. Canonical duplicate checks, candidate-row equivariance and public scalar decisions are audited.
* The structural paired estimates are independently recomputed from tracked decisions by the CPU-only public audit, without private checkpoints.
* All **24 main auxiliary head fits** are repeated from initialization in an isolated directory; selected checkpoint files are byte-identical. The refits reuse verified frozen features. Full independent feature-corpus extraction and complete historical retraining are **not** claimed.
* Actual pinned-Qwen native projection, deterministic evaluation, causal prefix and right-padding checks pass. Sixteen validation examples reproduce the cached decisions through full live policies. Parameter counts and required checkpoint-file hashes are verified.
* Full-policy latency includes tokenization, model forward, candidate extraction, head, final selection and synchronization. Peak MLX allocation is **3.817 GiB**, under 48 GiB. No remote GPU, full-backbone optimization or new transformer recurrence is used.

An unrestricted offline Hub metadata lookup initially rejected optional missing repository files; the original loader's explicit file patterns correctly located and verified the inference checkpoint. A helper initially targeted an uncreated temporary directory and was corrected before execution. The supplementary durable-I/O smoke initially exposed a relative-versus-absolute journal cleanup bug; it was corrected and covered by a relative-root regression and native MLX/NumPy serialization check. This wrapper was added after freezing and does not modify the frozen scientific source, objectives or models. Final test suites pass; no scientific-protocol amendment or final-based model reselection occurred.

## Security, privacy and artifact review

The scanner checks **all blobs reachable from canonical HEAD and public release tags**, plus every tracked/unignored proposed file. Gzip payloads are decompressed. The initial sweep covers 23 predecessor commits and 1,974 historical blobs; the final receipt includes all added working files. Checks cover recognizable credentials/private keys, tokens, private home paths/network addresses and accidental checkpoint/cache files. Findings are recorded as categories/paths only; secret values would never be printed. No findings or forbidden artifacts were detected. Canonical Git history uses the generic public research identity.

Human review also checks raw synthetic prompts, manifests, logs selected for publication, model cache exclusions, remotes and attribution. Infrastructure/tool names are fictional. No private employer data, internal hostnames, proprietary datasets, environment secrets, personal filesystem paths, model weights, optimizer arrays or hidden-state caches are included. `.gitignore` is supplemented by tracked-file/history inspection; no assumption is made that ignore rules remove committed content. Unrelated original private refs and verified backup bundles are not publication candidates.

The scanner is a bounded review, not a guarantee against every possible secret format. [Machine-readable security receipt](../results/experiment_05e/security_audit.json), [public analysis audit](../results/experiment_05e/public_analysis_audit.json), [dataset/protocol audit](../results/experiment_05e/dataset_and_protocol_audit.json), [checkpoint verification](../results/experiment_05e/model_checkpoint_verification.json), and [test receipt](../results/experiment_05e/test_verification.json) provide exact scope and limitations.

## Licensing and provenance

Original source and generated synthetic artifacts retain MIT. New graph/task solvers and candidate heads are original project code; no third-party implementation is vendored. Qwen is downloaded separately; its pinned official [LICENSE](https://huggingface.co/Qwen/Qwen3-0.6B/raw/c1899de289a04d12100db370d81485cdf75e47ca/LICENSE) was rechecked as Apache-2.0. Existing MLX/MLX-LM and dependency attributions remain in [third-party provenance](third_party_licenses.md). Papers are linked/paraphrased, not redistributed, and no official RecurTrace reproduction or architectural novelty is claimed.

## Review-ready material and remaining action

[Draft release notes](release_notes_v0.3.0.md), [reproduction guide](reproduction_v0.3.0.md), updated [overview](research_overview.md), [Rendezvous Markdown/HTML](../reports/experiment_05e/the_rendezvous.md), and [series conclusion](../reports/research_conclusion.md) are prepared locally. The completed branch is committed with a clean working tree and a verified external bundle at handoff. The final commit is identified by Git and the handoff, avoiding a self-referential hash inside its own content.

**Remaining publication action: user review and any later explicit release authorization. No publication is represented as completed.**
