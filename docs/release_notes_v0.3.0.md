# v0.3.0 — draft release notes

**Latent Loop Confidence — The Rendezvous: Scoring, Recurrence and Decision Making**

Prepared locally for review. **No v0.3.0 tag or GitHub release has been created or published.** Existing v0.1.0/v0.2.0 releases are unchanged. The proposed release includes the completed local Experiments 5C, 5D and 5E on top of v0.2.0.

## Research result

A learned candidate head can improve frozen Qwen3's constrained selected outputs on this new synthetic structural mixture: **27.50% → 33.11%, +5.61 percentage points [2.89, 8.31]** across three positive seeds. It **does not beat ordinary supervised candidate classification**, which reaches **42.58%** with the same architecture, features, labels and training budget. Score-only reaches **42.25%**, inconclusive against classification. Simple abstain priors, graph damage and presentation sensitivity limit interpretation. This is not evidence of improved general autoregressive generation. A supplementary paired option-description shift collapses Q-only to 26.78% and C2 to 24.50%; fusion versus BaseLM is inconclusive there.

## Included studies

* **5.2 / 5C — Break the Thinker:** frozen-grid baseline reproduced; larger boards, path lengths, depth extrapolation, untied/global controls, full episodes and a failure frontier. The primary 9×9 eight-versus-four-loop gain is **+5.67 pp [3.56, 8.00]**. Useful local propagation coexists with essentially failed long-path navigation. [Report](../reports/experiment_05c/scaling_and_failure_frontier.md)
* **5.3 / 5D — The Decision Bridge:** arbitrary graph recurrence, candidate-aware Q, separate frozen-base fitting, classifier and compute controls, distant counterfactual pairs, larger graphs and complete navigation. One-loop **79.39% → 84.89%** scoring improvement is seed-dominated and matched by direct classification. At eight loops validation disables scoring. [Report](../reports/experiment_05d/decision_bridge.md), [LLM handoff](../reports/experiment_05d/rendezvous_handoff.md)
* **5.4 / 5E — The Rendezvous:** pinned frozen Qwen3-0.6B, verified single-token labels, three public synthetic task families, 24 primary head fits plus 24 verification refits, paired structural confirmation and direct-classifier controls. Fusion corrects 536 and damages 334 seed-instance decisions over three heads × 1,200 independent structural problems. Matched direct classification beats selected fusion by **9.47 pp [6.64, 12.25]**. [Report](../reports/experiment_05e/the_rendezvous.md)
* **Research conclusion:** a recommendation to **PIVOT** toward ordinary constrained task adaptation and exact external solvers, rather than additional unsupported recurrent-LM complexity. [Conclusion](../reports/research_conclusion.md)

## Limitations and cost

The same trained recurrent network can benefit from more hidden-state updates, but this does not demonstrate recurrence superiority or scalable planning. Scoring can improve frozen four-choice decisions, but classifiers and candidate priors explain much of the value. Structural graph accuracy worsens; permutation stability is poor; synthetic absolute accuracy is low. Three seeds give limited training-distribution uncertainty. Confirmatory, exploratory, negative, privileged and inconclusive results remain distinguished in the detailed reports.

Final-study full batch-one latency is **25.326 ms BaseLM / 25.626 ms fusion / 25.684 ms C2** on 140–200-token validation prompts. Peak MLX allocation is **3.817 GiB**, below the 48 GiB cap. These are synchronized full-prefix measurements, not cached-feature speedups. No transformer recurrence is trained in 5E.

## Reproduce and review

[Reproduction guidance](reproduction_v0.3.0.md) · [Publication audit](publication_audit_v0.3.0.md) · [Research overview](research_overview.md) · [Limitations](limitations.md)

Public metrics, deterministic synthetic data, solver checks, figures and self-contained HTML reports are included. Checkpoints and large hidden-state tensors are excluded; instructions describe their regeneration. Final-study head refits are byte-identical from verified cached features. Full independent retraining of the historical series and complete independent feature re-extraction are not claimed. Original code/generated artifacts retain MIT; Qwen weights are separately downloaded under Apache-2.0.

There is no automatically scheduled next experiment or publication. Review the completed snapshot and audit before any future release action.
