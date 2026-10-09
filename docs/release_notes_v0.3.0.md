# Latent Loop Confidence v0.3.0 — Final Results

This release adds completed Experiments 5.2–5.4 to v0.2.0 and closes the research series with a **PIVOT** conclusion. Earlier releases and all historical measurements are preserved.

* **5.2 — Break the Thinker:** more recurrent computation improves larger-maze first decisions, but long-path navigation remains unreliable and strong feedforward alternatives are competitive. [Report](../reports/experiment_05c/scaling_and_failure_frontier.md)
* **5.3 — The Decision Bridge:** candidate scoring improves a shallow frozen graph policy, with seed-dominated gains matched by direct classification; validation disables scoring when deeper recurrence already solves ID decisions. [Report](../reports/experiment_05d/decision_bridge.md)
* **5.4 — The Rendezvous:** candidate-score fusion improves frozen Qwen3-0.6B structural choices from **27.50% to 33.11%**, a paired **+5.61 percentage points (95% CI [2.89, 8.31])**. A matched direct classifier reaches **42.58%**, beating fusion by **9.47 points (95% CI [6.64, 12.25])**. Novel candidate wording and structural changes expose substantial weaknesses. [Report](../reports/experiment_05e/the_rendezvous.md)

Tiny recurrent grid and graph networks benefit from extra hidden-state updates, but do not establish a general advantage over strong feedforward models. Recurrent Qwen experiments do not establish useful improvements over simpler architectures. Four-choice synthetic adaptation is not general-purpose reasoning, architectural novelty or robust confidence-guided generation.

[Research conclusion](../reports/research_conclusion.md) · [Research overview](research_overview.md) · [Reproduction instructions](reproduction_v0.3.0.md) · [Publication audit](publication_audit_v0.3.0.md) · [Limitations](limitations.md)

Public code, synthetic data, scalar decisions, figures and HTML reports are included; model weights and large caches are excluded. Existing solver, test, regeneration and cached-feature refit receipts are retained. Full independent historical retraining and complete independent feature re-extraction are not claimed. Original code is MIT; separately downloaded Qwen weights are Apache-2.0.
