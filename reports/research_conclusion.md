# Latent Loop Confidence — research conclusion

## Decision

**PIVOT — useful mechanisms exist, but the proposed scoring/recurrence architecture is not competitive with simpler alternatives.** Small supervised heads can improve constrained choices from a frozen representation. Shared local recurrence can propagate useful information. Neither result establishes a general reasoning architecture, a distinctive advantage over direct classification, or reliable long-horizon planning.

The closing [Rendezvous experiment](experiment_05e/the_rendezvous.md) improves frozen Qwen3's structural-mixture action accuracy **27.50% → 33.11%, +5.61 pp [2.89, 8.31]** across three positive seeds. The matched direct classifier reaches **42.58%**; fusion trails it **9.47 pp [6.64, 12.25]**. Score-only reaches **42.25%**, inconclusive against that classifier. A simple abstain prior reaches **44.75%** on the shifted mixture. This supports bounded supervised output adaptation while arguing against special architectural claims.

Experiments use different datasets, hardware procedures and training objectives. Their headline percentages are not a longitudinal leaderboard. Historical negative and inconclusive findings remain part of the result.

## A. Did the initial confidence-head idea work?

Partly, as a discrimination tool. The initial pilot showed that frozen hidden states contain useful information for task-specific prediction: one-pass accuracy **50.28%** versus frozen **44.83%**, paired +5.44 pp [2.11, 8.89]. Recurrence did not improve reliably over that one-pass head: −0.33 pp [−2.33, 1.67]. Early answer-prefix and shared-template limitations constrain this evidence.

Experiment 3 correctness AUROC reached **0.798** on its fresh-ID distribution. Experiment 4A's stronger task generators exposed poor transfer: historical C/V AUROC fell to **0.530/0.534**. Recognizing some incorrect answers is useful for diagnostics, but it does not identify a correct replacement or establish trustworthy calibrated confidence. The final experiment's Q-at-base-selected-answer correctness AUROC is **0.696 on fresh ID**, but **0.560 on structural shift**. Old C checkpoint weights are not assumed to transfer.

[Initial report](research_summary.md) · [Experiment 3](experiment_03/learning_when_to_think.md) · [Experiment 4A](experiment_04a/foundation_report.md)

## B. Did learned value-of-computation estimation work?

It predicted properties of some trajectories better than simple confidence controls. Experiment 3 immediate-gain AUROC **0.762**, versus confidence-only **0.695**, indicates information about next-loop supervised loss change. Its learned stopping accuracy **49.06%** nevertheless trailed one pass **51.28%**. Loss improvement and decision correction are different outcomes.

Experiment 4B made that distinction explicit. Wrong→right event AUROC/AP were **0.779/0.147** at prevalence **0.060**; right→wrong **0.794/0.229** at prevalence **0.080**. These rare-event prediction signals did not establish deployed benefit. D stopping **34.85%** versus one-pass **33.31%** had paired +1.54 pp [−0.95, 3.88]. D plus trajectory selection **34.88%** versus the final visited state differed only +0.03 pp [−1.13, 1.47]. Privileged label/future-state oracles are opportunity bounds, not deployable policies.

Prediction quality alone did not solve the problem that the available recurrent computation often damaged the decision.

[Experiment 4B](experiment_04b/decision_value_and_trajectory_selection.md)

## C. Did recurrence work?

**Useful local information propagation worked; general transformer reasoning and scalable planning did not follow.** These are separate systems and claims.

Experiment 5 trained lightweight shared middle-layer adapters in frozen Qwen3. Four-loop accuracy **36.00%** versus one-pass **32.87%** had an inconclusive paired interval. Depths 1–4 were **36.46%, 37.15%, 36.34%, 36.00%**. Depth four minus one was −0.46 pp [−3.13, 1.50]. Simultaneous multi-depth supervision performed near chance, generalization stayed weak, and recurrence cost more inference time.

The 18,884-parameter grid planner in Experiment 5B was much clearer: the same weights at 1/2/4/8 iterations scored **61.83/86.00/98.28/99.56%**. Four versus one improved **36.44 pp [32.55, 40.50]**. Four-block feedforward controls reached **97.72% and 98.33%**, so weight sharing was not established as uniquely valuable. Eight-loop 7×7 completion reached **95%**, but long-path completion only **54.44%**. Qwen maze recurrence scored **28.33%**, worse than one-pass **47.78%**. Snake horizon-six agreement **96.61%** was matched by feedforward computation; safety, ties and unavoidable outcomes complicate the planning interpretation.

Experiment 5C froze the original grid weights and reproduced archived outputs. On new 9×9 mazes, eight versus four loops improved action accuracy **88.72% → 94.39%, +5.67 pp [3.56, 8.00]**. Extra computation sometimes extended useful spatial influence. But 32 loops declined to **93.22%**, new recurrent and deep feedforward first-action accuracy were comparable, and long-path navigation remained essentially unsolved. The retrained recurrent planner's long-path completion was **0% at eight loops** and **1.39% at 16/32**, despite higher first-action accuracy. Local interventions and paired remote-goal changes support bounded propagation, not a learned general BFS algorithm.

Experiment 5D's arbitrary-graph network reached **100% fresh-ID at eight loops**, while larger/longer graphs failed: 64-node accuracy **68.19% at eight / 73.47% at sixteen**, and long-path **0% / 28.56%**. Strong untied feedforward controls remained competitive or better. Recurrence can make previously inaccessible information available, but useful information does not guarantee robust repeated action selection or full planning.

[Experiment 5](experiment_05/train_the_thinker.md) · [5B](experiment_05b/learning_to_navigate.md) · [5C](experiment_05c/scaling_and_failure_frontier.md) · [5D](experiment_05d/decision_bridge.md)

## D. Did candidate scoring change decisions for the better?

Yes in bounded conditions, with material qualifications. Q evaluates alternative candidates; historical C only estimates correctness of the selected decision. They are different targets.

In Experiment 5D, a frozen one-loop graph policy improved **79.39% → 84.89%**, but the gain was concentrated in one training seed; direct classification reached **85.39%**. At eight loops the base already reached 100% fresh-ID and validation chose lambda zero. The primary scoring intervention was therefore disabled. Scoring did not repair missing distant information.

Experiment 5E produces a new prospective positive comparison against frozen Qwen on a structural task mixture: **+5.61 pp [2.89, 8.31]**, with seed effects **+6.50, +4.67, +5.67 pp**. Across three heads on 1,200 independent instances it corrected **536** and damaged **334** seed-instance decisions. These are actual selected-output changes, not merely correctness AUROC.

But the same fusion **damaged directed-graph accuracy by 8.25 pp**, trails a fair classifier, and is highly presentation-sensitive. Semantic agreement after candidate relabeling is **51.11%** for fusion, and only **37.78%** after symbol-plus-candidate relabeling. Aggregate improvement does not establish reliable action selection under arbitrary shifts.

[5D handoff](experiment_05d/rendezvous_handoff.md) · [Rendezvous](experiment_05e/the_rendezvous.md)

An additional **exploratory post-primary paired wording diagnostic** exposes a further failure. On 600 new tasks with the same semantic candidates but novel option descriptions, Q-only falls **42.83% → 26.78%**, and C2 **42.33% → 24.50%**. Fusion versus BaseLM becomes inconclusive: **+2.17 pp [−0.67, 4.61]**. Both candidate models are template-sensitive; direct classification is not a robust deployment solution merely because it wins the original comparison.

## E. Is a scoring head preferable to ordinary supervised classification?

No advantage was established. In the final study, Q and the strong C2 control use the **same 197,057-parameter shared candidate architecture**, inputs, examples, optimizer budget and checkpoint-selection rules. Their losses and deployment rules differ. Calling one a scorer does not make it an architectural alternative to classification.

Q-only and C2 structural accuracies **42.25% / 42.58%** differ −0.33 pp [−2.08, 1.53], inconclusive. Selected logit fusion is significantly worse than C2. The effective no-logits prompt-hidden head only exceeds a candidate-description-only control by **0.69 pp [−0.89, 2.47]**. The incremental value of prompt-hidden information is not established by this ablation; candidate text priors explain substantial performance. A simple abstain prior beats all fitted methods on that structural mixture.

Base temperature scaling improves calibration without changing any action. Fusion improves top-one accuracy but worsens structural NLL **2.376 → 2.814** and ECE **0.350 → 0.422**. The fair control distinguishes ordinary supervised adaptation from confidence calibration and from a special new decision mechanism.

## F. Did any result justify architectural integration into an autoregressive LLM?

The evidence supports a **small external task-specific constrained-choice adapter**, if its real workload is validated. Its measured incremental batch-one cost was **0.30 ms (~1.2%)** with about **0.77 MiB** additional active Metal allocation for one head. That is practical engineering scale.

It does not support training a new LM architecture, recurrent decoder, halting controller or general free-text scorer. Four-label conditional next-token selection cannot establish better unrestricted language generation. Every head was trained with thousands of supervised synthetic labels. No intrinsic pretrained reasoning improvement or weight transfer from grid/graph systems was shown. The model-independent candidate interface and paired evaluation methodology are more reusable than these task-specific weights.

## G. What should not be pursued further without new evidence?

* Additional Qwen recurrent-layer, loop-memory or adaptive-halting complexity intended to rescue weak supervised task competence.
* Claims that correctness/value AUROC alone imply better final decisions or useful reasoning.
* Claims that shared recurrent weights outperform untied feedforward computation without matched data, parameters and measured latency.
* General planning claims based on first-action accuracy when complete navigation still cycles or fails.
* Architectural novelty claims for a shared candidate MLP that is mathematically an ordinary classifier.
* Deployment based on aggregate synthetic gains while ignoring damaged decisions, candidate-order sensitivity, abstain priors and absolute error rates.

## H. Smallest useful next step

If practical constrained routing remains useful, run **one preregistered replication of the logit-aware direct candidate classifier against candidate-description priors and an exact rule solver on fresh held-out fictional tool-capability compositions**, with unfamiliar option descriptions and reporting abstain/non-abstain strata separately and preserving realistic overall prevalence. Require paired net correction, presentation stability and a clearly specified deployment threshold. This tests ordinary task adaptation, not a new general reasoning architecture. It does not justify further recurrent LM training. No follow-up is started here.

## Reproducibility and publication status

Reports, public measurements, generators, configurations, solvers and tests are preserved. The final study's 13,680 main observations independently validate and regenerate byte-identically; all 24 heads refit to byte-identical checkpoints using the recorded frozen features. Native Qwen parity, causal behavior, live-versus-cached decisions, candidate-row equivariance, optimizer restore and checkpoint hashes are checked. Full historical end-to-end retraining and independent complete frozen-feature re-extraction were not performed. Model weights and large caches are excluded.

Experiments 5C/5D/5E and the consolidated **v0.3.0 draft remain local**. Public v0.1.0 and v0.2.0 are unchanged. This closes the requested research run with a bounded positive intervention result and substantial negative architectural/generalization findings.
