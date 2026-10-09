# Latent Loop Confidence v0.2.0 — Learning to Navigate with Recurrent Computation

Can a model improve its decision by repeatedly updating hidden state over the same input? This release preserves Experiments 1–4B and adds completed Experiments 5 and 5.1/5B.

**A tiny recurrent network learned to improve decisions across repeated hidden-state updates, but recurrence superiority and scalable planning were not established.**

- Experiment 5: frozen Qwen3-0.6B with shared middle-layer LoRA. Recurrent accuracy 36.00% versus one-pass 32.87%; paired +3.13 percentage points [−2.90, +8.91]. Depth-dependent benefits were not established; simultaneous supervision was near chance.
- Experiment 5B: 18,884-parameter grid planner reaches 61.83%, 86.00%, 98.28%, 99.56% at 1/2/4/8 internal loops. Four versus one: +36.44 points [32.55, 40.50]. The board remains unchanged within a decision.
- Four-block feedforward controls reach 97.72% and 98.33%; recurrence superiority is unestablished. Extra loops increase executed computation.
- Eight-loop full-maze completion is 95% on 7×7 but 54.44% on long paths. First-move accuracy overstates full navigation reliability.
- Qwen maze recurrence reaches 28.33% versus one-pass 47.78%; paired −19.44 points [−34.44, −7.78]. Grid success did not transfer to the transformer in this study.
- Separate prospective Snake study: four-loop horizon-six teacher agreement 96.61%, equaled by four-block feedforward. Full-episode survival improvement is inconclusive; food collection does not reliably improve with depth.

All studies retain seed-level results, failed variants, independent solver checks, paired uncertainty, timing and limitations. Results concern small synthetic tasks, not general reasoning. Reports contain historical local-status wording reflecting their execution date; this release publishes those completed records unchanged.

See [research overview](research_overview.md), [reproduction instructions](reproduction_v0.2.0.md), [publication audit](publication_audit_v0.2.0.md), [E5 report](../reports/experiment_05/train_the_thinker.md) and [E5B report](../reports/experiment_05b/learning_to_navigate.md). Public analyses regenerate from measurements without weights. Model inference and training require separately regenerated ignored checkpoints; full independent retraining was not repeated for release validation.

Next planned: Experiment 5.2, **Does Thinking Scale With Distance?** Fresh locked scale/distance/depth tests and stronger feedforward controls. No Experiment 5.2 work is included. The v0.1.0 tag is preserved.
