# Latent Loop Confidence v0.1.0 — Initial Research Results

The initial hypothesis asks whether small auxiliary networks can predict when additional latent computation will improve a frozen transformer's decision.

This release preserves the original confidence/recurrent-refinement pilots and middle-layer feasibility probe, Experiment 3 (Learning When to Think), Experiment 4A (Foundation and Generalization), and Experiment 4B (Decision Value and Trajectory Selection), including negative and inconclusive findings.

Correctness, loss improvement, and actual transitions have useful predictive signals on some distributions. Experiment 4B fresh-ID WR AUROC/AP is 0.779/0.147 (prevalence 0.060); RW is 0.794/0.229 (prevalence 0.080).

Decision accuracy is frozen 33.80%, one pass 33.31%, one recurrent step 34.70%, D stopping 34.85%, D + selection 34.88%. D minus one pass is +1.54 percentage points [−0.95, +3.88]; selection minus final visited state is +0.03 [−1.13, +1.47]. Neither establishes the intended primary advantage. Longer recurrence can damage decisions; transfer to stronger tasks weakens old probe signals. Prediction performance is not evidence of improved underlying reasoning.

The README and research overview link public measurements, manifests, reports, and figures. CPU report regeneration and metric audits require no weights. Python 3.12, pinned MLX/Metal dependencies, and the exact Apache-2.0 Qwen3 revision support Apple Silicon model runs. Tests and an isolated real-model smoke run were verified; full historical retraining was not rerun. Ignored checkpoints/caches must be regenerated in experiment dependency order, and consumed lockboxes require fresh protocols for independent refits.

Limitations include synthetic conditional four-choice tasks, one small backbone, three shared-item training seeds, near-chance performance on several tasks, scarce transition events, and small latency samples. Privileged oracles are not deployable systems. Original code uses MIT; third-party weights are excluded.

Planned Experiment 5 will train recurrent middle-layer computation and compare it with a parameter-matched one-pass adapter. No Experiment 5 implementation, unfinished training, or result is included in v0.1.0.
