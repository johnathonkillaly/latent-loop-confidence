# Latent Loop Confidence — research overview

Can auxiliary networks recognize when another latent computation will improve a decision? The historical work separates correctness (C), next-loop supervised loss gain (V), and correction/degradation value (D). Prediction signals can be useful even when the recurrent computation they describe is unreliable.

| Study | Research question | Architecture | Key recorded result | Important limitation | Report |
|---|---|---|---|---|---|
| Initial pilots and middle-layer probe | Can confidence/refinement recover decisions from a frozen representation? | Frozen Qwen3; final-vector heads, tied residual refinement, bounded memory; separate repeated-middle-block probe | One pass 50.28% versus frozen 44.83%; recurrence advantage unestablished | Initial answer-format artifact; shared templates; eight-loop degradation; middle probe trains only three scalars | [Detailed report](../reports/research_summary.md) |
| Experiment 3 | Does next-loop loss value differ from confidence? | Frozen parent refiners; C and immediate-gain probes; stopping controls | Correctness AUROC 0.798; gain AUROC 0.762 versus confidence-only gain 0.695 | Gain stopping 49.06% versus one pass 51.28%; weaker template transfer | [Detailed report](../reports/experiment_03/learning_when_to_think.md) |
| Experiment 4A | Do signals survive stronger tasks and structural shifts? | Historical frozen heads; fresh scalar controls/calibrators; causal trajectory exports | Fresh-ID historical C/V AUROC 0.530/0.534; scalar gain 0.580 | Weak transfer; poor task competence; no newly trained neural heads | [Detailed report](../reports/experiment_04a/foundation_report.md) |
| Experiment 4B | Can decision transitions guide paid stopping and state selection? | Fresh four-event D head, grouped OOF rankers, frozen parent trajectories | WR AUROC/AP 0.779/0.147; RW 0.794/0.229; D 34.85% | D-versus-one-pass and selection-versus-final paired intervals include zero | [Detailed report](../reports/experiment_04b/decision_value_and_trajectory_selection.md) |

| Experiment 5 | Can trained middle-layer recurrence improve Qwen decisions? | Frozen Qwen; shared layers 12–14, rank-8 LoRA and bounded gates | Rec4 36.00% versus one-pass 32.87%; interval includes zero | Depth4 versus depth1 −0.46 pp [−3.13, +1.50]; weak transfer | [Detailed report](../reports/experiment_05/train_the_thinker.md) |
| Experiment 5.1 / 5B | Can repeated unchanged-input computation help navigation? | Tiny shared local gated grid network; separate Qwen and Snake controls | Grid4 98.28%, grid8 99.56%; four-versus-one +36.44 pp [32.55, 40.50] | Feedforward controls tie; long-path completion 54.44%; negative Qwen transfer | [Detailed report](../reports/experiment_05b/learning_to_navigate.md) |

Historical naming is retained. The original pilots share one report; no separate Experiment 2 report is present. Data distributions change between studies, so the aggregate accuracies cannot be compared as successive model improvements.

```mermaid
flowchart LR
    X[Prompt] --> Q[Frozen Qwen3]
    Q --> H[Final hidden state]
    H --> R[Tied final-vector refinement]
    R --> R
    R --> C[C: current correctness]
    R --> V[V: next loss gain]
    R --> D[D: correction and degradation]
    C --> P[Label-free stopping and visited-state selection]
    V --> P
    D --> P
    P --> Y[Four-choice decision]
```

The diagram describes the primary historical architecture. The separate middle-layer probe repeats transformer layers before the late decoder; it does not reproduce RecurTrace's loop-time memory protocol.

The initial pilots and most secondary comparisons are exploratory. Experiment 4B prospectively identifies three primary comparisons: D versus one pass, selection versus the same visited final state, and event AP versus scalar C+V. The first two are inconclusive; the event-prediction comparison supports useful discrimination within this benchmark. Label-informed oracles are privileged opportunity bounds. Negative transfer and deeper-loop harm are retained.

## Roadmap

**Experiment 5 — Training Recurrent Computation (planned):** train the thinker itself using frozen-backbone middle-layer LoRA, bounded reinjection, and multi-depth supervision. Use fresh training/validation and a prospective final lockbox. Compare recurrence to trained one-pass adaptation at equal parameter counts and measured compute. Assess fixed depths before attempting C/V/D transfer or a new controller.

See [reproduction guidance](reproduction.md), [publication audit](publication_audit.md), [limitations](limitations.md), and [research references](literature_review.md).

## Completed v0.2.0 and next planned study

A tiny recurrent network learned to improve decisions across repeated hidden-state updates, but recurrence superiority and scalable planning were not established. Snake's 96.61% horizon-six agreement is matched by feedforward computation; ties and unavoidable-collision states limit interpretation. Qwen maze recurrence (28.33%) underperforms one-pass adaptation (47.78%). Full episodes expose failures hidden by first-action accuracy.

Experiment 5 is now completed; any earlier planned wording above describes the v0.1.0 roadmap. Planned Experiment 5.2, **Does Thinking Scale With Distance?**, will test larger boards, longer graph distances, extra loops and untied/global controls. New experimental results are excluded from this release.

[Release notes](release_notes_v0.2.0.md) · [Audit](publication_audit_v0.2.0.md) · [Reproduction](reproduction_v0.2.0.md)
