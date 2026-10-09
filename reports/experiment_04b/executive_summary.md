# Experiment 4B executive summary

A fresh learned D head ranks decision transitions, but deployed benefit must be judged separately. A fresh-ID accuracy advantage over one pass is not established by the paired interval. The prespecified ranker does not establish an improvement over the same visited final answer.

The third head ranks actual corrections (WR AUROC 0.779, AP 0.147) and harmful loops (RW AUROC 0.794, AP 0.229). Event prevalence is 0.060/0.080. WR AP versus scalar C+V: +0.07 [+0.04, +0.09] (absolute AP units). RW AP versus scalar C+V: +0.13 [+0.10, +0.17] (absolute AP units).

D stopping scores 34.85%; earlier-state selection scores 34.88%. One pass scores 33.31%, frozen 33.80%. A single recurrent step scores 34.70%; D versus it is +0.15 [-1.34, +1.44] points. Selection recovers 3.88% of problems and loses a correct final answer on 3.86%. Its net difference versus the same visited final state is +0.03 [-1.13, +1.47] percentage points. A model that selects earlier still pays all 1.673 visited loops.

Full live inference is 25.27 ms/item for D and 25.57 ms/item for D plus selector, versus 24.91 ms/item for one pass on the same balanced validation subset. Batch-8 amortized latencies are 27.33/27.44/27.10 ms/item. Peak recorded Metal/RSS: 3.892/1.553 GiB.

The confirmatory results are the prespecified fresh-ID D/one-pass, selector/final, and WR/RW AP/scalar-CV contrasts. Shift, alternate stacks/classifiers, utility, selectors, and oracles are secondary/descriptive. All thresholds, calibrators, checkpoints and validation winners were sealed before 6,480 new final problems were scored. Paired intervals cluster problem instances and parent seeds; three parents and unadjusted multiple comparisons limit inference.

| Split | One pass | D stop | D + selector |
|---|---:|---:|---:|
| Fresh ID | 33.31% | 34.85% | 34.88% |
| New format | 27.19% | 28.03% | 28.42% |
| New structure | 33.77% | 34.08% | 33.95% |
| Harder depth | 32.54% | 35.52% | 35.60% |
| New range | 31.25% | 30.53% | 31.02% |

Completed experiment count: **36 supervised neural fits**, 27 main and 9 grouped OOF; eight substantive neural conditions across **three parent seeds**, plus random/prevalence/statistical controls. Best D architecture by validation NLL: ordinary CE gated four-class head in all parents. Best policy/selector is the per-parent validation-selected procedure below; it is not chosen by final accuracy.

- Seed 11: `d_utility` with `final`; D direct threshold 0.
- Seed 22: `stack_CD` with `final`; D direct threshold 0.1.
- Seed 33: `d_utility` with `final`; D direct threshold 0.

That validation-selected procedure scores 34.88%, paying 1.567 loops. The primary deployable selector remains 4.55 points behind its best-visited oracle. D versus a label-free matched-loop control: +1.62 [-0.15, +4.48] points.

The most important failure is the gap between transition prediction and useful deployed decisions: preventing an observed damaging loop can also forgo a later correction, and an earlier-state ranker can replace good answers with wrong ones. Check the paired accuracy and structural-transfer tables before claiming a better decision-maker. More computation is justified only where the measured accuracy gain and full runtime meet an explicit application preference; a probe score alone cannot decide that.

Ready for public review as a bounded synthetic research result. Keep the transition dataset, prospective lockbox, OOF/causality checks, work accounting and one-pass control. State memory was not launched. This does not demonstrate new reasoning, broad benchmark gains, or architectural novelty. Negative results and every A–H category are preserved.

Branch: `research/exp04-decision-value`; base commit `29140bd101e8a0c80ba5ef4cadf7000100e13327`. Scientific commit hash: recorded by the local commit receipt after completing the scientific snapshot. No remote was created or pushed. All historical experiments are preserved.

Review: [full report](decision_value_and_trajectory_selection.md), [self-contained HTML](decision_value_and_trajectory_selection.html), [measured summary](../../results/experiment_04b/summary.json), [portable audit](../../results/experiment_04b/public_audit.json).
