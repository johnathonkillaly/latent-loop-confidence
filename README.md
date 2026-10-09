# Latent Loop Confidence

*Confidence, Value of Computation, and Recurrent Decision-Making in Small Language Models*

Can a small neural network predict whether additional latent computation will improve a transformer's decision? This independent research project studies that question with frozen **Qwen3-0.6B**, small auxiliary heads, recurrent refinement, bounded loop memory, learned stopping, and selection among already visited recurrent states.

**A tiny recurrent network learned to improve decisions across repeated hidden-state updates, but recurrence superiority and scalable planning were not established.** Earlier Qwen studies found useful prediction signals without a reliable recurrent decision advantage. Grid planning and language-model reasoning are separate experimental systems.

**Local v0.3.0 draft:** Experiments 5C–5E are completed locally and prepared for review. Public releases remain v0.1.0 and v0.2.0. Nothing from this draft has been pushed or published.

The closing frozen-Qwen study improves constrained structural-choice accuracy **27.50% → 33.11%, +5.61 pp [2.89, 8.31]**, but a matched direct classifier reaches **42.58%** and a simple abstain prior **44.75%**. Novel option wording also collapses the learned heads. The recommendation is **PIVOT** toward carefully validated task adaptation and exact solvers. See the [Rendezvous report](reports/experiment_05e/the_rendezvous.md) and [research conclusion](reports/research_conclusion.md).

Three predictions have different targets:

| Signal | Target | Distinction |
|---|---|---|
| C — confidence | Current decision correctness | Discrimination and calibration are separate from changing decisions. |
| V — value of computation | Expected next-loop supervised loss improvement | Lower loss can occur without correcting an answer. |
| D — decision value | Beneficial and harmful decision transitions | Wrong→right correction and right→wrong degradation are measured separately. |

Later experiments add **Q**, a shared candidate-quality head that evaluates alternatives and can directly change the selected action. Historical C estimates correctness of the already selected answer. Q is compared with ordinary supervised candidate classification using matched features and parameters.

## Research record

The [research overview](docs/research_overview.md) links the questions, architectures, results, and limitations. Historical naming is preserved: the initial format pilot, primary pilot, and small middle-layer probe share the original report. There is no separately numbered Experiment 2 report; none has been invented.

| Milestone | Recorded finding | Detailed report |
|---|---|---|
| Initial confidence/refinement pilots | One-pass accuracy 50.28% versus frozen 44.83%; paired gain +5.44 points [2.11, 8.89]. Recurrence versus one pass −0.33 points [−2.33, 1.67]. | [Original report](reports/research_summary.md), [HTML](reports/research_summary.html) |
| Experiment 3 — Learning When to Think | Fresh-ID correctness AUROC 0.798; immediate-gain AUROC 0.762, AP 0.641 versus prevalence 0.381. Gain stopping 49.06% versus one pass 51.28%. | [Report](reports/experiment_03/learning_when_to_think.md), [HTML](reports/experiment_03/learning_when_to_think.html) |
| Experiment 4A — Foundation and Generalization | On the stronger generator mixture, historical correctness/gain AUROC falls to 0.530/0.534. Gain stopping 33.74% versus one pass 33.10%; paired interval includes zero. | [Report](reports/experiment_04a/foundation_report.md), [HTML](reports/experiment_04a/foundation_report.html), [Handoff](reports/experiment_04a/experiment_04b_handoff.md) |
| Experiment 4B — Decision Value and Trajectory Selection | Fresh-ID WR AUROC/AP 0.779/0.147; RW 0.794/0.229. D stopping 34.85%; D + selection 34.88%. No established primary decision or selector advantage. | [Report](reports/experiment_04b/decision_value_and_trajectory_selection.md), [HTML](reports/experiment_04b/decision_value_and_trajectory_selection.html) |
| Experiment 5 — Train the Thinker | Qwen recurrent 36.00% versus one-pass 32.87%; paired +3.13 pp [−2.90, +8.91]. No established depth gain. | [Report](reports/experiment_05/train_the_thinker.md), [HTML](reports/experiment_05/train_the_thinker.html) |
| Experiment 5.1 / 5B — Think Again | 18,884-parameter grid planner: 61.83/86.00/98.28/99.56% at 1/2/4/8 loops; four-versus-one +36.44 pp [32.55, 40.50]. Comparable four-block feedforward controls reach 97.72/98.33%. Qwen recurrence underperforms one-pass. | [Report](reports/experiment_05b/learning_to_navigate.md), [HTML](reports/experiment_05b/learning_to_navigate.html) |
| Experiment 5.2 / 5C — Break the Thinker | Frozen 9×9 eight-versus-four-loop gain +5.67 pp [3.56, 8.00]; useful propagation, failed long-path completion. | [Report](reports/experiment_05c/scaling_and_failure_frontier.md), [HTML](reports/experiment_05c/scaling_and_failure_frontier.html) |
| Experiment 5.3 / 5D — Decision Bridge | One-loop graph rescoring 79.39% → 84.89%, seed-dominated and matched by a classifier; eight-loop validation disables Q. | [Report](reports/experiment_05d/decision_bridge.md), [HTML](reports/experiment_05d/decision_bridge.html), [Handoff](reports/experiment_05d/rendezvous_handoff.md) |
| Experiment 5.4 / 5E — Rendezvous | Frozen Qwen structural fusion +5.61 pp [2.89, 8.31]; matched C2 beats fusion by 9.47 pp [6.64, 12.25]. | [Report](reports/experiment_05e/the_rendezvous.md), [HTML](reports/experiment_05e/the_rendezvous.html) |

These rows use different datasets and are not a longitudinal accuracy leaderboard. Three seeds share each experiment's held-out problems. Reports distinguish prospective primary comparisons, exploratory findings, negative results, inconclusive comparisons, and privileged label-informed oracle bounds.

## Experiment 4B in detail

| Fresh-ID decision system | Accuracy |
|---|---:|
| Frozen backbone | 33.80% |
| Historical one-pass head | 33.31% |
| Single recurrent step | 34.70% |
| D stopping | 34.85% |
| D + trajectory selection | 34.88% |

WR event prevalence is **0.060**, versus AUROC **0.779** and AP **0.147**; RW prevalence is **0.080**, versus AUROC **0.794** and AP **0.229**. These are rounded recorded values, with full precision in the [saved summary](results/experiment_04b/summary.json).

The prospective primary D-versus-one-pass comparison is **+1.54 percentage points [−0.95, +3.88]**. Selection versus the final visited state is **+0.03 points [−1.13, +1.47]**. Both are inconclusive. Event AP improves over the prespecified scalar C+V control; a stronger scalar control and a plain parameter-matched classifier constrain architectural interpretations. Selection after four paid loops recovers damaged answers, but does not establish an advantage over simple maximum-C selection. Oracles use labels/future states and are unavailable at inference.

Most historical recurrence updates operate on the final decision vector; the original middle-layer experiment is a separate three-scalar feasibility probe. Useful correctness, loss-gain, and transition discrimination do not demonstrate better underlying reasoning. Eight-loop final-vector refinement damaged accuracy; template and structural transfer exposed substantial limitations. Full-path timing does not establish a deployment speedup.

## Reproduce and audit

Python **3.12** was used; package metadata permits Python ≥3.11. Model-scale runs require Apple Silicon, MLX/Metal, and the pinned public checkpoint. CPU tests, public metric audits, and report regeneration need no model or accelerator:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
.venv/bin/latent-loop --help
.venv/bin/python -m latent_loop.cli report --config configs/pilot.yaml
.venv/bin/python -m latent_loop.experiment03.run report
.venv/bin/python -m latent_loop.experiment04a.pipeline report
.venv/bin/python -m latent_loop.experiment04b.pipeline report
.venv/bin/python -m latent_loop.experiment04b.pipeline verify
```

Run report commands in an isolated checkout: regeneration updates derived reports/audit timestamps. For Apple model runs:

```sh
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e '.[apple,test]'
LATENT_LOOP_MLX_TESTS=1 .venv/bin/python -m pytest -q
.venv/bin/python -m latent_loop.cli download --config configs/pilot.yaml
.venv/bin/python -m latent_loop.cli run --config configs/pilot.yaml --run-dir artifacts/reproduction
```

The model is `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, in FP32 after a recorded BF16 parity failure. Training seeds are **11, 22, 33**; manifests record dataset seeds, independent solvers, configurations, and protocol hashes. See [complete reproduction guidance](docs/reproduction.md) and the [release audit](docs/publication_audit.md).

**Verified reproduction scope:** package installation, CPU and MLX integration tests, fresh tiny real-model execution, dataset/configuration loading, and public report/metric regeneration. Full historical training was not rerun for this release. Weights and large hidden tensors are excluded. Their regeneration requires the pilot → Experiment 3 → 4A → 4B dependency chain; independent refits can change checkpoint hashes and need a fresh protocol. Consumed lockboxes deliberately reject reselection. Exact initial adaptive rows also retain an older stopping fallback. Analysis reproducibility is stronger than verified end-to-end retraining reproducibility.

## Limitations and roadmap

Six synthetic task families, conditional four-choice predictions, one small frozen model, few seeds, shared algorithm families, scarce transition events, and small hardware timing samples limit interpretation. Several tasks remain near chance. Generalization across arbitrary task structures, reliable recurrence accuracy gains, useful trajectory selection, and broad reasoning improvement remain unestablished. Calibration, loss improvement, decision changes, and recurrent reasoning capability are distinct claims.

Experiment 5 trains Qwen middle-layer LoRA adapters; its paired gains remain inconclusive. Experiment 5B shows useful repeated local grid computation, while eight-loop completion is 95% on 7×7 and only 54.44% on the long-path subset. Snake horizon-six agreement is 96.61% at four loops, equaled by its four-block feedforward control. Qwen maze recurrence scores 28.33% versus one-pass 47.78%; no transfer of the grid benefit was established.

Experiments 5C, 5D and 5E are completed in this local draft. 5C exposes a failure frontier; 5D connects recurrence to candidate scoring; 5E returns to frozen language-model output intervention. Scoring improves some constrained decisions, but structural graph harm, label sensitivity and competitive ordinary classifiers rule out broad reasoning or architectural claims. No further experiment starts automatically.

See the **unpublished** [v0.3.0 draft notes](docs/release_notes_v0.3.0.md), [audit](docs/publication_audit_v0.3.0.md), [reproduction guide](docs/reproduction_v0.3.0.md), and [conclusion](reports/research_conclusion.md). The [v0.2.0 notes](docs/release_notes_v0.2.0.md), [audit](docs/publication_audit_v0.2.0.md) and [reproduction guide](docs/reproduction_v0.2.0.md) remain historical release records.

## Attribution and license

Original project code and generated synthetic artifacts use [MIT](LICENSE). The separately downloaded Qwen3 model uses Apache-2.0; its weights/tokenizer are not distributed here. Installed dependencies retain their own licenses. See [third-party provenance](docs/third_party_licenses.md).

This work is inspired by [Qwen3](https://arxiv.org/abs/2505.09388), [RecurTrace](https://arxiv.org/html/2609.03379v2), [Universal Transformers](https://arxiv.org/abs/1807.03819), [Adaptive Computation Time](https://arxiv.org/abs/1603.08983), [PonderNet](https://arxiv.org/abs/2107.05407), and [Encode-Think-Decode](https://arxiv.org/abs/2510.07358). The [literature review](docs/literature_review.md) explains the differences and additional references. No RecurTrace implementation was copied. This is an independent adaptation, with no claim of official reproduction or architectural novelty.
