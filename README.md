# latent-loop-confidence

A public research pilot asking whether confidence and decision-value heads can guide latent recurrence around a frozen language model, improving decisions without generated reasoning tokens.

**Main finding:** a small **one-pass head** was as effective as recurrence at lower additional compute. On 600 held-out synthetic decisions, its mean accuracy was **50.28%**, versus **44.83%** for frozen Qwen3-0.6B. The improvement was **5.44 percentage points**, with an item-paired bootstrap 95% interval of **[2.11, 8.89]** points. Recurrence versus the one-pass head was **−0.33 points**, interval **[−2.33, 1.67]**; a recurrence advantage was not established.

| Variant | Test accuracy, mean ± seed SD | OOD accuracy |
|---|---:|---:|
| Frozen conditional model | 44.83 ± 0.00% | 45.67% |
| Confidence-only selection | 49.11 ± 1.77% | 50.00% |
| One-pass head, width 64 | **50.28 ± 0.51%** | **51.67%** |
| One-pass head, width 256 | 49.61 ± 0.84% | 50.22% |
| Fixed recurrence, validation depth | 49.94 ± 1.00% | 50.78% |
| Recurrence with memory | 49.00 ± 1.61% | 49.11% |
| Recurrence with learned stopping | 50.06 ± 0.98% | 49.56% |

See the [full research report](reports/research_summary.md), [self-contained HTML report](reports/research_summary.html), and [recorded measurements](results/metrics/). All plotted values come from actual runs. Prior paper results are excluded from these tables and figures.

**Experiment 3 — Learning When to Think:** [research report](reports/experiment_03/learning_when_to_think.md), [HTML report](reports/experiment_03/learning_when_to_think.html), [configuration and audit](experiments/experiment_03/), and [recorded results](results/experiment_03/). Fifteen new neural fits across three seeds distinguish current correctness (fresh-ID AUROC 0.798) from immediate next-loop gain (0.762; confidence-only gain prediction 0.695). Gain stopping scores 49.06% at 1.894 loops versus the one-pass head's 51.28%; matched-compute accuracy intervals include zero. Full live inference measures 19.70 ms versus 19.35 ms for one pass on the same timed validation subset. Template transfer weakens the signal. Historical findings above remain unchanged.

**Experiment 4A — Foundation and Generalization:** [foundation report](reports/experiment_04a/foundation_report.md), [Experiment 4B handoff](reports/experiment_04a/experiment_04b_handoff.md), [frozen protocol and dataset manifests](experiments/experiment_04a/), and [public measurements](results/experiment_04a/). On 10,224 independently verified new problems, the frozen historical heads transfer weakly: fresh-ID confidence/gain AUROC is 0.530/0.534, versus 0.580 for a fresh scalar gain control. Gain stopping scores 33.74% versus one pass at 33.10%; the paired difference interval includes zero. All 122,688 trajectory states and 92,016 adjacent correction/degradation labels are exported, with hidden tensors excluded from Git. Three halting mechanisms are measured with actual executed-work accounting. No new neural models or 4B selector were trained; 4B needs a fresh prospective final lockbox. Historical findings remain unchanged.

**Experiment 4B — Decision Value and Trajectory Selection:** [executive summary](reports/experiment_04b/executive_summary.md), [research report](reports/experiment_04b/decision_value_and_trajectory_selection.md), [HTML report](reports/experiment_04b/decision_value_and_trajectory_selection.html), [prospective protocol](experiments/experiment_04b/), and [public measurements](results/experiment_04b/). Thirty-six neural fits across three parent seeds evaluate 6,480 new locked problems. Fresh-ID correction/damage AUROC is 0.779/0.794 and AP 0.147/0.229. D stopping scores 34.85% at 1.673 paid loops versus one pass at 33.31%; the paired difference is +1.54 points [−0.95, +3.88], so an accuracy advantage is not established. The prespecified selector scores 34.88%, adding no established benefit on D-visited states. After four paid loops, selection recovers accuracy from 28.78% to 35.11%, with simple maximum-C competitive. Full live D inference is 25.27 ms/item versus 24.91 ms for one pass. Event prediction improves, while a general decision or compute advantage remains unproven. Historical findings remain unchanged.

The hypothesis is interesting because a pretrained representation may contain useful decision information that its original readout does not recover. Confidence can estimate correctness; continuation can estimate whether another loop is worthwhile. Those are distinct learning targets. A useful uncertainty probe does not automatically improve decisions or justify extra computation.

We trained confidence, candidate reranking, tied residual refinement, three-state loop memory, learned continuation, and one-pass controls around a frozen backbone. Seven auxiliary fits per seed were completed across three seeds in the primary pilot. An initial format-failure pilot completed another 21 fits and is retained. A separate middle-layer experiment trained three injection scalars around a repeated frozen block. The total is **43 research fits**, plus isolated smoke verification.

What worked: supervised readouts recovered a modest decision gain; the confidence head improved correctness ranking from AUROC 0.761 to 0.808; learned stopping reduced average loop count. Temperature scaling improved calibration without changing answers.

What failed: eight refinement loops reduced accuracy to **25.67%** without memory and **31.22%** with memory. Confidence/probability fusion worsened NLL. The initial prompt suffered answer-letter bias, and BF16 failed strict numerical parity. Both failures are preserved. Memory and adaptive halting did not convincingly improve accuracy over the relevant trained controls. The minimal middle-layer injection fit did not beat simply repeating the frozen block twice.

What remains unknown: whether a stronger trained middle-layer module, more data, unseen formats, natural benchmarks, or a larger backbone provides a recurrence advantage. Multi-step task performance remained weak. We found readout improvements, not evidence of a new internal reasoning algorithm.

## Scope and research disclaimer

This is an **independent adaptation**, not an official [RecurTrace](https://arxiv.org/html/2609.03379v2) reproduction. The primary recurrence acts on the final hidden vector. The middle-layer probe is a separate, much smaller experiment. The [literature review](docs/literature_review.md) cites RecurTrace, Universal Transformers, ACT, PonderNet, Encode-Think-Decode, Think-at-Hard, Semantic Entropy Probes, and process supervision.

We used six deterministic task families, 1,200 training examples, 300 validation examples, 600 test examples, and 300 OOD examples. Three training seeds share the same held-out examples. Templates are shared; OOD combines unseen symbols/numbers with extra sequential depth. These exploratory comparisons have no multiple-comparison correction. Hyperparameter selection used validation; the answer prefix was selected on 60 training examples. Initial negative results remain visible.

Accuracy and NLL condition on the same four verified single-token option letters for every variant. Unconstrained next-token accuracy is a separate diagnostic: **44.17%** on test prompts. The model emits no reasoning tokens in these experiments. Latency measurements distinguish prompt inference from cached-head inference, and analytical matrix FLOPs are estimates.

The actual machine was an Apple M4 Max with 128 GiB unified memory, using MLX Metal and FP32. Peak recorded MLX allocation was **4.21 GiB**, under the configured 48 GiB budget. The primary command took about 62 seconds through evaluation; the middle-layer probe and timing took about 196 seconds. Literature, implementation, verification, and documentation took longer; execution manifests distinguish these times. No other workstation or training service was accessed or changed.

## Reproduce

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e '.[apple,test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m latent_loop.cli download --config configs/pilot.yaml
.venv/bin/python -m latent_loop.cli run --config configs/pilot.yaml --run-dir artifacts/reproduction
.venv/bin/python -m latent_loop.cli middle --config configs/pilot.yaml --run-dir artifacts/reproduction
.venv/bin/python -m latent_loop.cli report --config configs/pilot.yaml --run-dir artifacts/reproduction
```

Basic CI uses `pip install -e '.[test]'` and CPU-only tests, with no model or GPU. `make smoke` runs a separate tiny real-model pipeline; `make mlx-test` verifies head parity, gradients, isolation, stopping, and checkpoint resume on MLX CPU. `make report` regenerates the public report without a model or accelerator. Stage commands are `data`, `extract`, `train`, and `evaluate`; `train --seed 11` selects one seed. See [reproduction instructions](docs/reproduction.md), [architecture](docs/architecture.md), [methodology](docs/methodology.md), and [limitations](docs/limitations.md).

Auxiliary training resumes from epoch checkpoints after safe interruption. Feature extraction resumes from batch shards. Changed fitting configurations need a fresh run directory. The tiny middle-layer training stage restarts on interruption. The command budget defaults to six hours; model-scale work requires Metal and never silently uses CPU.

The pinned checkpoint is `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, under Apache-2.0. Original code and generated data/results use MIT. Model weights and checkpoints are excluded from git; [third-party provenance](docs/third_party_licenses.md) records licenses and tokenizer details. Git history is local only, with no remote repository or publication created.
