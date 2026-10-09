# Limitations

This is a synthetic, short-answer decision pilot on one small frozen model. It is not a RecurTrace reproduction or evidence for broad reasoning, hallucination prevention, general tool use, or long-form generation.

The head sees the final decision vector, not all token states. Candidate embeddings identify option letters rather than re-encoding full answer text; answer content is available indirectly through the shared prompt representation. Improvements can reflect learned readout, option biases, or calibration. No causal intervention establishes a new reasoning algorithm.

The primary test set contains 100 examples per task and shares generator templates with training. OOD mixes unseen symbols/numbers with extra sequential depth; aggregate OOD accuracy does not isolate those factors. Larger numbers, unseen formats, natural tasks, and robust alternative baselines require fresh held-out data. Seven prefix candidates were examined on training data after an initial failed-format pilot; all resulting claims remain exploratory.

Only three seeds were run. Many variants were compared, and intervals are descriptive without family-wise corrections. An advantage over the frozen conditional model does not imply an advantage over a trained one-pass head. Removing memory from a model trained to use it is not evidence that memory is better than a separately trained model without memory.

BF16 behavior was not robust enough for the selected strict parity checks, so all recorded backbone experiments use FP32. Head timing excludes backbone inference. Analytical matrix FLOPs omit normalization, activations, some attention cost, and synchronization. MLX peak allocation differs from process RSS and total unified-memory use. No CUDA experiment, larger-model comparison, profiler FLOP count, ACT objective, PonderNet objective, or full RecurTrace LMA experiment was run.

The middle-layer study trains only three injection scalars on 240 examples. One training seed, five epochs, and no memory/halting make it a feasibility probe. Its untrained repeated-layer control cannot establish how a properly trained middle module would perform. Checkpoint resume is supported at epoch boundaries for the auxiliary heads; interrupted middle-layer probes restart their tiny five-epoch training stage. No CPU fallback is used for model-scale jobs.


## Unpublished 5C–5E closing evidence

Useful local information propagation did not become robust long-horizon navigation. Untied/deep feedforward controls stayed competitive. The graph scorer’s one-loop gains were seed-dominated and unnecessary at eight loops. In the frozen-LM Rendezvous study, scoring improves structural aggregate choices by +5.61 pp [2.89, 8.31], but matched C2 beats fusion by 9.47 pp [6.64, 12.25], a simple abstain prior is stronger still, and directed-graph decisions degrade. Candidate-description priors explain substantial value; the incremental prompt-hidden contribution is inconclusive. Whole-policy semantic permutation invariance fails. Fusion confidence is poorly calibrated. Synthetic four-label selection cannot establish better general free-text generation or justify recurrent LM architecture changes.

These completed studies are prepared locally for v0.3.0 review, not published. See the [Rendezvous report](../reports/experiment_05e/the_rendezvous.md) and [conclusion](../reports/research_conclusion.md). Independent cached-feature head refits are verified; full historical retraining and independent complete feature re-extraction are not claimed.

A post-primary exploratory paired description shift further reduces Q-only from 42.83% to 26.78% and C2 from 42.33% to 24.50%; fusion versus BaseLM is inconclusive. The simpler classifier is also template-sensitive and must not be presumed deployable.
