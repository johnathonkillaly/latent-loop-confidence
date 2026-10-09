# Experiment 5E — The Rendezvous

This local experiment freezes Qwen3-0.6B at the recorded revision and trains only small, shared candidate heads. It does not train transformer recurrence or evaluate free-text generation.

## Development and prospective selection

Three 180-observation frozen-model development runs compared unordered labels with longer state programs, conventional alphabetical labels with shorter programs, and a plain assistant prefix. Original results remain in `results/experiment_05e/development_pilots.json`. Frozen competence was weak. Conventional label display and 1–3 state instructions were retained; the existing `Answer: (` prefix was retained after the prefix comparison tied in aggregate. No final outcomes informed these choices.

The configuration fixes 6,000 training observations, 1,200 validation observations, five independent 1,200-observation final sets, and 480 paired permutation views of 240 final instances. Each ordinary split has equal routing, state, and graph counts. Validation examples 0–599 choose checkpoints using action accuracy, then set NLL. Examples 600–1199 select the correctness-versus-ranking scorer and fusion weight, averaged over seeds 511, 622, and 733. Lower lambda wins an accuracy tie; correctness wins an objective tie. Temperature minimizes set NLL on the same policy-validation half.

Shared width-64 candidate MLPs receive RMS-normalized final prompt hidden states, mean frozen input embeddings of the observed candidate description, frozen label-token embeddings, and five conditional logit/probability statistics. Labels and candidate semantic identities are separate. Every head has the same allocated architecture. Ablations zero specified feature groups; their effective capacity is smaller and is disclosed. No head has separate A/B/C/D output weights. C2 uses exactly the scorer architecture and features with set-aware direct classification loss. It is a supervision comparison, not an independent architectural invention.

All eight conditions use the same supervised examples, 24 epochs, AdamW, batch size 128, and gradient clipping at 1. Correctness uses per-candidate BCE; ranking uses positive-versus-negative logistic ranking; C1/C2 use negative log probability mass on the full valid action set; the residual control uses set CE on base logits plus an unconstrained learned correction. The base model and its probabilities are never fitted.

## Primary test

**Population:** the 1,200-instance structural-generalization mixture: deny-overrides in fictional tool routing, copy/multiply register programs, and directed DAG routes. These structures are absent from training. Their component shift strengths differ and will be reported separately.

**Comparison:** selected BaseLM + Q fusion versus unchanged BaseLM on the same instances. The primary estimate averages the three independently trained heads. A paired two-way bootstrap resamples task instances and training seeds, with 10,000 draws. Three seeds provide limited evidence about training variability. The strength check compares the identical selected fusion with logit-aware C2 on those instances. An interval spanning zero does not establish an advantage. Other populations, objectives, ablations, and permutation results are secondary or exploratory.

Fusion is `base_logit + lambda * normalized_Q`, equivalent in argmax to `log P + lambda * normalized_Q`. Q is centered within each candidate set and divided by max(RMS, 1). Lambda zero is an explicit candidate and must remain disabled if selected. Score-only decisions are also evaluated. Temperature scaling retains the same argmax. Candidate-order views are clustered by parent instance, not treated as new independent observations.

## Integrity and deployment

Raw observations contain no solver distances, hidden solutions, or outcomes beyond the ordinary multiple-choice candidates. Independent solvers verify all valid action sets. Cross-split canonicalization rejects routing tool permutations, register renamings, and matching unrooted graph WL invariants; this is not complete semantic equivalence certification. Abstain prevalence and a simple abstain prior are reported.

Final extraction requires `protocol_freeze.json`; every extraction and scoring opening is logged. Training refuses to proceed once final data have been opened. Checkpoints and feature caches are ignored and resumable. The freeze records dataset, source, and selected-checkpoint hashes. Final reports may be edited; weights, lambda, and benchmark definitions may not be selected again.

Full-policy timing uses validation prompts of 140–200 tokens, six warmups and 36 randomized interleaved repetitions per method, batch 1 and 8, with synchronization. It includes tokenization, the frozen full-prefix forward, four-label projection, candidate features, head, host transfer, and final semantic action. Unrestricted next-token projection is a separate diagnostic. Head-only cached timing is labeled separately.

This is a bounded synthetic constrained-choice experiment. It cannot establish improved general language generation, pretrained reasoning, or superiority of a scorer over an equivalent classifier merely by renaming the head.
