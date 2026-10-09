# Experiment 5.3 — prospective protocol

Started 2026-10-09 16:31:13 UTC from local Experiment 5.2 commit
`1f2e4947689984643bbadccb3327a8baec4cba81`. No remote publication.

H1 (information), H2 (decision impact), H3 (complementarity), H4 (generalization),
and H5 (architectural value) are separate hypotheses. All comparisons except the
following primary test are secondary/exploratory, with unadjusted intervals.

Development amendment before final evaluation: the first random-pair generator
put 77% of reachable/unreachable examples combined at distance 1–2 and yielded
near-ceiling one-hop validation. Its 36 development fits, measurements, raw data
and source versions are preserved under `results/experiment_05d_pilot_easy` and
`pilot_easy_distribution/`. No final model predictions were made. The main study
uses 10/12-node graphs with fixed quotas: 25% distance 1–2, 40% distance 3–4,
20% distance 5–9, 15% unreachable. Start degree is at least two. Larger ID-style
tests retain these quotas; community and long-path shifts are separately defined.
These are diagnostic distributions, not naturally occurring graph prevalence.

**Primary:** frozen width-32 recurrent base at eight message-passing iterations,
with versus without a validation-selected correctness/ranking scorer and fusion
lambda, on 600 new ID graphs. Each graph topology is disjoint from base-training,
scorer-training, validation and other final populations under a conservative
unrooted Weisfeiler–Lehman invariant. This certifies absence of isomorphic graph
copies across splits by rejecting all matching invariants, including collisions;
it does not establish that every pair of distinct invariants is equally difficult.
Three independently trained seeds are crossed with identical graphs. Report mean
paired accuracy difference and a crossed graph/seed bootstrap 95% interval (3000
replicates). A zero-spanning interval is inconclusive. No final-set re-selection.

**Prespecified counterfactual confirmation:** paired graphs whose two start arms
are locally identical, but a distant goal attachment switches the optimal first
action. Arm distance five (goal distance six) is the target stratum: compare both-
members-correct at two versus eight iterations, using the same frozen recurrent
weights. Candidate readout sees neighbor states, so goal influence requires at
least goal-distance minus one updates. Earlier pair logits must agree. Other arm
lengths and depths are secondary. Pair, rather than member, is the resampling unit.

Base and scorer training examples/topologies are disjoint. Select each base
checkpoint by validation optimal-set NLL, then freeze its hash before fitting any
scorer. Each scorer is a shared per-candidate MLP with frozen base representations;
correctness uses BCE, ranking uses positive/negative pairwise logistic loss, and
the matched direct classifier uses optimal-action-set NLL. All have identical
observable features and parameter counts. No joint fitting, stopping controller,
distance auxiliary target, spatial coordinates, node IDs or oracle values enter
inference. Goal and start flags, adjacency, degree and an abstain flag are observed.

Fit one-hop, shared variable-depth, eight-block untied same-width, and narrow
eight-block approximately parameter-matched policies. Give a separate recurrent
control an additional scorer-data training stage with the same number of example
presentations as a scorer. This is an extra-supervision control, not the frozen
base. Fit correctness/ranking/direct scorers on one-hop and recurrent models;
fit a direct classifier on the same-width feedforward model. Three seeds.

Fusion is log-softmax(base logits) + lambda times the scorer's per-graph centered,
RMS-normalized raw scores (RMS floored at 1). Select lambda by validation action
accuracy, ties favor smaller lambda; select primary scorer objective by validation
fused accuracy, ties favor correctness. Score-only, temperature calibration
(argmax unchanged), constant scores and per-graph shuffled scores are controls.
Candidate BCE calibration is assessed separately on unnormalized sigmoid scores.

Before final opening, secondary depth-specific fusion weights are selected using
the same validation grid and the fixed selected scorer checkpoint/objective.
Depth 16 reuses the depth-eight weight. The primary depth-eight comparison remains
unchanged, including the possibility that validation disables scoring (lambda 0).
Also report the validation-selected best *nonzero* depth-eight weight as a forced
intervention diagnostic, never as a replacement primary comparison. No final-set
scoring weight or winning depth will be selected.

Final files are generated deterministically before fitting but cannot be loaded by
training APIs. Record hashes and selected checkpoints/settings before an explicit
lockbox opening receipt. All model/selection code is frozen with that receipt.
No fit permitted after opening. Full rollouts reset recurrent state per action,
use the raw graph only, and stop at goal, abstain, cycle or 128 actions. Abstaining
is successful only when no route exists; unsolvable outcomes are reported apart
from reachable goal completion. Individual rollout actions are not independent
statistical samples.

No Snake/Qwen training or Experiment 5.4 implementation. Secondary dependency
selection and adaptive stopping are optional and may be omitted to protect rigor.
