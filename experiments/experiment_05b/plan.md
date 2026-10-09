# Experiment 5.1 — Think Again

Started from local Experiment 5 commit `17ae4e4acac70ef099a55522f0964381a01516d9` on `research/exp05b-maze-planning`. Local work only; public v0.1.0 is immutable.

## Hypothesis and design

Repeated, shared, strictly cardinal one-hop computation over an unchanged observation can improve optimal first-action selection. Track A is a tiny local grid network; Track B is the existing frozen Qwen3 middle-layer mechanism. Success of A makes no claim about B or general reasoning.

Grid inputs contain wall/free/agent/goal channels only. Pointwise encoding initializes per-cell states. A gated normalized bounded candidate update exchanges four cardinal neighbors per iteration. A pointwise head reads only the agent cell. Thus a goal outside radius k cannot affect the agent readout in k loops. Walls have zero hidden states. The observation is reintroduced pointwise each loop, never advanced. At each environment action the hidden state resets.

Primary action supervision minimizes negative log total probability of the full optimal-action set. Any shortest-path action is correct. BFS distance maps are labels/evaluation only. No legality masking hides invalid learned decisions.

Train and validation have balanced nominal families; final balanced ID is the prespecified diagnostic benchmark, with a separate natural unfiltered occupancy distribution. Structural shift changes occupancy to connected frontier growth. D4-canonical WALL LAYOUT hashes exclude all rotations/reflections AND different start/goal placements of the same layout across splits. Independent examples have one starting decision per layout; rollout transitions are clustered by initial maze, never treated as independent samples.

Final-depth 4, sampled variable depths 1/2/4/8, staged difficulty/depth curriculum, and simultaneous multi-depth supervision are recorded, including unsuccessful variants. All receive 20 epochs with the same maximum minibatch count; the curriculum samples the currently eligible pool with replacement to retain update count. Curriculum exposure frequencies necessarily differ. Feedforward controls include exact parameter-matched one-hop, approximately parameter-matched four-hop narrow, and four-hop same-width compute controls. The latter has more parameters. These are local models, not unrestricted global CNNs.

Checkpoint selection minimizes depth-4 validation set loss (depth 1 for one-pass). Before final opening, choose the recurrent objective by mean validation accuracy at depth 4 (ties lower validation loss then lexical name). Primary depth comparison is 4 minus 1 for this validation-selected condition; 2 minus 1 and 4 minus 2 are prespecified secondary. Report all depths and all variants; do not select a winning final depth. Compare with the best validation-selected four-hop feedforward condition at depth 4. Bootstrap paired independent maze layouts and training seeds; only 3 seeds, no multiplicity adjustment. Fixed deployment depths 1/2/4/8, with 8 extrapolation for final4 and curriculum (variable/simultaneous train at 8).

All grid training uses the same 3,600 layouts and seeds 11/22/33. Qwen uses a prespecified first 720 training /120 validation subset for both controls; its results are a separate smaller-budget experiment. Qwen frozen backbone/revision, LoRA rank 8 q/v layers 12–14, full-prefix recomputation with no caches, fixed positions, existing open update gate. One-pass and recurrent trainable counts differ by one scalar. Text format is fixed before fitting; alternative serialization is an exploratory robustness check.

All final configurations, checkpoint hashes, code, and manifests must freeze before scoring sealed final sets. No final-driven architecture or checkpoint changes. Paired accuracy intervals use seed and maze resampling; rollout intervals resample whole episodes. Latency is actual synchronized inference, separate from parameter count, with tokenization included for Qwen and array conversion included for grid models.

Six-hour active budget including reporting; resumable epoch checkpoints in ignored artifacts. Track A and reproducibility take priority over Track B scale and Snake. No C/V/D controllers. Auxiliary distance supervision, across-action memory, and learned halting are optional separate future studies, not part of this primary action-only protocol. Snake utility order is frozen in config before evaluation; no food respawn, bounded actual-body search only.
