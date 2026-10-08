# Experiment 4B handoff — prepared by 4A

No decision-value head or recurrent trajectory selector was implemented. This handoff supplies data, integrity controls, evaluation baselines and measurement tools. 4A final outcomes are descriptive evidence; they must not select the next architecture.

## 1. Available trajectory datasets

10,224 unique instances, three parents, depths 1–4: 122,688 state rows and 92,016 adjacent transitions.

| Split | Instances | State rows across parents | RNG seed | Public trajectory pattern |
| --- | --- | --- | --- | --- |
| controller_train | 4032 | 48384 | 440001 | `results/experiment_04a/trajectories/controller_train_seed{11,22,33}.jsonl.gz` |
| controller_val | 1008 | 12096 | 440101 | `results/experiment_04a/trajectories/controller_val_seed{11,22,33}.jsonl.gz` |
| fresh_id | 1296 | 15552 | 440201 | `results/experiment_04a/trajectories/fresh_id_seed{11,22,33}.jsonl.gz` |
| new_template | 1296 | 15552 | 440301 | `results/experiment_04a/trajectories/new_template_seed{11,22,33}.jsonl.gz` |
| structural_shift | 1296 | 15552 | 440401 | `results/experiment_04a/trajectories/structural_shift_seed{11,22,33}.jsonl.gz` |
| harder_depth | 1296 | 15552 | 440501 | `results/experiment_04a/trajectories/harder_depth_seed{11,22,33}.jsonl.gz` |

Public generators: `src/latent_loop/experiment04a/data.py`; dataset recipes/manifests/exclusions: `experiments/experiment_04a/`. Public full instances: `results/experiment_04a/datasets/`. Per-parent metrics and raw probabilities/logits: `results/experiment_04a/{metrics,predictions}/`. Registries: `experiments/experiment_04a/{development,final}_trajectory_registry.json`. Hidden arrays are local ignored NPZs under `artifacts/experiment_04a/trajectories/`; public references include SHA256 and array coordinates. They are available on this machine, not distributed in Git.

## 2. Exact trajectory schema

Schema version: `latent-loop-trajectory-v1`. UTF-8 JSONL compressed with gzip. One row per instance × parent × depth, ordered instance-major then depth. No global row ID is needed: `(instance_id, parent_seed, depth)` is the primary key. This version has exactly the following namespaces:
Machine-readable JSON Schema: `experiments/experiment_04a/trajectory_schema.json`. The validator also independently checks cross-row arithmetic and labels, which JSON Schema alone cannot establish.

| Namespace | Fields / types |
| --- | --- |
| root | schema_version: string; identity: object; model: object; depth: integer 1–4; observation: object; offline_supervision: object |
| identity | instance_id: SHA256 string; structural_hash: SHA256 string; task_family: string; structural_generator_id: string; split: string; parent_seed: integer |
| model | backbone_id: string; backbone_revision: pinned commit string; refiner_sha256: SHA256 string |
| observation | predicted_answer_index: integer 0–3; candidate_logits: float[4]; candidate_probabilities: float[4]; entropy: float; margin: float; previous_state_change: float; hidden_state_reference: object |
| hidden_state_reference | artifact_path: repository-relative local path; sha256: SHA256 string; array: 'states'; depth_index: integer 0–3; instance_index: integer; dtype: 'float32'; dimensions: [1024] |
| offline_supervision | true_answer_index: integer 0–3; current_correct: boolean; current_loss_nats: float; next_transition: 'W->R'/'R->W'/'W->W'/'R->R' or null; signed_correctness_change: -1/0/+1 or null; signed_loss_gain_nats: float or null |

Loss is `logsumexp(current candidate logits) - true candidate logit`. `signed_loss_gain_nats = loss(depth) - loss(depth+1)`: positive means next loss decreased. Conventional signed next-minus-current loss change is its negative. `signed_correctness_change = int(next correct) - int(current correct)`. All adjacent fields are null at depth 4. Probabilities condition on four candidates; entropy uses natural logs, margin is largest minus second-largest probability. State change is `||current-previous|| / max(||previous||, 1e-6)`; depth 1 compares with the frozen backbone state. NPZ `states` has shape `[4,N,1024]`, and `backbone` has `[N,1024]`.

## 3. Split identities and allowed uses

Only `controller_train` and `controller_val` are development splits. They are balanced across 12 semantic generator approaches, six tasks and three nominal strata. Four final splits have 1,296 instances each: fresh ID (same 4A approaches, new instances), new template (held-out structured presentation IDs 2/3), structural shift (six withheld approaches), harder depth (longer sequential programs and larger graphs, with static-task limitations). All seeds are unique and all six splits reject alpha-renamed instances. Manifest hashes and public exclusions are immutable evidence, not permission to reuse consumed tests for selection.

## 4. Recommended causal input features

The established runtime observation is `hidden[1024], probabilities[4], entropy, margin, depth, previous_change`. `RuntimeObservation.vector()` yields 1,032 FP32 values: RMS-normalized hidden state, four probabilities, entropy, margin, completed relative change, depth/4. `runtime_observation(current_hidden, current_logits, previous_hidden, depth)` accepts completed-state values only. `runtime_from_fields` rejects labels, future state/loss/prediction, oracle depth, offline supervision, task metadata and unknown keys.

These are interface recommendations already established in the frozen 4A plan, not architecture selection from final performance. Evaluate hidden versus scalar ablations using development data. Frozen confidence/gain predictions can be optional derived current-state features under an explicit condition. Do not include ground-truth answers, future states, future probabilities, future losses, transition labels, oracle depths or task/difficulty metadata in the default deployed controller. Supervised targets remain in their separate offline namespace. Fit input normalizers on training data only.

## 5. Wrong-to-right prevalence

| Split | W→R event rate [95%], percentage scale | W→R given useful loss gain [95%] |
| --- | --- | --- |
| Validation | 5.908 [4.827, 7.055]% | 11.441 [8.465, 14.123]% |
| Fresh ID | 6.301 [5.247, 7.373]% | 12.898 [9.984, 15.484]% |
| New format | 5.838 [4.586, 6.962]% | 11.022 [8.013, 13.754]% |
| New structure | 6.996 [5.898, 8.136]% | 14.408 [11.798, 16.856]% |
| Harder depth | 6.919 [5.718, 8.034]% | 14.007 [11.156, 16.534]% |

Rates average three adjacent transitions and three parents, retaining all states of each independent problem in bootstrap resampling. A positive loss gain is often not a correction; do not substitute it for a decision-value target.

## 6. Right-to-wrong prevalence

| Split | R→W event rate [95%], percentage scale |
| --- | --- |
| Validation | 7.815 [5.919, 9.447]% |
| Fresh ID | 7.767 [5.657, 9.542]% |
| New format | 6.850 [4.938, 8.436]% |
| New structure | 8.136 [7.073, 9.285]% |
| Harder depth | 7.802 [6.250, 9.336]% |

Per-depth rates and the other two event classes are in `results/experiment_04a/summary.json`. Model correction and degradation separately or use their signed difference; avoid treating unchanged-correct states as correction events. The choice of 4B target must be developed and selected on development data.

## 7. Existing predictor performance

| Split | Confidence AUROC [95%] | Gain AUROC [95%] | Joint gain AUROC | Scalar gain AUROC |
| --- | --- | --- | --- | --- |
| Validation | 0.509 [0.481, 0.539] | 0.561 [0.530, 0.590] | 0.567 | 0.583 |
| Fresh ID | 0.530 [0.505, 0.556] | 0.534 [0.502, 0.559] | 0.560 | 0.580 |
| New format | 0.497 [0.470, 0.524] | 0.553 [0.509, 0.603] | 0.573 | 0.563 |
| New structure | 0.577 [0.552, 0.603] | 0.574 [0.544, 0.602] | 0.588 | 0.578 |
| Harder depth | 0.505 [0.481, 0.530] | 0.549 [0.521, 0.578] | 0.552 | 0.579 |

Confidence and immediate-gain probes transfer much more weakly than their historical synthetic results. The report includes Brier/ECE/log loss, AP, selective accuracy, risk-coverage, per-depth ranking and paired scalar comparisons. Task/difficulty diagnostics are not a deployable hidden-information baseline.

## 8. Recurrent depths and validation selections

| Parent | Validation-selected fixed depth | Confidence threshold | Gain threshold |
| --- | --- | --- | --- |
| 11 | 2 | 0 | 1 |
| 22 | 1 | 0 | 0.75 |
| 33 | 2 | 0 | 0.75 |

The verified computational horizon remains four, and every depth is retained. Final trajectories show degradation at deeper iterations; that is a reported limitation, not a license to select a 4B architecture or depth using these tests. Any new budget or policy is selected on train/validation and assessed on a fresh prospective final set.

## 9. Performance and calibration problems

Frozen backbone competence is low on several tasks, structural transfer is weak, nominal static predicates do not truly deepen, temperature/recalibration may reduce error without informative ranking, and useful loss gain differs from argmax correction. Parents can have different correction/degradation dynamics. The head trained on historical structures and freshly fitted scalar controls are not equal-training hidden-versus-scalar interventions. Conditional four-option performance cannot imply natural-language reasoning competence.

## 10. Reusable frozen checkpoints

Backbone `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, FP32, hidden width 1,024, option IDs 32–35, frozen prefix `Answer: (`. Per-parent refiner and one-pass checkpoints are `artifacts/checkpoints/{recurrence,feedforward}_seed{11,22,33}/best.safetensors`; original E3 confidence/improvement/joint probes are `artifacts/experiment_03/checkpoints/{confidence,improvement,joint}_seed{11,22,33}/best.safetensors`. Probe width 64, current input dimension 1,032, joint outputs 3. Exact hashes are in `experiments/experiment_04a/frozen_protocol.json` and the historical audit. All neural weights remain unchanged. Weights are excluded from Git; retraining creates a new checkpoint identity and must be declared.

## 11. Reproduction commands

From the repository root, using the public project dependencies:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e '.[apple,test]'
.venv/bin/python -m pytest -q
LATENT_LOOP_MLX_TESTS=1 .venv/bin/python -m pytest -q
.venv/bin/python -m latent_loop.experiment04a.pipeline report
```

The report command is CPU-only, recomputes every public solver label/trajectory transition and all primary metrics, and regenerates tables/plots. It needs no weights, hidden tensors, model download or GPU. Run it against already published 4A predictions to reproduce analysis, without model selection.

For local frozen-model extraction/evaluation (requires the exact historical checkpoints and cached pinned backbone), stage commands are:

```sh
.venv/bin/python -m latent_loop.experiment04a.pipeline audit
.venv/bin/python -m latent_loop.experiment04a.pipeline data
.venv/bin/python -m latent_loop.experiment04a.pipeline development
.venv/bin/python -m latent_loop.experiment04a.pipeline freeze
.venv/bin/python -m latent_loop.experiment04a.pipeline final
.venv/bin/python -m latent_loop.experiment04a.pipeline timing
.venv/bin/python -m latent_loop.experiment04a.pipeline report
```

Do not rerun development or data over the consumed lockbox: those stages deliberately refuse. Exact local caches resume feature extraction by batch and trajectory collection by parent/split; signatures reject changed datasets/model/prefix/checkpoints. SIGINT/SIGTERM stops at safe batch/stage boundaries. The original three-hour deadline is recorded in config. A new model reproduction requires an isolated checkout/output archive and a fresh `started_utc`/resource budget configured before data generation and protocol freezing; remove only that reproduction's derived manifests/selections/freeze/open/cache/output files, retaining source and the public historical exclusion list. Do not overwrite the recorded study. Recreated neural weights may differ; document the new model hashes and create a new evaluation protocol. Original model-training commands are in `docs/reproduction.md` and `reports/experiment_03/learning_when_to_think.md`.

## 12. Locked final-test protocol

4A frozen `2026-10-08T22:07:16.662885+00:00`, opened `2026-10-08T22:07:17.080770+00:00`. Public `frozen_protocol.json` binds core source/config/data/selection/model hashes; `lockbox_opened.json` binds the opening to that protocol. These **4A final sets are consumed** and may only support descriptive audit/regression checking, not further architecture/threshold/checkpoint selection. Keep them out of 4B training/validation to preserve interpretability. Create a new independently seeded lockbox, reject alpha structures from all prior data, write and hash its task mix/strata/primary comparisons before scoring, and record its opening. Development on 4A train/validation is allowed. Natural-task validation may require its own independently reserved sets.

## 13. Measurement methodology

Full input tokenization, array materialization, right-padded frozen backbone, label decoding, live auxiliary inference and host stopping barriers, final Metal sync and output conversion. Model loading excluded. Auxiliary-only uses h/base already on device and includes output conversion.

Batch sizes 1/8, the same 24 task-balanced validation prompts, three parents, three warmups and five repeats. Scalar halting synchronizes per active row/depth; compact halting extracts one Boolean vector per depth and really removes inactive rows; masked halting executes all four loops and cannot claim saved compute. Diagnostic threshold 0.5 is frozen separately from selected accuracy policies. All outputs/depths match cached trajectories, and executed row counts are asserted. Load, peak process RSS/MLX allocation, components and raw samples are public. A future trajectory selector must pay for every state acquired and every controller operation; selecting an earlier answer after inspecting later states does not recover spent computation.

## 14. Scientific risks and proceed decision

4B can proceed as a new controlled experiment because complete causal observations and correction/degradation labels now exist. There is no demonstrated transferable confidence/gain advantage strong enough to guarantee success. Event imbalance, low recoverable competence, negative recurrent drift, metadata/format effects, calibration transfer, small parent-seed count and timing overhead are central risks. Compare to frozen, one-pass, fixed depths and fresh scalar controls with paired problem/seed uncertainty. Do not compare an oracle to a paid controller as if it were deployable. Preserve this negative evidence, preregister a small hypothesis set, and open the new lockbox only after selection is complete.
