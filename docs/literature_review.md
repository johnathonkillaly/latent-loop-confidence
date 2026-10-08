# Literature review

Reviewed on 2026-10-08. The implementations and results in this repository are independent pilot work; no claim of novelty or official reproduction is made.

## RecurTrace

[Wang et al., RecurTrace, arXiv:2609.03379v2](https://arxiv.org/html/2609.03379v2) repeats a frozen middle block between an encoder and decoder. Each looped layer reads its own previous outputs at the same token position. QK-normalized loop attention uses a three-loop window, signed relative-distance biases, and near-closed gates. Input reinjection begins after the first loop, with a zero-initialized scalar. This preserves the original one-loop computation.

Training samples clamped log-normal-Poisson depths. A separately distilled continuation head predicts whether deeper supervised loss improves by a margin, rather than whether the current answer is correct. Checkpoints, halting thresholds, and minimum depth require held-out selection. The appendix distinguishes matched training axes from identical FLOPs and includes separate halting-distillation cost. Experiments use eight H800 GPUs, much larger training mixtures, and multiple seeds. Ablations address QK normalization, gates, distance bias, memory window, plain looping, non-loop capacity, and over-depth degradation. Reported limitations include sequence-level stopping, bounded same-position memory, frozen Qwen3 architectures, and direct-answer tasks.

The paper and targeted title/author/GitHub searches did not identify a verifiable author-released implementation. This does not prove none exists. No RecurTrace code was copied; there is no third-party implementation license to incorporate.

## Other implementation-relevant work

| Work | What informs this experiment | Difference from our pilot |
|---|---|---|
| [Dehghani et al., Universal Transformers, 1807.03819](https://arxiv.org/abs/1807.03819) | Weight sharing across depth and adaptive per-position computation. | Our cheap head iterates only a final decision vector, not sequence-wide transformer computation. |
| [Graves, Adaptive Computation Time, 1603.08983](https://arxiv.org/abs/1603.08983) | Differentiable halting and a ponder cost make computation allocation trainable. | We fit explicit future-improvement labels after representation training; we do not implement ACT or infer its failure generally. |
| [Banino et al., PonderNet, 2107.05407](https://arxiv.org/abs/2107.05407) | A learned stopping distribution regularized toward a geometric prior. | Our deterministic threshold policy has no geometric-prior objective. |
| [Koishekenov et al., Encode, Think, Decode, 2510.07358](https://arxiv.org/abs/2510.07358) | A selected middle block provides recurrent capacity without a new full backbone. | Our middle experiment trains only three injection scalars, rather than reproducing its mid-training recipe. |
| [Fu et al., Think-at-Hard, 2511.08577v4](https://arxiv.org/abs/2511.08577v4) | Selective iteration and wrong-to-right/right-to-wrong accounting address latent overthinking; depth-aware LoRA and duo-causal attention support token-level refinement. | We use sequence decisions and loss-improvement continuation, without LoRA or duo-causal attention. The latest paper title is *Dynamic Looped Transformers for Improved Reasoning*. |
| [Kossen et al., Semantic Entropy Probes, 2406.15927](https://arxiv.org/abs/2406.15927) | Hidden states can support cheap uncertainty probes. | A correctness classifier is not a semantic-entropy probe; our labels are deterministic candidate outcomes, not semantic clusters. |
| [Lightman et al., Let's Verify Step by Step, 2305.20050](https://arxiv.org/abs/2305.20050) | Process supervision differs from final-outcome supervision; intermediate value signals require their own targets. | We use candidate correctness and future loss reduction, not human reasoning-step reward labels. |

## Our mathematical choices

For candidate (c), baseline probability is (p(c\mid x,C)=\operatorname{softmax}_C(s_c)). The confidence head predicts (q_c=\sigma(f(h,e_c,\log p_c,\log p^{vocab}_c,H^{vocab},m,H_C))). Correctness BCE supervises each candidate. Reranking uses (\log p_c+\beta\log q_c), with validation selecting \(\beta\); the classifier-only comparator uses \(\log q_c\).

The residual head is (z_{k+1}=z_k+\tanh(g)U(z_k,z_0,M_k)+\tanh(a)\operatorname{RMS}(z_0)), initialized with (g=a=0). Scores are (s^{(k)}=s^{(0)}+(z_k-z_0)E_C^T). This formulation preserves cached baseline logits exactly at initialization despite projection rounding.

Our loop memory attends across a bounded history of final-position states, never across examples or future tokens. Its scope is deliberately smaller than a layer-by-token trajectory. Continuation targets are (y_k=\mathbf1[\min_{j>k}\ell_j<\ell_k-0.01]) over training depths 1–4. Correctness, improvement opportunity, stopping policy, and correctness-oracle upper bounds remain distinct.

## Feasible scope and compute limits

On this M4 Max, a small frozen backbone, cached features, tiny auxiliary heads, multiple seeds, and a small middle-layer probe are practical. We chose FP32 after BF16 parity failed. Useful controls include one-pass heads with equal parameter count or comparable matrix operations, memory removal, randomized updates, and deeper-than-trained evaluation.

A faithful large-mixture, multi-scale reproduction, full-model fine-tuning controls, long-generation cache behavior, and extensive paired benchmark replication would require substantially more compute. Our synthetic decisions cannot establish broad language-model reasoning gains, and recurrence count alone is not a FLOP or latency measurement.
