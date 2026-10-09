# Experiment 5 implementation references

Primary sources were revisited directly; source dates and versions are those linked below. These references motivate controls, not evidence for our results.

[RecurTrace v2](https://arxiv.org/html/2609.03379v2) uses a shared middle block, input reinjection, same-token loop memory, variable-depth training, and a separately distilled halting head. Its initialization limits disruption and its training protocol distinguishes controlled budget axes from measured wall time. We reviewed its method and implementation appendix for causal alignment, initialization, stability, and halting objectives.

Our study instead trains query/value LoRA and scalar gates on a small synthetic mixture, supervises fixed depths 1–4, uses bounded convex reinjection/state updates, and compares against one-pass adaptation. It omits LMA, the paper's large training mixture, clamped stochastic-depth recipe, full-fine-tuning baseline, and halting distillation. We use FP32 MLX on one Mac, rather than claiming protocol equivalence. No author implementation is copied. This is not an official reproduction.

| Primary reference | Implementation relevance |
|---|---|
| [Universal Transformers](https://arxiv.org/abs/1807.03819) | Shared-depth self-attentive computation and adaptive positions |
| [Adaptive Computation Time](https://arxiv.org/abs/1603.08983) | Differentiable computation allocation; deferred here |
| [PonderNet](https://arxiv.org/abs/2107.05407) | Learned halting distribution and regularization; deferred here |
| [Encode, Think, Decode](https://arxiv.org/abs/2510.07358) | Reusing a pretrained middle block |
| [Deeply-Supervised Nets](https://arxiv.org/abs/1409.5185) | Intermediate supervised objectives |
| [LoRA](https://arxiv.org/abs/2106.09685) | Low-rank adaptation with frozen pretrained arrays |

The installed `mlx-lm` Qwen3 and LoRALinear APIs provide the original transformer operations and adapter implementation under MIT. Our wrapper invokes those dependencies; it does not vendor or relicense Apple source. Actual layer count, head dimensions, GQA groups, QK normalization, RoPE, residuals and tied embedding configuration are recorded from the pinned model. Qwen3 retains its separately downloaded Apache-2.0 license. See the historical third-party provenance document.
