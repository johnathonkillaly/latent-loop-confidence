"""Causal full-prefix Qwen3 recurrence with shared middle-layer LoRA.

No KV caches are accepted. Sequence positions remain unchanged across loops.
Original pretrained arrays remain frozen; only LoRA and scalar gates train.
"""

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.lora import LoRALinear


def rms(x):
    return mx.sqrt(mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True) + 1e-8)


class Thinker(nn.Module):
    def __init__(self, backbone, label_ids, start, count, rank=8,
                 reinjection="none", update_initial=0.1, norm_cap=2.0):
        super().__init__()
        layers = backbone.layers
        if not 0 < start < start + count < len(layers):
            raise ValueError("Recurrence must lie strictly between encoder and decoder")
        if reinjection not in ("none", "fixed", "learned"):
            raise ValueError("Invalid reinjection mechanism")
        if not 0 < update_initial < 1 or norm_cap < 1:
            raise ValueError("Require a nonclosed update gate and norm cap >= 1")
        # Reusing the frozen model in a new condition must never nest adapters.
        for layer in layers:
            for key in ("q_proj", "v_proj"):
                projection = getattr(layer.self_attn, key)
                if isinstance(projection, LoRALinear):
                    setattr(layer.self_attn, key, projection.linear)
        if rank:
            for layer in layers[start:start + count]:
                for key in ("q_proj", "v_proj"):
                    base = getattr(layer.self_attn, key)
                    setattr(layer.self_attn, key, LoRALinear.from_base(base, r=rank, scale=1.0))
        backbone.freeze()
        if rank:
            for layer in layers[start:start + count]:
                for key in ("q_proj", "v_proj"):
                    getattr(layer.self_attn, key).unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        self.backbone = backbone
        self.label_ids = list(label_ids)
        self.start = start
        self.count = count
        self.rank = rank
        self.reinjection = reinjection
        self.norm_cap = norm_cap
        # This open gate preserves adapter gradients. Depth one ignores it,
        # giving exact baseline parity when zero-output LoRA is initialized.
        self.update_logit = mx.array(math.log(update_initial / (1 - update_initial)))
        if reinjection == "learned":
            # 0.25*sigmoid(logit) starts at 0.05 and remains bounded.
            self.reinjection_logit = mx.array(math.log(0.2 / 0.8))
        if not rank:
            self.freeze()

    def parameter_counts(self):
        trainable = tree_flatten(self.trainable_parameters())
        for key, _ in trainable:
            if not (key.endswith(("lora_a", "lora_b")) or key in ("update_logit", "reinjection_logit")):
                raise AssertionError(f"Unexpected trainable pretrained weight: {key}")
        return {"trainable": sum(v.size for _, v in trainable),
                "total": sum(v.size for _, v in tree_flatten(self.parameters())),
                "trainable_keys": [k for k, _ in trainable]}

    def __call__(self, ids, lengths=None, depths=(1,), raw_recurrence=False,
                 reinjection_override=None, return_tokens=False):
        if not depths or min(depths) < 1 or max(depths) > 8:
            raise ValueError("Depths must be in 1..8; 5..8 are diagnostics only")
        if lengths is None:
            lengths = mx.full((ids.shape[0],), ids.shape[1], dtype=mx.int32)
        h = self.backbone.model.embed_tokens(ids)
        mask = create_attention_mask(h, None)
        for layer in self.backbone.layers[:self.start]:
            h = layer(h, mask=mask, cache=None)
        h0 = mx.stop_gradient(h)
        z = h0
        previous = h0
        first = None
        results = {}
        for k in range(1, max(depths) + 1):
            if k > 1:
                mode = reinjection_override or self.reinjection
                r = (0.25 * mx.sigmoid(self.reinjection_logit) if mode == "learned"
                     else mx.array(0.05 if mode == "fixed" else 0.0))
                # Same-token anchor with current RMS; no cross-position mixing.
                anchor = h0 * (rms(z) / rms(h0))
                h = z + r * (anchor - z)
            else:
                h = h0
            for layer in self.backbone.layers[self.start:self.start + self.count]:
                h = layer(h, mask=mask, cache=None)
            if k == 1:
                z = h
                first = z
            elif raw_recurrence:
                z = h
            else:
                # Token-local radial cap plus convex update gives a finite norm
                # bound relative to depth one, without closing the gradient path.
                h = h * mx.minimum(1.0, self.norm_cap * rms(first) / rms(h))
                z = z + mx.sigmoid(self.update_logit) * (h - z)
            if k in depths:
                out = z
                for layer in self.backbone.layers[self.start + self.count:]:
                    out = layer(out, mask=mask, cache=None)
                out = self.backbone.model.norm(out)
                selected = out if return_tokens else out[mx.arange(ids.shape[0]), lengths - 1]
                if self.backbone.args.tie_word_embeddings:
                    weight = self.backbone.model.embed_tokens.weight[mx.array(self.label_ids)]
                else:
                    weight = self.backbone.lm_head.weight[mx.array(self.label_ids)]
                logits = selected @ weight.T
                current = z[mx.arange(ids.shape[0]), lengths - 1]
                prev = previous[mx.arange(ids.shape[0]), lengths - 1]
                origin = first[mx.arange(ids.shape[0]), lengths - 1]
                results[k] = {"logits": logits,
                              "state_norm": mx.linalg.norm(current, axis=-1),
                              "state_distance": mx.linalg.norm(current - prev, axis=-1) / mx.maximum(mx.linalg.norm(prev, axis=-1), 1e-6),
                              "state_drift": mx.linalg.norm(current - origin, axis=-1) / mx.maximum(mx.linalg.norm(origin, axis=-1), 1e-6)}
            previous = z
        return results


def objective(model, ids, lengths, labels, reference_logits, depths, weights, alpha=0.02):
    trace = model(ids, lengths, depths)
    q = mx.stop_gradient(mx.softmax(reference_logits, axis=-1))
    protect = mx.stop_gradient(mx.argmax(reference_logits, axis=-1) == labels)
    total = mx.array(0.0)
    for k, weight in zip(depths, weights):
        logits = trace[k]["logits"]
        loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        kl = mx.sum(q * (mx.log(mx.maximum(q, 1e-8)) - lp), axis=-1)
        preservation = mx.mean(mx.where(protect, kl, 0.0))
        total = total + weight * (loss + alpha * preservation)
    return total / sum(weights)
