"""Small shared candidate heads; no language-model parameters are trainable."""

import mlx.core as mx
import mlx.nn as nn


class CandidateHead(nn.Module):
    def __init__(self, dimension, width=64):
        super().__init__()
        self.input = nn.Linear(dimension, width)
        self.output = nn.Linear(width, 1)

    def __call__(self, x):
        return self.output(nn.silu(self.input(x)))[..., 0]


def view(x, condition):
    """Same allocated parameter budget; ablated inputs explicitly zeroed."""
    if condition in ("c1", "no_logits"):
        return mx.concatenate([x[..., :-5], mx.zeros_like(x[..., -5:])], axis=-1)
    if condition == "logits_only":
        # Keep label embedding and original logits, remove prompt and semantic-description states.
        d = (x.shape[-1] - 5) // 3
        return mx.concatenate([mx.zeros_like(x[..., :2*d]), x[..., 2*d:]], axis=-1)
    if condition == "description_only":
        d = (x.shape[-1] - 5) // 3
        return mx.concatenate([mx.zeros_like(x[..., :d]), x[..., d:3*d], mx.zeros_like(x[..., -5:])], axis=-1)
    return x


def objective(scores, y, condition, base):
    if condition in ("correctness", "no_logits", "logits_only", "description_only"):
        return mx.mean(mx.maximum(scores, 0) - scores*y + mx.log1p(mx.exp(-mx.abs(scores))))
    if condition == "ranking":
        difference = scores[:, :, None] - scores[:, None, :]
        mask = y[:, :, None] * (1-y[:, None, :])
        return mx.sum(nn.softplus(-difference)*mask) / mx.maximum(mx.sum(mask), 1)
    if condition == "residual":
        scores = scores + base
    positive = mx.where(y > 0, scores, -1e9)
    return mx.mean(mx.logsumexp(scores, axis=-1) - mx.logsumexp(positive, axis=-1))
