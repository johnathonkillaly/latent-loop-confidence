"""Small current-state probes; inference APIs accept no supervision."""

import mlx.core as mx
import mlx.nn as nn

from latent_loop.models import Confidence, rms


class Probe(nn.Module):
    def __init__(self, inputs, width=64, outputs=1):
        super().__init__()
        self.hidden = nn.Linear(inputs, width)
        self.output = nn.Linear(width, outputs)

    def __call__(self, current_features):
        return self.output(nn.silu(self.hidden(current_features)))


class CandidateProbe(nn.Module):
    """Historical candidate-confidence architecture, adapted to current correctness."""

    def __init__(self, dim, decoder, width=64):
        super().__init__()
        self.head = Confidence(dim, width)
        self.freeze_decoder = decoder
        self.freeze(keys=["freeze_decoder"])
        self.dim = dim

    def __call__(self, current_features):
        x = current_features
        p = x[:, self.dim : self.dim + 4]
        lp = mx.log(mx.maximum(p, 1e-30))
        ent, margin, depth = x[:, -4], x[:, -3], x[:, -1]
        stats = mx.stack(
            [
                lp,
                mx.broadcast_to(ent[:, None], lp.shape),
                mx.broadcast_to(margin[:, None], lp.shape),
                mx.broadcast_to(depth[:, None], lp.shape),
            ],
            axis=-1,
        )
        logits = self.head(x[:, : self.dim], self.freeze_decoder, lp, stats)
        selected = logits[mx.arange(len(x)), mx.argmax(p, axis=-1)]
        return selected[:, None]


def live_features(z, scores, previous, depth):
    p = mx.softmax(scores, axis=-1)
    ordered = mx.sort(p, axis=-1)
    entropy = -mx.sum(p * mx.log(mx.maximum(p, 1e-30)), axis=-1, keepdims=True)
    margin = ordered[:, -1:] - ordered[:, -2:-1]
    change = mx.sqrt(mx.sum((z - previous) ** 2, axis=-1, keepdims=True)) / mx.maximum(
        mx.sqrt(mx.sum(previous**2, axis=-1, keepdims=True)), 1e-6
    )
    return mx.concatenate(
        [rms(z), p, entropy, margin, change, mx.full((len(z), 1), depth / 4)], axis=-1
    )
