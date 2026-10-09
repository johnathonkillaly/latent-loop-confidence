"""Small current-state neural models, with no label or trajectory arguments."""

import mlx.core as mx
import mlx.nn as nn


class GatedHead(nn.Module):
    def __init__(self, inputs=1032, width=32, outputs=4):
        super().__init__()
        self.value = nn.Linear(inputs, width)
        self.gate = nn.Linear(inputs, width)
        self.output = nn.Linear(width, outputs)

    def __call__(self, current_features):
        return self.output(
            nn.silu(self.value(current_features))
            * mx.sigmoid(self.gate(current_features))
        )


class MatchedHead(nn.Module):
    def __init__(self, inputs=1032, width=64, outputs=4):
        super().__init__()
        self.hidden = nn.Linear(inputs, width)
        self.output = nn.Linear(width, outputs)

    def __call__(self, current_features):
        return self.output(nn.silu(self.hidden(current_features)))


def make_model(name, config, seed):
    mx.random.seed(seed + 10000 + sum((i + 1) * ord(c) for i, c in enumerate(name)))
    if name == "matched":
        return MatchedHead(width=config["matched_width"])
    if name.startswith("event_"):
        return GatedHead(width=config["event_width"], outputs=1)
    if name.startswith("ranker_"):
        return GatedHead(1036, config["ranker_width"], 1)
    return GatedHead(width=config["head_width"])
