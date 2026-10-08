"""Native MLX heads. This module is not imported by CPU CI."""

import mlx.core as mx
import mlx.nn as nn


def rms(x):
    return x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)


class Confidence(nn.Module):
    def __init__(self, dim, width=64):
        super().__init__()
        self.hidden = nn.Linear(2 * dim + 5, width)
        self.output = nn.Linear(width, 1)

    def __call__(self, h, emb, logits, stats):
        n, c = logits.shape
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        x = mx.concatenate(
            [
                mx.broadcast_to(rms(h)[:, None, :], (n, c, h.shape[-1])),
                mx.broadcast_to(rms(emb)[None, :, :], (n, c, emb.shape[-1])),
                logp[:, :, None],
                stats[:, :, None] if stats.ndim == 2 else stats,
            ],
            axis=-1,
        )
        return self.output(nn.silu(self.hidden(x))).squeeze(-1)


class Refiner(nn.Module):
    def __init__(self, dim, width=64, memory=False, window=3):
        super().__init__()
        self.memory = memory
        self.window = window
        self.hidden = nn.Linear(2 * dim, width)
        self.output = nn.Linear(width, dim, bias=False)
        self.gate = mx.array(0.0, mx.float32)
        self.injection = mx.array(0.0, mx.float32)
        if memory:
            self.query = nn.Linear(dim, 32, bias=False)
            self.key = nn.Linear(dim, 32, bias=False)
            self.memory_gate = mx.array(0.0, mx.float32)
            self.distance = mx.array(0.1, mx.float32)

    def step(self, z, z0, history):
        context = rms(z0)
        if self.memory and history:
            past = mx.stack(history[-self.window :], axis=1)
            q = rms(self.query(rms(z)))[:, None, :]
            key = rms(self.key(rms(past)))
            distances = mx.arange(len(history[-self.window :]), 0, -1)
            attention = mx.softmax(
                mx.sum(q * key, axis=-1) / 32**0.5 - self.distance * distances, axis=-1
            )
            read = mx.sum(attention[:, :, None] * past, axis=1)
            context = context + mx.tanh(self.memory_gate) * rms(read - z)
        update = self.output(
            nn.silu(self.hidden(mx.concatenate([rms(z), context], axis=-1)))
        )
        return z + mx.tanh(self.gate) * update + mx.tanh(self.injection) * rms(z0)

    def __call__(self, h, base, decoder, depth=4, return_states=False):
        z = h
        history = []
        logits = []
        states = []
        for _ in range(depth):
            next_z = self.step(z, h, history)
            history = (history + [z])[-self.window :]
            z = next_z
            logits.append(base + (z - h) @ decoder.T)
            states.append(z)
        return (mx.stack(logits), mx.stack(states)) if return_states else logits[-1]


class Continuation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden = nn.Linear(dim + 5, 32)
        self.output = nn.Linear(32, 1)

    def __call__(self, z, logits, depth):
        p = mx.softmax(logits, axis=-1)
        x = mx.concatenate([rms(z), p, mx.full((z.shape[0], 1), depth / 4)], axis=-1)
        return self.output(nn.silu(self.hidden(x))).squeeze(-1)
