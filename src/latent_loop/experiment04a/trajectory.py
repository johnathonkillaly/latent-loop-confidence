"""Stable offline trajectory records and an isolated deployment feature API."""

from dataclasses import dataclass

import numpy as np
from scipy.special import softmax

SCHEMA_VERSION = "latent-loop-trajectory-v1"
RUNTIME_FIELDS = frozenset(
    {"hidden", "probabilities", "entropy", "margin", "depth", "previous_change"}
)


@dataclass(frozen=True)
class RuntimeObservation:
    hidden: np.ndarray
    probabilities: np.ndarray
    entropy: float
    margin: float
    depth: int
    previous_change: float

    def vector(self):
        h = self.hidden / np.sqrt(np.mean(self.hidden * self.hidden) + 1e-6)
        return np.r_[
            h,
            self.probabilities,
            self.entropy,
            self.margin,
            self.previous_change,
            self.depth / 4,
        ].astype(np.float32)


def runtime_observation(current_hidden, current_logits, previous_hidden, depth):
    """Accepts only completed-state inputs. No trajectory metadata or labels."""
    if depth not in (1, 2, 3, 4):
        raise ValueError("Depth outside deployed budget")
    h, past = (
        np.asarray(current_hidden, dtype=np.float32),
        np.asarray(previous_hidden, dtype=np.float32),
    )
    if h.shape != (1024,) or past.shape != h.shape:
        raise ValueError("Expected current and previous 1024-dimensional states")
    p = softmax(np.asarray(current_logits, dtype=np.float64))
    if p.shape != (4,):
        raise ValueError("Exactly four conditional option scores required")
    h = h.copy()
    p = p.copy()
    h.flags.writeable = False
    p.flags.writeable = False
    ordered = np.sort(p)
    return RuntimeObservation(
        h,
        p,
        float(-(p * np.log(p.clip(1e-30))).sum()),
        float(ordered[-1] - ordered[-2]),
        depth,
        float(np.linalg.norm(h - past) / max(np.linalg.norm(past), 1e-6)),
    )


def runtime_from_fields(fields):
    if set(fields) != RUNTIME_FIELDS:
        raise ValueError(
            "Runtime observations reject supervision, future values, metadata, and unknown fields"
        )
    return RuntimeObservation(**fields)


def transition(current_correct, next_correct):
    return ("R" if current_correct else "W") + "->" + ("R" if next_correct else "W")


def record(
    row, seed, depth, logits, hidden, previous, loss, correct, state_ref, hashes
):
    k = depth - 1
    obs = runtime_observation(hidden[k], logits[k], previous, depth)
    supervision = {
        "true_answer_index": row["label"],
        "current_correct": bool(correct[k]),
        "current_loss_nats": float(loss[k]),
        "next_transition": None,
        "signed_correctness_change": None,
        "signed_loss_gain_nats": None,
    }
    if depth < 4:
        supervision.update(
            next_transition=transition(correct[k], correct[k + 1]),
            signed_correctness_change=int(correct[k + 1]) - int(correct[k]),
            signed_loss_gain_nats=float(loss[k] - loss[k + 1]),
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "instance_id": row["id"],
            "structural_hash": row["structural_hash"],
            "task_family": row["task"],
            "structural_generator_id": row["generator_id"],
            "split": row["split"],
            "parent_seed": seed,
        },
        "model": hashes,
        "depth": depth,
        "observation": {
            "predicted_answer_index": int(np.argmax(logits[k])),
            "candidate_logits": logits[k].tolist(),
            "candidate_probabilities": obs.probabilities.tolist(),
            "entropy": obs.entropy,
            "margin": obs.margin,
            "previous_state_change": obs.previous_change,
            "hidden_state_reference": state_ref,
        },
        "offline_supervision": supervision,
    }
