"""CPU-only public-artifact verification, clustered summaries, figures and handoff."""

import gzip
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import logsumexp, softmax
from scipy.stats import rankdata

from latent_loop.experiment04a.common import preservation, utc
from latent_loop.experiment04a.data import EXP, FINAL, OUT, TASKS, reference, sha
from latent_loop.experiment04a.science import binary, bootstrap
from latent_loop.experiment04a.trajectory import SCHEMA_VERSION, transition
from latent_loop.training import atomic_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORT = Path("reports/experiment_04a")
SPLITS = ("controller_train", "controller_val", *FINAL)
SHORT = {
    "controller_train": "Train",
    "controller_val": "Validation",
    "fresh_id": "Fresh ID",
    "new_template": "New format",
    "structural_shift": "New structure",
    "harder_depth": "Harder depth",
}


def read_json(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(Path(path).read_text())


def public_predictions(split, seed):
    x = read_json(OUT / "predictions" / f"{split}_seed{seed}.json.gz")
    for k in (
        "labels",
        "base_logits",
        "one_pass_logits",
        "logits",
        "correct",
        "loss",
        "gain",
        "lengths",
    ):
        x[k] = np.asarray(x[k])
    for k in ("confidence", "gain_predictions", "depths"):
        x[k] = {name: np.asarray(v) for name, v in x[k].items()}
    return x


def decisions(x, name):
    if name in ("frozen", "one_pass"):
        logits = x["base_logits" if name == "frozen" else "one_pass_logits"]
    else:
        logits = x["logits"][x["depths"][name] - 1, np.arange(len(x["labels"]))]
    return logits.argmax(-1) == x["labels"]


def auc(p, y):
    p, y = np.asarray(p).ravel(), np.asarray(y, bool).ravel()
    npos, nneg = y.sum(), (~y).sum()
    if not npos or not nneg:
        return float("nan")
    return float((rankdata(p)[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def ranking_interval(p, y, comparator=None, repeats=400):
    """Paired problem and parent-seed bootstrap; every problem retains all depths."""
    p, y = np.asarray(p), np.asarray(y)
    other = None if comparator is None else np.asarray(comparator)
    rng = np.random.default_rng(440711)

    def estimate(seeds, ix):
        scores = [
            auc(p[s][..., ix], y[s][..., ix])
            - (auc(other[s][..., ix], y[s][..., ix]) if other is not None else 0)
            for s in seeds
        ]
        return float(np.nanmean(scores))

    values = [
        estimate(
            rng.integers(0, len(p), len(p)), rng.integers(0, p.shape[-1], p.shape[-1])
        )
        for _ in range(repeats)
    ]
    return {
        "estimate": estimate(range(len(p)), np.arange(p.shape[-1])),
        "cluster_seed_bootstrap95": np.quantile(values, [0.025, 0.975]).tolist(),
        "unique_problems": p.shape[-1],
        "parent_seeds": len(p),
        "repeats": repeats,
    }


def ratio_interval(numerator, denominator, repeats):
    numerator, denominator = np.asarray(numerator), np.asarray(denominator)
    rng = np.random.default_rng(440712)
    samples = []
    for _ in range(repeats):
        seeds = rng.integers(0, len(numerator), len(numerator))
        ix = rng.integers(0, numerator.shape[-1], numerator.shape[-1])
        samples.append(
            numerator[seeds][:, :, ix].sum()
            / max(1, denominator[seeds][:, :, ix].sum())
        )
    return {
        "estimate": float(numerator.sum() / max(1, denominator.sum())),
        "cluster_seed_bootstrap95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "unique_problems": numerator.shape[-1],
        "parent_seeds": len(numerator),
        "repeats": repeats,
    }


def verify_public(config, predictions, recorded):
    """Recompute scores and every trajectory label from portable public artifacts."""
    manifest = read_json(EXP / "dataset_manifest.json")
    registries = read_json(EXP / "development_trajectory_registry.json") + read_json(
        EXP / "final_trajectory_registry.json"
    )
    registry = {(r["split"], r["seed"]): r for r in registries}
    unique_ids, unique_shapes, prompts = set(), set(), set()
    states = 0
    checks = 0
    for split in SPLITS:
        path = OUT / "datasets" / f"{split}.jsonl.gz"
        assert sha(path) == manifest["splits"][split]["sha256"]
        rows = [json.loads(line) for line in gzip.open(path, "rt")]
        for r in rows:
            assert (
                r["id"] not in unique_ids
                and r["structural_hash"] not in unique_shapes
                and r["prompt"] not in prompts
            )
            unique_ids.add(r["id"])
            unique_shapes.add(r["structural_hash"])
            prompts.add(r["prompt"])
            answer = reference(r["generator_id"], r["spec"])
            assert r["choices"][r["label"]] == (
                r["entities"][answer] if isinstance(answer, int) else answer
            )
        for seed in config["parent_seeds"]:
            x, m = predictions[split][seed], recorded[split][seed]
            assert x["ids"] == [r["id"] for r in rows]
            assert np.array_equal(x["labels"], [r["label"] for r in rows])
            logits, y = x["logits"], x["labels"]
            correct = logits.argmax(-1) == y
            loss = logsumexp(logits, axis=-1) - logits[:, np.arange(len(y)), y]
            assert np.array_equal(correct, x["correct"])
            assert np.allclose(loss, x["loss"], atol=1e-10)
            assert np.allclose(loss[:-1] - loss[1:], x["gain"], atol=1e-10)
            for name, values in x["confidence"].items():
                calculated = binary(values, correct)
                for field in (
                    "auroc",
                    "brier",
                    "ece",
                    "log_loss",
                    "aurc",
                    "accuracy_at_50pct_coverage",
                ):
                    assert np.isclose(calculated[field], m["confidence"][name][field])
                    checks += 1
            for name, values in x["gain_predictions"].items():
                calculated = binary(values[:3], x["gain"] > config["gain_delta_nats"])
                assert np.isclose(calculated["auroc"], m["gain"][name]["auroc"])
                checks += 1
            for name in m["policies"]:
                assert np.isclose(
                    decisions(x, name).mean(), m["policies"][name]["accuracy"]
                )
                checks += 1
            reg = registry[(split, seed)]
            assert sha(reg["public_trajectory"]) == reg["public_sha256"]
            with gzip.open(reg["public_trajectory"], "rt") as f:
                for i, row in enumerate(rows):
                    for k in range(4):
                        r = json.loads(next(f))
                        assert set(r) == {
                            "schema_version",
                            "identity",
                            "model",
                            "depth",
                            "observation",
                            "offline_supervision",
                        }
                        assert (
                            r["schema_version"] == SCHEMA_VERSION
                            and r["depth"] == k + 1
                        )
                        assert r["identity"]["instance_id"] == row["id"]
                        assert (
                            r["identity"]["structural_hash"] == row["structural_hash"]
                        )
                        assert r["identity"]["task_family"] == row["task"]
                        assert (
                            r["identity"]["structural_generator_id"]
                            == row["generator_id"]
                        )
                        assert (
                            r["identity"]["parent_seed"] == seed
                            and r["identity"]["split"] == split
                        )
                        assert r["model"] == reg["model_hashes"]
                        obs, sup = r["observation"], r["offline_supervision"]
                        assert set(sup) == {
                            "true_answer_index",
                            "current_correct",
                            "current_loss_nats",
                            "next_transition",
                            "signed_correctness_change",
                            "signed_loss_gain_nats",
                        }
                        assert set(obs) == {
                            "predicted_answer_index",
                            "candidate_logits",
                            "candidate_probabilities",
                            "entropy",
                            "margin",
                            "previous_state_change",
                            "hidden_state_reference",
                        }
                        p = softmax(logits[k, i])
                        assert np.allclose(
                            obs["candidate_probabilities"], p, atol=1e-10
                        )
                        assert np.array_equal(obs["candidate_logits"], logits[k, i])
                        assert obs["predicted_answer_index"] == int(
                            logits[k, i].argmax()
                        )
                        assert np.isclose(
                            obs["entropy"], -(p * np.log(p.clip(1e-30))).sum()
                        )
                        assert np.isclose(
                            obs["margin"], np.sort(p)[-1] - np.sort(p)[-2]
                        )
                        ref = obs["hidden_state_reference"]
                        assert (
                            ref["sha256"] == reg["hidden_sha256"]
                            and ref["depth_index"] == k
                            and ref["instance_index"] == i
                        )
                        assert sup["true_answer_index"] == y[i] and sup[
                            "current_correct"
                        ] == bool(correct[k, i])
                        assert np.isclose(sup["current_loss_nats"], loss[k, i])
                        if k < 3:
                            assert sup["next_transition"] == transition(
                                correct[k, i], correct[k + 1, i]
                            )
                            assert sup["signed_correctness_change"] == int(
                                correct[k + 1, i]
                            ) - int(correct[k, i])
                            assert np.isclose(
                                sup["signed_loss_gain_nats"],
                                loss[k, i] - loss[k + 1, i],
                            )
                        else:
                            assert all(
                                sup[v] is None
                                for v in (
                                    "next_transition",
                                    "signed_correctness_change",
                                    "signed_loss_gain_nats",
                                )
                            )
                        states += 1
                assert f.read() == ""
    value = {
        "completed_utc": utc(),
        "independent_reference_labels": len(unique_ids),
        "unique_problem_and_alpha_structure_count": len(unique_shapes),
        "trajectory_state_rows": states,
        "adjacent_supervised_transitions": states // 4 * 3,
        "public_metric_recomputation_checks": checks,
        "all_passed": True,
        "hidden_tensors_needed": False,
        "preservation": preservation(),
    }
    atomic_json(OUT / "public_verification.json", value)
    return value


def summarize(config, predictions, recorded):
    result = {}
    repeats = config["bootstrap_repeats"]
    for split in SPLITS:
        examples = [predictions[split][s] for s in config["parent_seeds"]]
        reports = [recorded[split][s] for s in config["parent_seeds"]]
        correct = np.stack([x["correct"] for x in examples])
        gains = np.stack([x["gain"] for x in examples])
        useful = gains > config["gain_delta_nats"]
        up = ~correct[:, :-1] & correct[:, 1:]
        down = correct[:, :-1] & ~correct[:, 1:]
        item = {
            "n": len(examples[0]["labels"]),
            "confidence": {},
            "gain": {},
            "policies": {},
            "events": {},
            "paired_accuracy": {},
        }
        for kind in ("confidence", "gain"):
            for name in reports[0][kind]:
                fields = (
                    "auroc",
                    "brier",
                    "ece",
                    "log_loss",
                    "accuracy_at_50pct_coverage",
                    "aurc",
                    "average_precision",
                    "prevalence",
                )
                item[kind][name] = {
                    field: float(np.mean([r[kind][name][field] for r in reports]))
                    for field in fields
                }
                p = np.stack(
                    [
                        x["confidence" if kind == "confidence" else "gain_predictions"][
                            name
                        ]
                        for x in examples
                    ]
                )
                if kind == "gain":
                    p = p[:, :3]
                item[kind][name]["auroc_interval"] = ranking_interval(
                    p, correct if kind == "confidence" else useful
                )
        for name in reports[0]["policies"]:
            d = np.stack([decisions(x, name) for x in examples])
            item["policies"][name] = {
                field: float(np.mean([r["policies"][name][field] for r in reports]))
                for field in ("accuracy", "macro_task_accuracy", "nll", "mean_loops")
            }
            item["policies"][name]["accuracy_interval"] = bootstrap(d, repeats)
            item["policies"][name]["seed_accuracies"] = [float(v.mean()) for v in d]
        for a, b in [
            ("improvement", "one_pass"),
            ("confidence", "one_pass"),
            ("fixed_selected", "one_pass"),
            ("improvement", "fixed_selected"),
            ("one_pass", "frozen"),
        ]:
            item["paired_accuracy"][f"{a}_minus_{b}"] = bootstrap(
                np.stack(
                    [decisions(x, a).astype(float) - decisions(x, b) for x in examples]
                ),
                repeats,
            )
        pred = np.stack([x["logits"].argmax(-1) for x in examples])
        pmax = np.stack([softmax(x["logits"], axis=-1).max(-1) for x in examples])
        events = {
            "wrong_to_right": up,
            "right_to_wrong": down,
            "wrong_to_wrong": ~correct[:, :-1] & ~correct[:, 1:],
            "right_to_right": correct[:, :-1] & correct[:, 1:],
            "meaningful_loss_gain": useful,
            "positive_loss_gain": gains > 0,
            "loss_increase": gains < 0,
            "meaningful_loss_harm": gains < -config["gain_delta_nats"],
            "unchanged_prediction": pred[:, :-1] == pred[:, 1:],
            "confidence_increase_without_correction": (pmax[:, 1:] > pmax[:, :-1])
            & ~up,
        }
        for name, values in events.items():
            item["events"][name] = bootstrap(values, repeats)
            item["events"][name]["by_depth"] = [
                bootstrap(values[:, k], repeats) for k in range(3)
            ]
        item["events"]["correction_given_useful_loss_gain"] = ratio_interval(
            up & useful, useful, repeats
        )
        item["events"]["any_correct_depth"] = bootstrap(correct.any(1), repeats)
        item["paired_gain_auc"] = {}
        existing = np.stack(
            [x["gain_predictions"]["existing_gain"][:3] for x in examples]
        )
        for name in (
            "confidence_only",
            "current_depth",
            "scalar_logistic",
            "task_difficulty_diagnostic",
        ):
            other = np.stack([x["gain_predictions"][name][:3] for x in examples])
            item["paired_gain_auc"][f"existing_gain_minus_{name}"] = ranking_interval(
                existing, useful, other
            )
        item["confidence_by_depth"] = {
            str(k + 1): {
                name: {
                    field: float(
                        np.mean(
                            [
                                binary(x["confidence"][name][k], x["correct"][k])[field]
                                for x in examples
                            ]
                        )
                    )
                    for field in ("auroc", "brier", "ece", "log_loss")
                }
                for name in examples[0]["confidence"]
            }
            for k in range(4)
        }
        item["gain_by_depth"] = {
            str(k + 1): {
                name: float(
                    np.mean(
                        [r["gain_by_depth"][str(k + 1)][name]["auroc"] for r in reports]
                    )
                )
                for name in reports[0]["gain"]
            }
            for k in range(3)
        }
        item["strata"] = {
            field: {
                value: {
                    "n": reports[0]["strata"][field][value]["n"],
                    "accuracy": {
                        name: float(
                            np.mean(
                                [
                                    r["strata"][field][value]["accuracy"][name]
                                    for r in reports
                                ]
                            )
                        )
                        for name in reports[0]["policies"]
                    },
                    "confidence_auroc": float(
                        np.mean(
                            [
                                r["strata"][field][value]["confidence_auroc"]
                                for r in reports
                            ]
                        )
                    ),
                    "gain_auroc": float(
                        np.mean(
                            [r["strata"][field][value]["gain_auroc"] for r in reports]
                        )
                    ),
                }
                for value in reports[0]["strata"][field]
            }
            for field in reports[0]["strata"]
        }
        item["unrestricted_backbone"] = reports[0]["unrestricted_backbone"]
        result[split] = item
        print(f"public clustered summary: {split}", flush=True)
    atomic_json(OUT / "summary.json", result)
    return result


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(REPORT / "figures" / f"{name}.png", dpi=170, bbox_inches="tight")
    fig.savefig(REPORT / "figures" / f"{name}.svg", bbox_inches="tight")
    svg = REPORT / "figures" / f"{name}.svg"
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def figures(summary, recorded, timing):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, task in zip(axes.flat, TASKS):
        for split in ("controller_val", *FINAL):
            path = OUT / "datasets" / f"{split}.jsonl.gz"
            rows = [json.loads(line) for line in gzip.open(path, "rt")]
            raw = read_json(OUT / "predictions" / f"{split}_seed11.json.gz")
            correct = np.asarray(raw["base_logits"]).argmax(-1) == raw["labels"]
            names = ["easy", "moderate", "hard", "extrapolation"]
            xx, yy = [], []
            for k, difficulty in enumerate(names):
                mask = np.array(
                    [
                        r["task"] == task and r["difficulty_stratum"] == difficulty
                        for r in rows
                    ]
                )
                if mask.any():
                    xx.append(k)
                    yy.append(correct[mask].mean())
            ax.plot(xx, yy, "o-", label=SHORT[split], alpha=0.85)
        ax.axhline(0.25, color="grey", ls=":")
        ax.set_title(task.capitalize())
        ax.set_ylim(0, 1)
        ax.set_xticks(range(4), ["Easy", "Moderate", "Hard", "Extra."], rotation=20)
    axes[0, 0].set_ylabel("Frozen four-choice accuracy")
    axes[1, 0].set_ylabel("Frozen four-choice accuracy")
    axes[0, 2].legend(fontsize=8)
    fig.suptitle(
        "Balanced nominal strata; static predicates do not deepen in every task", y=1.02
    )
    save_figure(fig, "accuracy_task_difficulty")
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True, sharey=True)
    for ax, split in zip(axes, FINAL):
        for name in (
            "raw_probability",
            "temperature",
            "existing_confidence",
            "confidence_recalibrated",
        ):
            bins = list(recorded[split].values())[0]["confidence"][name]["calibration"]
            # One parent plotted to avoid averaging bins with different occupancy; metrics average all parents.
            populated = [b for b in bins if b["count"]]
            xx = [b["confidence"] for b in populated]
            yy = [b["accuracy"] for b in populated]
            (line,) = ax.plot(xx, yy, "-", label=name.replace("_", " "))
            ax.scatter(
                xx,
                yy,
                s=[
                    5 + 180 * b["count"] / sum(v["count"] for v in bins)
                    for b in populated
                ],
                color=line.get_color(),
            )
        ax.plot([0, 1], [0, 1], color="grey", ls=":")
        ax.set_title(SHORT[split])
        ax.set_xlabel("Predicted correctness")
    axes[0].set_ylabel("Observed correctness")
    axes[0].legend(fontsize=7)
    fig.suptitle("Reliability, parent seed 11, depths 1–4 pooled descriptively", y=1.03)
    save_figure(fig, "confidence_calibration")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for name in (
        "existing_gain",
        "joint_gain",
        "confidence_only",
        "current_depth",
        "scalar_logistic",
        "task_difficulty_diagnostic",
    ):
        values = [
            summary[s]["gain"][name]["auroc_interval"]
            for s in ("controller_val", *FINAL)
        ]
        yy = np.array([v["estimate"] for v in values])
        ci = np.array([v["cluster_seed_bootstrap95"] for v in values]).T
        ax.errorbar(
            range(5),
            yy,
            yerr=np.maximum(0, [yy - ci[0], ci[1] - yy]),
            marker="o",
            capsize=2,
            label=name.replace("_", " "),
        )
    ax.axhline(0.5, color="grey", ls=":")
    ax.set_xticks(range(5), [SHORT[s] for s in ("controller_val", *FINAL)])
    ax.set_ylabel("Next-loss-gain AUROC (δ = 0.01 nat)")
    ax.legend(fontsize=8, ncol=2)
    save_figure(fig, "gain_generalization")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, split in zip(axes, FINAL):
        for name, offset in (("wrong_to_right", -0.16), ("right_to_wrong", 0.16)):
            vals = summary[split]["events"][name]["by_depth"]
            yy = np.array([v["estimate"] for v in vals])
            ci = np.array([v["cluster_seed_bootstrap95"] for v in vals]).T
            ax.bar(
                np.arange(3) + offset,
                yy,
                0.3,
                label=name.replace("_", " "),
                yerr=np.maximum(0, [yy - ci[0], ci[1] - yy]),
                capsize=2,
            )
        ax.set_xticks(range(3), ["1→2", "2→3", "3→4"])
        ax.set_title(SHORT[split])
    axes[0].set_ylabel("Fraction of all adjacent transitions")
    axes[0].legend(fontsize=8)
    save_figure(fig, "corrected_degraded")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for split in FINAL:
        ax.plot(
            range(1, 5),
            [summary[split]["policies"][f"fixed_{k}"]["accuracy"] for k in range(1, 5)],
            "o-",
            label=SHORT[split],
        )
    ax.axhline(0.25, color="grey", ls=":")
    ax.set_xticks(range(1, 5))
    ax.set_xlabel("Executed recurrence depth")
    ax.set_ylabel("Four-choice accuracy, mean over parents")
    ax.legend()
    save_figure(fig, "recurrence_depth")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = [
        "frozen",
        "one_pass",
        "fixed_4",
        "confidence:scalar",
        "confidence:compact",
        "confidence:masked",
        "improvement:scalar",
        "improvement:compact",
        "improvement:masked",
    ]
    for ax, batch in zip(axes, ("1", "8")):
        vals = timing["summary"][batch]
        ax.barh(
            range(len(names)),
            [vals[n]["amortized_ms_per_item"] for n in names],
            color=["#395b83"] * 3 + ["#3a8278"] * 3 + ["#b1783c"] * 3,
        )
        ax.set_yticks(range(len(names)), [n.replace(":", " / ") for n in names])
        ax.invert_yaxis()
        ax.set_xlabel("Full inference ms / item (batch amortized)")
        ax.set_title(f"Batch {batch}; synchronized live inference")
    save_figure(fig, "full_latency")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = [f"improvement:{m}@0.5" for m in ("scalar", "compact", "masked")]
    for ax, batch in zip(axes, ("1", "8")):
        vals = timing["summary"][batch]
        ax.bar(
            np.arange(3) - 0.16,
            [vals[n]["mean_returned_depth"] for n in names],
            0.3,
            label="Returned depth",
        )
        ax.bar(
            np.arange(3) + 0.16,
            [vals[n]["mean_executed_refiner_rows_per_item"] for n in names],
            0.3,
            label="Executed rows / item",
        )
        ax.set_xticks(range(3), ["Scalar", "Compacted", "Masked"])
        ax.set_title(f"Batch {batch}; diagnostic gain threshold 0.5")
        ax.set_ylim(0, 4.5)
    axes[0].legend()
    axes[0].set_ylabel("Loops or executed row-equivalents")
    save_figure(fig, "paid_work")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    for ax, split in zip(axes, FINAL):
        for name in (
            "raw_probability",
            "temperature",
            "existing_confidence",
            "confidence_recalibrated",
        ):
            curves = [
                r["confidence"][name]["risk_coverage"] for r in recorded[split].values()
            ]
            ax.plot(
                [v["coverage"] for v in curves[0]],
                np.mean([[v["risk"] for v in c] for c in curves], axis=0),
                label=name.replace("_", " "),
            )
        ax.set_title(SHORT[split])
        ax.set_xlabel("Coverage")
    axes[0].set_ylabel("Decision error among retained states")
    axes[0].legend(fontsize=7)
    save_figure(fig, "risk_coverage")


def percentage(x):
    return f"{100 * x:.2f}%"


def timing_analysis(timing, config):
    result = {
        "bootstrap_units": "Parent seeds and paired prompt groups; batch 8 has only three groups, not 24 independent timing units.",
        "load_note": "Timing ran under existing system load and overlapped CPU public-artifact verification/bootstrap analysis for part of the run. No unrelated jobs were changed. Interleaving balances within-run comparisons but does not establish idle-system performance.",
        "paired_differences": {},
        "padding_ratio": {},
    }
    for batch in (1, 8):
        records = [r for r in timing["records"] if r["batch_size"] == batch]
        index = {(r["seed"], r["batch_index"], r["variant"]): r for r in records}
        groups = sorted({r["batch_index"] for r in records})
        seed = config["parent_seeds"][0]
        lengths = [index[(seed, k, "frozen")]["prompt_lengths"] for k in groups]
        result["padding_ratio"][str(batch)] = float(
            sum(max(v) * len(v) for v in lengths) / sum(sum(v) for v in lengths)
        )
        result["paired_differences"][str(batch)] = {}
        for a, b in [
            ("improvement:compact", "one_pass"),
            ("improvement:compact", "fixed_4"),
            ("improvement:compact@0.5", "improvement:scalar@0.5"),
        ]:
            values = np.array(
                [
                    [
                        (
                            np.median(index[(s, k, a)]["full_ms"])
                            - np.median(index[(s, k, b)]["full_ms"])
                        )
                        / batch
                        for k in groups
                    ]
                    for s in config["parent_seeds"]
                ]
            )
            v = bootstrap(values, config["bootstrap_repeats"])
            v["unique_prompt_groups"] = v.pop("unique_problems")
            v["unit"] = "amortized milliseconds per item"
            result["paired_differences"][str(batch)][f"{a}_minus_{b}"] = v
    atomic_json(OUT / "timing_analysis.json", result)
    return result


def interval(v, percent=False, difference=False):
    factor = 100 if percent else 1
    suffix = (" pp" if difference else "%") if percent else ""
    lo, hi = v["cluster_seed_bootstrap95"]
    return (
        f"{factor * v['estimate']:.3f} [{factor * lo:.3f}, {factor * hi:.3f}]{suffix}"
    )


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def report(config):
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(exist_ok=True)
    predictions = {
        split: {s: public_predictions(split, s) for s in config["parent_seeds"]}
        for split in SPLITS
    }
    recorded = {
        split: {
            s: read_json(OUT / "metrics" / f"{split}_seed{s}.json")
            for s in config["parent_seeds"]
        }
        for split in SPLITS
    }
    verification = verify_public(config, predictions, recorded)
    summary = summarize(config, predictions, recorded)
    timing = read_json(OUT / "latency.json")
    figures(summary, recorded, timing)
    write_documents(config, summary, recorded, timing, verification)
    print(
        "public verification, eight figures, foundation report and 4B handoff complete",
        flush=True,
    )


def write_documents(config, summary, recorded, timing, verification):
    timing_stats = timing_analysis(timing, config)
    manifest = read_json(EXP / "dataset_manifest.json")
    audit = read_json(EXP / "existing_audit.json")
    frozen = read_json(EXP / "frozen_protocol.json")
    opened = read_json(EXP / "lockbox_opened.json")
    selected = {
        s: read_json(EXP / f"selection_seed{s}.json") for s in config["parent_seeds"]
    }
    extraction = [
        read_json(OUT / name)
        for name in ("development_extraction.json", "final_extraction.json")
    ]
    peak_mlx = (
        max([timing["peak_mlx_bytes"]] + [x["peak_mlx_bytes"] for x in extraction])
        / 1024**3
    )
    peak_rss = (
        max(
            [timing["peak_process_rss_bytes"]]
            + [x["peak_process_rss_bytes"] for x in extraction]
        )
        / 1024**3
    )
    elapsed = (
        datetime.fromisoformat(utc()) - datetime.fromisoformat(config["started_utc"])
    ).total_seconds()
    resource_summary = {
        "reported_utc": utc(),
        "elapsed_wall_seconds_at_report": elapsed,
        "peak_recorded_mlx_gib": peak_mlx,
        "peak_recorded_process_rss_gib": peak_rss,
        "memory_budget_gib": config["memory_gib"],
        "timing_command_seconds": timing["command_seconds"],
        "extraction_command_seconds": [x["command_seconds"] for x in extraction],
        "statistical_fits": 36,
        "new_neural_fits": 0,
        "model_weight_mutation": False,
        "prior_work_preserved": True,
        "remote_workstation_accessed": False,
    }
    atomic_json(OUT / "execution_summary.json", resource_summary)
    totals = sum(config["sizes"].values())
    lines = [
        "# Experiment 4A: foundation, generalization and measurement hardening",
        "",
        f"Recorded {utc()}. Starting commit `{config['start_commit']}`; branch `research/exp04-foundation`. All original conclusions and files are preserved; this is a new robustness evaluation.",
        "",
        "## Findings",
        "",
        "The stronger benchmark substantially weakens the historical confidence and immediate-gain signals. Frozen Experiment 3 weights were reused for all three parents; no neural models were trained or selected on these final results. The tables below distinguish signal ranking, calibration, actual decision changes and paid computation. The historical 0.798 correctness / 0.762 gain AUROCs are exploratory reference points, not expectations for this distribution.",
        "",
        table(
            [
                "Split",
                "Frozen",
                "One pass",
                "Gain stopping",
                "Loops",
                "Confidence AUROC",
                "Gain AUROC",
                "Scalar gain AUROC",
            ],
            [
                [
                    SHORT[s],
                    percentage(summary[s]["policies"]["frozen"]["accuracy"]),
                    percentage(summary[s]["policies"]["one_pass"]["accuracy"]),
                    percentage(summary[s]["policies"]["improvement"]["accuracy"]),
                    f"{summary[s]['policies']['improvement']['mean_loops']:.3f}",
                    f"{summary[s]['confidence']['existing_confidence']['auroc']:.3f}",
                    f"{summary[s]['gain']['existing_gain']['auroc']:.3f}",
                    f"{summary[s]['gain']['scalar_logistic']['auroc']:.3f}",
                ]
                for s in ("controller_val", *FINAL)
            ],
        ),
        "",
        "Fresh ID means in-distribution relative to the *new 4A generator mixture*. It is already a transfer distribution for the historical refiner and heads. Structural shift additionally withholds one semantic generator per task; new format uses held-out structured presentations for the same 4A algorithms. These are different shifts, not controlled estimates of a single causal template effect.",
        "",
        "## Historical audit and equal information",
        "",
        f"Recomputed {audit['historical_policy_accuracy_and_exact_nll_checks']} historical stopping-policy accuracy/NLL entries from saved predictions. Verified three distinct checkpoint hashes for each frozen component and {verification['preservation']['historical_files']} historical tracked files (only a README link may change). Reviewed old generators, refiner, confidence/gain probes, deployment features, stopping code and reproduction instructions. Historical artifacts were not rewritten.",
        "",
        "All variants share each prompt, the same four candidates, the same pinned Qwen3-0.6B FP32 backbone representation and label-token decoder. The one-pass head and refiner retain their original weights. Current argmax decisions are held fixed when comparing correctness estimators; calibration does not rerank candidates. Runtime probes see only current/completed states. Task/difficulty controls are explicitly offline diagnostics. Every live policy must pay for the loops it executes.",
        "",
        "Original candidate NLL clips tiny probabilities; this experiment uses stable, unclipped four-choice log-sum-exp NLL. Conditional accuracy is separate from unrestricted next-token accuracy; the latter here means the immediate next token after the frozen answer prefix, not free-form completion accuracy.",
        "",
        table(
            [
                "Final split",
                "Four-choice frozen accuracy",
                "Unrestricted next-token accuracy",
                "Option-token rate",
            ],
            [
                [
                    SHORT[s],
                    percentage(summary[s]["policies"]["frozen"]["accuracy"]),
                    percentage(summary[s]["unrestricted_backbone"]["accuracy"]),
                    percentage(
                        summary[s]["unrestricted_backbone"]["option_token_rate"]
                    ),
                ]
                for s in FINAL
            ],
        ),
        "",
        "## Independent data and prospective lockbox",
        "",
        table(
            ["Split", "Instances", "Per task", "RNG seed", "Role"],
            [
                [
                    s,
                    config["sizes"][s],
                    config["sizes"][s] // 6,
                    config["split_seeds"][s],
                    "Fit controls"
                    if s == "controller_train"
                    else "Calibrate/select"
                    if s == "controller_val"
                    else "Final, now consumed",
                ]
                for s in SPLITS
            ],
        ),
        "",
        table(
            [
                "Task",
                "Development approaches",
                "Withheld structural approach",
                "Independent solver",
            ],
            [
                [
                    "Arithmetic",
                    "Accumulator; expression tree",
                    "Parallel ledgers",
                    "Affine algebra; constructed integer expression; signed account balance",
                ],
                [
                    "Comparison",
                    "Mixed units; resource filtering",
                    "Value per cost",
                    "Integer byte normalization; declarative eligibility; exact Fraction ratios",
                ],
                [
                    "State",
                    "Register operations; inventory transfers",
                    "Boolean program",
                    "Homogeneous transition matrices; independent bit masks",
                ],
                [
                    "Pointer",
                    "Functional graph; alternating maps",
                    "Symbolic modular walk",
                    "Transition-matrix powers; direct modular composition",
                ],
                [
                    "Classification",
                    "Numeric conjunction; disjunction with negation",
                    "Weighted conditions",
                    "Independent declarative predicates",
                ],
                [
                    "Routing",
                    "Capability constraints; incomplete/conflicting requests",
                    "Ordered rules",
                    "Eligible-set filters and priority rules",
                ],
            ],
        ),
        "",
        "Labels are generated by a direct simulator and checked by a separate deterministic solver. All public examples have four distinct choices. Entity names and candidate permutations are removed from structural fingerprints; constants, operations and queries are retained. Register renamings are minimized over permutations; directed rooted graph invariants conservatively reject isomorphic instances. A conservative collision can reject a distinct graph. ID/lexical tests intentionally share abstract algorithm archetypes; literal-free skeleton disjointness is not claimed.",
        "",
        f"All {totals:,} generated instances have distinct prompt, instance and alpha-structure fingerprints across all six splits. Rejections: `{json.dumps(manifest['integrity']['generation_rejections'], sort_keys=True)}`. A pre-freeze historical audit found one renamed pointer graph in the unopened draft fresh-ID set. The public historical fingerprint exclusion list removed it; development data remained unchanged. Comparable old accumulator, graph and normalized memory-max instances are excluded. Historical modulo-register, Boolean-only classifier and verb-registry semantics differ from the new generators. The integrity amendment is recorded without exposing final scores.",
        "",
        f"Protocol frozen at `{frozen['frozen_utc']}`; lockbox first opened at `{opened['opened_utc']}`. Source, configuration, dataset manifest, exclusion list, plan, selections and checkpoint hashes were fixed before final extraction/scoring. Train/validation were the only selection data. Final sets are now consumed evaluation data; **4B requires a fresh independently seeded prospective final set**. No final-driven changes to heads, tasks, thresholds or architecture were made.",
        "",
        "## Competence and benchmark difficulty",
        "",
        "Tasks, approaches and nominal difficulty strata are balanced by a fixed schedule. Easy/moderate/hard sequential tasks use 1/3/5 updates, and extrapolation uses 8. Expression trees use 1/3/7/15 operators at depth 1/2/3/4. Graphs expand from 8/10/12 to 16 nodes. Arithmetic includes negative values and different operand distributions. Actual operation count, depth, maximum absolute numeric input and token length are retained and evaluated independently.",
        "",
        "Static comparison and classification approaches do not increase their predicate count across every nominal stratum; harder-depth is therefore a mixed extrapolation condition. Numbers and entity indices contribute to `numeric_max`, so it is a descriptive input-range measure. Program length also includes instructions that can be irrelevant to the queried variable. Difficulty labels do not establish minimum reasoning depth. No poor-performing task was dropped or replaced after observing model outcomes.",
        "",
        table(
            ["Task", *[SHORT[s] for s in FINAL]],
            [
                [
                    task,
                    *[
                        percentage(
                            summary[s]["strata"]["task"][task]["accuracy"]["frozen"]
                        )
                        for s in FINAL
                    ],
                ]
                for task in TASKS
            ],
        ),
        "",
        "Balanced micro accuracy equals the mean over six task accuracies. Task-level gains can cancel substantial losses. Full per-generator, actual-depth, operation, numeric-range, template and token-length results are in the public metrics and summary JSON; low baseline competence limits claims about additional reasoning.",
        "",
        "![Frozen accuracy by task and nominal difficulty](figures/accuracy_task_difficulty.png)",
        "",
        "## Confidence: ranking versus calibration",
        "",
        "Eight estimators use the same current decisions: raw maximum candidate probability, temperature-scaled probability, entropy and margin logistic calibration, frozen confidence and joint heads, and separately recalibrated versions of both heads. One temperature and four scalar logistic calibrators per parent are fitted only on validation states. Training-only gain controls contribute seven scalar fits per parent; total new statistical fits = 36, new neural fits = 0.",
        "",
        table(
            [
                "Parent",
                "Temperature",
                "Confidence threshold",
                "Gain threshold",
                "Selected fixed depth",
            ],
            [
                [
                    s,
                    f"{selected[s]['calibrators']['temperature']:.3f}",
                    selected[s]["policies"]["confidence"]["threshold"],
                    selected[s]["policies"]["improvement"]["threshold"],
                    selected[s]["policies"]["fixed_selected"]["depth"],
                ]
                for s in config["parent_seeds"]
            ],
        ),
        "",
        table(
            [
                "Final split / estimator",
                "AUROC [95%]",
                "Brier",
                "ECE",
                "Binary log loss",
                "Accuracy at 50% coverage",
                "AURC",
            ],
            [
                [
                    f"{SHORT[s]} / {name}",
                    interval(summary[s]["confidence"][name]["auroc_interval"]),
                    *[
                        f"{summary[s]['confidence'][name][k]:.3f}"
                        for k in (
                            "brier",
                            "ece",
                            "log_loss",
                            "accuracy_at_50pct_coverage",
                            "aurc",
                        )
                    ],
                ]
                for s in FINAL
                for name in (
                    "raw_probability",
                    "temperature",
                    "entropy_calibration",
                    "margin_calibration",
                    "existing_confidence",
                    "confidence_recalibrated",
                    "joint_confidence",
                    "joint_recalibrated",
                )
            ],
        ),
        "",
        "Ranking AUROC measures discrimination; Brier/ECE/log loss measure probability quality. Temperature minimizes candidate NLL and preserves each answer, but multiclass maximum-probability rankings can change between examples. Recalibration can improve Brier while leaving an uninformative ranking. Report all metrics rather than equating lower calibration error with better decisions. Reliability plots show parent 11 so bins with different occupancy are not averaged; marker areas indicate occupancy and sparse bins are imprecise. Tables average all parents. Risk-coverage curves average parents descriptively. Tied-score ordering makes selective metrics sensitive to tie resolution, especially constant controls.",
        "",
        "![Confidence calibration by split](figures/confidence_calibration.png)",
        "",
        "![Risk coverage](figures/risk_coverage.png)",
        "",
        "Temperature scaling is a validation-based post-hoc method discussed by [Guo et al., 2017](https://arxiv.org/abs/1706.04599); that paper does not establish transfer calibration for this benchmark.",
        "",
        "## Immediate gain and actual corrections",
        "",
        "A useful loss gain is current exact conditional NLL minus next-loop NLL greater than 0.01 nat. It can improve probability on the true answer without changing the argmax decision. Gain supervision covers depths 1→2, 2→3 and 3→4; no fabricated depth-4 continuation label is included. Training controls predict gain from frozen confidence alone, entropy, margin, depth, all current scalar features, task/difficulty metadata, or metadata plus the frozen hidden-state gain score. The last two are diagnostic conditions, not deployable policies.",
        "",
        table(
            ["Final split / predictor", "Gain AUROC [95%]", "AP", "Brier", "Log loss"],
            [
                [
                    f"{SHORT[s]} / {name}",
                    interval(summary[s]["gain"][name]["auroc_interval"]),
                    *[
                        f"{summary[s]['gain'][name][k]:.3f}"
                        for k in ("average_precision", "brier", "log_loss")
                    ],
                ]
                for s in FINAL
                for name in (
                    "existing_gain",
                    "joint_gain",
                    "confidence_only",
                    "entropy",
                    "margin",
                    "current_depth",
                    "scalar_logistic",
                    "task_difficulty_diagnostic",
                    "diagnostic_plus_hidden_gain",
                    "train_prior",
                )
            ],
        ),
        "",
        table(
            [
                "Split",
                "Useful gain",
                "W→R [95%]",
                "R→W [95%]",
                "W→R given useful gain [95%]",
                "Unchanged prediction",
                "Confidence ↑ without correction",
            ],
            [
                [
                    SHORT[s],
                    percentage(
                        summary[s]["events"]["meaningful_loss_gain"]["estimate"]
                    ),
                    interval(summary[s]["events"]["wrong_to_right"], True),
                    interval(summary[s]["events"]["right_to_wrong"], True),
                    interval(
                        summary[s]["events"]["correction_given_useful_loss_gain"], True
                    ),
                    percentage(
                        summary[s]["events"]["unchanged_prediction"]["estimate"]
                    ),
                    percentage(
                        summary[s]["events"]["confidence_increase_without_correction"][
                            "estimate"
                        ]
                    ),
                ]
                for s in FINAL
            ],
        ),
        "",
        "All four transition classes and all positive/harmful loss rates have problem/seed intervals in `summary.json`. Confidence increase means maximum candidate probability increased; an unchanged correct answer counts as no correction. Conditional correction rates retain the same clustered resampling in numerator and denominator.",
        "",
        "![Improvement AUROC under distribution shifts](figures/gain_generalization.png)",
        "",
        "![Corrected versus degraded transitions](figures/corrected_degraded.png)",
        "",
        "Frozen hidden-state probes do not isolate a causal hidden-information effect: those probes were trained on historical data whereas scalar controls fit fresh 4A training data. The metadata controls diagnose task/difficulty mixtures; adding the frozen gain score to those controls tests incremental transfer information, not a newly trained hidden-state controller. Per-depth AUROCs help reveal easy ranking from current depth alone. No larger neural head was adopted.",
        "",
        "## Paired policy comparisons and privileged trajectory ceilings",
        "",
        table(
            [
                "Final split",
                "Gain minus one pass [95%], pp",
                "Selected fixed minus one pass [95%], pp",
                "Gain minus selected fixed [95%], pp",
            ],
            [
                [
                    SHORT[s],
                    *[
                        interval(
                            summary[s]["paired_accuracy"][name], True, difference=True
                        )
                        for name in (
                            "improvement_minus_one_pass",
                            "fixed_selected_minus_one_pass",
                            "improvement_minus_fixed_selected",
                        )
                    ],
                ]
                for s in FINAL
            ],
        ),
        "",
        table(
            [
                "Final split",
                "Depth 1",
                "Depth 2",
                "Depth 3",
                "Depth 4",
                "Oracle min-loss",
                "Oracle first-correct",
                "First-correct paid loops",
            ],
            [
                [
                    SHORT[s],
                    *[
                        percentage(summary[s]["policies"][f"fixed_{d}"]["accuracy"])
                        for d in range(1, 5)
                    ],
                    percentage(summary[s]["policies"]["oracle_best_loss"]["accuracy"]),
                    percentage(
                        summary[s]["policies"]["oracle_first_correct"]["accuracy"]
                    ),
                    f"{summary[s]['policies']['oracle_first_correct']['mean_loops']:.3f}",
                ]
                for s in FINAL
            ],
        ),
        "",
        "Both oracles use labels/future information and are privileged evaluation bounds. Min-loss depth need not select a correct argmax even when another depth is correct. First-correct chooses the earliest correct depth or depth 1 when none is correct, paying every loop needed to reach that depth; its stopping knowledge is unavailable at inference. A deployed selector would also pay to acquire any later states it inspects. These ceilings do not establish an achievable value-head improvement.",
        "",
        "![Recurrence depth behavior](figures/recurrence_depth.png)",
        "",
        "Intervals jointly resample the same independent problem indices across all parents and all depths, and resample three parent seeds. Mean/rate and paired accuracy use 1,000 replicates; AUROC and paired AUROC use 400. Depth states and duplicated backbone results are not independent samples. Only three parents weakly characterize training variability; intervals are descriptive and no multiplicity-adjusted significance claim is made. Comparisons from training or calibration validation are apparent/development results.",
        "",
        "## Full inference timing and executed work",
        "",
        timing["scope"],
        "",
        f"Three warmups and {timing['repeats']} repeats per variant/batch, interleaved deterministic order, {len(timing['timed_validation_ids'])} validation prompts balanced across six tasks and three parent seeds. Batch 1 and batch 8 use the same prompts. Prompt-length padding changes work; amortized batch time is throughput, not single-request latency. Model load and model download are excluded. No unrelated process was terminated or reconfigured.",
        "",
        table(
            [
                "Variant",
                "Batch-1 ms/item",
                "Batch-8 amortized ms/item",
                "Batch-8 auxiliary ms/batch",
                "Returned depth",
                "Paid rows/item",
                "Host barriers/batch",
            ],
            [
                [
                    name,
                    f"{timing['summary']['1'][name]['amortized_ms_per_item']:.3f}",
                    f"{timing['summary']['8'][name]['amortized_ms_per_item']:.3f}",
                    f"{timing['summary']['8'][name]['aux_mean_batch_median_ms']:.3f}",
                    f"{timing['summary']['8'][name]['mean_returned_depth']:.3f}",
                    f"{timing['summary']['8'][name]['mean_executed_refiner_rows_per_item']:.3f}",
                    f"{timing['summary']['8'][name]['mean_midloop_host_barriers_per_batch']:.3f}",
                ]
                for name in timing["summary"]["1"]
            ],
        ),
        "",
        "Scalar halting performs a host extraction per active example/depth. Compaction extracts one Boolean continue-vector per completed depth and actually shrinks active GPU rows; it still pays synchronization, indexing and Python assembly overhead. Masked selection stays on device during loops but executes all four refinements and three probe invocations for every row. Early returned depth is not saved computation in that condition. Neutral threshold-0.5 variants are diagnostic and never substituted for validation-selected policies.",
        "",
        table(
            ["Paired full-latency difference, ms/item [95%]", "Batch 1", "Batch 8"],
            [
                [
                    name,
                    *[
                        interval(timing_stats["paired_differences"][b][name])
                        for b in ("1", "8")
                    ],
                ]
                for name in timing_stats["paired_differences"]["1"]
            ],
        ),
        "",
        f"Padding token-work ratio is {timing_stats['padding_ratio']['1']:.3f} for batch 1 and {timing_stats['padding_ratio']['8']:.3f} for batch 8. Batch 8 did not improve amortized full latency on this heterogeneous sample. At diagnostic gain threshold 0.5 it reduces scalar host barriers from {timing['summary']['8']['improvement:scalar@0.5']['mean_midloop_host_barriers_per_batch']:.2f} to {timing['summary']['8']['improvement:compact@0.5']['mean_midloop_host_barriers_per_batch']:.2f} per batch; auxiliary cost changes from {timing['summary']['8']['improvement:scalar@0.5']['aux_mean_batch_median_ms']:.3f} to {timing['summary']['8']['improvement:compact@0.5']['aux_mean_batch_median_ms']:.3f} ms per batch. Full backbone cost remains dominant. The paired timing bootstrap uses parent seeds and prompt groups; batch 8 has only three groups and cannot support strong precision claims.",
        "",
        timing_stats["load_note"],
        "",
        "Every timed output/depth is checked against independently cached trajectories; all three halting mechanisms agree within 0.002 logit absolute tolerance. Compact/scalar paid-row counts match actual returned depths; masked counts are four rows per item. Separate controller-only samples and component synchronization diagnostics are stored; extra diagnostic barriers do not represent additive normal-path cost. [MLX lazy evaluation documentation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html) explains why explicit evaluation/synchronization is needed for measurement.",
        "",
        f"System load averages: start `{timing['start_load_average']}`, end `{timing['end_load_average']}`. Full warmup/repeat records, p10/p90 across batch medians, prompt lengths and executed rows are public. Twenty-four prompts are too few for hardware-wide latency claims; load, thermal state, dispatch noise and prompt lengths limit small speed differences. This is FP32 final-vector recurrence on one M4 Max, not a general kernel optimization result.",
        "",
        "![End-to-end latency](figures/full_latency.png)",
        "",
        "![Returned depth versus paid computation](figures/paid_work.png)",
        "",
        "## Trajectory deliverable, verification and resources",
        "",
        f"Published {verification['trajectory_state_rows']:,} versioned state records and {verification['adjacent_supervised_transitions']:,} adjacent transition labels for {totals:,} distinct problems × three parents × four depths. The public verifier independently checked all solver labels, every trajectory transition/loss/probability, and {verification['public_metric_recomputation_checks']:,} metric entries without hidden arrays or MLX. All checks passed. Hidden states and checkpoint files remain outside Git with SHA256 references.",
        "",
        f"Measured peak MLX allocation: **{peak_mlx:.3f} GiB**; measured peak process RSS across extraction/timing: **{peak_rss:.3f} GiB**, under the configured {config['memory_gib']} GiB allocation limit. RSS is process memory, not total machine usage; peak MLX and RSS have different scopes. Extraction commands took {extraction[0]['command_seconds']:.1f} s (development) and {extraction[1]['command_seconds']:.1f} s (final); timing took {timing['command_seconds']:.1f} s. Wall time through this report: {elapsed / 60:.1f} min. Later verification/commit time is recorded in completion metadata.",
        "",
        "The first development metrics pass hit a Boolean-label log-loss dtype error after feature/trajectory extraction. It was fixed before protocol freeze; cached features were reused. Final data were still sealed. No neural retraining, workstation connection or external publication occurred.",
        "",
        "## Reproduction and limitations",
        "",
        "See [4B handoff](experiment_04b_handoff.md) for exact paths/schema and commands. CPU report regeneration uses only committed public files; model extraction uses pinned public dependencies plus locally regenerated historical checkpoints. Historical model weights are not distributed. Their original hashes support local exact reuse; independent retraining can differ and needs a new declared protocol rather than silently accepting different weights.",
        "",
        "The benchmark is synthetic, answer-conditioned and operates on the final hidden vector without generated reasoning. Constants/name rejection does not make tasks natural or prove unseen algorithms. Structured templates simultaneously change syntax, length and representation. Static nominal strata are not matched increasing depths. Poor backbone competence, historical refiner transfer failure, three seeds, four-loop horizon, few timing prompts and multiple exploratory summaries restrict inference. Calibrators trained on validation may fail under new structures. Event scarcity and differing correction/degradation rates complicate the decision-value target.",
        "",
        "The infrastructure is ready for 4B, with scientifically useful negative transfer evidence and explicit correction labels. It does not validate the proposed decision-value architecture. Keep architecture/feature selection on development data, compare against one-pass and scalar controls, and establish a new prospective final lockbox before any 4B scoring. No 4B model or trajectory selector was implemented.",
        "",
    ]
    (REPORT / "foundation_report.md").write_text("\n".join(lines))
    handoff(config, summary, timing, verification, selected, frozen, opened)


def handoff(config, summary, timing, verification, selected, frozen, opened):
    """The fourteen requested handoff topics; descriptive results are not model selection."""
    lines = [
        "# Experiment 4B handoff — prepared by 4A",
        "",
        "No decision-value head or recurrent trajectory selector was implemented. This handoff supplies data, integrity controls, evaluation baselines and measurement tools. 4A final outcomes are descriptive evidence; they must not select the next architecture.",
        "",
        "## 1. Available trajectory datasets",
        "",
        f"{sum(config['sizes'].values()):,} unique instances, three parents, depths 1–4: {verification['trajectory_state_rows']:,} state rows and {verification['adjacent_supervised_transitions']:,} adjacent transitions.",
        "",
        table(
            [
                "Split",
                "Instances",
                "State rows across parents",
                "RNG seed",
                "Public trajectory pattern",
            ],
            [
                [
                    s,
                    config["sizes"][s],
                    config["sizes"][s] * 12,
                    config["split_seeds"][s],
                    f"`results/experiment_04a/trajectories/{s}_seed{{11,22,33}}.jsonl.gz`",
                ]
                for s in SPLITS
            ],
        ),
        "",
        "Public generators: `src/latent_loop/experiment04a/data.py`; dataset recipes/manifests/exclusions: `experiments/experiment_04a/`. Public full instances: `results/experiment_04a/datasets/`. Per-parent metrics and raw probabilities/logits: `results/experiment_04a/{metrics,predictions}/`. Registries: `experiments/experiment_04a/{development,final}_trajectory_registry.json`. Hidden arrays are local ignored NPZs under `artifacts/experiment_04a/trajectories/`; public references include SHA256 and array coordinates. They are available on this machine, not distributed in Git.",
        "",
        "## 2. Exact trajectory schema",
        "",
        f"Schema version: `{SCHEMA_VERSION}`. UTF-8 JSONL compressed with gzip. One row per instance × parent × depth, ordered instance-major then depth. No global row ID is needed: `(instance_id, parent_seed, depth)` is the primary key. This version has exactly the following namespaces:",
        "Machine-readable JSON Schema: `experiments/experiment_04a/trajectory_schema.json`. The validator also independently checks cross-row arithmetic and labels, which JSON Schema alone cannot establish.",
        "",
        table(
            ["Namespace", "Fields / types"],
            [
                [
                    "root",
                    "schema_version: string; identity: object; model: object; depth: integer 1–4; observation: object; offline_supervision: object",
                ],
                [
                    "identity",
                    "instance_id: SHA256 string; structural_hash: SHA256 string; task_family: string; structural_generator_id: string; split: string; parent_seed: integer",
                ],
                [
                    "model",
                    "backbone_id: string; backbone_revision: pinned commit string; refiner_sha256: SHA256 string",
                ],
                [
                    "observation",
                    "predicted_answer_index: integer 0–3; candidate_logits: float[4]; candidate_probabilities: float[4]; entropy: float; margin: float; previous_state_change: float; hidden_state_reference: object",
                ],
                [
                    "hidden_state_reference",
                    "artifact_path: repository-relative local path; sha256: SHA256 string; array: 'states'; depth_index: integer 0–3; instance_index: integer; dtype: 'float32'; dimensions: [1024]",
                ],
                [
                    "offline_supervision",
                    "true_answer_index: integer 0–3; current_correct: boolean; current_loss_nats: float; next_transition: 'W->R'/'R->W'/'W->W'/'R->R' or null; signed_correctness_change: -1/0/+1 or null; signed_loss_gain_nats: float or null",
                ],
            ],
        ),
        "",
        "Loss is `logsumexp(current candidate logits) - true candidate logit`. `signed_loss_gain_nats = loss(depth) - loss(depth+1)`: positive means next loss decreased. Conventional signed next-minus-current loss change is its negative. `signed_correctness_change = int(next correct) - int(current correct)`. All adjacent fields are null at depth 4. Probabilities condition on four candidates; entropy uses natural logs, margin is largest minus second-largest probability. State change is `||current-previous|| / max(||previous||, 1e-6)`; depth 1 compares with the frozen backbone state. NPZ `states` has shape `[4,N,1024]`, and `backbone` has `[N,1024]`.",
        "",
        "## 3. Split identities and allowed uses",
        "",
        "Only `controller_train` and `controller_val` are development splits. They are balanced across 12 semantic generator approaches, six tasks and three nominal strata. Four final splits have 1,296 instances each: fresh ID (same 4A approaches, new instances), new template (held-out structured presentation IDs 2/3), structural shift (six withheld approaches), harder depth (longer sequential programs and larger graphs, with static-task limitations). All seeds are unique and all six splits reject alpha-renamed instances. Manifest hashes and public exclusions are immutable evidence, not permission to reuse consumed tests for selection.",
        "",
        "## 4. Recommended causal input features",
        "",
        "The established runtime observation is `hidden[1024], probabilities[4], entropy, margin, depth, previous_change`. `RuntimeObservation.vector()` yields 1,032 FP32 values: RMS-normalized hidden state, four probabilities, entropy, margin, completed relative change, depth/4. `runtime_observation(current_hidden, current_logits, previous_hidden, depth)` accepts completed-state values only. `runtime_from_fields` rejects labels, future state/loss/prediction, oracle depth, offline supervision, task metadata and unknown keys.",
        "",
        "These are interface recommendations already established in the frozen 4A plan, not architecture selection from final performance. Evaluate hidden versus scalar ablations using development data. Frozen confidence/gain predictions can be optional derived current-state features under an explicit condition. Do not include ground-truth answers, future states, future probabilities, future losses, transition labels, oracle depths or task/difficulty metadata in the default deployed controller. Supervised targets remain in their separate offline namespace. Fit input normalizers on training data only.",
        "",
        "## 5. Wrong-to-right prevalence",
        "",
        table(
            [
                "Split",
                "W→R event rate [95%], percentage scale",
                "W→R given useful loss gain [95%]",
            ],
            [
                [
                    SHORT[s],
                    interval(summary[s]["events"]["wrong_to_right"], True),
                    interval(
                        summary[s]["events"]["correction_given_useful_loss_gain"], True
                    ),
                ]
                for s in ("controller_val", *FINAL)
            ],
        ),
        "",
        "Rates average three adjacent transitions and three parents, retaining all states of each independent problem in bootstrap resampling. A positive loss gain is often not a correction; do not substitute it for a decision-value target.",
        "",
        "## 6. Right-to-wrong prevalence",
        "",
        table(
            ["Split", "R→W event rate [95%], percentage scale"],
            [
                [SHORT[s], interval(summary[s]["events"]["right_to_wrong"], True)]
                for s in ("controller_val", *FINAL)
            ],
        ),
        "",
        "Per-depth rates and the other two event classes are in `results/experiment_04a/summary.json`. Model correction and degradation separately or use their signed difference; avoid treating unchanged-correct states as correction events. The choice of 4B target must be developed and selected on development data.",
        "",
        "## 7. Existing predictor performance",
        "",
        table(
            [
                "Split",
                "Confidence AUROC [95%]",
                "Gain AUROC [95%]",
                "Joint gain AUROC",
                "Scalar gain AUROC",
            ],
            [
                [
                    SHORT[s],
                    interval(
                        summary[s]["confidence"]["existing_confidence"][
                            "auroc_interval"
                        ]
                    ),
                    interval(summary[s]["gain"]["existing_gain"]["auroc_interval"]),
                    f"{summary[s]['gain']['joint_gain']['auroc']:.3f}",
                    f"{summary[s]['gain']['scalar_logistic']['auroc']:.3f}",
                ]
                for s in ("controller_val", *FINAL)
            ],
        ),
        "",
        "Confidence and immediate-gain probes transfer much more weakly than their historical synthetic results. The report includes Brier/ECE/log loss, AP, selective accuracy, risk-coverage, per-depth ranking and paired scalar comparisons. Task/difficulty diagnostics are not a deployable hidden-information baseline.",
        "",
        "## 8. Recurrent depths and validation selections",
        "",
        table(
            [
                "Parent",
                "Validation-selected fixed depth",
                "Confidence threshold",
                "Gain threshold",
            ],
            [
                [
                    s,
                    selected[s]["policies"]["fixed_selected"]["depth"],
                    selected[s]["policies"]["confidence"]["threshold"],
                    selected[s]["policies"]["improvement"]["threshold"],
                ]
                for s in config["parent_seeds"]
            ],
        ),
        "",
        "The verified computational horizon remains four, and every depth is retained. Final trajectories show degradation at deeper iterations; that is a reported limitation, not a license to select a 4B architecture or depth using these tests. Any new budget or policy is selected on train/validation and assessed on a fresh prospective final set.",
        "",
        "## 9. Performance and calibration problems",
        "",
        "Frozen backbone competence is low on several tasks, structural transfer is weak, nominal static predicates do not truly deepen, temperature/recalibration may reduce error without informative ranking, and useful loss gain differs from argmax correction. Parents can have different correction/degradation dynamics. The head trained on historical structures and freshly fitted scalar controls are not equal-training hidden-versus-scalar interventions. Conditional four-option performance cannot imply natural-language reasoning competence.",
        "",
        "## 10. Reusable frozen checkpoints",
        "",
        "Backbone `Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca`, FP32, hidden width 1,024, option IDs 32–35, frozen prefix `Answer: (`. Per-parent refiner and one-pass checkpoints are `artifacts/checkpoints/{recurrence,feedforward}_seed{11,22,33}/best.safetensors`; original E3 confidence/improvement/joint probes are `artifacts/experiment_03/checkpoints/{confidence,improvement,joint}_seed{11,22,33}/best.safetensors`. Probe width 64, current input dimension 1,032, joint outputs 3. Exact hashes are in `experiments/experiment_04a/frozen_protocol.json` and the historical audit. All neural weights remain unchanged. Weights are excluded from Git; retraining creates a new checkpoint identity and must be declared.",
        "",
        "## 11. Reproduction commands",
        "",
        "From the repository root, using the public project dependencies:",
        "",
        "```sh\npython3.12 -m venv .venv\n.venv/bin/python -m pip install -r requirements-lock.txt\n.venv/bin/python -m pip install -e '.[apple,test]'\n.venv/bin/python -m pytest -q\nLATENT_LOOP_MLX_TESTS=1 .venv/bin/python -m pytest -q\n.venv/bin/python -m latent_loop.experiment04a.pipeline report\n```",
        "",
        "The report command is CPU-only, recomputes every public solver label/trajectory transition and all primary metrics, and regenerates tables/plots. It needs no weights, hidden tensors, model download or GPU. Run it against already published 4A predictions to reproduce analysis, without model selection.",
        "",
        "For local frozen-model extraction/evaluation (requires the exact historical checkpoints and cached pinned backbone), stage commands are:",
        "",
        "```sh\n.venv/bin/python -m latent_loop.experiment04a.pipeline audit\n.venv/bin/python -m latent_loop.experiment04a.pipeline data\n.venv/bin/python -m latent_loop.experiment04a.pipeline development\n.venv/bin/python -m latent_loop.experiment04a.pipeline freeze\n.venv/bin/python -m latent_loop.experiment04a.pipeline final\n.venv/bin/python -m latent_loop.experiment04a.pipeline timing\n.venv/bin/python -m latent_loop.experiment04a.pipeline report\n```",
        "",
        "Do not rerun development or data over the consumed lockbox: those stages deliberately refuse. Exact local caches resume feature extraction by batch and trajectory collection by parent/split; signatures reject changed datasets/model/prefix/checkpoints. SIGINT/SIGTERM stops at safe batch/stage boundaries. The original three-hour deadline is recorded in config. A new model reproduction requires an isolated checkout/output archive and a fresh `started_utc`/resource budget configured before data generation and protocol freezing; remove only that reproduction's derived manifests/selections/freeze/open/cache/output files, retaining source and the public historical exclusion list. Do not overwrite the recorded study. Recreated neural weights may differ; document the new model hashes and create a new evaluation protocol. Original model-training commands are in `docs/reproduction.md` and `reports/experiment_03/learning_when_to_think.md`.",
        "",
        "## 12. Locked final-test protocol",
        "",
        f"4A frozen `{frozen['frozen_utc']}`, opened `{opened['opened_utc']}`. Public `frozen_protocol.json` binds core source/config/data/selection/model hashes; `lockbox_opened.json` binds the opening to that protocol. These **4A final sets are consumed** and may only support descriptive audit/regression checking, not further architecture/threshold/checkpoint selection. Keep them out of 4B training/validation to preserve interpretability. Create a new independently seeded lockbox, reject alpha structures from all prior data, write and hash its task mix/strata/primary comparisons before scoring, and record its opening. Development on 4A train/validation is allowed. Natural-task validation may require its own independently reserved sets.",
        "",
        "## 13. Measurement methodology",
        "",
        timing["scope"],
        "",
        "Batch sizes 1/8, the same 24 task-balanced validation prompts, three parents, three warmups and five repeats. Scalar halting synchronizes per active row/depth; compact halting extracts one Boolean vector per depth and really removes inactive rows; masked halting executes all four loops and cannot claim saved compute. Diagnostic threshold 0.5 is frozen separately from selected accuracy policies. All outputs/depths match cached trajectories, and executed row counts are asserted. Load, peak process RSS/MLX allocation, components and raw samples are public. A future trajectory selector must pay for every state acquired and every controller operation; selecting an earlier answer after inspecting later states does not recover spent computation.",
        "",
        "## 14. Scientific risks and proceed decision",
        "",
        "4B can proceed as a new controlled experiment because complete causal observations and correction/degradation labels now exist. There is no demonstrated transferable confidence/gain advantage strong enough to guarantee success. Event imbalance, low recoverable competence, negative recurrent drift, metadata/format effects, calibration transfer, small parent-seed count and timing overhead are central risks. Compare to frozen, one-pass, fixed depths and fresh scalar controls with paired problem/seed uncertainty. Do not compare an oracle to a paid controller as if it were deployable. Preserve this negative evidence, preregister a small hypothesis set, and open the new lockbox only after selection is complete.",
        "",
    ]
    (REPORT / "experiment_04b_handoff.md").write_text("\n".join(lines))
