"""CPU-only recomputation, paired analyses and public research report."""

import base64
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from latent_loop.experiment03.data import EXP, OUT, utc, verify_preservation
from latent_loop.experiment03.science import (
    binary_metrics,
    paired_bootstrap,
    policy_depth,
    targets,
)
from latent_loop.training import atomic_json

LABELS = {
    "frozen": "Frozen model",
    "one_pass": "One-pass head",
    "confidence": "Confidence stop",
    "confidence_matched": "Confidence, matched on validation",
    "improvement": "Immediate-gain classifier stop",
    "joint": "Joint stop",
    "voc": "Expected-gain / VOC stop",
    "old_continuation": "Old any-future continuation",
    "oracle_next": "Oracle next gain",
    "oracle_best": "Oracle best loss depth",
}
SPLITS = [
    "historical_test",
    "historical_ood",
    "fresh_id",
    "template_shift",
    "depth_shift",
]


def average(values):
    present = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(present)) if present else None


def pick(logits, depth):
    return np.asarray(logits)[np.asarray(depth) - 1, np.arange(len(depth))]


def correctness(record, name):
    logits = (
        record["base_logits"]
        if name == "frozen"
        else record["one_pass_logits"]
        if name == "one_pass"
        else pick(record["logits"], record["depths"][name])
    )
    return np.asarray(logits).argmax(-1) == record["labels"]


def summarize(config):
    seeds = config["seeds"]
    metrics = {s: json.loads((OUT / f"seed{s}.json").read_text()) for s in seeds}
    predictions = {
        s: json.load(gzip.open(OUT / f"predictions_seed{s}.json.gz", "rt"))
        for s in seeds
    }
    summary = {
        "completed_utc": utc(),
        "seeds": seeds,
        "fits": len(list((OUT / "training").glob("*.json"))),
        "splits": {},
        "paired": {},
        "information": {},
        "checks": {},
        "scope": "Means across three parent/controller seeds on shared unique examples; depths and seeds do not multiply independent observations.",
    }
    for split in ["controller_val", *SPLITS]:
        item = {"policies": {}, "predictors": {}}
        sample = metrics[seeds[0]]["splits"][split]
        for name in sample["policies"]:
            rows = [metrics[s]["splits"][split]["policies"][name] for s in seeds]
            item["policies"][name] = {
                key: average([r[key] for r in rows])
                for key in [
                    "accuracy",
                    "nll",
                    "mean_loops",
                    "ece",
                    "wrong_to_right",
                    "right_to_wrong",
                ]
            }
            item["policies"][name]["accuracy_seed_sd"] = float(
                np.std([r["accuracy"] for r in rows], ddof=1)
            )
            item["policies"][name]["per_task"] = {
                task: {
                    key: average([r["per_task"][task][key] for r in rows])
                    for key in [
                        "accuracy",
                        "nll",
                        "mean_loops",
                        "wrong_to_right",
                        "right_to_wrong",
                    ]
                }
                for task in rows[0]["per_task"]
            }
        for kind in ["confidence", "classification", "regression"]:
            item["predictors"][kind] = {}
            for name in sample["predictors"][kind]:
                keys = (
                    [
                        "auroc",
                        "average_precision",
                        "prevalence",
                        "brier",
                        "ece",
                        "precision_at_0.5",
                        "recall_at_0.5",
                        "accuracy_at_50pct_coverage",
                        "aurc",
                    ]
                    if kind != "regression"
                    else ["mae", "rmse", "spearman"]
                )
                item["predictors"][kind][name] = {
                    key: average(
                        [
                            metrics[s]["splits"][split]["predictors"][kind][name][key]
                            for s in seeds
                        ]
                    )
                    for key in keys
                }
        summary["splits"][split] = item
        summary["paired"][split] = {}
        for a, b in [
            ("improvement", "confidence"),
            ("improvement", "confidence_matched"),
            ("improvement", "one_pass"),
            ("joint", "confidence"),
            ("voc", "confidence"),
            ("improvement", "fixed_2"),
            ("improvement", "fixed_3"),
        ]:
            differences = [
                correctness(predictions[s][split], a).astype(float)
                - correctness(predictions[s][split], b)
                for s in seeds
            ]
            summary["paired"][split][f"{a}_vs_{b}"] = paired_bootstrap(differences)
        for name in ["improvement", "joint", "voc"]:
            differences = []
            for s in seeds:
                r = predictions[s][split]
                mixture = metrics[s]["splits"][split]["matched_fixed_mixtures"][name]
                lo, hi, w = (
                    mixture["floor"],
                    mixture["ceiling"],
                    mixture["ceiling_probability"],
                )
                baseline = (1 - w) * correctness(r, f"fixed_{lo}") + w * correctness(
                    r, f"fixed_{hi}"
                )
                differences.append(correctness(r, name) - baseline)
            summary["paired"][split][f"{name}_vs_matched_fixed_mixture"] = (
                paired_bootstrap(differences)
            )
        information = {}
        for a, b in [
            ("improvement", "confidence_gain"),
            ("joint_gain", "confidence_gain"),
            ("control_plus_gain", "task_depth_control"),
        ]:
            differences = []
            for s in seeds:
                r = predictions[s][split]
                useful = np.asarray(r["gain"]) > config["delta"]
                scores = r["scores"]
                # Aggregate repeated depths within each unique example before resampling.
                differences.append(
                    (
                        (np.asarray(scores[b])[:3] - useful) ** 2
                        - (np.asarray(scores[a])[:3] - useful) ** 2
                    ).mean(0)
                )
            information[f"{a}_brier_reduction_vs_{b}"] = paired_bootstrap(differences)
        information["confidence_vs_predicted_gain_spearman"] = average(
            [
                float(
                    spearmanr(
                        np.asarray(predictions[s][split]["scores"]["confidence"])[
                            :3
                        ].ravel(),
                        np.asarray(predictions[s][split]["scores"]["improvement"])[
                            :3
                        ].ravel(),
                    ).statistic
                )
                for s in seeds
            ]
        )
        information["observed_gain_correlations"] = {
            key: average(
                [
                    metrics[s]["splits"][split]["predictors"]["observed_correlations"][
                        key
                    ]
                    for s in seeds
                ]
            )
            for key in sample["predictors"]["observed_correlations"]
        }
        summary["information"][split] = information
        # Independently rebuild losses, immediate labels, streaming stops and accuracy.
        for s in seeds:
            r = predictions[s][split]
            loss, _, gain, useful = targets(r["logits"], r["labels"], config["delta"])
            assert np.allclose(gain, r["gain"], atol=1e-8)
            for name, setting in metrics[s]["settings"].items():
                depth = policy_depth(
                    name,
                    {k: np.asarray(v) for k, v in r["scores"].items()},
                    setting,
                    np.asarray(r["logits"]),
                    np.asarray(r["labels"]),
                )
                assert np.array_equal(depth, r["depths"][name]), (s, split, name)
                assert (
                    abs(
                        correctness(r, name).mean()
                        - metrics[s]["splits"][split]["policies"][name]["accuracy"]
                    )
                    < 1e-12
                )
                row = metrics[s]["splits"][split]["policies"][name]
                chosen_loss = loss[depth - 1, np.arange(len(depth))]
                assert abs(float(depth.mean()) - row["mean_loops"]) < 1e-12
                assert abs(float(chosen_loss.mean()) - row["exact_nll"]) < 1e-8
                assert (
                    abs(
                        float(np.minimum(chosen_loss, -np.log(1e-12)).mean())
                        - row["nll"]
                    )
                    < 1e-8
                )
            assert (
                abs(
                    binary_metrics(np.asarray(r["scores"]["improvement"])[:3], useful)[
                        "auroc"
                    ]
                    - metrics[s]["splits"][split]["predictors"]["classification"][
                        "improvement"
                    ]["auroc"]
                )
                < 1e-12
            )
    val = summary["splits"]["controller_val"]
    summary["best_confidence_validation"] = max(
        val["predictors"]["confidence"],
        key=lambda n: val["predictors"]["confidence"][n]["auroc"],
    )
    gain_candidates = [
        "improvement",
        "joint_gain",
        "confidence_gain",
        "entropy",
        "margin",
        "change",
        "scalar_all",
        "old_continuation",
    ]
    summary["best_gain_validation"] = max(
        gain_candidates, key=lambda n: val["predictors"]["classification"][n]["auroc"]
    )
    deployment = ["confidence", "improvement", "joint", "voc"]
    summary["best_policy_validation"] = min(
        deployment,
        key=lambda n: (
            -val["policies"][n]["accuracy"],
            val["policies"][n]["mean_loops"],
            val["policies"][n]["nll"],
        ),
    )
    summary["checks"] = {
        "independent_loss_targets_stops_accuracy_and_gain_auc_recomputed": True,
        "historical_preservation": verify_preservation(),
        "opportunity_precedes_every_fit": True,
    }
    opportunity_time = json.loads(
        (OUT / "opportunity_before_training.json").read_text()
    )["completed_utc"]
    for path in (OUT / "training").glob("*.json"):
        assert json.loads(path.read_text())["started_utc"] > opportunity_time
    atomic_json(OUT / "summary.json", summary)
    return summary, metrics, predictions


def figures(summary, metrics, predictions, latency, config):
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    seeds, split = config["seeds"], "fresh_id"
    paths = []

    def save(fig, name, rect=None):
        fig.tight_layout(rect=rect)
        path = directory / (name + ".png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in [
        "confidence",
        "joint_confidence",
        "candidate_confidence",
        "correctness_scalar",
    ]:
        p = np.concatenate(
            [np.asarray(predictions[s][split]["scores"][name]).ravel() for s in seeds]
        )
        y = np.concatenate(
            [
                np.asarray(predictions[s][split]["logits"]).argmax(-1).ravel()
                == np.tile(predictions[s][split]["labels"], 4)
                for s in seeds
            ]
        )
        bins = binary_metrics(p, y)["calibration"]
        axes[0].plot(
            [r["confidence"] for r in bins],
            [r["accuracy"] for r in bins],
            "o-",
            label=name,
        )
        axes[1].bar(
            name, summary["splits"][split]["predictors"]["confidence"][name]["auroc"]
        )
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(
        xlabel="Predicted current correctness",
        ylabel="Observed correctness",
        title="Current correctness calibration (all four depths)",
    )
    axes[0].legend(fontsize=8)
    axes[1].set(ylabel="AUROC", ylim=(0.5, 0.9), title="Correctness discrimination")
    axes[1].tick_params(axis="x", rotation=30)
    save(fig, "confidence_correctness")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ["regression", "joint_regression", "confidence_gain_reg", "train_mean"]:
        reports = [
            metrics[s]["splits"][split]["predictors"]["regression"][name] for s in seeds
        ]
        x = np.mean(
            [[r["predicted"] for r in m["calibration"]] for m in reports], axis=0
        )
        y = np.mean(
            [[r["observed"] for r in m["calibration"]] for m in reports], axis=0
        )
        axes[0].plot(x, y, "o-", label=name)
    axes[0].plot([-0.4, 0.7], [-0.4, 0.7], "k--", lw=1)
    axes[0].set(
        xlabel="Predicted next-loop gain (nats)",
        ylabel="Observed gain (nats)",
        title="Signed gain calibration, depths 1–3",
    )
    axes[0].legend(fontsize=8)
    for name in ["improvement", "joint_gain", "confidence_gain"]:
        probs = np.concatenate(
            [
                np.asarray(predictions[s][split]["scores"][name])[:3].ravel()
                for s in seeds
            ]
        )
        y = np.concatenate(
            [
                (np.asarray(predictions[s][split]["gain"]) > config["delta"]).ravel()
                for s in seeds
            ]
        )
        bins = binary_metrics(probs, y)["calibration"]
        axes[1].plot(
            [r["confidence"] for r in bins],
            [r["accuracy"] for r in bins],
            "o-",
            label=name,
        )
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set(
        xlabel="Predicted P(gain > 0.01)",
        ylabel="Observed useful-gain fraction",
        title="Gain classification calibration",
    )
    axes[1].legend(fontsize=8)
    save(fig, "predicted_observed_gain")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in range(3):
        c = np.concatenate(
            [
                np.asarray(predictions[s][split]["scores"]["confidence"])[k]
                for s in seeds
            ]
        )
        gain = np.concatenate(
            [np.asarray(predictions[s][split]["gain"])[k] for s in seeds]
        )
        groups = np.array_split(np.argsort(c), 10)
        axes[0].plot(
            [c[g].mean() for g in groups],
            [gain[g].mean() for g in groups],
            "o-",
            label=f"{k + 1}→{k + 2}",
        )
        axes[1].plot(
            [c[g].mean() for g in groups],
            [(gain[g] > config["delta"]).mean() for g in groups],
            "o-",
            label=f"{k + 1}→{k + 2}",
        )
    axes[0].axhline(0, c="k", lw=1)
    axes[0].set(
        xlabel="Current correctness confidence",
        ylabel="Observed gain (nats)",
        title="Confidence versus signed future gain",
    )
    axes[1].set(
        xlabel="Current correctness confidence",
        ylabel="Meaningful improvement probability",
        title="Confidence is not immediate gain",
    )
    for a in axes:
        a.legend()
    save(fig, "confidence_vs_gain")

    variants = [
        "fixed_1",
        "fixed_2",
        "fixed_3",
        "fixed_4",
        "one_pass",
        "confidence",
        "confidence_matched",
        "improvement",
        "joint",
        "voc",
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for split in ["fresh_id", "template_shift", "depth_shift"]:
        policies = summary["splits"][split]["policies"]
        ax = axes[["fresh_id", "template_shift", "depth_shift"].index(split)]
        fixed = [policies[f"fixed_{d}"]["accuracy"] * 100 for d in range(1, 5)]
        ax.plot(range(1, 5), fixed, "k--o", label="fixed depths")
        for name in variants[4:]:
            m = policies[name]
            ax.scatter(
                m["mean_loops"], m["accuracy"] * 100, label=LABELS.get(name, name), s=45
            )
        ax.set(xlabel="Mean paid recurrence loops", ylabel="Accuracy (%)", title=split)
    axes[-1].legend(fontsize=7, loc="best")
    save(fig, "accuracy_loops")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Both axes use accuracy/NLL on the actual timed validation subset, not all test items.
    for index, name in enumerate(latency["summary"]):
        rows = [r for r in latency["records"] if r["variant"] == name]
        outcomes, nlls = [], []
        for r in rows:
            raw = predictions[r["seed"]]["controller_val"]
            i = raw["ids"].index(r["id"])
            logits = (
                np.asarray(raw["base_logits"])[i]
                if name == "frozen"
                else np.asarray(raw["one_pass_logits"])[i]
                if name == "one_pass"
                else np.asarray(raw["logits"])[r["depth"][0] - 1, i]
            )
            y = raw["labels"][i]
            outcomes.append(logits.argmax() == y)
            nlls.append(float(np.logaddexp.reduce(logits) - logits[y]))
        x = latency["summary"][name]["full_mean_of_item_medians_ms"]
        color = plt.get_cmap("tab20")(index)
        axes[0].scatter(
            x,
            np.mean(outcomes) * 100,
            color=color,
            label=f"{index + 1}. {LABELS.get(name, name)}",
        )
        axes[1].scatter(x, np.mean(nlls), color=color)
        offsets = {"joint": (-12, 5), "improvement": (5, -10)}
        for ax in axes:
            ax.annotate(
                str(index + 1),
                (x, np.mean(outcomes) * 100 if ax == axes[0] else np.mean(nlls)),
                fontsize=7,
                xytext=offsets.get(name, (3, 4)),
                textcoords="offset points",
            )
    axes[0].set(
        xlabel="Full measured latency (ms)",
        ylabel="Timed-subset accuracy (%)",
        title="Same 24 validation prompts × 3 seeds",
    )
    axes[1].set(
        xlabel="Full measured latency (ms)",
        ylabel="Timed-subset exact NLL",
        title="Loss versus measured end-to-end latency",
    )
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        fontsize=8,
    )
    save(fig, "accuracy_full_latency", rect=(0, 0.22, 1, 1))

    fig, ax = plt.subplots(figsize=(11, 4))
    names = [
        "fixed_2",
        "fixed_3",
        "fixed_4",
        "confidence_matched",
        "improvement",
        "joint",
        "voc",
    ]
    policies = summary["splits"]["fresh_id"]["policies"]
    positions = np.arange(len(names))
    ax.bar(
        positions - 0.18,
        [policies[n]["wrong_to_right"] for n in names],
        0.36,
        label="Wrong→right",
    )
    ax.bar(
        positions + 0.18,
        [policies[n]["right_to_wrong"] for n in names],
        0.36,
        label="Right→wrong",
    )
    ax.set_xticks(positions, names, rotation=20)
    ax.set(
        ylabel="Mean count per 600 items, versus depth one",
        title="Correction and degradation are separate from loss gain",
    )
    ax.legend()
    save(fig, "correction_degradation")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for k in range(3):
        gain = np.concatenate(
            [np.asarray(predictions[s]["fresh_id"]["gain"])[k] for s in seeds]
        )
        axes[0].hist(
            gain,
            bins=45,
            density=True,
            histtype="step",
            lw=1.5,
            label=f"{k + 1}→{k + 2}",
        )
        axes[1].bar(k - 0.15, (gain > 0.01).mean(), 0.3)
        axes[1].bar(k + 0.15, (gain < -0.01).mean(), 0.3)
    axes[0].axvline(0, c="k", lw=1)
    axes[0].set(
        xlabel="Signed loss gain (nats)",
        ylabel="Density",
        title="Beneficial and harmful loss distribution",
    )
    axes[0].legend()
    axes[1].set_xticks(range(3), ["1→2", "2→3", "3→4"])
    axes[1].set(ylabel="Fraction", title="Gain > 0.01 (blue); harm < −0.01 (orange)")
    save(fig, "gain_harm_distribution")

    fig, ax = plt.subplots(figsize=(10, 4))
    for name in [
        "confidence",
        "confidence_matched",
        "improvement",
        "joint",
        "voc",
        "oracle_best",
    ]:
        counts = (
            np.mean(
                [
                    metrics[s]["splits"]["fresh_id"]["policies"][name][
                        "stopping_counts"
                    ]
                    for s in seeds
                ],
                axis=0,
            )
            / 600
        )
        ax.plot(range(1, 5), counts, "o-", label=LABELS.get(name, name))
    ax.set(
        xlabel="Paid stopping depth",
        ylabel="Fraction",
        xticks=[1, 2, 3, 4],
        title="Stopping depths; oracle is label-informed",
    )
    ax.legend(fontsize=8)
    save(fig, "stopping_depth")

    fig, ax = plt.subplots(figsize=(9, 4))
    for name in [
        "confidence",
        "joint_confidence",
        "candidate_confidence",
        "correctness_scalar",
        "raw_probability",
    ]:
        curves = [
            metrics[s]["splits"]["fresh_id"]["predictors"]["confidence"][name][
                "risk_coverage"
            ]
            for s in seeds
        ]
        risk = np.mean([[r["risk"] for r in curve] for curve in curves], axis=0)
        ax.plot([r["coverage"] for r in curves[0]], risk, label=name)
    ax.set(
        xlabel="Coverage",
        ylabel="Error rate among retained decisions",
        title="Risk–coverage, all four current decisions",
    )
    ax.legend(fontsize=8)
    save(fig, "risk_coverage")
    return paths


def pct(x):
    return f"{100 * x:.2f}%"


def behavior_supplement(config, predictions):
    from latent_loop.experiment03.run import behavior

    examples = json.load(gzip.open(OUT / "examples.json.gz", "rt"))
    reports, cases = {}, []
    for split in SPLITS:
        reports[split] = {}
        for seed in config["seeds"]:
            r = predictions[seed][split]
            loss, correct, gain, useful = targets(
                r["logits"], r["labels"], config["delta"]
            )
            t = {
                "logits": np.asarray(r["logits"]),
                "loss": loss,
                "correct": correct,
                "gain": gain,
                "useful": useful,
            }
            categories, selected = behavior(
                t,
                {k: np.asarray(v) for k, v in r["scores"].items()},
                examples[split],
                0.5,
            )
            reports[split][str(seed)] = categories
            cases.extend(
                [
                    {**row, "seed": seed, "split": split, "threshold": 0.5}
                    for row in selected
                ]
            )
    atomic_json(
        OUT / "behavior_diagnostic.json",
        {
            "threshold": 0.5,
            "scope": "Fixed probability 0.5 for post-evaluation behavior analysis only; does not alter any deployment threshold or training target.",
            "categories": reports,
            "cases": cases,
        },
    )
    return reports


def interval(row):
    lo, hi = row["hierarchical_seed_item_bootstrap95"]
    return f"{row['mean_delta'] * 100:+.2f} pp [{lo * 100:+.2f}, {hi * 100:+.2f}]"


def report(config):
    import markdown

    summary, metrics, predictions = summarize(config)
    diagnostic_behavior = behavior_supplement(config, predictions)
    latency = json.loads((OUT / "latency.json").read_text())
    summary["paired_latency_ms"] = {}
    for a, b in [
        ("improvement", "one_pass"),
        ("improvement", "fixed_4"),
        ("improvement", "confidence_matched"),
        ("voc", "fixed_4"),
    ]:
        arrays = []
        for seed in config["seeds"]:
            ar = {
                r["id"]: float(np.median(r["full_ms"]))
                for r in latency["records"]
                if r["seed"] == seed and r["variant"] == a
            }
            br = {
                r["id"]: float(np.median(r["full_ms"]))
                for r in latency["records"]
                if r["seed"] == seed and r["variant"] == b
            }
            assert ar.keys() == br.keys()
            arrays.append([ar[k] - br[k] for k in ar])
        summary["paired_latency_ms"][f"{a}_vs_{b}"] = paired_bootstrap(arrays)
    atomic_json(OUT / "summary.json", summary)
    paths = figures(summary, metrics, predictions, latency, config)
    id_result = summary["splits"]["fresh_id"]
    confidence = summary["best_confidence_validation"]
    gain = summary["best_gain_validation"]
    policy = summary["best_policy_validation"]
    c, g, p = (
        id_result["predictors"]["confidence"][confidence],
        id_result["predictors"]["classification"][gain],
        id_result["policies"][policy],
    )
    timing = latency["summary"]
    audit = json.loads((EXP / "audit.json").read_text())
    lines = [
        "# Experiment 3: Learning When to Think",
        "",
        "## Findings",
        "",
        f"A small frozen-state probe distinguishes current correctness from immediate next-loop loss improvement in this synthetic pilot. On 600 fresh original-template examples, the validation-selected confidence estimator ({confidence}) reaches AUROC **{c['auroc']:.3f}**; the validation-selected gain predictor ({gain}) reaches AUROC **{g['auroc']:.3f}**, AP **{g['average_precision']:.3f}** against useful-gain prevalence **{g['prevalence']:.3f}**. Confidence alone as a supervised gain predictor reaches AUROC **{id_result['predictors']['classification']['confidence_gain']['auroc']:.3f}**. This is evidence of a distinguishable predictive signal, not evidence of internal reasoning or broad transfer.",
        "",
        f"The stopping family selected by validation accuracy is **{LABELS[policy]}**. It scores **{pct(p['accuracy'])}** at **{p['mean_loops']:.3f} loops**, versus frozen **{pct(id_result['policies']['frozen']['accuracy'])}** and the existing one-pass head **{pct(id_result['policies']['one_pass']['accuracy'])}**. Its accuracy difference from confidence stopping is {interval(summary['paired']['fresh_id'][policy + '_vs_confidence'])}. Accuracy gains remain uncertain with only three seeds and 600 shared items. The one-pass head remains stronger on fresh-ID raw accuracy.",
        "",
        f"Full measured latency on a prespecified balanced 24-item controller-validation subset is **{timing[policy]['full_mean_of_item_medians_ms']:.2f} ms** for this controller and **{timing['one_pass']['full_mean_of_item_medians_ms']:.2f} ms** for the one-pass head. The complete timed subset, costs, and raw repetitions are recorded. These timings include tokenization, backbone, live stopping, and Metal synchronization. They do not establish a deployment speedup on another device or prompt distribution.",
        "",
        "## Audit and preserved baseline",
        "",
        f"Recomputed {audit['recomputed_rows']} historical accuracy/NLL rows from saved logits. All passed. Recovered depths 1–4 from the three original refiner checkpoints; all historical test and OOD logits matched archived trajectories exactly. Verified dataset hashes, cache signatures, shapes, FP32 dtype, and distinct parent checkpoint hashes. Historical tracked files remain intact except an added README link; original reports and results are untouched.",
        "",
        "The source audit found no invalidating final-test leakage, causal-cache, stopping-budget, or probability-normalization error. The earlier continuation target was **any later permitted loss improvement**, trained on refiner-training examples. This experiment uses **immediate** loss gain and fresh out-of-refiner-training examples. Historical candidate confidence was a depth-zero candidate classifier; the new primary confidence head predicts current recurrent argmax correctness. The adapted historical architecture is a separate baseline, fitted to current correctness with conditional statistics and depth.",
        "",
        "Historical NLL clips probabilities at 1e-12. New target loss uses stable, unclipped logsumexp; both historical-convention NLL and exact NLL are saved for policies. The answer prefix `Answer: (` and public frozen FP32 Qwen3-0.6B revision are retained. Output probabilities are conditional on four labels; this is not language-model perplexity or free-generation accuracy.",
        "",
        "## Frozen design and supervision",
        "",
        "The pre-fit opportunity analysis completed before every new fit (timestamps verified). Independent fresh splits have 1,200 controller-training, 300 controller-validation, and 600 items each for fresh ID, rewritten templates, and harder depth. Independent reference solvers validate every answer. Exact instance IDs and prompts are disjoint across all historical and new splits. Templates/algorithm families and alpha-renamed structures remain shared; unseen structural generalization is not claimed. Template shift changes wording while retaining original distributions. Depth shift uses existing OOD depths 5–8 and numeric ranges; three easy task families remain depth one.",
        "",
        "At depth k, each probe observes RMS-normalized current state (1,024 values), current probabilities (four), option entropy, margin, k/4, and relative change from the previous completed state: 1,032 features. The previous state at depth one is the backbone state. No live API accepts labels, future hidden states, or future loss. Backbone and three existing refiners remain frozen. Hidden vectors and weights stay in ignored artifacts.",
        "",
        "Targets are current argmax correctness; gain classification `L_k − L_(k+1) > 0.01 nat`; and signed regression `E[L_k − L_(k+1)]`. Confidence trains at depths 1–4. Gain losses train at depths 1–3 with terminal masking. The joint head has separate confidence/classification/regression outputs with summed BCE/BCE/MSE, using a training-only gain standard deviation to scale regression. MSE targets retain both signs and all magnitudes; no positive-example selection. Width 64, AdamW 0.001, decay 0.01, clipping 1, batch 64, 30 epochs; each checkpoint chosen by its validation loss. Three parent/controller seeds [11,22,33]; 15 new neural fits; no new backbone/refiner fits. Epoch-level resumable optimizer/last/best states and immutable data/parent/config signatures are saved.",
        "",
        "Scalar logistic and ridge controls fit on controller training only: entropy, margin, completed change, confidence alone, all scalar observables with confidence, and a diagnostic task/depth/difficulty control with and without the dedicated gain score. Task/difficulty metadata is used only in diagnostic controls. They are never deployed as controller inputs. Fitting these controls to training predictions is not cross-fitting; held-out metrics remain the basis of claims.",
        "",
        "## Opportunity before training",
        "",
        "| Transition | Meaningful gain | Harmful loss | Changed answer | Wrong→right | Right→wrong | Mean gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    opportunities = json.loads((OUT / "opportunity_before_training.json").read_text())[
        "results"
    ]
    for k in range(3):
        rows = [
            opportunities[str(s)]["controller_val"]["transitions"][k]
            for s in config["seeds"]
        ]
        mean = lambda key: float(np.mean([r[key] for r in rows]))
        lines.append(
            f"| {k + 1}→{k + 2} | {pct(mean('meaningful_gain'))} | {pct(mean('harmful_loss'))} | {mean('changed'):.1f}/300 | {mean('wrong_to_right'):.1f}/300 | {mean('right_to_wrong'):.1f}/300 | {mean('mean_gain'):+.3f} nat |"
        )
    lines += [
        "",
        "Counts are seed means on 300 shared validation items. Gains are not extremely rare, but most loss reductions do not correct an answer. All transition quantiles, positive/meaningful/harmful frequencies, confidence increases on wrong outcomes, and historical opportunities are in `opportunity_before_training.json`; final-split opportunities are reported separately after fitting.",
        "",
        "## Predictor results",
        "",
        "Models/families below are selected on controller validation AUROC; checkpoints use validation objective. Tables show final results without using them to reselect models.",
        "",
        "| Split | Confidence AUROC | Gain AUROC | Gain AP | Prevalence | Confidence-only gain AUROC | Scalar gain AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        pred = summary["splits"][split]["predictors"]
        m = pred["classification"][gain]
        lines.append(
            f"| {split} | {pred['confidence'][confidence]['auroc']:.3f} | {m['auroc']:.3f} | {m['average_precision']:.3f} | {m['prevalence']:.3f} | {pred['classification']['confidence_gain']['auroc']:.3f} | {pred['classification']['scalar_all']['auroc']:.3f} |"
        )
    lines += [
        "",
        "| Fresh-ID confidence estimate | AUROC | Brier | ECE | Accuracy at 50% coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in id_result["predictors"]["confidence"].items():
        lines.append(
            f"| {name} | {m['auroc']:.3f} | {m['brier']:.3f} | {m['ece']:.3f} | {pct(m['accuracy_at_50pct_coverage'])} |"
        )
    lines += [
        "",
        "| Fresh-ID gain classifier | AUROC | AP | Brier | Precision at 0.5 | Recall at 0.5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in id_result["predictors"]["classification"].items():
        precision = (
            f"{m['precision_at_0.5']:.3f}" if m["precision_at_0.5"] is not None else "—"
        )
        recall = f"{m['recall_at_0.5']:.3f}" if m["recall_at_0.5"] is not None else "—"
        lines.append(
            f"| {name} | {m['auroc']:.3f} | {m['average_precision']:.3f} | {m['brier']:.3f} | {precision} | {recall} |"
        )
    lines += [
        "",
        "AP is a tied-score-aware average precision, not trapezoidal PR-AUC. A constant predictor has AP equal to prevalence and AUROC 0.5. Precision/recall use fixed probability 0.5; deployable thresholds are selected separately. Confidence metrics pool four current decisions; gain metrics pool three transitions. These repeated observations are correlated.",
        "",
        "| Fresh-ID signed-gain regressor | MAE (nat) | RMSE (nat) | Spearman |",
        "|---|---:|---:|---:|",
    ]
    for name, m in id_result["predictors"]["regression"].items():
        rho = f"{m['spearman']:.3f}" if m["spearman"] is not None else "—"
        lines.append(f"| {name} | {m['mae']:.3f} | {m['rmse']:.3f} | {rho} |")
    lines += [
        "",
        "Regression calibration curves, per-depth prevalence/AUROC/AP, per-task gain metrics, and confidence/entropy/margin/change versus observed-gain correlations are saved in each seed JSON. Poor signed-regression calibration limits the VOC interpretation even when gain classification ranks opportunities well.",
        "",
        "## Information beyond confidence and task difficulty",
        "",
        "Brier reductions below are paired across the same items and depths; positive means the augmented predictor is better. Hierarchical intervals resample seeds and unique items, averaging repeated depths within items. Intervals are exploratory, with three seeds and no multiplicity adjustment.",
        "",
        "| Split | Joint gain vs confidence-only gain: Brier reduction | Task/depth control + dedicated gain vs control |",
        "|---|---:|---:|",
    ]
    for split in SPLITS:
        info = summary["information"][split]
        format_brier = lambda r: (
            f"{r['mean_delta']:+.4f} [{r['hierarchical_seed_item_bootstrap95'][0]:+.4f}, {r['hierarchical_seed_item_bootstrap95'][1]:+.4f}]"
        )
        lines.append(
            f"| {split} | {format_brier(info['joint_gain_brier_reduction_vs_confidence_gain'])} | {format_brier(info['control_plus_gain_brier_reduction_vs_task_depth_control'])} |"
        )
    lines += [
        "",
        f"On fresh ID, current confidence versus dedicated gain probability has Spearman **{summary['information']['fresh_id']['confidence_vs_predicted_gain_spearman']:.3f}**. The two are related but do not make identical predictions. The task/depth/difficulty control tests whether gain predictions add information beyond broad easy/hard task detection. Its held-out augmentation gains on original templates weaken or reverse under template shift; this limits claims about general information sufficiency. This is an observational probe comparison, not a proof of which information is encoded or used causally.",
        "",
        "## Stopping, matched computation and utility",
        "",
        "Each fixed policy pays depths 1–4. Confidence stops when current correctness probability reaches its selected threshold. The gain classifier continues only above its selected threshold. Joint continuation requires its gain score above a threshold and its current confidence below a threshold. These thresholds maximize validation accuracy, then minimize loops, then NLL. Confidence stops at depth one under that rule for all seeds on this validation set. The additional matched-compute confidence threshold is chosen solely to approach the gain policy's validation mean loops; it is a fairness diagnostic, not a reselected primary policy.",
        "",
        "VOC continues when predicted signed next-loop gain exceeds cost. The configurable nat/loop cost grid is selected by validation `NLL + 0.01 * loops`; all grid points are retained. The measured auxiliary increment converts this constant cost to lambda × milliseconds. This conversion does not estimate instance-dependent incremental latency. The old continuation baseline retains its different any-future supervision but uses a newly validation-selected threshold. Oracle-next knows true immediate gain; oracle-best knows all future losses and chooses a depth while paying all intervening loops. Both are diagnostic and excluded from deployment selection.",
        "",
        "| Policy | Historical test | Fresh ID | Template shift | Depth shift | Fresh loops | Fresh NLL |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in id_result["policies"]:
        m = id_result["policies"][name]
        lines.append(
            f"| {LABELS.get(name, name)} | {pct(summary['splits']['historical_test']['policies'][name]['accuracy'])} | {pct(m['accuracy'])} | {pct(summary['splits']['template_shift']['policies'][name]['accuracy'])} | {pct(summary['splits']['depth_shift']['policies'][name]['accuracy'])} | {m['mean_loops']:.3f} | {m['nll']:.3f} |"
        )
    lines += [
        "",
        "| Fresh-ID paired accuracy comparison | Mean delta and hierarchical 95% interval |",
        "|---|---:|",
    ]
    for name, m in summary["paired"]["fresh_id"].items():
        lines.append(f"| {name} | {interval(m)} |")
    lines += [
        "",
        "The random fixed-depth comparator chooses floor/ceiling depth independently of labels with probability matching each controller's observed mean loops. Its reported accuracy/NLL are exact expectations of that mixture; it is not an executed policy and has no measured latency. Matched loops are an analytical comparison and do not match synchronization/controller overhead. Final-set mean loops can differ for confidence and gain despite matching on validation. All pairs and achieved loops are recorded.",
        "",
        "## Measured cost",
        "",
        latency["scope"],
        "",
        "| Variant | Full mean of item medians (ms) | p10–p90 (ms) | Auxiliary-only (ms) | Timed mean loops |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, m in timing.items():
        lines.append(
            f"| {LABELS.get(name, name)} | {m['full_mean_of_item_medians_ms']:.3f} | {m['full_p10_ms']:.3f}–{m['full_p90_ms']:.3f} | {m['aux_mean_of_item_medians_ms']:.3f} | {m['mean_loops_timed_subset']:.3f} |"
        )
    head_only = {
        name: average(
            [r["median_ms"] for r in latency["head_only"] if r["head"] == name]
        )
        for name in [
            "confidence",
            "improvement",
            "regression",
            "joint",
            "candidate_confidence",
        ]
    }
    lines += [
        "",
        f"Controller-only median times, averaged across seeds: `{json.dumps(head_only, sort_keys=True)}` ms. These exclude feature construction and refiner/backbone. Full timing interleaves policy order, uses three warmup calls per variant/seed and {config['latency_repeats']} synchronized repetitions on each of {config['latency_examples']} shared validation prompts per seed. Full and cached-live outputs/stopping depths agree within FP32 tolerance; streaming features agree with cached construction. Prompt length and each raw time are retained. p10–p90 describes spread across item medians, not a confidence interval. GPU time varies with load and thermals; small sub-millisecond full-path differences should be interpreted cautiously. There are no FLOP-based speedup claims.",
        "",
        "The latency chart evaluates accuracy/NLL on the exact timed subset. The broader accuracy-versus-loops figure uses all 600 final items. This prevents mixing test accuracy and timing from a different prompt population into a spurious measured frontier.",
        "",
        f"Relative to the one-pass head, gain stopping adds {summary['paired_latency_ms']['improvement_vs_one_pass']['mean_delta']:.3f} ms on this timed subset (paired hierarchical interval {summary['paired_latency_ms']['improvement_vs_one_pass']['hierarchical_seed_item_bootstrap95'][0]:.3f} to {summary['paired_latency_ms']['improvement_vs_one_pass']['hierarchical_seed_item_bootstrap95'][1]:.3f} ms). It also adds {summary['paired_latency_ms']['improvement_vs_fixed_4']['mean_delta']:.3f} ms versus fixed depth four despite using fewer loops. These within-run intervals do not capture between-run thermal or load variation. Controller and host synchronization overhead exceed the measured saved refiner work in this implementation.",
        "",
        "## Behavior, tasks and representative cases",
        "",
        "Categories use the validation-selected confidence threshold; final outcomes never define the threshold. A category can be empty, especially when confidence stopping selects a threshold of zero. That degenerate operational partition is reported honestly; a separate fixed 0.5-confidence diagnostic is included in the behavior supplement for interpretable category analysis.",
        "",
        "| Fresh-ID task | Frozen | One-pass | Gain policy | Confidence matched | Gain corrections/degradations |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task in id_result["policies"]["frozen"]["per_task"]:
        m = id_result["policies"]["improvement"]["per_task"][task]
        lines.append(
            f"| {task} | {pct(id_result['policies']['frozen']['per_task'][task]['accuracy'])} | {pct(id_result['policies']['one_pass']['per_task'][task]['accuracy'])} | {pct(m['accuracy'])} | {pct(id_result['policies']['confidence_matched']['per_task'][task]['accuracy'])} | {m['wrong_to_right']:.1f} / {m['right_to_wrong']:.1f} |"
        )
    lines += [
        "",
        "Cases retain prompt, correct label, depth predictions/probabilities/losses, predicted/observed immediate gain, and paid oracle depth. They include wrong/confident corrections, uncertain examples never correct within four loops, and correct answers damaged by further loops. Case selection uses outcomes only after final evaluation and cannot validate internal reasoning. The benchmark's near-chance arithmetic/state/pointer performance remains a major limit.",
        "",
        "The fixed-0.5 diagnostic below aggregates categories across seeds and depths 1–3, weighting by category counts. These repeated states are not independent sample sizes. Saved oracle depth distributions include the global best loss depth and the best remaining depth from the current state.",
        "",
        "| Fresh-ID category | Repeated states | P(useful gain) | P(loss harm) | P(correction) | P(degradation) | Predicted gain | Observed gain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category in [
        "correct_confident",
        "wrong_uncertain",
        "wrong_confident",
        "correct_uncertain",
    ]:
        rows = [
            diagnostic_behavior["fresh_id"][str(s)][category] for s in config["seeds"]
        ]
        n = sum(r["n"] for r in rows)
        weighted = lambda key: sum(r.get(key, 0) * r["n"] for r in rows) / max(1, n)
        lines.append(
            f"| {category} | {n} | {pct(weighted('meaningful_improvement_probability'))} | {pct(weighted('harmful_loss_probability'))} | {pct(weighted('correction_probability'))} | {pct(weighted('degradation_probability'))} | {weighted('mean_predicted_gain'):+.3f} | {weighted('mean_actual_gain'):+.3f} |"
        )
    lines += [
        "",
        "## Direct answers to the research questions",
        "",
        f"1. **Is current confidence predictable?** Yes in this pilot: fresh-ID AUROC {c['auroc']:.3f}; calibration, Brier and risk–coverage are reported separately.",
        f"2. **Is future immediate improvement predictable?** Yes on original templates: {gain} AUROC {g['auroc']:.3f}, AP {g['average_precision']:.3f} versus prevalence {g['prevalence']:.3f}; transfer weakens on rewritten templates.",
        "3. **Are the signals distinct?** They are related but gain prediction outperforms confidence-only gain prediction on ID. The diagnostic controls and their paired Brier intervals support a limited incremental signal; template shift prevents a broad claim.",
        f"4. **Does improvement stopping beat confidence stopping?** Fresh-ID accuracy changes by {interval(summary['paired']['fresh_id']['improvement_vs_confidence'])}; the interval includes zero. Confidence selects one loop on validation, so this unrestricted comparison also spends more compute.",
        f"5. **Does it improve accuracy at matched compute?** Against confidence matched on validation: {interval(summary['paired']['fresh_id']['improvement_vs_confidence_matched'])}; against an exact matched-loops fixed-depth mixture: {interval(summary['paired']['fresh_id']['improvement_vs_matched_fixed_mixture'])}. Neither establishes an accuracy advantage. The gain and matched-confidence policies achieve {id_result['policies']['improvement']['mean_loops']:.3f} and {id_result['policies']['confidence_matched']['mean_loops']:.3f} final mean loops, respectively.",
        f"6. **Does it reduce harmful loops?** On fresh ID, gain stopping damages {id_result['policies']['improvement']['right_to_wrong']:.1f} decisions versus {id_result['policies']['fixed_4']['right_to_wrong']:.1f} at depth four and {id_result['policies']['confidence_matched']['right_to_wrong']:.1f} for matched confidence, per 600 items averaged across seeds. It also corrects {id_result['policies']['improvement']['wrong_to_right']:.1f} decisions. This is descriptive harm reduction with remaining regressions, not a guaranteed safeguard.",
        "7. **Which tasks benefit?** Gain stopping reaches 36.67% arithmetic and 87.00% routing accuracy, but only 25.33% pointer and 18.33% state accuracy on fresh ID. The one-pass head reaches 31.67% pointer and 29.33% state. Within-task gain metrics remain in the public files; aggregate gains do not establish reliable multistep reasoning.",
        "8. **Where does it fail?** Template transfer weakens, signed gain calibration is imperfect, and the one-pass head remains stronger on fresh-ID accuracy. Incorrect low-confidence examples often remain uncorrected within four loops.",
        "9. **Is this genuine gain prediction or easy/hard detection?** Task/depth/difficulty controls and their augmented variants test this; ID incremental performance persists descriptively, while template transfer degrades. This rules out neither shortcuts nor unmeasured confounders.",
        "10. **Is VOC useful?** It provides an explicit loss/compute utility knob and a recorded cost sweep, but signed prediction errors, host synchronization, and backbone-dominated cost prevent a demonstrated general accuracy/latency advantage.",
        "",
        "## Limitations and next experiment",
        "",
        "This is an exploratory frozen-final-vector synthetic pilot, not an official RecurTrace reproduction. Only three seeds, 300 validation items, related templates, reused algorithm families, several policy comparisons, and a small timing sample limit inference. Confidence and gain AUROC comparisons reflect correlated trajectories; bootstrap resamples unique items. The maximum deployed depth is four. Earlier-layer probes and new eight-loop experiments are deferred; old eight-loop harm remains a historical result. No large architectural sweep is justified by these results.",
        "",
        "The next experiment should preregister a larger controller-validation set and richer independently generated wording/task structures, use out-of-fold scalar stacking, and evaluate a calibration-focused joint predictor with batched GPU halting to reduce host synchronization. Keep the frozen refiner and one-pass control initially. Establish transfer and full measured cost before adding recurrence complexity.",
        "",
        "## Reproduction and publication",
        "",
        "Run from the repository root with the existing editable environment:",
        "",
        "```sh",
        "python -m latent_loop.experiment03.run audit",
        "python -m latent_loop.experiment03.run prepare",
        "python -m latent_loop.experiment03.run run",
        "python -m latent_loop.experiment03.run timing",
        "python -m latent_loop.experiment03.run report",
        "```",
        "",
        "`report` is CPU-only and recomputes checks/figures from public saved predictions. The numerical pipeline reuses the original ignored parent checkpoints/features. Those are not vendored: obtain them by the existing primary reproduction workflow before Experiment 3. The attached config records the original four-hour deadline; for a later independent reproduction copy the config and set `started_utc` to the new start, preserving scientific hyperparameters. Signatures deliberately prevent silently mixing changed configs with old checkpoints; use a separate checkout/cache directory for an independent run. Completed fits reuse best weights; unfinished fits resume from the last completed epoch and optimizer state. The reported fit count excludes scalar statistical fits, historical fits, and verification tests.",
        "",
        f"Peak recorded numerical MLX allocation is **{max(latency['peak_mlx_bytes'], json.loads((OUT / 'extraction.json').read_text())['peak_mlx_bytes'], *[json.loads(p.read_text())['peak_mlx_bytes'] for p in (OUT / 'training').glob('*.json')]) / 1024**3:.2f} GiB**, below the 48 GiB allocator budget. OS process RSS is separately saved. No remote workstation or unrelated process was touched. Public synthetic data, model source/license references, configuration, hashes, metrics, plots, and reports are included; hidden tensors, weights, environments, and personal paths are excluded. See the publication audit for the tracked-file scan and size check. Work is committed locally on `research/value-of-computation`; no remote publication is performed.",
        "",
        "## Recorded figures",
        "",
    ]
    for path in paths:
        lines += [f"![{path.stem.replace('_', ' ')}](../../{path.as_posix()})", ""]
    directory = Path("reports/experiment_03")
    directory.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    (directory / "learning_when_to_think.md").write_text(content)
    rendered = markdown.markdown(content, extensions=["tables", "fenced_code"])
    for path in paths:
        rendered = rendered.replace(
            "../../" + path.as_posix(),
            "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(),
        )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Learning When to Think</title><style>body{max-width:1120px;margin:40px auto;padding:0 22px;font:16px/1.6 system-ui;color:#162333}h1,h2{line-height:1.25}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}th,td{border-bottom:1px solid #d9e1e8;padding:7px 10px;text-align:left}th{background:#eef3f8}img{max-width:100%}pre{overflow:auto;background:#f1f4f7;padding:14px}code{font-size:13px}a{color:#135e99}</style><body>'
        + rendered
        + "</body></html>"
    )
    (directory / "learning_when_to_think.html").write_text(document)
    print(
        json.dumps(
            {
                "best_confidence": confidence,
                "best_gain": gain,
                "best_policy": policy,
                "fresh_id": {"confidence": c, "gain": g, "policy": p},
            },
            indent=2,
        )
    )
