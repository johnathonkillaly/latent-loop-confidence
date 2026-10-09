"""Portable public-data audit, clustered uncertainty, measured figures and research report."""

import base64
import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

from latent_loop.experiment04a.data import TASKS, reference, sha
from latent_loop.experiment04a.science import bootstrap
from latent_loop.experiment04b.common import EXP, FINAL, OUT, preservation, utc
from latent_loop.experiment04b.science import (
    outcomes,
    stop_depth,
    transition_metrics,
    visited_selection,
)
from latent_loop.training import atomic_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT = Path("reports/experiment_04b")
SHORT = {
    "fresh_id": "Fresh ID",
    "new_template": "New format",
    "structural_shift": "New structure",
    "harder_depth": "Harder depth",
    "numeric_range": "New range",
}


def read(path):
    path = Path(path)
    if not path.exists() and Path(str(path) + ".gz").exists():
        with gzip.open(str(path) + ".gz", "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def public(split, seed):
    with gzip.open(OUT / "predictions" / f"{split}_seed{seed}.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    x = {
        key: np.array([r[key] for r in rows])
        for key in ("label", "frozen_correct", "one_pass_correct")
    }
    for key in ("correct", "loss", "C", "V", "legacy_gain", "logits"):
        x[key] = np.swapaxes(np.array([r[key] for r in rows]), 0, 1)
    for key in ("quality", "transition_probabilities", "stack_events"):
        x[key] = {
            name: np.swapaxes(np.array([r[key][name] for r in rows]), 0, 1)
            for name in rows[0][key]
        }
    x["paths"] = {
        name: {
            s: np.array([r["paths"][name][s] for r in rows])
            for s in rows[0]["paths"][name]
        }
        for name in rows[0]["paths"]
    }
    x["rows"] = rows
    return x


def decided(x, policy, selector="final"):
    if policy in ("frozen", "one_pass"):
        return x[policy + "_correct"]
    p = x["paths"][policy]
    if selector == "oracle":
        return (x["correct"] & (np.arange(4)[:, None] < p["paid"])).any(0)
    return x["correct"][p[selector] - 1, np.arange(len(x["label"]))]


def rank_metric(p, y, kind):
    p, y = np.asarray(p).ravel(), np.asarray(y, bool).ravel()
    pos, neg = y.sum(), (~y).sum()
    if not pos or not neg:
        return np.nan
    if kind == "auroc":
        return float((rankdata(p)[y].sum() - pos * (pos + 1) / 2) / (pos * neg))
    order = np.argsort(-p, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    tp = np.cumsum(y[order])[ends]
    return float(np.sum(tp / (ends + 1) * np.diff(np.r_[0, tp / pos])))


def ranking(p, y, kind, other=None, repeats=400):
    p, y = np.asarray(p), np.asarray(y)
    rng = np.random.default_rng(450911)

    def estimate(seeds, ix):
        return float(
            np.nanmean(
                [
                    rank_metric(p[s][..., ix], y[s][..., ix], kind)
                    - (
                        rank_metric(other[s][..., ix], y[s][..., ix], kind)
                        if other is not None
                        else 0
                    )
                    for s in seeds
                ]
            )
        )

    vals = [
        estimate(
            rng.integers(len(p), size=len(p)),
            rng.integers(p.shape[-1], size=p.shape[-1]),
        )
        for _ in range(repeats)
    ]
    return {
        "estimate": estimate(range(len(p)), np.arange(p.shape[-1])),
        "cluster_seed_bootstrap95": np.quantile(vals, [0.025, 0.975]).tolist(),
        "unique_problems": p.shape[-1],
        "parent_seeds": len(p),
        "repeats": repeats,
    }


def audit(config, data=None):
    if data is None:
        data = {s: [public(s, seed) for seed in config["parent_seeds"]] for s in FINAL}
    frozen = read(EXP / "frozen_protocol.json")
    for p, h in frozen["hashes"]["files"].items():
        assert sha(p) == h, p
    all_ids, all_shapes, all_prompts = set(), set(), set()
    checks = 0
    manifest = read(EXP / "dataset_manifest.json")
    for split in FINAL:
        path = OUT / "datasets" / f"{split}.jsonl.gz"
        assert sha(path) == manifest["splits"][split]["sha256"]
        rows = [json.loads(line) for line in gzip.open(path, "rt")]
        for r in rows:
            assert (
                r["id"] not in all_ids
                and r["structural_hash"] not in all_shapes
                and r["prompt"] not in all_prompts
            )
            all_ids.add(r["id"])
            all_shapes.add(r["structural_hash"])
            all_prompts.add(r["prompt"])
            answer = reference(r["generator_id"], r["spec"])
            expected = r["entities"][answer] if isinstance(answer, int) else answer
            assert r["choices"][r["label"]] == expected
        for seed, x in zip(config["parent_seeds"], data[split], strict=True):
            chosen = read(EXP / f"selection_seed{seed}.json")
            metric = read(OUT / "metrics" / f"{split}_seed{seed}.json")
            assert (
                sha(OUT / "predictions" / f"{split}_seed{seed}.jsonl.gz")
                == metric["predictions_sha256"]
            )
            assert [r["instance_id"] for r in x["rows"]] == [r["id"] for r in rows]
            assert np.array_equal(x["label"], [r["label"] for r in rows])
            yy = x["label"]
            logits = x["logits"]
            assert np.array_equal(logits.argmax(-1) == yy, x["correct"])
            loss = logsumexp(logits, axis=-1) - logits[:, np.arange(len(yy)), yy]
            assert np.allclose(loss, x["loss"], atol=1e-10)
            scores = {k: x[k] for k in ("C", "V", "legacy_gain")}
            scores["D"] = x["transition_probabilities"][chosen["D_model"]]
            scores.update(x["stack_events"])
            for name, p in x["paths"].items():
                if name in chosen["rules"]:
                    assert np.array_equal(
                        p["paid"], stop_depth(scores, chosen["rules"][name])
                    )
                if name.startswith("fixed_") and name != "fixed_selected":
                    assert np.all(p["paid"] == int(name[-1]))
                for selector, selected in p.items():
                    if selector == "paid":
                        continue
                    assert np.all((selected >= 1) & (selected <= p["paid"]))
                    if selector == "final":
                        assert np.array_equal(selected, p["paid"])
                    else:
                        assert np.array_equal(
                            selected,
                            visited_selection(x["quality"][selector], p["paid"]),
                        )
                    assert np.isclose(
                        decided(x, name, selector).mean(),
                        metric["selection"]["all"][name][selector]["accuracy"],
                    )
                    checks += 1
                for i, d in enumerate(p["paid"]):
                    expected = [
                        "continue" if k < d else "stop" if k == d else "unvisited"
                        for k in range(1, 5)
                    ]
                    assert x["rows"][i]["decisions"][name] == expected
            for name, p in x["transition_probabilities"].items():
                assert (
                    np.isfinite(p).all()
                    and np.all(p >= 0)
                    and np.allclose(p.sum(-1), 1, atol=1e-6)
                )
                calc = transition_metrics(p[:3], outcomes(x["correct"]))
                recorded = metric["transitions"]["all"][name]
                for field in ("four_class_log_loss", "multiclass_brier"):
                    assert np.isclose(calc[field], recorded[field])
                    checks += 1
                for ev in ("WR", "RW"):
                    for field in ("auroc", "average_precision", "brier", "ece"):
                        assert np.isclose(
                            calc["classes"][ev][field], recorded["classes"][ev][field]
                        )
                        checks += 1
    registry = read(EXP / "final_trajectory_registry.json")
    state_count = 0
    for r in registry:
        assert sha(r["public_trajectory"]) == r["public_sha256"]
        x = data[r["split"]][config["parent_seeds"].index(r["seed"])]
        labels = outcomes(x["correct"])
        lines = 0
        with gzip.open(r["public_trajectory"], "rt") as stream:
            for line in stream:
                row = json.loads(line)
                i, k = divmod(lines, 4)
                assert row["identity"]["instance_id"] == x["rows"][i]["instance_id"]
                assert row["depth"] == k + 1
                assert np.array_equal(
                    row["observation"]["candidate_logits"], x["logits"][k, i]
                )
                sup = row["offline_supervision"]
                assert sup["true_answer_index"] == x["label"][i]
                assert sup["current_correct"] == x["correct"][k, i]
                assert np.isclose(sup["current_loss_nats"], x["loss"][k, i])
                if k < 3:
                    assert (
                        sup["next_transition"]
                        == ("W->W", "W->R", "R->W", "R->R")[labels[k, i]]
                    )
                    assert sup["signed_correctness_change"] == int(
                        x["correct"][k + 1, i]
                    ) - int(x["correct"][k, i])
                    assert np.isclose(
                        sup["signed_loss_gain_nats"],
                        x["loss"][k, i] - x["loss"][k + 1, i],
                    )
                else:
                    assert sup["next_transition"] is None
                lines += 1
        assert lines == r["state_rows"]
        state_count += lines
    infos = [read(p) for p in sorted((OUT / "training").glob("*.json"))]
    assert len(infos) == 36 and all(i["complete"] for i in infos)
    for seed in config["parent_seeds"]:
        oof = read(EXP / f"selection_seed{seed}.json")["OOF"]
        assignment = np.array(oof["problem_fold_assignment"])
        assert set(assignment) == {0, 1, 2} and len(assignment) == 4032
        for fold in range(3):
            info = read(OUT / "training" / f"oof{fold}_seed{seed}.json")
            assert info["training_items"] == 3 * int((assignment != fold).sum())
    value = {
        "checked_utc": utc(),
        "verified_unique_final_problems": len(all_ids),
        "new_state_rows": state_count,
        "new_adjacent_transitions": state_count // 4 * 3,
        "public_metric_checks": checks,
        "neural_fits": len(infos),
        "all_selectors_within_paid_prefix": True,
        "all_stop_continue_logs_recomputed": True,
        "all_final_labels_rechecked_by_independent_solvers": True,
        "frozen_public_source_and_metadata_hashes_verified": True,
        "OOF_problem_group_integrity_verified": True,
        "historical_preservation": preservation(),
        "weights_required_for_public_audit": False,
    }
    atomic_json(OUT / "public_audit.json", value)
    print("Public audit:", json.dumps(value), flush=True)
    return value


def summarize(config, data):
    result = {}
    for split, xx in data.items():
        metrics = [
            read(OUT / "metrics" / f"{split}_seed{s}.json")
            for s in config["parent_seeds"]
        ]
        selections = [
            read(EXP / f"selection_seed{s}.json") for s in config["parent_seeds"]
        ]
        policies = {}
        for name in xx[0]["paths"]:
            policies[name] = {}
            for selector in [s for s in xx[0]["paths"][name] if s != "paid"] + [
                "oracle"
            ]:
                correct = np.stack([decided(x, name, selector) for x in xx])
                row = {
                    "accuracy": bootstrap(correct, config["bootstrap_repeats"]),
                    "mean_paid_loops": float(
                        np.mean([x["paths"][name]["paid"] for x in xx])
                    ),
                    "mean_selected_depth": None
                    if selector == "oracle"
                    else float(np.mean([x["paths"][name][selector] for x in xx])),
                    "seed_accuracy": correct.mean(-1).tolist(),
                }
                policies[name][selector] = row
        baselines = {
            name: bootstrap(
                np.stack([decided(x, name) for x in xx]), config["bootstrap_repeats"]
            )
            for name in ("frozen", "one_pass")
        }
        model_names = list(xx[0]["transition_probabilities"]) + ["selected_D"]
        transition = {}
        for name in model_names:
            values = [
                m["transitions"]["all"][s["D_model"] if name == "selected_D" else name]
                for m, s in zip(metrics, selections, strict=True)
            ]
            transition[name] = {
                "four_class_log_loss": float(
                    np.mean([v["four_class_log_loss"] for v in values])
                ),
                "multiclass_brier": float(
                    np.mean([v["multiclass_brier"] for v in values])
                ),
                "events": {
                    ev: {
                        field: float(
                            np.mean(
                                [
                                    v["classes"][ev][field]
                                    for v in values
                                    if v["classes"][ev][field] is not None
                                ]
                            )
                        )
                        if any(v["classes"][ev][field] is not None for v in values)
                        else None
                        for field in (
                            "prevalence",
                            "auroc",
                            "average_precision",
                            "brier",
                            "ece",
                            "precision_at_0.5",
                            "recall_at_0.5",
                            "selected_fraction_at_0.5",
                            "precision_at_top_10pct",
                            "recall_at_top_10pct",
                        )
                    }
                    for ev in ("WR", "RW")
                },
                "conditional_ranking": {
                    ev: {
                        field: float(
                            np.mean(
                                [
                                    v["conditional_event_ranking"][ev][field]
                                    for v in values
                                ]
                            )
                        )
                        for field in ("prevalence", "auroc", "average_precision")
                    }
                    for ev in ("WR_given_current_wrong", "RW_given_current_right")
                },
            }
        comparisons = {}
        pairs = [
            ("D_minus_frozen", "d_direct", "final", "frozen", "final"),
            ("one_pass_minus_frozen", "one_pass", "final", "frozen", "final"),
            ("D_minus_one_pass", "d_direct", "final", "one_pass", "final"),
            ("D_minus_fixed1", "d_direct", "final", "fixed_1", "final"),
            ("D_minus_matched", "d_direct", "final", "matched_d_direct", "final"),
            ("D_minus_fixed_selected", "d_direct", "final", "fixed_selected", "final"),
            ("selector_minus_final", "d_direct", "ranker_CVD", "d_direct", "final"),
            ("selector_minus_max_C", "d_direct", "ranker_CVD", "d_direct", "max_C"),
            ("selector_minus_one_pass", "d_direct", "ranker_CVD", "one_pass", "final"),
            (
                "selector_minus_matched_selector",
                "d_direct",
                "ranker_CVD",
                "matched_d_direct",
                "ranker_CVD",
            ),
            (
                "fixed4_selector_minus_final",
                "fixed_4",
                "ranker_CVD",
                "fixed_4",
                "final",
            ),
            (
                "fixed4_selector_minus_max_C",
                "fixed_4",
                "ranker_CVD",
                "fixed_4",
                "max_C",
            ),
            (
                "fixed4_CVD_minus_ranker_C",
                "fixed_4",
                "ranker_CVD",
                "fixed_4",
                "ranker_C",
            ),
            ("CVD_minus_CV", "stack_CVD", "final", "stack_CV", "final"),
            ("oracle_gap", "d_direct", "oracle", "d_direct", "ranker_CVD"),
        ]
        for label, a, sa, b, sb in pairs:
            delta = np.stack(
                [decided(x, a, sa).astype(float) - decided(x, b, sb) for x in xx]
            )
            comparisons[label] = bootstrap(delta, config["bootstrap_repeats"])
        best = np.stack(
            [
                decided(
                    x,
                    s["validation_best_combination"]["policy"],
                    s["validation_best_combination"]["selector"],
                )
                for x, s in zip(xx, selections, strict=True)
            ]
        )
        best_paid = float(
            np.mean(
                [
                    x["paths"][s["validation_best_combination"]["policy"]]["paid"]
                    for x, s in zip(xx, selections, strict=True)
                ]
            )
        )
        baselines["validation_best_combination"] = bootstrap(
            best, config["bootstrap_repeats"]
        )
        baselines["validation_best_mean_paid_loops"] = best_paid
        truth = np.stack([outcomes(x["correct"]) for x in xx])
        intervals = {}
        for ev, k in [("WR", 1), ("RW", 2)]:
            pp = np.stack(
                [
                    x["transition_probabilities"][s["D_model"]][:3, :, k]
                    for x, s in zip(xx, selections, strict=True)
                ]
            )
            cc = np.stack(
                [x["transition_probabilities"]["scalar_CV"][:3, :, k] for x in xx]
            )
            allscalar = np.stack(
                [x["transition_probabilities"]["scalar_all"][:3, :, k] for x in xx]
            )
            stackcv = np.stack(
                [x["transition_probabilities"]["stack_CV"][:3, :, k] for x in xx]
            )
            stackcvd = np.stack(
                [x["transition_probabilities"]["stack_CVD"][:3, :, k] for x in xx]
            )
            intervals[ev] = {}
            for kind in ("auroc", "average_precision"):
                for label, p, other in [
                    ("selected_D", pp, None),
                    ("D_minus_scalar_CV", pp, cc),
                    ("D_minus_scalar_all", pp, allscalar),
                    ("CVD_minus_CV", stackcvd, stackcv),
                ]:
                    intervals[ev][kind + "_" + label] = ranking(
                        p, truth == k, kind, other, config["auc_bootstrap_repeats"]
                    )
        task = {
            task: {
                "one_pass": float(
                    np.mean(
                        [m["baselines"][task]["one_pass_accuracy"] for m in metrics]
                    )
                ),
                "D": float(
                    np.mean(
                        [
                            m["selection"][task]["d_direct"]["final"]["accuracy"]
                            for m in metrics
                        ]
                    )
                ),
                "D_selector": float(
                    np.mean(
                        [
                            m["selection"][task]["d_direct"]["ranker_CVD"]["accuracy"]
                            for m in metrics
                        ]
                    )
                ),
                "WR_prevalence": float(
                    np.mean(
                        [
                            m["transitions"][task][s["D_model"]]["classes"]["WR"][
                                "prevalence"
                            ]
                            for m, s in zip(metrics, selections, strict=True)
                        ]
                    )
                ),
                "RW_prevalence": float(
                    np.mean(
                        [
                            m["transitions"][task][s["D_model"]]["classes"]["RW"][
                                "prevalence"
                            ]
                            for m, s in zip(metrics, selections, strict=True)
                        ]
                    )
                ),
            }
            for task in TASKS
        }
        failure = {
            category: {
                key: float(
                    np.mean(
                        [m["failures_D_ranker_CVD"][category][key] for m in metrics]
                    )
                )
                if all(
                    m["failures_D_ranker_CVD"][category][key] is not None
                    for m in metrics
                )
                else None
                for key in (
                    "count",
                    "frequency",
                    "mean_C",
                    "mean_predicted_D",
                    "mean_actual_D",
                    "mean_selected_depth",
                    "mean_paid_loops",
                )
            }
            for category in "ABCDEFGH"
        }
        result[split] = {
            "baselines": baselines,
            "policies": policies,
            "transition_models": transition,
            "paired_accuracy_differences": comparisons,
            "paired_event_intervals": intervals,
            "tasks": task,
            "failure_categories": failure,
            "class_prevalence": np.mean(
                [
                    np.bincount(outcomes(x["correct"]).ravel(), minlength=4)
                    / outcomes(x["correct"]).size
                    for x in xx
                ],
                axis=0,
            ).tolist(),
        }
        print("summarized", split, flush=True)
    atomic_json(
        OUT / "summary.json",
        {
            "completed_utc": utc(),
            "splits": result,
            "uncertainty": "Two-way paired problem-instance and parent-seed bootstrap; all depths remain grouped. Only three parents; intervals conditional on one trained backbone/refiner family. No multiplicity correction.",
            "primary_comparisons": "Fresh ID D-final minus one pass, D-ranker_CVD minus same visited final, WR/RW AP minus scalar CV.",
        },
    )
    return result


def save(fig, name, layout_rect=None):
    folder = REPORT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    if fig.get_layout_engine() is None:
        fig.tight_layout(rect=layout_rect)
    for ext in ("png", "svg"):
        path = folder / f"{name}.{ext}"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def figures(config, summary, data, latency):
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fid = summary["fresh_id"]
    xx = data["fresh_id"]
    selections = [read(EXP / f"selection_seed{s}.json") for s in config["parent_seeds"]]
    names = [
        "selected_D",
        "binary_pair",
        "matched",
        "stack_C",
        "stack_V",
        "stack_CV",
        "stack_CVD",
        "scalar_all",
        "task_diagnostic",
        "prevalence",
        "random_raw",
    ]
    for ev in ("WR", "RW"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, field in zip(axes, ("auroc", "average_precision"), strict=True):
            values = [fid["transition_models"][n]["events"][ev][field] for n in names]
            ax.barh(
                names[::-1],
                values[::-1],
                color=[
                    "#416b92" if n == "selected_D" else "#a3b8c8" for n in names[::-1]
                ],
            )
            ax.axvline(
                0.5
                if field == "auroc"
                else fid["class_prevalence"][1 if ev == "WR" else 2],
                color="black",
                linestyle="--",
                label="chance/prevalence",
            )
            ax.set(
                xlabel=field.replace("_", " ").upper(),
                xlim=(0, 1 if field == "auroc" else max(values) * 1.3),
                title=f"{ev}: fresh ID, original prevalence",
            )
            ci = fid["paired_event_intervals"][ev][field + "_selected_D"][
                "cluster_seed_bootstrap95"
            ]
            ax.errorbar(
                values[0],
                len(names) - 1,
                xerr=[[values[0] - ci[0]], [ci[1] - values[0]]],
                fmt="none",
                color="black",
                capsize=3,
            )
            ax.legend(fontsize=8)
        save(fig, "01_correction" if ev == "WR" else "02_degradation")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, k, ev in zip(axes, (1, 2), ("WR", "RW"), strict=True):
        for name, label in [
            ("selected_D", "D calibrated"),
            ("d4_plain_raw", "D raw"),
            ("binary_pair", "Binary pair"),
        ]:
            p = np.concatenate(
                [
                    x["transition_probabilities"][
                        s["D_model"] if name == "selected_D" else name
                    ][:3, :, k].ravel()
                    for x, s in zip(xx, selections, strict=True)
                ]
            )
            y = np.concatenate([(outcomes(x["correct"]) == k).ravel() for x in xx])
            groups = np.array_split(np.argsort(p), 10)
            ax.plot(
                [p[g].mean() for g in groups],
                [y[g].mean() for g in groups],
                "o-",
                label=label,
            )
        ax.plot([0, 0.5], [0, 0.5], "k--")
        ax.set(
            xlabel="Mean predicted event probability",
            ylabel="Observed event rate",
            title=f"{ev} calibration (equal-count bins)",
        )
        ax.legend()
    save(fig, "03_calibration")
    fig, ax = plt.subplots(figsize=(7, 5))
    p = np.concatenate(
        [
            (
                x["transition_probabilities"][s["D_model"]][:3, :, 1]
                - x["transition_probabilities"][s["D_model"]][:3, :, 2]
            ).ravel()
            for x, s in zip(xx, selections, strict=True)
        ]
    )
    y = np.concatenate(
        [
            (
                (outcomes(x["correct"]) == 1).astype(int)
                - (outcomes(x["correct"]) == 2)
            ).ravel()
            for x in xx
        ]
    )
    groups = np.array_split(np.argsort(p), 10)
    ax.plot(
        [p[g].mean() for g in groups],
        [y[g].mean() for g in groups],
        "o-",
        label="Observed mean in score bins",
    )
    lo, hi = min(p.min(), -0.1), max(p.max(), 0.1)
    ax.plot([lo, hi], [lo, hi], "k--", label="Ideal")
    ax.axhline(0, color="grey", lw=0.7)
    ax.set(
        xlabel="Predicted P(WR)−P(RW)",
        ylabel="Observed signed correctness change",
        title="Decision value: fresh ID",
    )
    ax.legend()
    save(fig, "04_decision_value")
    fig, ax = plt.subplots(figsize=(8, 5))
    for selector, color in [
        ("final", "#888888"),
        ("max_C", "#d99a36"),
        ("ranker_CVD", "#416b92"),
        ("oracle", "#6c8d67"),
    ]:
        rr = [fid["policies"][f"fixed_{k}"][selector] for k in range(1, 5)]
        ax.plot(
            range(1, 5),
            [r["accuracy"]["estimate"] * 100 for r in rr],
            "o-",
            label=f"Fixed: {selector}",
            color=color,
        )
    for name, selector, label in [
        ("d_direct", "final", "D stop"),
        ("d_direct", "ranker_CVD", "D + ranker"),
        ("stack_CV", "final", "CV stop"),
        ("stack_CVD", "ranker_CVD", "CVD + ranker"),
    ]:
        r = fid["policies"][name][selector]
        ax.scatter(
            r["mean_paid_loops"], 100 * r["accuracy"]["estimate"], s=70, label=label
        )
    ax.axhline(
        100 * fid["baselines"]["one_pass"]["estimate"],
        color="black",
        linestyle="--",
        label="One pass",
    )
    ax.set(
        xlabel="Paid recurrent loops (selected depth never refunds cost)",
        ylabel="Accuracy (%)",
        title="Accuracy versus actual visited budget",
    )
    ax.legend(fontsize=8, ncol=2)
    save(fig, "05_paid_loops")
    # Accuracy is separately measured on final ID; latency is validation measured, not final latency.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, size in zip(axes, ("1", "8"), strict=True):
        for name in (
            "one_pass",
            "fixed_1",
            "fixed_4",
            "d_direct",
            "d_direct:ranker_CVD",
            "fixed_4:ranker_CVD",
            "stack_CVD:ranker_CVD",
            "validation_best",
        ):
            if name == "one_pass":
                acc = fid["baselines"]["one_pass"]["estimate"]
            elif name == "validation_best":
                acc = fid["baselines"]["validation_best_combination"]["estimate"]
            else:
                pol, _, sel = name.partition(":")
                acc = fid["policies"][pol][sel or "final"]["accuracy"]["estimate"]
            ms = latency["summary"][size][name]["amortized_ms_per_item"]
            ax.scatter(ms, acc * 100, label=name.replace("_", " ").replace(":", " + "))
        ax.set(
            xlabel="Full validation inference, ms/item",
            ylabel="Independent final ID accuracy (%)",
            title=f"Batch {size}; different accuracy/timing datasets",
        )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        fontsize=8,
    )
    save(fig, "06_full_latency", layout_rect=(0, 0.16, 1, 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    selectors = [
        "final",
        "max_C",
        "ranker_C",
        "ranker_CV",
        "ranker_CD",
        "ranker_CVD",
        "oracle",
    ]
    for ax, pol in zip(axes, ("fixed_4", "d_direct"), strict=True):
        vals = [
            fid["policies"][pol][s]["accuracy"]["estimate"] * 100 for s in selectors
        ]
        ax.bar(selectors, vals, color="#8ca9bd")
        ax.tick_params(axis="x", rotation=40)
        ax.set(
            ylabel="Accuracy (%)",
            title=f"{pol}: same paid trajectories for all selectors",
        )
    save(fig, "07_selectors")
    examples = []
    for seed in config["parent_seeds"]:
        examples.extend(
            [
                r
                for r in read(OUT / "examples" / f"fresh_id_seed{seed}.json")
                if r["category"] == "G"
            ]
        )
    fig, axes = plt.subplots(
        max(1, min(3, len(examples))),
        1,
        figsize=(10, 3 * max(1, min(3, len(examples)))),
        squeeze=False,
    )
    if not examples:
        axes[0, 0].text(
            0.5,
            0.5,
            "No earlier-state recoveries in the prespecified path.",
            ha="center",
        )
    for ax, r in zip(axes[:, 0], examples[:3]):
        colors = ["#70a17a" if c else "#bb7777" for c in r["correct_by_depth"]]
        paid = r["paid_loops"]
        ax.bar(
            range(1, paid + 1),
            r["C_by_depth"][:paid],
            color=colors[:paid],
            alpha=0.7,
            label="Calibrated C",
        )
        source = next(
            x for s, x in zip(config["parent_seeds"], xx, strict=True) if s == r["seed"]
        )
        index = next(
            i
            for i, row in enumerate(source["rows"])
            if row["instance_id"] == r["instance_id"]
        )
        ax.plot(
            range(1, paid + 1),
            source["quality"]["ranker_CVD"][:paid, index],
            "o-",
            color="#264e78",
            label="Ranker quality score",
        )
        ax.axvline(r["selected_depth"], color="#264e78", ls="--", label="Selected")
        ax.axvline(r["paid_loops"], color="black", ls=":", label="Paid through")
        ax.set(
            title=f"Seed {r['seed']}, {r['instance_id'][:12]}: visited answers="
            + str(r["predictions_by_depth"][:paid])
            + "; correct="
            + str(r["correct_by_depth"][:paid]),
            xlabel="Depth",
            ylabel="Calibrated C",
        )
        ax.legend()
    save(fig, "08_recovery_examples")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout="constrained")
    for ax, key, title in zip(
        axes,
        ("one_pass", "D", "D_selector"),
        ("One pass", "D stop", "D + ranker"),
        strict=True,
    ):
        matrix = np.array(
            [[summary[s]["tasks"][task][key] * 100 for s in FINAL] for task in TASKS]
        )
        im = ax.imshow(matrix, vmin=0, vmax=65, cmap="viridis", aspect="auto")
        ax.set_xticks(range(5), [SHORT[s] for s in FINAL], rotation=40, ha="right")
        ax.set_yticks(range(6), TASKS)
        ax.set_title(title)
        for i in range(6):
            for j in range(5):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] < 35 else "black",
                    fontsize=9,
                )
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Accuracy (%)", fraction=0.02)
    save(fig, "09_task_generalization")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    comparisons = fid["paired_accuracy_differences"]
    labels = list(comparisons)
    estimates = np.array([comparisons[k]["estimate"] for k in labels]) * 100
    cis = np.array([comparisons[k]["cluster_seed_bootstrap95"] for k in labels]) * 100
    axes[0].errorbar(
        estimates,
        np.arange(len(labels)),
        xerr=np.abs(cis.T - estimates),
        fmt="o",
        color="#416b92",
    )
    axes[0].set_yticks(
        range(len(labels)), [label.replace("_", " ") for label in labels]
    )
    axes[0].axvline(0, color="black", ls="--")
    axes[0].set_xlabel("Paired accuracy difference (percentage points)")
    ff = fid["failure_categories"]
    axes[1].bar(list(ff), [ff[c]["frequency"] * 100 for c in ff], color="#a27878")
    axes[1].set(
        xlabel="Prespecified failure/transition category",
        ylabel="Occurrences per 100 denominator units",
        title="A–D: transitions; E–H: problems",
    )
    save(fig, "10_errors_uncertainty")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ev, k in zip(axes, ("WR", "RW"), (1, 2), strict=True):
        for name in ("selected_D", "scalar_CV", "scalar_all", "stack_CVD"):
            ax.plot(
                range(5),
                [
                    summary[s]["transition_models"][name]["events"][ev][
                        "average_precision"
                    ]
                    for s in FINAL
                ],
                "o-",
                label=name,
            )
        ax.plot(
            range(5),
            [summary[s]["class_prevalence"][k] for s in FINAL],
            "k--",
            label="Prevalence",
        )
        ax.set_xticks(range(5), [SHORT[s] for s in FINAL], rotation=30)
        ax.set(ylabel="Average precision", title=f"{ev} transfer")
        ax.legend(fontsize=8)
    save(fig, "11_event_generalization")


def percent(v):
    return f"{100 * v:.2f}%"


def interval(v, scale=100):
    lo, hi = v["cluster_seed_bootstrap95"]
    return f"{v['estimate'] * scale:+.2f} [{lo * scale:+.2f}, {hi * scale:+.2f}]"


def report(config):
    REPORT.mkdir(parents=True, exist_ok=True)
    archive_metrics()
    data = {s: [public(s, seed) for seed in config["parent_seeds"]] for s in FINAL}
    audit_value = audit(config, data)
    summary = summarize(config, data)
    extra = supplemental(config, data)
    for split in FINAL:
        summary[split]["failure_categories"] = extra[split]["failure_categories"]
        summary[split]["stopping_behavior"] = extra[split]["behavior_mean_counts"]
    stored = read(OUT / "summary.json")
    stored["splits"] = summary
    stored["failure_definition"] = (
        "Broad user categories: any signed WW loss change; H covers any correct later visited state. Meaningful-change subgroups retained in per-seed metrics."
    )
    atomic_json(OUT / "summary.json", stored)
    latency = read(OUT / "latency.json")
    figures(config, summary, data, latency)
    # Research prose is assembled only from this locked evaluation and validation choices.
    write_report(config, summary, data, latency, audit_value)


def write_report(config, summary, data, latency, audit_value):
    import resource

    import markdown

    fid = summary["fresh_id"]
    trans = fid["transition_models"]
    pol = fid["policies"]
    selected = [read(EXP / f"selection_seed{s}.json") for s in config["parent_seeds"]]
    infos = [read(p) for p in sorted((OUT / "training").glob("*.json"))]
    frozen = read(EXP / "frozen_protocol.json")
    opened = read(EXP / "lockbox_opened.json")
    extraction = read(OUT / "final_extraction.json")
    receipt_path = REPORT / "local_commit_receipt.json"
    commit_note = (
        f"Scientific code/results commit: `{read(receipt_path)['scientific_commit']}` (local only; this receipt is a subsequent documentation commit)."
        if receipt_path.exists()
        else "Scientific commit hash: recorded by the local commit receipt after completing the scientific snapshot."
    )
    peak = (
        max(
            [latency["peak_mlx_bytes"], extraction["peak_mlx_bytes"]]
            + [i["peak_mlx_bytes"] for i in infos]
        )
        / 2**30
    )
    rss = (
        max(
            latency["peak_process_rss_bytes"],
            extraction["peak_process_rss_bytes"],
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        / 2**30
    )
    d = pol["d_direct"]["final"]
    ds = pol["d_direct"]["ranker_CVD"]
    delta = fid["paired_accuracy_differences"]
    behavior = fid["stopping_behavior"]
    lost_final = float(
        np.mean(
            [
                np.mean(
                    ~decided(x, "d_direct", "ranker_CVD")
                    & decided(x, "d_direct", "final")
                )
                for x in data["fresh_id"]
            ]
        )
    )
    notes = []
    for ev in ("WR", "RW"):
        r = fid["paired_event_intervals"][ev]["average_precision_D_minus_scalar_CV"]
        notes.append(
            f"{ev} AP versus scalar C+V: {interval(r, 1)} (absolute AP units)."
        )
    stop_supported = delta["D_minus_one_pass"]["cluster_seed_bootstrap95"][0] > 0
    selection_supported = (
        delta["selector_minus_final"]["cluster_seed_bootstrap95"][0] > 0
    )
    measured = "A fresh learned D head ranks decision transitions, but deployed benefit must be judged separately."
    stopping_verdict = (
        "D stopping improves fresh-ID accuracy over one pass with a positive paired interval."
        if stop_supported
        else "A fresh-ID accuracy advantage over one pass is not established by the paired interval."
    )
    selector_verdict = (
        "The prespecified ranker improves the same visited final answer with a positive paired interval."
        if selection_supported
        else "The prespecified ranker does not establish an improvement over the same visited final answer."
    )
    lines = [
        "# D: Learning Whether Another Thought Changes the Answer",
        "",
        "Experiment 4B — decision-value prediction and recurrent trajectory selection.",
        "",
        f"{measured} {stopping_verdict} {selector_verdict}",
        "",
        f"The locked fresh-ID result is D stopping **{percent(d['accuracy']['estimate'])}**, D plus the CVD ranker **{percent(ds['accuracy']['estimate'])}**, and one pass **{percent(fid['baselines']['one_pass']['estimate'])}**. A single recurrent step already reaches **{percent(pol['fixed_1']['final']['accuracy']['estimate'])}**; D minus that control is {interval(delta['D_minus_fixed1'])} percentage points. D pays {d['mean_paid_loops']:.3f} loops; selection pays the identical {ds['mean_paid_loops']:.3f} loops even when it returns depth {ds['mean_selected_depth']:.3f} on average.",
        "",
        "## Boundaries and prospective status",
        "",
        f"Parent scientific commit: `{config['start_commit']}`. Branch: `research/exp04-decision-value`. {commit_note} Protocol frozen at `{frozen['frozen_utc']}`; final lockbox opened at `{opened['opened_utc']}`. All 296 historical tracked files remain unchanged except the README addition. No backbone/refiner weights were trained, no workstation was accessed, and no remote was created or pushed.",
        "",
        "The confirmatory targets were fixed in the written plan: fresh-ID direct D versus one pass, CVD ranker versus the final state on the same D-visited trajectory, and correction/degradation AP versus a scalar C+V control. Everything else is prespecified secondary or descriptive: shift tests, seven signal stacks, alternative classifiers, utility, max-C, ranker masks, failure examples, and oracle diagnostics. The same final examples evaluate several policies; intervals are not adjusted for multiple comparisons. No final ranking selected a deployment winner.",
        "",
        "The 4A final lockbox was already consumed. 4B generated five independently seeded sealed sets, each with 1,296 problems and 216 per task: fresh ID, new linguistic templates, unseen structural generator families, harder reasoning depth, and numerical-range changes. All 6,480 instance IDs, normalized structures, and prompts are distinct from prior data and from each other. Training reuses 4,032 4A problems; validation reuses 1,008, partitioned by task into 336 checkpoint, 336 calibration/stack, and 336 policy examples. Sealed final labels were not used for any choice.",
        "",
        "Numerical-range tests scale actual operands, thresholds, resources, capacities, or inventories by 17 where the task permits it. Pointer graphs remain fresh nominal-range controls; this split is not a uniform range intervention across all six tasks. Tree labels are independently recursively solved, sequential tasks independently simulated, and invalid/nonunique choices rejected. Abstract algorithms overlap within ID/format/range shifts; alpha-equivalent exact instances do not. Structural shift switches to a held-out generator approach. Harder-depth extrapolation also changes prompt lengths and difficulty.",
        "",
        "## Architecture, targets and fitting",
        "",
        "The frozen FP32 Qwen3-0.6B checkpoint and three frozen E3 tied residual refiners supply four decision states. Inputs are the current RMS-normalized 1,024-dimensional hidden vector, four candidate probabilities, entropy, margin, relative change from the already observed previous state, and depth/4: 1,032 features. Conditional logits cover the same single-token option letters A–D. No generated reasoning tokens are used.",
        "",
        "D is a width-32 gated MLP: `Linear_out(silu(Linear_value(x)) * sigmoid(Linear_gate(x)))`, with four outputs WW, WR, RW, RR. Its target is `2*current_correct + next_correct` for depths 1→2, 2→3, 3→4. Expected decision gain is `P(WR)-P(RW)` in accuracy-probability units. Correctness supervises training and offline diagnostics only; runtime APIs have no correctness, loss, label, or future-state argument.",
        "",
        "Ordinary cross entropy and inverse-square-root class-weighted cross entropy are compared on original-distribution checkpoint NLL. All three parents chose ordinary CE. A width-64 plain MLP is parameter matched. Independently trained width-16 WR/RW gated binary heads have nearly equal total capacity. Every neural model uses AdamW, learning rate 0.001, weight decay 0.01, batches of 512, at most 40 epochs, gradient norm clipping at 1, and epoch checkpoints with optimizer state. Original prevalence is retained at evaluation.",
        "",
        "| Neural condition | Fits | Parameters per parent |",
        "|---|---:|---:|",
    ]
    for name, label, count in [
        ("d4_plain", "Ordinary four-class D", 3),
        ("d4_weighted", "Weighted four-class D", 3),
        ("matched", "Plain parameter-matched classifier", 3),
        ("event_WR", "Independent WR head", 3),
        ("event_RW", "Independent RW head", 3),
        ("oof0", "Three grouped OOF D folds", 9),
        ("ranker_C", "Ranker C", 3),
        ("ranker_CV", "Ranker CV", 3),
        ("ranker_CD", "Ranker CD", 3),
        ("ranker_CVD", "Ranker CVD", 3),
    ]:
        info = next(i for i in infos if i["name"] == name)
        lines.append(f"| {label} | {count} | {info['parameters']:,} |")
    lines += [
        "",
        "This is 36 supervised neural fits: 27 main fits plus 9 OOF nuisance fits, across three parent seeds (11, 22, 33). Three untouched random initializations are controls and do not count as fits. Eight substantive fitted neural conditions are compared, plus the OOF D procedure. Small statistical calibrators and event policies are additional fits recorded completely in each selection JSON; they are not folded into the neural count.",
        "",
        "C is the frozen E3 correctness head, recalibrated separately at depths 1–4 by logistic fitting. V is the frozen E3 regression head predicting expected next conditional-loss improvement, restored to its training loss scale; it is not the binary useful-loss head. That older binary gain controller is retained separately as `legacy_gain`.",
        "",
        "Four-class calibration fits a positive temperature plus centered class offsets by NLL on calibration validation, with a small offset penalty. Binary WR/RW logits receive separate logistic calibration. Two event probabilities are projected if their sum exceeds one; the residual WW/RR mass is split using the training non-event prior. Their four-class NLL/Brier is therefore a diagnostic completion, not a learned estimate of both residual classes. Event AP/AUROC and expected decision gain remain the meaningful comparisons.",
        "",
        "Seven tiny validation-trained event stacks use C, V, D, CV, CD, VD, and CVD. Their inputs are only head predictions. Logistic scalar controls trained on 4A training data use C+V, entropy, margin, all scalar observables (including depth and completed hidden change), and task/difficulty metadata. Task metadata is an offline diagnostic control, excluded from deployment. Different fitting sample sizes and frozen versus fresh heads limit causal attribution; parameter-matched fresh models address head capacity separately.",
        "",
        "For ranker training, all states of each training problem stay within one of three OOF folds. A D model trained on the other folds supplies that problem’s D features; its architecture and fixed epoch budget were selected using checkpoint validation. Calibration/stack fitting also cross-fits calibration transforms within its own 336-problem group. The final deployment calibrators use all calibration validation. Policy thresholds use only the disjoint 336-problem policy group.",
        "",
        "The shared width-16 ranker scores a current state independently, using the same 1,032 base features plus four masked extra slots C, V, P(WR), P(RW). All four ranker variants have equal capacity. C/CV/CD/CVD names describe extra signals; their base hidden/candidate features remain identical. The architecture is permutation safe when state rows and explicit depth tags are permuted. Pointwise correctness BCE plus 0.5 times pairwise correct-versus-incorrect softplus loss is trained on random visible prefixes of lengths 1–4. Ranker checkpoints minimize this combined validation objective, although the generic training field is named `best_validation_nll`. At terminal depth 4, unavailable continuation V/D features are zero.",
        "",
        "## Validation choices and stopping rules",
        "",
        "Direct D continues when `P(WR)-P(RW)>threshold`. C continues below a correctness threshold; V above a next-loss-gain threshold; each event stack above its WR−RW threshold. Thresholds maximize policy-validation accuracy, with lower paid loops then lower NLL as ties. The utility alternative continues when `D-lambda*1>0`: incremental cost is one normalized recurrent loop, and lambda is accuracy probability per loop. Lambda selection maximizes the prespecified validation objective `accuracy-0.01*mean_paid_loops` (0.01 is one percentage point per loop). It is an illustrative utility preference, not a monetary or latency-optimal cost.",
        "",
        "| Parent | D | Direct D threshold | Utility lambda | Validation-selected combination | Validation accuracy |",
        "|---|---|---:|---:|---|---:|",
    ]
    for s in selected:
        b = s["validation_best_combination"]
        lines.append(
            f"| {s['seed']} | {s['D_model']} | {s['rules']['d_direct']['threshold']} | {s['rules']['d_utility']['lambda']} | {b['policy']} + {b['selector']} | {percent(b['accuracy'])} |"
        )
    lines += [
        "",
        "The validation-selected best combination is a per-parent selection procedure, not one universal post-test champion. All three selected final-state answers. The prespecified CVD ranker remains evaluated even though validation did not choose it.",
        "",
        "## Transition discrimination and calibration",
        "",
        "| Predictor | WR AUROC | WR AP | RW AUROC | RW AP | Four-class NLL | Multiclass Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "selected_D",
        "d4_weighted",
        "matched",
        "binary_pair",
        "stack_C",
        "stack_V",
        "stack_D",
        "stack_CV",
        "stack_CD",
        "stack_VD",
        "stack_CVD",
        "scalar_CV",
        "entropy_control",
        "margin_control",
        "scalar_all",
        "task_diagnostic",
        "prevalence",
        "no_transition",
        "random_raw",
    ):
        r = trans[name]
        e = r["events"]
        lines.append(
            f"| {name} | {e['WR']['auroc']:.3f} | {e['WR']['average_precision']:.3f} | {e['RW']['auroc']:.3f} | {e['RW']['average_precision']:.3f} | {r['four_class_log_loss']:.3f} | {r['multiclass_brier']:.3f} |"
        )
    lines += [
        "",
        f"Fresh-ID original class prevalence WW/WR/RW/RR: {', '.join(percent(v) for v in fid['class_prevalence'])}. A constant predictor’s AP equals event prevalence; four-class accuracy alone can conceal useless transition prediction. Train/validation prevalence by task/depth is recorded before fitting in `results/experiment_04b/prevalence/`; final shift prevalence is measured after the freeze, never used to set class weights.",
        "",
        "| Event | D AUROC, 95% interval | D AP, 95% interval | AP difference versus scalar CV, 95% interval |",
        "|---|---:|---:|---:|",
    ]
    for ev in ("WR", "RW"):
        r = fid["paired_event_intervals"][ev]
        lines.append(
            f"| {ev} | {interval(r['auroc_selected_D'], 1)} | {interval(r['average_precision_selected_D'], 1)} | {interval(r['average_precision_D_minus_scalar_CV'], 1)} |"
        )
    lines += [
        "",
        "These intervals resample independent problems and parent seeds jointly, preserving all depths of a problem. Unconditional discrimination can partly identify whether the current answer is right. The conditional tests below use ground truth only to define evaluation subsets, never as a runtime input.",
        "",
        "| D conditional diagnostic | Event prevalence | AUROC | AP |",
        "|---|---:|---:|---:|",
    ]
    for ev, r in trans["selected_D"]["conditional_ranking"].items():
        lines.append(
            f"| {ev} | {r['prevalence']:.3f} | {r['auroc']:.3f} | {r['average_precision']:.3f} |"
        )
    lines += [
        "",
        "| D event | Precision at 0.5 | Recall at 0.5 | Fraction selected | Precision at top 10% | Recall at top 10% | ECE | Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ev, r in trans["selected_D"]["events"].items():

        def number(v):
            return "undefined (no predictions)" if v is None else f"{v:.3f}"

        lines.append(
            "| "
            + ev
            + " | "
            + " | ".join(
                number(r[k])
                for k in (
                    "precision_at_0.5",
                    "recall_at_0.5",
                    "selected_fraction_at_0.5",
                    "precision_at_top_10pct",
                    "recall_at_top_10pct",
                    "ece",
                    "brier",
                )
            )
            + " |"
        )
    lines += [
        "",
        "Top-10% is a prespecified descriptive coverage diagnostic, not a test-tuned deployment threshold. Proper four-class NLL/Brier and raw-versus-calibrated event results are retained in the metric files. Equal-count plots expose signed-gain calibration; low ECE for rare events alone does not establish useful discrimination.",
        "",
        "![WR discrimination](figures/01_correction.png)",
        "",
        "![RW discrimination](figures/02_degradation.png)",
        "",
        "![Event calibration](figures/03_calibration.png)",
        "",
        "![Signed decision value](figures/04_decision_value.png)",
        "",
        "## Deployed accuracy, paid computation and selection",
        "",
        "| System | Fresh-ID accuracy | Mean paid loops | Mean selected depth |",
        "|---|---:|---:|---:|",
        f"| Frozen | {percent(fid['baselines']['frozen']['estimate'])} | 0 | 0 |",
        f"| One pass | {percent(fid['baselines']['one_pass']['estimate'])} | 1 | 1 |",
    ]
    for name in [
        "fixed_1",
        "fixed_2",
        "fixed_3",
        "fixed_4",
        "fixed_selected",
        "confidence",
        "value",
        "legacy_gain",
        "stack_C",
        "stack_V",
        "stack_D",
        "stack_CV",
        "stack_CD",
        "stack_VD",
        "stack_CVD",
        "d_direct",
        "d_utility",
        "matched_d_direct",
    ]:
        r = pol[name]["final"]
        lines.append(
            f"| {name} + final | {percent(r['accuracy']['estimate'])} | {r['mean_paid_loops']:.3f} | {r['mean_selected_depth']:.3f} |"
        )
    for name, selector in [
        ("fixed_4", "max_C"),
        ("fixed_4", "ranker_C"),
        ("fixed_4", "ranker_CV"),
        ("fixed_4", "ranker_CD"),
        ("fixed_4", "ranker_CVD"),
        ("d_direct", "max_C"),
        ("d_direct", "ranker_C"),
        ("d_direct", "ranker_CV"),
        ("d_direct", "ranker_CD"),
        ("d_direct", "ranker_CVD"),
        ("stack_CVD", "ranker_CVD"),
    ]:
        r = pol[name][selector]
        lines.append(
            f"| {name} + {selector} | {percent(r['accuracy']['estimate'])} | {r['mean_paid_loops']:.3f} | {r['mean_selected_depth']:.3f} |"
        )
    lines += [
        "",
        f"Validation-selected combination: {percent(fid['baselines']['validation_best_combination']['estimate'])} at {fid['baselines']['validation_best_mean_paid_loops']:.3f} paid loops.",
        "",
        "| Paired comparison | Difference [95% interval], percentage points |",
        "|---|---:|",
    ]
    for label, r in delta.items():
        lines.append(f"| {label} | {interval(r)} |")
    lines += [
        "",
        "Matched controls randomize adjacent fixed depths from a stable ID hash and the adaptive mean paid depth, without labels, task identity, or per-problem controller outputs. The mean uses final paid work only as a descriptive compute match, frozen in the algorithm before final labels opened. Finite-sample realized means can differ slightly and are explicitly recorded. Controllers carry additional head/synchronization cost, so matched loops do not imply matched latency.",
        "",
        f"The best-visited oracle after D stopping reaches {percent(pol['d_direct']['oracle']['accuracy']['estimate'])} at {d['mean_paid_loops']:.3f} loops; the deployable CVD selector leaves {100 * delta['oracle_gap']['estimate']:.2f} points of that opportunity unexploited. The best-allowed four-depth oracle reaches {percent(pol['fixed_4']['oracle']['accuracy']['estimate'])}, charged four loops. Both use supervised correctness and are privileged diagnostics. A selected depth of two after visiting four states still costs four states.",
        "",
        "![Paid loops](figures/05_paid_loops.png)",
        "",
        "![Selectors](figures/07_selectors.png)",
        "",
        "## Full native MLX timing and memory",
        "",
        "Timing reuses 4A’s right-padded backbone, live state-feature, synchronization, and active-compaction mechanics. The measured path includes tokenization, input arrays, frozen backbone, candidate projection, needed C/V/D heads, native calibration and event policies, selector scores, host continuation barriers, final Metal synchronization, and output conversion. Model loading is excluded. Auxiliary timings start from h/base already on device. Three warmups and five repeats cover 24 task-balanced validation problems at batches 1 and 8, across three parents. Variant order is randomized; every repeat is checked against cached label-free paid/selected depths and logits. No bootstrap runs during timing.",
        "",
        "| Variant | Full batch-1 ms/item | Full batch-8 ms/item | Auxiliary batch-1 ms | Timed paid loops |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in (
        "frozen",
        "one_pass",
        "fixed_1",
        "fixed_2",
        "fixed_3",
        "fixed_4",
        "confidence",
        "value",
        "d_direct",
        "d_utility",
        "stack_CV",
        "stack_CD",
        "stack_CVD",
        "fixed_4:max_C",
        "fixed_4:ranker_CVD",
        "d_direct:max_C",
        "d_direct:ranker_CVD",
        "stack_CVD:ranker_CVD",
        "validation_best",
    ):
        a, b = latency["summary"]["1"][name], latency["summary"]["8"][name]
        lines.append(
            f"| {name} | {a['amortized_ms_per_item']:.2f} | {b['amortized_ms_per_item']:.2f} | {a['aux_mean_batch_median_ms']:.3f} | {a['mean_paid_loops']:.3f} |"
        )
    lines += [
        "",
        f"Peak recorded Metal allocation is {peak:.3f} GiB; peak recorded extraction/timing/report process RSS is {rss:.3f} GiB, under the 48-GiB configurable budget. Full timing command: {latency['command_seconds']:.1f} seconds; final extraction: {extraction['command_seconds']:.1f} seconds; neural fitting: {sum(i['training_seconds'] for i in infos):.1f} seconds. Implementation/reporting time is additional. Timed paid depths can differ from final average depths because timing uses validation examples. The accuracy-versus-latency figure therefore pairs independent final accuracy with validation latency and labels that distinction. It is not a final-set latency claim. Prompt-length distributions and native host barriers remain in the raw timing records.",
        "",
        "![Full measured latency](figures/06_full_latency.png)",
        "",
        "## Generalization and error analysis",
        "",
        "| Split | Frozen | One pass | D stop | D + ranker | Paid D loops | D versus one pass [95% points] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, s in summary.items():
        p = s["policies"]["d_direct"]
        lines.append(
            f"| {SHORT[split]} | {percent(s['baselines']['frozen']['estimate'])} | {percent(s['baselines']['one_pass']['estimate'])} | {percent(p['final']['accuracy']['estimate'])} | {percent(p['ranker_CVD']['accuracy']['estimate'])} | {p['final']['mean_paid_loops']:.3f} | {interval(s['paired_accuracy_differences']['D_minus_one_pass'])} |"
        )
    lines += [
        "",
        "Task-level metrics, difficulty strata, transition depth, event prevalence, and all classifier/selector combinations are preserved in the per-seed JSON results. Structural and harder-depth shifts are distinct from new wording. A numerical-range split with unchanged pointer ranges should not be described as universal range robustness.",
        "",
        "![Generalization by task](figures/09_task_generalization.png)",
        "",
        "![Event transfer](figures/11_event_generalization.png)",
        "",
        "| Category | Occurrences, mean per parent | Rate | Mean C | Predicted D | Actual D | Selected depth | Paid loops |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, r in fid["failure_categories"].items():
        vals = [f"{r['count']:.1f}", percent(r["frequency"])] + [
            "—" if r[k] is None else f"{r[k]:.3f}"
            for k in (
                "mean_C",
                "mean_predicted_D",
                "mean_actual_D",
                "mean_selected_depth",
                "mean_paid_loops",
            )
        ]
        lines.append("| " + category + " | " + " | ".join(vals) + " |")
    lines += [
        "",
        "A=WR; B=RW; C=WW with any positive loss gain; D=WW with any negative loss gain; E=correct stopped state whose next allowed loop would damage it; F=damage on a visited continuation; G=correct earlier state replacing wrong final visited state; H=wrong earlier state despite any correct later visited state. A–D rates divide all three eligible transitions per problem. E–H count events per problem; a problem can contain multiple F events. A–D diagnostics include unvisited transitions only offline; controller/selection remain visited-only. For G/H the D diagnostic refers to the selected state’s next eligible transition. The broad table is recomputed from frozen public traces in supplemental_analysis.json; raw per-seed metrics also preserve the meaningful ±0.01-nat C/D subgroups and the narrower wrong-earlier/correct-final H subgroup. Broad H includes later intermediate correct states even if the final answer is wrong.",
        "",
        "All rows in this failure table refer to direct D plus the prespecified CVD ranker. Representative examples use the lexicographically earliest instance IDs within each category and parent/split, rather than choosing dramatic outcomes. Public prompts, choices, logits, correctness/loss traces, controller predictions, paid loops, selected depth, and stop/continue logs are saved in `results/experiment_04b/examples/`. They illustrate mechanisms and do not prove aggregate performance.",
        "",
        "![Deterministic recovery examples](figures/08_recovery_examples.png)",
        "",
        "![Errors and paired uncertainty](figures/10_errors_uncertainty.png)",
        "",
        "## Answers to the research questions",
        "",
        f"1. **Can D predict WR corrections?** Fresh-ID AUROC/AP is {trans['selected_D']['events']['WR']['auroc']:.3f}/{trans['selected_D']['events']['WR']['average_precision']:.3f}, compared with WR prevalence {fid['class_prevalence'][1]:.3f}. {notes[0]}",
        f"2. **Can D predict RW damage?** Fresh-ID AUROC/AP is {trans['selected_D']['events']['RW']['auroc']:.3f}/{trans['selected_D']['events']['RW']['average_precision']:.3f}; prevalence is {fid['class_prevalence'][2]:.3f}. {notes[1]}",
        f"3. **Beyond C and V?** Event tables compare all seven stacks; the paired CVD-minus-CV WR/RW AP differences are {interval(fid['paired_event_intervals']['WR']['average_precision_CVD_minus_CV'], 1)} and {interval(fid['paired_event_intervals']['RW']['average_precision_CVD_minus_CV'], 1)}. D minus the stronger scalar entropy/margin/depth/change control gives WR/RW AP differences {interval(fid['paired_event_intervals']['WR']['average_precision_D_minus_scalar_all'], 1)} and {interval(fid['paired_event_intervals']['RW']['average_precision_D_minus_scalar_all'], 1)}. The plain parameter-matched classifier is competitive; this does not establish a gated-architecture advantage. These are predictive information tests, not causal attribution.",
        f"4. **Does D stopping improve accuracy?** {stopping_verdict} Difference versus one pass: {interval(delta['D_minus_one_pass'])} points.",
        f"5. **Does it prevent harmful recurrence?** Per 1,296 fresh-ID problems (mean across parents), stopping avoids {behavior['avoided_RW_by_stopping']:.1f} reachable damage events and also misses {behavior['missed_WR_by_stopping']:.1f} reachable corrections. Continued steps deliver {behavior['continued_WR']:.1f} corrections and {behavior['continued_RW']:.1f} degradations, almost cancelling. Immediate damage is protected in category E ({percent(fid['failure_categories']['E']['frequency'])}); visited damage remains in F ({percent(fid['failure_categories']['F']['frequency'])} occurrences per problem). Aggregate accuracy and matched controls determine utility.",
        f"6. **Can selection recover earlier answers?** Yes when category G occurs: {percent(fid['failure_categories']['G']['frequency'])} of problems in the primary path. It also loses a correct final answer on {percent(lost_final)}. Broader category H misses a correct later visited state on {percent(fid['failure_categories']['H']['frequency'])}; some of those later correct states are intermediate rather than final.",
        f"7. **Does it beat the final state?** {selector_verdict} Paired difference: {interval(delta['selector_minus_final'])} points. After four paid loops, CVD ranker versus final is {interval(delta['fixed4_selector_minus_final'])} points; versus simple maximum-C it is {interval(delta['fixed4_selector_minus_max_C'])}. Extra V/D signals versus the equally sized C ranker give {interval(delta['fixed4_CVD_minus_ranker_C'])} points.",
        f"8. **Does it beat one pass?** D plus ranker minus one pass is {interval(delta['selector_minus_one_pass'])} points.",
        f"9. **At matched compute?** D versus its adjacent-depth mixture is {interval(delta['D_minus_matched'])} points; D plus ranker versus the mixture plus the same ranker is {interval(delta['selector_minus_matched_selector'])} points. Measured full latency includes overhead.",
        "10. **New templates and structures?** The five-split table and task heatmap report transfer without changing any threshold. Claims must be bounded by the structural and harder-depth results, not the old synthetic pilot.",
        f"11. **How close to the visited oracle?** The primary selector remains {100 * delta['oracle_gap']['estimate']:.2f} points below its privileged visited-state oracle.",
        "12. **What is worth retaining?** Retain the independent lockbox, transition labels, causal visited-prefix controls, grouped OOF procedure, full timing/work accounting, and simple one-pass reference. Predictive event heads merit further investigation only where their held-out signal survives strong scalar controls. Avoid adding state memory in this core study; no deployable reasoning or architectural novelty claim follows from these probes.",
        "",
        "## Limitations and publication readiness",
        "",
        "These are six public synthetic algorithms around one small frozen backbone and three historical refiners. Three seeds give weak estimates of training variability; the parent representations were learned on older synthetic templates. Fresh auxiliary supervision is local to the 4A training families. Conditional four-choice accuracy does not measure open-ended generation. No natural-language benchmark, new backbone scale, model retraining, learned history attention, or out-of-budget depth is tested. Utility uses a chosen normalized-loop preference, not measured latency costs. Oracle access to labels and complete future trajectories is unavailable at deployment. Selecting states is not evidence of reasoning, and better probe confidence is not necessarily better decisions.",
        "",
        "This is ready for public review as a bounded synthetic experiment with preserved negative results and a prospective final protocol. It is not sufficient for broad language-model reasoning or architectural novelty claims. All code, recipes, public datasets, trajectory labels, prediction/policy logs, confidence intervals, figures and portable audits are included; model weights and large hidden caches remain excluded. No personal paths, credentials, private company data, or external unlicensed implementation were added.",
        "",
        "## Reproduction",
        "",
        'Use Python 3.12 and the pinned `requirements-lock.txt`; install the package with `pip install -e ".[apple,test]"`. A compatible 4A local workspace is required for frozen checkpoints and development hidden caches. To regenerate those from scratch, follow the preserved [4A handoff](../experiment_04a/experiment_04b_handoff.md) and its staged reproduction commands first; do not substitute consumed 4A final data for the new 4B lockbox.',
        "",
        "```sh",
        ".venv/bin/python -m pytest -q",
        "LATENT_LOOP_MLX_TESTS=1 .venv/bin/python -m pytest -q",
        ".venv/bin/python -m latent_loop.experiment04b.pipeline prepare --restart-budget",
        ".venv/bin/python -m latent_loop.experiment04b.pipeline development --restart-budget",
        ".venv/bin/python -m latent_loop.experiment04b.pipeline freeze",
        ".venv/bin/python -m latent_loop.experiment04b.pipeline final --restart-budget",
        ".venv/bin/python -m latent_loop.experiment04b.pipeline timing --restart-budget",
        ".venv/bin/python -m latent_loop.experiment04b.pipeline report",
        ".venv/bin/python -m latent_loop.experiment04b.pipeline verify",
        "```",
        "",
        "The checked-in protocol is already frozen and consumed: development refuses reselection after consumption. For an independent rerun, use a new copy/run namespace with fresh independent final seeds and new protocol manifests; keep the recorded snapshot intact. Budget restart changes only the runtime clock, never the scientific protocol. Existing completed public results can be audited and reports regenerated with CPU scientific dependencies alone; the public audit does not require weights or hidden caches. Fits verify source/config/input hashes before resume; extraction verifies dataset/model signatures, resumes batch shards, and rejects mismatched caches. Timing preserves complete batches and verifies its protocol/source/config signature on resume. Safe interrupts stop at a batch/epoch boundary. Reporting never changes frozen model or policy choices.",
        "",
        f"Portable audit rechecked {audit_value['verified_unique_final_problems']:,} final problems, {audit_value['new_state_rows']:,} new trajectory states, {audit_value['new_adjacent_transitions']:,} adjacent labels, and {audit_value['public_metric_checks']:,} metric comparisons. OOF integrity, visited-state boundaries, and every stop/continue log are checked. The new exports reuse 4A’s trajectory schema; the 60,480 development states remain referenced in the compatibility audit rather than duplicated.",
        "",
        "## Prior work and provenance",
        "",
        "The preserved [literature review](../../docs/literature_review.md) includes [RecurTrace](https://arxiv.org/html/2609.03379v2), [Universal Transformers](https://arxiv.org/abs/1807.03819), [ACT](https://arxiv.org/abs/1603.08983), [PonderNet](https://arxiv.org/abs/2107.05407), Encode-Think-Decode, Think-at-Hard, uncertainty probes and process supervision. This continues an independent final-vector adaptation; it is not an official RecurTrace reproduction. Probability calibration follows the general post-hoc calibration question studied by [Guo et al., On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599); our temperature-plus-offset fit is broader than scalar temperature scaling. These citations motivate methods and do not provide empirical evidence for the present results. Qwen model provenance and Apache-2.0 terms remain in the preserved third-party license document; original code and generated synthetic artifacts use the repository MIT license.",
        "",
        "Source measurements: [summary](../../results/experiment_04b/summary.json), [public audit](../../results/experiment_04b/public_audit.json), [latency](../../results/experiment_04b/latency.json), [protocol](../../experiments/experiment_04b/frozen_protocol.json), [dataset manifest](../../experiments/experiment_04b/dataset_manifest.json).",
        "",
    ]
    text = "\n".join(lines)
    (REPORT / "decision_value_and_trajectory_selection.md").write_text(text)
    htmltext = markdown.markdown(text, extensions=["tables", "fenced_code"])
    for path in sorted((REPORT / "figures").glob("*.png")):
        htmltext = htmltext.replace(
            f'src="figures/{path.name}"',
            f'src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"',
        )
    document = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Experiment 4B: decision value</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#223247}table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}th,td{border:1px solid #c9d1d9;padding:6px 9px}img{max-width:100%;height:auto}pre{overflow-x:auto;background:#f3f5f7;padding:16px}h1,h2{line-height:1.2}a{color:#245e8f}</style><body>'
        + htmltext
        + "</body></html>"
    )
    (REPORT / "decision_value_and_trajectory_selection.html").write_text(document)
    exec_lines = [
        "# Experiment 4B executive summary",
        "",
        f"{measured} {stopping_verdict} {selector_verdict}",
        "",
        f"The third head ranks actual corrections (WR AUROC {trans['selected_D']['events']['WR']['auroc']:.3f}, AP {trans['selected_D']['events']['WR']['average_precision']:.3f}) and harmful loops (RW AUROC {trans['selected_D']['events']['RW']['auroc']:.3f}, AP {trans['selected_D']['events']['RW']['average_precision']:.3f}). Event prevalence is {fid['class_prevalence'][1]:.3f}/{fid['class_prevalence'][2]:.3f}. {notes[0]} {notes[1]}",
        "",
        f"D stopping scores {percent(d['accuracy']['estimate'])}; earlier-state selection scores {percent(ds['accuracy']['estimate'])}. One pass scores {percent(fid['baselines']['one_pass']['estimate'])}, frozen {percent(fid['baselines']['frozen']['estimate'])}. A single recurrent step scores {percent(pol['fixed_1']['final']['accuracy']['estimate'])}; D versus it is {interval(delta['D_minus_fixed1'])} points. Selection recovers {percent(fid['failure_categories']['G']['frequency'])} of problems and loses a correct final answer on {percent(lost_final)}. Its net difference versus the same visited final state is {interval(delta['selector_minus_final'])} percentage points. A model that selects earlier still pays all {d['mean_paid_loops']:.3f} visited loops.",
        "",
        f"Full live inference is {latency['summary']['1']['d_direct']['amortized_ms_per_item']:.2f} ms/item for D and {latency['summary']['1']['d_direct:ranker_CVD']['amortized_ms_per_item']:.2f} ms/item for D plus selector, versus {latency['summary']['1']['one_pass']['amortized_ms_per_item']:.2f} ms/item for one pass on the same balanced validation subset. Batch-8 amortized latencies are {latency['summary']['8']['d_direct']['amortized_ms_per_item']:.2f}/{latency['summary']['8']['d_direct:ranker_CVD']['amortized_ms_per_item']:.2f}/{latency['summary']['8']['one_pass']['amortized_ms_per_item']:.2f} ms/item. Peak recorded Metal/RSS: {peak:.3f}/{rss:.3f} GiB.",
        "",
        "The confirmatory results are the prespecified fresh-ID D/one-pass, selector/final, and WR/RW AP/scalar-CV contrasts. Shift, alternate stacks/classifiers, utility, selectors, and oracles are secondary/descriptive. All thresholds, calibrators, checkpoints and validation winners were sealed before 6,480 new final problems were scored. Paired intervals cluster problem instances and parent seeds; three parents and unadjusted multiple comparisons limit inference.",
        "",
        "| Split | One pass | D stop | D + selector |",
        "|---|---:|---:|---:|",
    ]
    for split, s in summary.items():
        exec_lines.append(
            f"| {SHORT[split]} | {percent(s['baselines']['one_pass']['estimate'])} | {percent(s['policies']['d_direct']['final']['accuracy']['estimate'])} | {percent(s['policies']['d_direct']['ranker_CVD']['accuracy']['estimate'])} |"
        )
    exec_lines += [
        "",
        "Completed experiment count: **36 supervised neural fits**, 27 main and 9 grouped OOF; eight substantive neural conditions across **three parent seeds**, plus random/prevalence/statistical controls. Best D architecture by validation NLL: ordinary CE gated four-class head in all parents. Best policy/selector is the per-parent validation-selected procedure below; it is not chosen by final accuracy.",
        "",
    ]
    for s in selected:
        b = s["validation_best_combination"]
        exec_lines.append(
            f"- Seed {s['seed']}: `{b['policy']}` with `{b['selector']}`; D direct threshold {s['rules']['d_direct']['threshold']}."
        )
    exec_lines += [
        "",
        f"That validation-selected procedure scores {percent(fid['baselines']['validation_best_combination']['estimate'])}, paying {fid['baselines']['validation_best_mean_paid_loops']:.3f} loops. The primary deployable selector remains {100 * delta['oracle_gap']['estimate']:.2f} points behind its best-visited oracle. D versus a label-free matched-loop control: {interval(delta['D_minus_matched'])} points.",
        "",
        "The most important failure is the gap between transition prediction and useful deployed decisions: preventing an observed damaging loop can also forgo a later correction, and an earlier-state ranker can replace good answers with wrong ones. Check the paired accuracy and structural-transfer tables before claiming a better decision-maker. More computation is justified only where the measured accuracy gain and full runtime meet an explicit application preference; a probe score alone cannot decide that.",
        "",
        "Ready for public review as a bounded synthetic research result. Keep the transition dataset, prospective lockbox, OOF/causality checks, work accounting and one-pass control. State memory was not launched. This does not demonstrate new reasoning, broad benchmark gains, or architectural novelty. Negative results and every A–H category are preserved.",
        "",
        f"Branch: `research/exp04-decision-value`; base commit `{config['start_commit']}`. {commit_note} No remote was created or pushed. All historical experiments are preserved.",
        "",
        "Review: [full report](decision_value_and_trajectory_selection.md), [self-contained HTML](decision_value_and_trajectory_selection.html), [measured summary](../../results/experiment_04b/summary.json), [portable audit](../../results/experiment_04b/public_audit.json).",
        "",
    ]
    (REPORT / "executive_summary.md").write_text("\n".join(exec_lines))


def supplemental(config, data):
    """Required broad failure definitions recomputed from frozen public traces, no choices changed."""
    result = {}
    examples = []
    for split, xx in data.items():
        rows = [
            json.loads(line)
            for line in gzip.open(OUT / "datasets" / f"{split}.jsonl.gz", "rt")
        ]
        per_seed = []
        for seed, x in zip(config["parent_seeds"], xx, strict=True):
            chosen = read(EXP / f"selection_seed{seed}.json")
            p = x["paths"]["d_direct"]
            paid, selected = p["paid"], p["ranker_CVD"]
            y = outcomes(x["correct"])
            actual = (y == 1).astype(int) - (y == 2)
            prob = x["transition_probabilities"][chosen["D_model"]]
            dv = prob[:3, :, 1] - prob[:3, :, 2]
            gain = x["loss"][:-1] - x["loss"][1:]
            n = len(paid)
            index = np.arange(n)
            final = x["correct"][paid - 1, index]
            selected_correct = x["correct"][selected - 1, index]
            later_correct = np.any(
                x["correct"]
                & (np.arange(1, 5)[:, None] > selected)
                & (np.arange(1, 5)[:, None] <= paid),
                axis=0,
            )
            masks = {
                "A": y == 1,
                "B": y == 2,
                "C": (y == 0) & (gain > 0),
                "D": (y == 0) & (gain < 0),
                "E": (y == 2) & (np.arange(1, 4)[:, None] == paid),
                "F": (y == 2) & (np.arange(1, 4)[:, None] < paid),
                "G": (selected_correct & ~final & (selected < paid))[None],
                "H": (~selected_correct & later_correct)[None],
            }
            category = {}
            for c, mask in masks.items():
                kk, ii = np.where(mask)
                if c in ("G", "H"):
                    kk = selected[ii] - 1
                assert np.all(kk < 3) or c not in ("G", "H")
                category[c] = {
                    "count": int(len(ii)),
                    "frequency": float(len(ii) / (3 * n if c in "ABCD" else n)),
                    "unique_problem_fraction": float(len(set(ii)) / n),
                    **{
                        key: float(value.mean()) if len(ii) else None
                        for key, value in {
                            "mean_C": x["C"][kk, ii],
                            "mean_predicted_D": dv[np.minimum(kk, 2), ii],
                            "mean_actual_D": actual[np.minimum(kk, 2), ii],
                            "mean_selected_depth": selected[ii],
                            "mean_paid_loops": paid[ii],
                        }.items()
                    },
                }
                for z in sorted(
                    range(len(ii)), key=lambda j: (rows[ii[j]]["id"], int(kk[j]))
                )[:3]:
                    i, k = int(ii[z]), int(kk[z])
                    examples.append(
                        {
                            "category": c,
                            "split": split,
                            "seed": seed,
                            "instance_id": rows[i]["id"],
                            "prompt": rows[i]["prompt"],
                            "choices": rows[i]["choices"],
                            "gold_label": int(x["label"][i]),
                            "event_depth": k + 1,
                            "correct_by_depth": x["correct"][:, i].tolist(),
                            "predictions_by_depth": x["logits"][:, i]
                            .argmax(-1)
                            .tolist(),
                            "loss_by_depth": x["loss"][:, i].tolist(),
                            "C_by_depth": x["C"][:, i].tolist(),
                            "predicted_D_by_depth": dv[:, i].tolist(),
                            "actual_D_by_depth": actual[:, i].tolist(),
                            "selected_depth": int(selected[i]),
                            "paid_loops": int(paid[i]),
                            "definition_version": "Broad user definitions: any signed loss change, any correct later visited state",
                            "example_choice": "earliest stable instance ID per category/parent/split",
                        }
                    )
            reached = np.arange(1, 4)[:, None] <= paid
            go = np.arange(1, 4)[:, None] < paid
            behavior = {
                "reachable_WR": int(((y == 1) & reached).sum()),
                "continued_WR": int(((y == 1) & go).sum()),
                "missed_WR_by_stopping": int(((y == 1) & reached & ~go).sum()),
                "reachable_RW": int(((y == 2) & reached).sum()),
                "continued_RW": int(((y == 2) & go).sum()),
                "avoided_RW_by_stopping": int(((y == 2) & reached & ~go).sum()),
                "useful_loss_events": int((gain > 0.01).sum()),
                "WR_among_useful_loss": int(((gain > 0.01) & (y == 1)).sum()),
                "WW_among_useful_loss": int(((gain > 0.01) & (y == 0)).sum()),
            }
            per_seed.append(
                {
                    "seed": seed,
                    "failure_categories": category,
                    "stopping_behavior": behavior,
                }
            )
        aggregate = {
            c: {
                key: float(np.mean([p["failure_categories"][c][key] for p in per_seed]))
                if all(p["failure_categories"][c][key] is not None for p in per_seed)
                else None
                for key in per_seed[0]["failure_categories"][c]
            }
            for c in "ABCDEFGH"
        }
        result[split] = {
            "per_seed": per_seed,
            "failure_categories": aggregate,
            "behavior_mean_counts": {
                k: float(np.mean([p["stopping_behavior"][k] for p in per_seed]))
                for k in per_seed[0]["stopping_behavior"]
            },
        }
    atomic_json(
        OUT / "supplemental_analysis.json",
        {
            "completed_utc": utc(),
            "splits": result,
            "status": "Descriptive recomputation of frozen public predictions; no model, threshold or selector choice changed.",
            "C_D_definition": "Any strictly positive/negative next-loop loss difference for WW, rather than the 0.01-nat meaningful-change subgroups in raw metric files.",
            "H_definition": "Wrong earlier selected state despite any correct later visited state; includes correct intermediate later states even if final is wrong.",
        },
    )
    atomic_json(OUT / "examples" / "broad_categories.json", examples)
    return result


def archive_metrics():
    """Lossless deterministic public packaging; scientific measurements are unchanged."""
    import hashlib

    previous = OUT / "metrics_archive_manifest.json"
    entries = read(previous)["entries"] if previous.exists() else {}
    for path in sorted((OUT / "metrics").glob("*.json")):
        content = path.read_bytes()
        compressed = Path(str(path) + ".gz")
        with gzip.GzipFile(filename=str(compressed), mode="wb", mtime=0) as stream:
            stream.write(content)
        with gzip.open(compressed, "rb") as stream:
            assert stream.read() == content
        entries[path.as_posix()] = {
            "compressed_path": compressed.as_posix(),
            "compressed_sha256": sha(compressed),
            "original_content_sha256": hashlib.sha256(content).hexdigest(),
            "original_bytes": len(content),
        }
        path.unlink()
    for entry in entries.values():
        assert sha(entry["compressed_path"]) == entry["compressed_sha256"]
        with gzip.open(entry["compressed_path"], "rb") as stream:
            assert (
                hashlib.sha256(stream.read()).hexdigest()
                == entry["original_content_sha256"]
            )
    atomic_json(
        previous,
        {
            "entries": entries,
            "format": "Lossless gzip of exact recorded UTF-8 JSON; read helper resolves logical .json paths to .json.gz.",
        },
    )
