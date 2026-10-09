"""CPU-only audits, paired uncertainty, measured figures, and a local research report."""

import base64
import datetime
import gzip
import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp, softmax

from latent_loop.experiment04a.data import TASKS, reference as solve, sha
from latent_loop.experiment04a.science import bootstrap
from latent_loop.experiment05.data import EXP, FINAL, OUT
from latent_loop.training import atomic_json

REPORT = Path("reports/experiment_05")
CONDITIONS = ("one_pass", "one_pass_compute", "recurrent", "reinjection", "multi_depth")


def read_gzip(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def trace(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def cube(rows, field="logits"):
    depths = sorted(map(int, rows[0]["depths"]))
    return np.array([[r["depths"][str(k)][field] for r in rows] for k in depths])


def statistic(logits, labels):
    p = softmax(logits, axis=-1)
    correct = logits.argmax(-1) == labels
    confidence = p.max(-1)
    ece = 0.0
    for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (confidence >= a) & (confidence < b if b < 1 else confidence <= b)
        if mask.any():
            ece += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return {"accuracy": float(correct.mean()), "nll": float(np.mean(logsumexp(logits, axis=-1) - logits[np.arange(len(labels)), labels])),
            "ece": float(ece), "brier": float(np.mean(np.sum((p - np.eye(4)[labels])**2, axis=-1)))}


def audit(config):
    records, datasets, checked = 0, 0, 0
    final_ids = set()
    expected_labels = {}
    split_rows = {}
    sources = {}
    manifest = json.loads((EXP / "dataset_manifest.json").read_text())
    frozen = json.loads((EXP / "frozen_protocol.json").read_text())
    for name, expected in frozen["source_hashes"].items():
        assert sha(name) == expected, f"Frozen source changed: {name}"
    for condition, seeds in frozen["fits"].items():
        for seed, record in seeds.items():
            p = OUT / "training" / f"main_{condition}_seed{seed}.json"
            assert sha(p) == record["training_record_sha256"]
            sources[str(p)] = sha(p)
            fit = json.loads(p.read_text())
            epochs = config["compute_control_epochs"] if condition == "one_pass_compute" else config["epochs"]
            assert fit["complete"] and fit["examples_processed"] == config["sizes"]["train"] * epochs
            assert fit["signature"]["config_sha256"] == sha(EXP / "config.json")
            assert len(fit["epochs"]) == epochs and fit["best_epoch"] == min(fit["epochs"], key=lambda r: r["validation_mean_nll"])["epoch"]
            assert len(fit["batch_records"]) == epochs * int(np.ceil(config["sizes"]["train"] / config["batch_size"]))
            assert all(np.isfinite([b["loss"], b["gradient_norm"], b["seconds"]]).all() and b["seconds"] > 0 for b in fit["batch_records"])
    matched_prefixes = {}
    for seed in config["seeds"]:
        shorter = json.loads((OUT / "training" / f"main_one_pass_seed{seed}.json").read_text())
        longer = json.loads((OUT / "training" / f"main_one_pass_compute_seed{seed}.json").read_text())
        for a, b in zip(shorter["batch_records"], longer["batch_records"]):
            assert (a["epoch"], a["batch"], a["depths"]) == (b["epoch"], b["batch"], b["depths"])
            assert a["loss"] == b["loss"] and a["gradient_norm"] == b["gradient_norm"]
        assert all(a["validation_by_depth"] == b["validation_by_depth"] for a, b in zip(shorter["epochs"], longer["epochs"]))
        matched_prefixes[str(seed)] = len(shorter["batch_records"])
    for name in ("final_execution.json", "latency.json"):
        assert json.loads((OUT / name).read_text())["protocol_sha256"] == sha(EXP / "frozen_protocol.json")
    prior = json.loads((EXP / "class_prior_diagnostic.json").read_text())
    assert prior["recorded_utc"] < frozen["frozen_utc"]
    for split in ("train", "val", *FINAL):
        p = OUT / "datasets" / f"{split}.jsonl.gz"
        rows = trace(p)
        split_rows[split] = rows
        datasets += len(rows)
        assert len(rows) == config["sizes"][split]
        assert sha(p) == manifest["splits"][split]["sha256"]
        for row in rows:
            assert row["id"] not in expected_labels, "Cross-split instance overlap"
            expected_labels[row["id"]] = (row["label"], row["task"])
            answer = solve(row["generator_id"], row["spec"])
            text = row["entities"][answer] if isinstance(answer, int) else answer
            assert row["choices"][row["label"]] == text
            assert len(set(row["choices"])) == 4
            if split in FINAL:
                assert row["id"] not in final_ids
                final_ids.add(row["id"])
        sources[str(p)] = sha(p)
    training_labels = np.array([r["label"] for r in split_rows["train"]])
    assert prior["training_counts"] == {str(k): int((training_labels == k).sum()) for k in range(4)}
    assert np.array_equal(np.array(prior["fixed_probabilities"]), np.bincount(training_labels, minlength=4) / len(training_labels))
    expected_paths = {}
    baseline_by_id = {}
    for split in FINAL:
        rows = split_rows[split]
        baseline_path = OUT / "baselines" / f"{split}.json.gz"
        base = read_gzip(baseline_path)
        assert base["ids"] == [r["id"] for r in rows]
        assert base["labels"] == [r["label"] for r in rows]
        baseline_by_id.update(zip(base["ids"], base["frozen_logits"]))
        sources[str(baseline_path)] = sha(baseline_path)
        for c in ("untrained_plain", "untrained_bounded"):
            expected_paths[OUT / "trajectories" / f"{c}_{split}.jsonl.gz"] = (base["ids"], [1, 2, 3, 4])
        for seed in config["seeds"]:
            for c in CONDITIONS:
                expected_paths[OUT / "trajectories" / f"{c}_{split}_seed{seed}.jsonl.gz"] = (base["ids"], [1] if c.startswith("one_pass") else [1, 2, 3, 4])
            for c in ("reinjection", "multi_depth"):
                for mode in ("none", "fixed"):
                    expected_paths[OUT / "ablations" / f"{c}_{mode}_{split}_seed{seed}.jsonl.gz"] = (base["ids"], [1, 2, 3, 4])
    rows = split_rows["fresh_id"]
    selected = sorted(i for task in TASKS for i in sorted([i for i, r in enumerate(rows) if r["task"] == task], key=lambda j: rows[j]["id"])[:4])
    for c in ("untrained_plain", "untrained_bounded"):
        expected_paths[OUT / "stress" / f"{c}.jsonl.gz"] = ([rows[i]["id"] for i in selected], list(range(1, 9)))
    for seed in config["seeds"]:
        for c in ("recurrent", "reinjection", "multi_depth"):
            expected_paths[OUT / "stress" / f"{c}_seed{seed}.jsonl.gz"] = ([rows[i]["id"] for i in selected], list(range(1, 9)))
    actual_paths = {p for folder in ("trajectories", "ablations", "stress") for p in (OUT / folder).glob("*.jsonl.gz")}
    assert actual_paths == set(expected_paths), "Missing or unexpected trajectory files"
    for folder in ("trajectories", "ablations", "stress"):
        for p in sorted((OUT / folder).glob("*.jsonl.gz")):
            sources[str(p)] = sha(p)
            rows = trace(p)
            ids, depths = expected_paths[p]
            assert [r["id"] for r in rows] == ids
            for row in rows:
                assert expected_labels[row["id"]] == (row["label"], row["task"])
                assert sorted(map(int, row["depths"])) == depths
                assert row["initial_logits"] == baseline_by_id[row["id"]]
                initial = np.array(row["initial_logits"])
                assert row["initial_prediction"] == int(initial.argmax())
                assert np.isclose(row["initial_correct_answer_probability"], softmax(initial)[row["label"]], atol=2e-6)
                for k, r in row["depths"].items():
                    l = np.array(r["logits"])
                    probabilities = softmax(l)
                    assert r["prediction"] == int(l.argmax())
                    assert np.isclose(r["conditional_loss"], logsumexp(l) - l[row["label"]], atol=2e-6)
                    assert np.isclose(r["correct_answer_probability"], probabilities[row["label"]], atol=2e-6)
                    assert np.isclose(r["confidence"], probabilities.max(), atol=2e-6)
                    ordered = np.sort(probabilities)
                    assert np.isclose(r["margin"], ordered[-1] - ordered[-2], atol=2e-6)
                    assert np.isclose(r["entropy"], -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))), atol=2e-6)
                    assert np.isfinite([r["state_norm"], r["state_distance"], r["state_drift"]]).all()
                    checked += 1
                records += 1
    atomic_json(OUT / "public_audit.json", {"all_passed": True, "independent_labels_checked": datasets,
                                            "distinct_final_problems": len(final_ids), "trajectory_examples_checked": records,
                                            "state_metric_checks": checked, "complete_trajectory_files": len(expected_paths), "identical_one_pass_training_prefix_batches": matched_prefixes, "record_hashes": sources,
                                            "weights_or_hidden_arrays_required": False})


def summarize(config, frozen):
    result = {}
    arrays = {}
    prior = json.loads((EXP / "class_prior_diagnostic.json").read_text())
    for split in FINAL:
        rows = trace(OUT / "datasets" / f"{split}.jsonl.gz")
        labels = np.array([r["label"] for r in rows])
        base = read_gzip(OUT / "baselines" / f"{split}.json.gz")
        assert base["ids"] == [r["id"] for r in rows]
        baselogits = np.array(base["frozen_logits"])
        initial = baselogits.argmax(-1) == labels
        systems = {}
        logits_by = {}
        corrected_by = {}
        for condition in CONDITIONS:
            allrows = [trace(OUT / "trajectories" / f"{condition}_{split}_seed{seed}.jsonl.gz") for seed in config["seeds"]]
            assert all([r["id"] for r in rr] == base["ids"] for rr in allrows)
            logits = np.stack([cube(rr) for rr in allrows])
            logits_by[condition] = logits
            correct = logits.argmax(-1) == labels[None, None, :]
            corrected_by[condition] = correct
            systems[condition] = {"depths": {}}
            for k in range(logits.shape[1]):
                stats = [statistic(l[k], labels) for l in logits]
                systems[condition]["depths"][str(k + 1)] = {**{name: float(np.mean([s[name] for s in stats])) for name in stats[0]},
                                                           "accuracy_interval": bootstrap(correct[:, k], config["bootstrap_repeats"], 550201 + k),
                                                           "accuracy_seed_sd": float(correct[:, k].mean(-1).std(ddof=1)),
                                                           "accuracy_by_seed": {str(seed): float(correct[j, k].mean()) for j, seed in enumerate(config["seeds"])},
                                                           "prediction_class_frequencies": (np.bincount(logits[:, k].argmax(-1).ravel(), minlength=4).astype(float) / (logits.shape[0] * logits.shape[2])).tolist(),
                                                           "mean_option_entropy": float(np.mean([-np.sum(softmax(v[k],axis=-1)*np.log(np.maximum(softmax(v[k],axis=-1),1e-12)),axis=-1).mean() for v in logits])),
                                                           "state_norm": float(np.mean([cube(rr, "state_norm")[k].mean() for rr in allrows])),
                                                           "state_drift": float(np.mean([cube(rr, "state_drift")[k].mean() for rr in allrows]))}
            transitions = {"initial_adaptation_WR": float((~initial[None, :] & correct[:, 0]).mean()),
                           "initial_adaptation_RW": float((initial[None, :] & ~correct[:, 0]).mean())}
            if correct.shape[1] > 1:
                prev, nxt = correct[:, :-1], correct[:, 1:]
                events = {"WR": ~prev & nxt, "RW": prev & ~nxt, "WW": ~prev & ~nxt, "RR": prev & nxt}
                for name, values in events.items():
                    transitions[name] = bootstrap(values, config["bootstrap_repeats"], 550601)
                transitions["WR_given_wrong"] = float(events["WR"].sum() / max((~prev).sum(), 1))
                transitions["RW_given_right"] = float(events["RW"].sum() / max(prev.sum(), 1))
                transitions["adjacent_accuracy_differences"] = {f"{k + 1}_minus_{k}": bootstrap(correct[:, k].astype(float) - correct[:, k - 1], config["bootstrap_repeats"], 550501 + k) for k in range(1, 4)}
            systems[condition]["transitions"] = transitions
            systems[condition]["tasks"] = {}
            for task in TASKS:
                ix = np.array([i for i, r in enumerate(rows) if r["task"] == task])
                systems[condition]["tasks"][task] = {str(k + 1): float(correct[:, k, ix].mean()) for k in range(correct.shape[1])}
            systems[condition]["difficulty"] = {name: {str(k + 1): float(correct[:, k, [i for i, r in enumerate(rows) if r["difficulty_stratum"] == name]].mean()) for k in range(correct.shape[1])} for name in sorted({r["difficulty_stratum"] for r in rows})}
        selected = frozen["selected_recurrent_family"]
        chosen = corrected_by[selected][:, 3].astype(float)
        comparisons = {f"selected_minus_{control}": bootstrap(chosen - corrected_by[control][:, 0], config["bootstrap_repeats"], 550801) for control in ("one_pass", "one_pass_compute")}
        comparisons.update({f"selected_depth1_minus_{control}_matched_inference": bootstrap(corrected_by[selected][:, 0].astype(float) - corrected_by[control][:, 0], config["bootstrap_repeats"], 550801) for control in ("one_pass", "one_pass_compute")})
        comparisons["selected_minus_frozen"] = bootstrap(chosen - initial[None, :], config["bootstrap_repeats"], 550801)
        comparisons["selected_depth4_minus_depth1"] = bootstrap(chosen - corrected_by[selected][:, 0], config["bootstrap_repeats"], 550801)
        untrained = {}
        for condition in ("untrained_plain", "untrained_bounded"):
            rr = trace(OUT / "trajectories" / f"{condition}_{split}.jsonl.gz")
            l = cube(rr)
            untrained[condition] = {str(k + 1): {**statistic(l[k], labels), "state_drift": float(cube(rr, "state_drift")[k].mean())} for k in range(4)}
            comparisons[f"selected_minus_{condition}"] = bootstrap(chosen - (l[3].argmax(-1) == labels)[None, :], config["bootstrap_repeats"], 550801)
        result[split] = {"n": len(rows), "frozen": statistic(baselogits, labels),
                         "training_answer_position_prior": statistic(np.tile(np.log(prior["fixed_probabilities"]), (len(rows), 1)), labels),
                         "historical_one_pass": {name: float(np.mean([statistic(np.array(l), labels)[name] for l in base["historical_one_pass_logits"].values()])) for name in ("accuracy", "nll", "ece", "brier")},
                         "systems": systems, "untrained": untrained, "paired_comparisons": comparisons}
        arrays[split] = (rows, labels, logits_by)
    return result, arrays


def timing_summary(value):
    output = {}
    for r in value["records"]:
        key = f"{r['condition']}:d{r['depth']}:b{r['batch_size']}"
        output.setdefault(key, []).append(r)
    return {key: {"ms_per_item": float(np.mean([r["amortized_ms_per_item"] for r in rows])),
                  "timed_accuracy": float(np.mean([r["subset_accuracy"] for r in rows]))} for key, rows in output.items()}


def figures(config, summary, timed, fits, frozen, ablations, stress):
    folder = REPORT / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    def save(name):
        plt.tight_layout()
        plt.savefig(folder / (name + ".png"), dpi=160)
        plt.savefig(folder / (name + ".svg"))
        plt.close()
    fid = summary["fresh_id"]
    colors = dict(zip(CONDITIONS, ["#577590", "#273c75", "#e09f3e", "#9e548a", "#2a9d8f"]))
    plt.figure(figsize=(8, 5))
    for condition in CONDITIONS:
        d = fid["systems"][condition]["depths"]
        x = list(map(int, d)); y = [100 * d[str(k)]["accuracy"] for k in x]
        bounds = np.array([d[str(k)]["accuracy_interval"]["cluster_seed_bootstrap95"] for k in x]) * 100
        plt.errorbar(x, y, yerr=np.maximum(np.array([y - bounds[:, 0], bounds[:, 1] - y]), 0), fmt="o-", capsize=3, label=condition, color=colors[condition])
    plt.plot([1, 2, 3, 4], [100 * fid["untrained"]["untrained_plain"][str(k)]["accuracy"] for k in range(1, 5)], "--", color="gray", label="untrained plain")
    plt.axhline(100 * fid["frozen"]["accuracy"], color="black", linestyle=":", label="frozen")
    plt.xticks([1, 2, 3, 4]);plt.xlabel("Recurrent depth");plt.ylabel("Fresh-ID accuracy (%)");plt.legend(fontsize=8);save("accuracy_depth")
    plt.figure(figsize=(8, 5))
    for condition in CONDITIONS:
        keys = [f"{condition}:d{k}:b1" for k in ((1,) if condition.startswith("one_pass") else (1, 2, 3, 4))]
        plt.plot([timed[k]["ms_per_item"] for k in keys], [100 * timed[k]["timed_accuracy"] for k in keys], "o-", label=condition, color=colors[condition])
    baseline = timed["frozen:d1:b1"]
    plt.scatter(baseline["ms_per_item"], 100 * baseline["timed_accuracy"], marker="D", color="black", label="frozen")
    plt.xlabel("Synchronized full inference ms/item, batch 1");plt.ylabel("Accuracy on the SAME 24 validation items (%)");plt.title("Timing subset; not a locked-test frontier");plt.legend(fontsize=8);save("accuracy_latency")
    plt.figure(figsize=(8, 5))
    for i, condition in enumerate(CONDITIONS):
        count = fits[condition][0]["trainable_parameters"]["trainable"]
        depth = "1" if condition.startswith("one_pass") else "4"
        acc = 100 * fid["systems"][condition]["depths"][depth]["accuracy"]
        plt.scatter(count / 1000, acc, color=colors[condition]);plt.annotate(condition, (count / 1000, acc), xytext=(10, i * 8 - 15), textcoords="offset points", fontsize=8)
    counts = [fits[c][0]["trainable_parameters"]["trainable"] / 1000 for c in CONDITIONS]
    span = max(max(counts) - min(counts), min(counts) * .0006)
    plt.xlim(min(counts) - span * .3, max(counts) + span * .9);plt.ticklabel_format(axis="x", style="plain", useOffset=False)
    plt.xlabel("Trainable parameters (thousands; almost identical)");plt.ylabel("Fresh-ID accuracy at prespecified depth (%)");save("parameters_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax in axes:
        for condition in ("recurrent", "reinjection", "multi_depth"):
            ax.plot([1, 2, 3, 4], [fid["systems"][condition]["depths"][str(k)]["state_drift"] for k in range(1, 5)], "o-", label=condition)
        ax.plot([1, 2, 3, 4], [fid["untrained"]["untrained_bounded"][str(k)]["state_drift"] for k in range(1, 5)], "--", label="untrained_bounded")
        ax.set_xlabel("Depth");ax.set_ylabel("State distance / depth-1 norm");ax.set_xticks([1, 2, 3, 4]);ax.legend(fontsize=7)
    axes[0].plot([1, 2, 3, 4], [fid["untrained"]["untrained_plain"][str(k)]["state_drift"] for k in range(1, 5)], "--", color="gray", label="untrained_plain")
    axes[0].legend(fontsize=7);axes[0].set_title("Including plain frozen recurrence")
    axes[1].set_title("Bounded architectures only")
    save("state_drift")
    plt.figure(figsize=(8, 5))
    conditions = ["recurrent", "reinjection", "multi_depth"]
    x = np.arange(3)
    plt.bar(x - .18, [100 * fid["systems"][c]["transitions"]["WR"]["estimate"] for c in conditions], .36, label="Wrong→right")
    plt.bar(x + .18, [100 * fid["systems"][c]["transitions"]["RW"]["estimate"] for c in conditions], .36, label="Right→wrong")
    plt.xticks(x, conditions);plt.ylabel("Fraction of all adjacent transitions (%)");plt.legend();save("transitions")
    plt.figure(figsize=(9, 5))
    for condition in CONDITIONS:
        length = min(len(f["epochs"]) for f in fits[condition])
        x = np.arange(1, length + 1)
        plt.plot(x, [np.mean([f["epochs"][j]["train_loss"] for f in fits[condition]]) for j in range(length)], label=condition + " train", color=colors[condition])
        plt.plot(x, [np.mean([f["epochs"][j]["validation_mean_nll"] for f in fits[condition]]) for j in range(length)], "--", label=condition + " val", color=colors[condition])
    plt.xlabel("Epoch (one-pass compute control gets more updates)");plt.ylabel("Conditional loss; training includes mild KL");plt.legend(fontsize=7, ncol=2);save("training_validation_loss")
    selected, onepass = frozen["selected_recurrent_family"], frozen["selected_one_pass_family"]
    matrix = np.array([[100 * (summary[s]["systems"][selected]["tasks"][task]["4"] - summary[s]["systems"][onepass]["tasks"][task]["1"]) for task in TASKS] for s in FINAL])
    plt.figure(figsize=(10, 5));plt.imshow(matrix, cmap="RdBu", vmin=-max(abs(matrix).max(), 1), vmax=max(abs(matrix).max(), 1));plt.colorbar(label="Recurrent − one pass, percentage points")
    plt.xticks(range(6), TASKS);plt.yticks(range(5), FINAL)
    for i in range(5):
        for j in range(6):plt.text(j, i, f"{matrix[i,j]:+.1f}", ha="center", va="center", fontsize=9)
    plt.title("Small task strata; descriptive, no per-cell significance claim");save("generalization_tasks")
    plt.figure(figsize=(8, 5))
    for condition, values in stress.items():plt.plot(range(1, 9), [100 * values[str(k)] for k in range(1, 9)], "o--" if condition.startswith("untrained") else "o-", label=condition)
    plt.xlabel("Depth (5–8 are extrapolation only)");plt.ylabel("Accuracy on 24 prespecified stress items (%)");plt.legend();save("stress_depth")


def interval(value):
    low, high = value["cluster_seed_bootstrap95"]
    def fmt(number):
        return f"{Decimal(str(number * 100)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP):+}"
    return f"{fmt(value['estimate'])} [{fmt(low)}, {fmt(high)}]"


def main():
    import markdown
    config = json.loads((EXP / "config.json").read_text())
    frozen = json.loads((EXP / "frozen_protocol.json").read_text())
    assert (OUT / "final_execution.json").exists()
    audit(config)
    summary, arrays = summarize(config, frozen)
    fits = {c: [json.loads((OUT / "training" / f"main_{c}_seed{s}.json").read_text()) for s in config["seeds"]] for c in CONDITIONS}
    latency = json.loads((OUT / "latency.json").read_text())
    timed = timing_summary(latency)
    ablations = {}
    for condition in ("reinjection", "multi_depth"):
        ablations[condition] = {}
        for mode in ("none", "fixed"):
            rr = [trace(OUT / "ablations" / f"{condition}_{mode}_fresh_id_seed{s}.jsonl.gz") for s in config["seeds"]]
            correct = np.stack([cube(r).argmax(-1) == np.array([q["label"] for q in r]) for r in rr])
            ablations[condition][mode] = {str(k + 1): float(correct[:, k].mean()) for k in range(4)}
    stress = {}
    for condition in ("recurrent", "reinjection", "multi_depth"):
        rr = [trace(OUT / "stress" / f"{condition}_seed{s}.jsonl.gz") for s in config["seeds"]]
        correct = np.stack([cube(r).argmax(-1) == np.array([q["label"] for q in r]) for r in rr])
        stress[condition] = {str(k + 1): float(correct[:, k].mean()) for k in range(8)}
    for condition in ("untrained_plain", "untrained_bounded"):
        rr = trace(OUT / "stress" / f"{condition}.jsonl.gz")
        correct = cube(rr).argmax(-1) == np.array([q["label"] for q in rr])
        stress[condition] = {str(k + 1): float(correct[k].mean()) for k in range(8)}
    validation = json.loads((OUT / "implementation_validation.json").read_text())
    final_execution = json.loads((OUT / "final_execution.json").read_text())
    pilots = [json.loads(p.read_text()) for p in (OUT / "training").glob("pilot_*.json")]
    peak = max([f["peak_mlx_bytes"] for values in fits.values() for f in values] + [p["peak_mlx_bytes"] for p in pilots] + [validation["peak_mlx_bytes"], latency["peak_mlx_bytes"], final_execution["peak_mlx_bytes"]]) / 2**30
    rss = max([f["peak_process_rss_bytes"] for values in fits.values() for f in values] + [p["peak_process_rss_bytes"] for p in pilots] + [latency["peak_process_rss_bytes"], final_execution["peak_process_rss_bytes"]]) / 2**30
    completion_path = OUT / "report_completion.json"
    if completion_path.exists():
        duration = json.loads(completion_path.read_text())["experiment_duration_seconds"]
    else:
        duration = (datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(config["started_utc"])).total_seconds()
        atomic_json(completion_path, {"first_report_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "experiment_duration_seconds": duration})
    budget = json.loads((EXP / "resume_budget.json").read_text())
    active_duration = duration - budget["excluded_idle_seconds"]
    output = {"completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "experiment_duration_seconds": duration,
              "splits": summary, "timing": timed, "reinjection_inference_ablations": ablations, "stress": stress,
              "selected_recurrent_family": frozen["selected_recurrent_family"], "selected_one_pass_family": frozen["selected_one_pass_family"],
              "seeds": config["seeds"], "peak_mlx_gib": peak, "peak_process_rss_gib": rss,
              "active_experiment_seconds": active_duration, "excluded_idle_seconds": budget["excluded_idle_seconds"],
              "fit_seconds": {c: sum(f["training_seconds"] for f in fits[c]) for c in CONDITIONS},
              "trainable_parameters": {c: fits[c][0]["trainable_parameters"]["trainable"] for c in CONDITIONS}}
    output["training_diagnostics"] = {c: {"gradient_norm_median": float(np.median([b["gradient_norm"] for f in fits[c] for b in f["batch_records"]])),
                                           "gradient_norm_max": float(max(b["gradient_norm"] for f in fits[c] for b in f["batch_records"])),
                                           "completed_batches": sum(len(f["batch_records"]) for f in fits[c]),
                                           "checkpoint_diagnostics": {seed: r["checkpoint_diagnostics"] for seed, r in frozen["fits"][c].items()}} for c in CONDITIONS}
    atomic_json(OUT / "summary.json", output)
    figures(config, summary, timed, fits, frozen, ablations, stress)
    selected, onepass = frozen["selected_recurrent_family"], frozen["selected_one_pass_family"]
    fid = summary["fresh_id"]
    paired = fid["paired_comparisons"]
    main_comparison = paired[f"selected_minus_{onepass}"]
    if main_comparison["cluster_seed_bootstrap95"][0] > 0:
        conclusion = "The paired fresh-ID interval supports a decision advantage over the validation-selected one-pass control in this small study. It does not establish general reasoning capability."
    else:
        conclusion = "Trainable recurrence did not establish an accuracy advantage over the validation-selected one-pass adapter. This is the main result; better supervised loss or a favorable point estimate does not establish useful repeated reasoning."
    lines = ["# Experiment 5 — Train the Loop, Not the Judge", "", "## Executive summary", "", conclusion, "",
             f"Selected before final opening: recurrent family **{selected}**, fixed depth **4**; one-pass family **{onepass}**, depth **1**. Fresh-ID accuracies are **{100*fid['systems'][selected]['depths']['4']['accuracy']:.2f}%** versus **{100*fid['systems'][onepass]['depths']['1']['accuracy']:.2f}%**; paired difference **{interval(main_comparison)} percentage points**.", "",
             f"Completed 15 main fits (five conditions × three seeds), plus two separate pilot fits. Seeds {config['seeds']}. Active study time through report: {active_duration/60:.1f} minutes, within the six-hour active budget. Calendar elapsed time: {duration/3600:.2f} hours, including a documented {budget['excluded_idle_seconds']/3600:.2f}-hour session idle interval after training finished. See [budget receipt](../../experiments/experiment_05/resume_budget.json). Synchronized fitting excludes evaluation/checkpoint/report overhead. Peak MLX allocation {peak:.2f} GiB; process RSS {rss:.2f} GiB. All Experiment 5 work is local and excluded from the public v0.1.0 release.", "",
             "## Architecture", "", "Frozen pinned Qwen3-0.6B: 28 layers, hidden width 1,024, 16 query heads, eight KV heads, 128-dimensional attention heads, SwiGLU width 3,072, tied embeddings. These are inspected model configuration values, not loader assumptions. Early layers 0–11 feed shared middle layers 12–14, followed by late layers 15–27. Rank-8 query/value LoRA is shared across loops. All original arrays stay frozen. No final-state MLP substitutes for the recurrent transformer block.", "",
             "Depth one runs the adapted block once. Later passes RMS-match the original block input to the current token state, optionally convexly reinject it with r in [0, 0.25], apply the same adapted transformer block, cap proposal norm at twice that token's depth-one norm, and mix using sigmoid(u), initialized to 0.1. Reinjection initializes to 0.05. Gates are shared global learned scalars, not input-dependent networks. This bounds growth relative to the learned first state; it does not guarantee preservation of the original model at later depths. LoRA zero-output B gives depth-one initialization parity, while the open update path permits learning.", "",
             "```mermaid\nflowchart LR\n X[Tokens] --> E[Frozen early layers]\n E --> H[Original state h0]\n H --> F[Shared middle transformer + LoRA]\n F --> U[Bounded gated recurrent update]\n H --> U\n U --> F\n U --> L[Frozen late layers]\n L --> Y[Four-label decision]\n```", "",
             "## Research motivation", "", "Earlier probes recognized correctness, loss gains and decision transitions, while additional recurrence often damaged answers. This experiment trains the computation itself. A gain can reflect task-specific adaptation, calibration, extra capacity or computation; shared weights and additional loops do not by themselves demonstrate reasoning.", "",
             "## Differences from RecurTrace", "", "See the [primary-source review](../../experiments/experiment_05/research_references.md). This is a small LoRA/gate study, with bounded updates and four-depth CE rather than a reproduction of loop memory, stochastic-depth large-mixture training, or halting distillation. No author implementation is copied. There is no memory or controller in the primary study.", "",
             "## Training methodology", "", f"AdamW, learning rate {config['learning_rate']}, decay {config['weight_decay']}, clipping 1, batch {config['batch_size']}. D/E/F/G train {config['epochs']} epochs. E/F uniformly sample one supervised depth 1–4 per batch; G averages equal-weight losses at all four. Configured weights are normalized by their sum. No model trains only at deepest depth. Mild KL preservation (alpha {config['preservation_alpha']}) applies only where frozen training decisions are correct, averaging zero for other examples over the full batch. Labels supervise fitting and offline metrics; the model inference API receives tokens/lengths only. Confidence increases are not rewarded independently of correctness.", "",
             f"The additional one-pass control receives {config['compute_control_epochs']} epochs, fixed before main training as ceil(3 × pilot multi-depth/one-pass batch time ratio). This approximates optimization cost, not exact FLOPs or equal wall time. Measured fit seconds are retained. It may select an earlier checkpoint; extra optimization can overfit. All conditions share initial adapter seeds, example order, rank and optimizer. Checkpoints minimize mean validation NLL over their supervised inference depths. Family selection uses mean validation NLL at the prespecified final inference depth, not final accuracy.", "",
             "## Dataset methodology", "", f"Fresh 4A generators: {config['sizes']['train']} training, {config['sizes']['val']} validation, {sum(config['sizes'][s] for s in FINAL)} prospective final problems across five splits. Six task families and nominal difficulty cells stay balanced; no poor-performing task is removed. Independent reference solvers verify labels. Public historical IDs/prompts/alpha structures and compatible original structure exclusions reject prior instances. ID/template shifts share abstract algorithms; structural shift holds out a generator per family. Static predicates do not always become deeper under nominal hard strata. Numeric-range pointer items are unchanged-range controls.", "",
             f"Frozen competence before fitting: training {100*json.loads((OUT/'baseline_competence.json').read_text())['train']['accuracy']:.2f}%, validation {100*json.loads((OUT/'baseline_competence.json').read_text())['val']['accuracy']:.2f}%. Several tasks remain near/below four-choice chance. This limits the study and may make calibration dominate adaptation. Final labels did not select data, hyperparameters, seeds, checkpoints, or depths.", "",
             "## Causal correctness checks", "", f"Real depth-one max absolute logit error {validation['depth_one_max_abs_error']:.8f}; trained depth-four prefix error {validation['trained_causal_prefix_max_abs_error']['4']:.8f}. Tiny-Qwen checks compare identical prefixes with different future suffixes, truncated prefixes and repeated calls at depths 1/2/4/8. The original GQA/QK normalization/RoPE/residual behavior remains in mlx-lm. Every pass uses cache=None and the same causal positions. Only right padding is used; outputs gather the last real token. Incremental KV inference is not implemented.", "",
             "## Gradient-flow verification", "", "Rank 8 and 16 pass numerical compatibility/count checks. Initial B and gate gradients are nonzero; A is initially zero-gradient because B is zero, as expected. Shared adapters are single arrays reused at every depth, not cloned. Original frozen-weight samples remain identical after updates. Deterministic evaluation, exact real adapter reload and exact tiny-model optimizer resume passed. The complete suite has 33 passing tests with MLX integration enabled.", "",
             "## Experimental configurations", "", "| Condition | Objective / reinjection | Trainable parameters | Measured fit seconds, three seeds | Training examples/s |", "|---|---|---:|---:|---:|"]
    for condition in CONDITIONS:
        lines.append(f"| {condition} | {'depth 1' if condition.startswith('one_pass') else 'equal depths 1–4' if condition=='multi_depth' else 'uniform sampled depth'} / {'learned' if condition in ('reinjection','multi_depth') else 'none'} | {output['trainable_parameters'][condition]:,} | {output['fit_seconds'][condition]:.1f} | {sum(f['examples_processed'] for f in fits[condition])/output['fit_seconds'][condition]:.2f} |")
    lines += ["", "One-pass and recurrent LoRA matrices have exactly equal capacity; recurrence adds one update scalar and gated variants add one reinjection scalar. The one-pass update scalar is frozen and excluded from trainable counts. This negligible difference is disclosed. Parameter count alone does not match inference cost.", "",
              f"The inspected model contains {validation['rank_parameter_counts']['8']['total']:,} parameters including rank-8 adapters and two scalar gates; rank 16 would train {validation['rank_parameter_counts']['16']['trainable']:,} parameters and was compatibility-tested only.", "",
              "| Family | Mean validation NLL at prespecified depth | Best epoch per seed (one-based) |", "|---|---:|---|"]
    for condition in CONDITIONS:
        lines.append(f"| {condition} | {frozen['validation_selection_scores'][condition]:.6f} | {', '.join(str(f['best_epoch'] + 1) for f in fits[condition])} |")
    lines += ["", "These validation scores, recorded before opening final data, select the reported families. Epoch selection is by NLL, so a later epoch with higher accuracy can remain unselected.", "", "## Accuracy results", "", "Every value below is measured on the new Experiment 5 problems. The old one-pass head is a transferred frozen historical checkpoint, not a reused historical test score.", "", "| Split | Frozen | Old one-pass head (transfer) | Trained one pass | Extra-budget one pass | Selected recurrence, depth 4 | Difference vs selected one pass [95%], pp |", "|---|---:|---:|---:|---:|---:|---:|"]
    for split in FINAL:
        s = summary[split]
        lines.append(f"| {split} | {100*s['frozen']['accuracy']:.2f}% | {100*s['historical_one_pass']['accuracy']:.2f}% | {100*s['systems']['one_pass']['depths']['1']['accuracy']:.2f}% | {100*s['systems']['one_pass_compute']['depths']['1']['accuracy']:.2f}% | {100*s['systems'][selected]['depths']['4']['accuracy']:.2f}% | {interval(s['paired_comparisons']['selected_minus_'+onepass])} |")
    lines += ["", "The old head was trained on a different distribution, so it provides a transfer reference and cannot isolate where adaptation belongs.", "", "| Fresh-ID paired comparison | Difference [95% interval], pp |", "|---|---:|"]
    lines += [f"| {name} | {interval(value)} |" for name, value in paired.items()]
    lines += ["", "| Training seed | One pass | Selected recurrence, depth 1 | Selected recurrence, depth 4 |", "|---|---:|---:|---:|"]
    for seed in config["seeds"]:
        lines.append(f"| {seed} | {100*fid['systems'][onepass]['depths']['1']['accuracy_by_seed'][str(seed)]:.2f}% | {100*fid['systems'][selected]['depths']['1']['accuracy_by_seed'][str(seed)]:.2f}% | {100*fid['systems'][selected]['depths']['4']['accuracy_by_seed'][str(seed)]:.2f}% |")
    lines += ["", "Intervals jointly resample three training seeds and the same unique problem indices across all conditions/depths; repeated depths stay clustered. The 288 fresh-ID problems are not 864 independent seed examples. Intervals condition on this training/validation set and the frozen selection. Three seeds and small task strata limit power; multiple comparisons have no multiplicity adjustment. Shift results are secondary. Depth-one comparisons match the forward architecture and parameter budget, testing variable-depth training effects without additional inference recurrence; they are secondary descriptive controls, not the primary hypothesis. Conditional option NLL is not language-model perplexity or free generation accuracy.", "", "| Fresh-ID system / depth | Accuracy | NLL | ECE | Brier |", "|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for k, r in fid["systems"][condition]["depths"].items():
            lines.append(f"| {condition} / {k} | {100*r['accuracy']:.2f}% | {r['nll']:.4f} | {r['ece']:.4f} | {r['brier']:.4f} |")
    lines += ["", "## Recurrence depth results", "", "![Accuracy versus depth](figures/accuracy_depth.png)", "", "The selected depth is fixed at four before final opening. Every trained depth is reported; the highest final point is not a newly selected inference policy. Adjacent paired differences are stored for each family. Eight-loop results below use only 24 prespecified balanced items and are extrapolation diagnostics, never selection data.", "", "![Eight-loop diagnostic](figures/stress_depth.png)", "", "## Compute comparisons", "", latency["scope"], "", "| System | Depth | Batch-1 ms/item | Batch-8 amortized ms/item |", "|---|---:|---:|---:|"]
    for condition in ("frozen", *CONDITIONS):
        for k in ((1,) if condition.startswith("one_pass") or condition == "frozen" else (1, 2, 3, 4)):
            lines.append(f"| {condition} | {k} | {timed[f'{condition}:d{k}:b1']['ms_per_item']:.2f} | {timed[f'{condition}:d{k}:b8']['ms_per_item']:.2f} |")
    d1, d4 = timed[f'{selected}:d1:b1']["ms_per_item"], timed[f'{selected}:d4:b1']["ms_per_item"]
    lines += ["", f"For the selected recurrent family, depth four adds {d4-d1:.2f} ms per batch-one request over depth one ({d4/d1:.2f}× latency). Only the requested depth is decoded during timing; recording all four analysis outputs is excluded. Every timing group also has one untimed reference call for exact repetition checks."]
    lines += ["", f"Timing uses 24 validation prompts, batches 1/8, {config['timing_warmups']} warmups and {config['timing_repeats']} synchronized repeats, three seeds, and randomized condition order. Condition runs remain sequential, so between-run thermals/load are not fully controlled. Batch-8 amortization measures throughput; variable prompt lengths add padding work. Model load is excluded. This is full four-candidate decision inference, not open-ended generation with a full-vocabulary output projection. The latency figure uses accuracy on the **same timed validation items**, not final accuracy from different prompts. Matrix FLOPs are not used to claim speedups.", "", "![Accuracy versus end-to-end latency](figures/accuracy_latency.png)", "", "![Parameters versus accuracy](figures/parameters_accuracy.png)", "", "## Generalization", "", "![Generalization by task family](figures/generalization_tasks.png)", "", "All families, nominal strata, withheld structural approaches and numeric shifts remain in the public summary. Cells contain few examples and are descriptive. Added compute, calibrated answer probabilities, memorized templates, or task-specific readouts remain alternative explanations for any accuracy gain.", "", "## Corrected/degraded decision analysis", "", "| Family | W→R rate [95%] | R→W rate [95%] | W→R given wrong | R→W given right |", "|---|---:|---:|---:|---:|"]
    insertion = lines.index("## Corrected/degraded decision analysis")
    descriptive = ["| Fresh-ID nominal difficulty | Selected one pass | Selected recurrence, depth 4 |", "|---|---:|---:|"]
    for stratum in (s for s in ("easy", "moderate", "hard", "extrapolation") if s in fid["systems"][selected]["difficulty"]):
        descriptive.append(f"| {stratum} | {100*fid['systems'][onepass]['difficulty'][stratum]['1']:.2f}% | {100*fid['systems'][selected]['difficulty'][stratum]['4']:.2f}% |")
    descriptive += ["", "Difficulty is a predefined generator descriptor, not an independently established reasoning-depth scale. Harder-depth shift and every task family are also reported without filtering.", ""]
    lines[insertion:insertion] = descriptive
    for condition in ("recurrent", "reinjection", "multi_depth"):
        tr = fid["systems"][condition]["transitions"]
        lines.append(f"| {condition} | {interval(tr['WR'])}% | {interval(tr['RW'])}% | {100*tr['WR_given_wrong']:.2f}% | {100*tr['RW_given_right']:.2f}% |")
    lines += ["", "Rates cover three adjacent recurrent transitions per problem, retaining their dependence in bootstrap. Initial frozen→trained-depth-one transitions are reported separately as adaptation, not evidence for additional loops. WW/RR, loss, correct-answer probability, confidence, margin, entropy, state distance and norm are recorded per example/depth. No labels or future states enter inference.", "", "![Corrections and degradations](figures/transitions.png)", "", "![Recurrent drift](figures/state_drift.png)", "", "## Failure analysis", "", conclusion, "", "Low backbone competence and CE near the four-choice uniform baseline restrict interpretation. A lower NLL may largely soften incorrect overconfident decisions. Norm bounds prevent unbounded growth but can also restrict useful adaptation; first-state norms are learned and do not guarantee closeness to the original model. Validation-selected models can generalize poorly from 144 items. Reinjection ablations below use the same trained weights, so they are diagnostics rather than independently fitted causal comparisons.", "", "| Family | Learned gate depth-4 accuracy | Fixed gate ablation | No reinjection ablation |", "|---|---:|---:|---:|"]
    for condition in ("reinjection", "multi_depth"):
        lines.append(f"| {condition} | {100*fid['systems'][condition]['depths']['4']['accuracy']:.2f}% | {100*ablations[condition]['fixed']['4']:.2f}% | {100*ablations[condition]['none']['4']:.2f}% |")
    lines += ["", f"Exploratory answer-position diagnostic, specified during validation before final opening: a constant prior fitted solely from training labels achieves {100*fid['training_answer_position_prior']['accuracy']:.2f}% fresh-ID accuracy and NLL {fid['training_answer_position_prior']['nll']:.4f}. Uniform four-option CE is {np.log(4):.4f}. The prior does not select any trained model.", "",
              "| Fresh-ID family at reported depth | Prediction frequencies A/B/C/D (%) | Mean option entropy (nats) | Accuracy SD across seeds, pp |", "|---|---|---:|---:|"]
    for c in CONDITIONS:
        r = fid["systems"][c]["depths"]["1" if c.startswith("one_pass") else "4"]
        lines.append(f"| {c} | {' / '.join(f'{100*v:.1f}' for v in r['prediction_class_frequencies'])} | {r['mean_option_entropy']:.4f} | {100*r['accuracy_seed_sd']:.2f} |")
    lines += ["", "| Family | Selected update gates by seed | Selected reinjection gates by seed | Median / maximum gradient norm before clipping |", "|---|---|---|---|"]
    for c in CONDITIONS:
        diag = output["training_diagnostics"][c]
        values = list(diag["checkpoint_diagnostics"].values())
        def gates(name):
            return " / ".join("—" if d[name] is None else f"{d[name]:.4f}" for d in values)
        lines.append(f"| {c} | {gates('update')} | {gates('reinjection')} | {diag['gradient_norm_median']:.3f} / {diag['gradient_norm_max']:.3f} |")
    lines += ["", "Gate values and nonzero saved LoRA B norms are checkpoint measurements. A gate staying near initialization is a limitation to investigate, not proof that it caused a failure. All reported gradients were clipped to norm one before the optimizer update."]
    lines += ["", "![Training and validation loss](figures/training_validation_loss.png)", "", "## Limitations", "", "This is a completed small controlled study, not an exhaustive hyperparameter sweep or broad reasoning benchmark. Only rank-8 attention adapters, one block, three short main epochs, one optimizer schedule and few synthetic instances were tested. The additional one-pass control uses approximate training-cost matching; exact hardware budgets differ. A fixed-depth compute control is also available by comparing the same adapted model at depths 1–4, but there is no separate larger nonshared backbone. No full-model fine-tuning, natural task suite, incremental cache optimization, controller retraining, trajectory objective or loop memory was run. Saved weights and large hidden arrays remain ignored. The preserved historical release and reports are unchanged.", "", "## Future work", "", "Use a larger fresh training/validation set and a separately reported diagnostic task mixture where the frozen representation has intermediate competence. Retain the original hard mixture as primary. Test longer training and a small gate-strength ablation before investing in memory/halting. Compare against matched one-pass optimization, and require paired structural-transfer evidence before attributing gains to recurrence. Old C/V/D probes are not assumed compatible with the newly trained states; transfer/recalibration is deferred until useful fixed-depth computation is established. This experiment prioritizes the thinker.", "", "## Reproduction and local status", "", "```sh\nLATENT_LOOP_MLX_TESTS=1 .venv/bin/python -m pytest -q\n.venv/bin/python -m latent_loop.experiment05.validate\n.venv/bin/python -m latent_loop.experiment05.train pilot\n.venv/bin/python -m latent_loop.experiment05.train main\n.venv/bin/python -m latent_loop.experiment05.evaluate freeze\n.venv/bin/python -m latent_loop.experiment05.evaluate final\n.venv/bin/python -m latent_loop.experiment05.evaluate timing\n.venv/bin/python -m latent_loop.experiment05.report\n```", "", "The completed config/protocol is consumed: fitting refuses after final opening. Use a new independent namespace, fresh seeds/runtime clock and manifests for a new experiment, rather than overwriting these files. CPU report regeneration reads saved measurements without MLX. Training resumes adapter/optimizer checkpoints and deterministic batch/depth schedules after safe interruptions. Source/config/data hashes reject incompatible resume. No Experiment 5 commit is pushed. Public v0.1.0 is `0a637eb97d66eebbd740b612ab92e9095c496931`.", "", "[Protocol](../../experiments/experiment_05/frozen_protocol.json) · [Dataset manifest](../../experiments/experiment_05/dataset_manifest.json) · [Summary](../../results/experiment_05/summary.json) · [Public audit](../../results/experiment_05/public_audit.json) · [Primary references](../../experiments/experiment_05/research_references.md)"]
    pos = lines.index("## Architecture")
    lines[pos:pos] = [f"The selected recurrent model scores {100*fid['systems'][selected]['depths']['1']['accuracy']:.2f}% at depth one. Depth four minus depth one is {interval(paired['selected_depth4_minus_depth1'])} percentage points. The simultaneous multi-depth condition scores {100*fid['systems']['multi_depth']['depths']['4']['accuracy']:.2f}% at depth four; its unfavorable result remains part of the main study.", ""]
    pos = lines.index("## Failure analysis")
    lines[pos:pos] = ["The bounded untrained control helps inspect stabilization from the cap and gated update. It uses fixed 0.05 reinjection and therefore matches the gated families' initialization more closely than the selected no-reinjection model. Differences against the latter do not isolate adaptation alone. Small state drift does not demonstrate better decisions.", ""]
    pos = lines.index("## Limitations") + 2
    lines[pos:pos] = ["Checkpoint and family selection by NLL can favor calibrated probabilities over decision accuracy. That criterion was preserved after final opening; this study does not evaluate final accuracy of unselected checkpoints.", ""]
    pos = lines.index("## Reproduction and local status") + 2
    env = json.loads((OUT / "runtime_environment.json").read_text())
    lines[pos:pos] = [f"Verified Python {env['python']}, MLX {env['training_versions']['mlx']}, mlx-lm {env['training_versions']['mlx-lm']}, FP32 on Apple Silicon. [Exact dependency versions](../../results/experiment_05/runtime_environment.json) include the clean CPU report environment, which has no MLX installed.", ""]
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / "train_the_thinker.md"
    path.write_text("\n".join(lines) + "\n")
    body = markdown.markdown(path.read_text(), extensions=["tables", "fenced_code"])
    def embed(match):
        image = path.parent / match.group(1)
        if image.exists() and image.suffix == ".png":
            return 'src="data:image/png;base64,' + base64.b64encode(image.read_bytes()).decode() + '"'
        return match.group(0)
    body = re.sub(r'src="([^"]+)"', embed, body)
    path.with_suffix(".html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Experiment 5 — Train the Loop, Not the Judge</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 24px}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ddd;padding:6px}img{max-width:100%}pre{overflow:auto;background:#f5f5f5;padding:16px}</style>'+body+'</html>')
    atomic_json(OUT / "report_artifact_manifest.json", {"generator_sha256": sha(__file__),
                "input_hashes": {str(p): sha(p) for p in (OUT / "summary.json", OUT / "public_audit.json", OUT / "latency.json", OUT / "runtime_environment.json", EXP / "frozen_protocol.json")},
                "artifact_hashes": {str(p): sha(p) for p in sorted(REPORT.rglob("*")) if p.is_file()},
                "figure_source": "Saved measured logits, trajectories, epoch records and synchronized timings; no synthetic result points"})
    print("Local report complete:", conclusion, "paired", interval(main_comparison), flush=True)


if __name__ == "__main__":
    main()
