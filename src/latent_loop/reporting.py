"""Regenerate all scientific plots and reports from recorded, public measurements."""

import base64
import gzip
import json
from pathlib import Path

import numpy as np

from latent_loop.training import atomic_json

MAIN = [
    "baseline",
    "temperature",
    "constant_calibration",
    "confidence_calibration",
    "confidence_only",
    "reranking",
    "feedforward",
    "wide_feedforward",
    "recurrence_selected",
    "memory_selected",
    "recurrence_adaptive",
    "memory_adaptive",
]
NAMES = {
    "baseline": "Frozen Qwen3",
    "temperature": "Temperature scaling",
    "constant_calibration": "Constant correctness prior",
    "confidence_calibration": "Confidence head (same decisions)",
    "confidence_only": "Confidence-only selection",
    "reranking": "Probability + confidence fusion",
    "feedforward": "One-pass head, width 64",
    "wide_feedforward": "One-pass head, width 256",
    "recurrence_selected": "Recurrence, validation depth",
    "memory_selected": "Memory, validation depth",
    "recurrence_adaptive": "Recurrence + learned stopping",
    "memory_adaptive": "Memory + learned stopping",
}


def report():
    import matplotlib

    matplotlib.use("Agg")
    import markdown
    import matplotlib.pyplot as plt
    from scipy.special import softmax

    from latent_loop.datasets import load

    inputs = [
        json.loads(p.read_text())
        for p in sorted(Path("results/metrics").glob("seed*.json"))
    ]
    if not inputs:
        raise RuntimeError("No evaluated checkpoints to report")
    rows = [r for d in inputs for r in d["results"]]
    seeds = [d["results"][0]["seed"] for d in inputs]
    config = (
        json.loads(Path("experiments/active_config.json").read_text())
        if Path("experiments/active_config.json").exists()
        else {
            "width": 64,
            "sizes": {"train": 1200, "val": 300, "test": 600, "ood": 300},
            "train_batch": 64,
            "epochs": 30,
            "learning_rate": 0.001,
        }
    )
    width = config["width"]
    NAMES["feedforward"] = f"One-pass head, width {width}"
    NAMES["wide_feedforward"] = f"One-pass head, width {width * 4}"

    def select(v, split="test"):
        return [r for r in rows if r["variant"] == v and r["split"] == split]

    def mean(v, key="accuracy", split="test"):
        return float(np.mean([r[key] for r in select(v, split)]))

    def sd(v, key="accuracy", split="test"):
        return (
            float(np.std([r[key] for r in select(v, split)], ddof=1))
            if len(inputs) > 1
            else 0.0
        )

    out = Path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / f"{name}.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for name, color in [("recurrence", "#275dad"), ("memory", "#a44a3f")]:
        x = np.arange(1, 9)
        y = [mean(f"{name}_d{k}") * 100 for k in x]
        err = [sd(f"{name}_d{k}") * 100 for k in x]
        ax.errorbar(x, y, yerr=err, marker="o", label=name, color=color)
    ax.axhline(mean("baseline") * 100, ls=":", color="gray", label="frozen baseline")
    ax.axhline(
        mean("feedforward") * 100, ls="--", color="#287a55", label="one-pass head"
    )
    ax.axvspan(4.5, 8.5, color="gray", alpha=0.1)
    ax.set(
        xlabel="Refinement loops",
        ylabel="Test accuracy (%)",
        title="Extra depth eventually harms decisions (±1 seed SD)",
        xticks=range(1, 9),
    )
    ax.legend(fontsize=8)
    save(fig, "accuracy_depth")
    # Analytical incremental matrix FLOPs; no claim these are measured hardware FLOPs.
    hidden = 1024
    perloop = 2 * (3 * hidden * width + hidden * 4)
    costs = {
        "baseline": 0.0,
        "confidence_only": 2 * 4 * ((2 * hidden + 5) * width + width),
        "feedforward": perloop,
        "wide_feedforward": 2 * (3 * hidden * width * 4 + hidden * 4),
        "recurrence_selected": perloop * mean("recurrence_selected", "average_depth"),
        "memory_selected": (perloop + 2 * hidden * 32 * 4)
        * mean("memory_selected", "average_depth"),
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, c in costs.items():
        ax.errorbar(c / 1e6, mean(v) * 100, yerr=sd(v) * 100, fmt="o")
        ax.annotate(
            NAMES.get(v, v),
            (c / 1e6, mean(v) * 100),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.set(
        xlabel="Estimated extra matrix FLOPs / decision (millions)",
        ylabel="Test accuracy (%)",
        title="The one-pass head is competitive at lower extra compute",
    )
    save(fig, "accuracy_cost")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([0, 1], [0, 1], ls="--", color="gray")
    for v in [
        "baseline",
        "temperature",
        "confidence_calibration",
        "feedforward",
        "recurrence_adaptive",
    ]:
        bins = select(v)[0]["calibration"]
        ax.plot(
            [b["confidence"] for b in bins],
            [b["accuracy"] for b in bins],
            marker="o",
            label=NAMES[v],
        )
    ax.set(
        xlabel="Reported correctness confidence",
        ylabel="Observed correctness",
        title=f"Calibration on test data (seed {seeds[0]})",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(fontsize=7)
    save(fig, "calibration")
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, 5)
    for j, v in enumerate(["recurrence_adaptive", "memory_adaptive"]):
        hist = np.mean([r["depth_histogram"][1:5] for r in select(v)], axis=0)
        ax.bar(x + (j - 0.5) * 0.32, hist, width=0.32, label=NAMES[v])
    ax.set(
        xlabel="Stopping depth",
        ylabel="Mean number of test examples",
        xticks=x,
        title="Learned stopping on a budget of four loops",
    )
    ax.legend(fontsize=8)
    save(fig, "halting_depth")
    fig, ax = plt.subplots(figsize=(7, 4))
    vs = [
        "feedforward",
        "recurrence_selected",
        "memory_selected",
        "recurrence_d8",
        "memory_d8",
    ]
    x = np.arange(len(vs))
    for j, (key, label) in enumerate(
        [("wrong_to_right", "Corrected"), ("right_to_wrong", "Degraded")]
    ):
        vals = [
            np.mean([r["transitions_vs_baseline"][key] for r in select(v)]) for v in vs
        ]
        ax.bar(x + (j - 0.5) * 0.35, vals, width=0.35, label=label)
    ax.set(
        xticks=x,
        xticklabels=["one pass", "recurrence", "memory", "recurrence 8", "memory 8"],
        ylabel="Mean test example count",
        title="Both corrections and regressions occur",
    )
    ax.legend()
    save(fig, "prediction_changes")
    fig, ax = plt.subplots(figsize=(7, 4))
    vs = [
        "recurrence_selected",
        "memory_selected",
        "memory_disabled",
        "memory_adaptive",
    ]
    x = np.arange(4)
    ax.bar(
        x,
        [mean(v) * 100 for v in vs],
        yerr=[sd(v) * 100 for v in vs],
        color=["#275dad", "#a44a3f", "gray", "#e39a60"],
    )
    ax.set(
        xticks=x,
        xticklabels=[
            "no memory\ntrained separately",
            "with memory",
            "memory removed\nafter training",
            "memory +\nlearned stopping",
        ],
        ylabel="Test accuracy (%)",
        title="Memory dependence does not establish a memory advantage",
    )
    save(fig, "memory_effect")
    # Paired uncertainty resamples examples, keeping seed predictions together.
    predictions = []
    for seed in seeds:
        with gzip.open(f"results/metrics/predictions_seed{seed}.json.gz", "rt") as f:
            predictions.append(json.load(f))
    comparisons = {}
    rng = np.random.default_rng(901)

    def differences(a, b, split):
        return np.stack(
            [
                (
                    np.array(p[split][a]["logits"]).argmax(-1) == p[split]["labels"]
                ).astype(float)
                - (
                    np.array(p[split][b]["logits"]).argmax(-1) == p[split]["labels"]
                ).astype(float)
                for p in predictions
            ]
        )

    for a, b in [
        ("feedforward", "baseline"),
        ("recurrence_selected", "baseline"),
        ("memory_selected", "recurrence_selected"),
        ("recurrence_selected", "feedforward"),
        ("recurrence_adaptive", "recurrence_selected"),
        ("memory_adaptive", "memory_selected"),
        ("confidence_only", "baseline"),
    ]:
        comparisons[a + "_vs_" + b] = {}
        for split in ["test", "ood"]:
            d = differences(a, b, split)
            n = d.shape[1]
            boots = []
            hier = []
            for _ in range(5000):
                ix = rng.integers(0, n, n)
                boots.append(d[:, ix].mean())
                s = rng.integers(0, len(seeds), len(seeds))
                hier.append(d[s][:, ix].mean())
            comparisons[a + "_vs_" + b][split] = {
                "mean_delta": float(d.mean()),
                "paired_item_bootstrap95": np.quantile(boots, [0.025, 0.975]).tolist(),
                "hierarchical_seed_item_bootstrap95": np.quantile(
                    hier, [0.025, 0.975]
                ).tolist(),
                "caution": "Three seeds, shared examples, exploratory comparisons; no multiplicity correction.",
            }
    atomic_json("results/metrics/paired_comparisons.json", comparisons)
    backbone = json.loads(Path("results/metrics/backbone.json").read_text())
    latency = (
        json.loads(Path("results/metrics/latency_backbone.json").read_text())
        if Path("results/metrics/latency_backbone.json").exists()
        else None
    )
    middle = (
        json.loads(Path("results/metrics/middle.json").read_text())
        if Path("results/metrics/middle.json").exists()
        else []
    )
    baseline_timing_text = (
        f"Baseline prompt median was {latency['median_ms']:.2f} ms."
        if latency
        else "Baseline prompt timing was not measured in this run."
    )
    format_metrics = (
        json.loads(Path("results/metrics/baseline_format.json").read_text())
        if Path("results/metrics/baseline_format.json").exists()
        else None
    )
    format_text = (
        f"The unrestricted next-token baseline scored {100 * format_metrics['test']['unconstrained_next_token_accuracy']:.2f}%; its vocabulary argmax was an option letter on {100 * format_metrics['test']['unconstrained_argmax_is_option']:.2f}% of test prompts. Full-vocabulary label NLL was {format_metrics['test']['full_vocabulary_label_nll']:.3f}. These are separate format diagnostics, not the primary conditional objective."
        if format_metrics
        else "Full-vocabulary format diagnostics were not exported in this run."
    )
    table = "| Variant | Test accuracy, mean ± SD | OOD accuracy | Test NLL | Correctness ECE | Loops |\n|---|---:|---:|---:|---:|---:|\n"
    for v in MAIN:
        table += f"| {NAMES[v]} | {100 * mean(v):.2f} ± {100 * sd(v):.2f}% | {100 * mean(v, split='ood'):.2f}% | {mean(v, 'nll'):.3f} | {mean(v, 'ece'):.3f} | {mean(v, 'average_depth'):.2f} |\n"
    confidence_table = "| Confidence estimate on original decisions | Correctness Brier | Correctness AUROC | Accuracy at 50% coverage |\n|---|---:|---:|---:|\n"
    for v in [
        "baseline",
        "constant_calibration",
        "temperature",
        "margin_calibration",
        "entropy_calibration",
        "confidence_calibration",
    ]:
        confidence_table += f"| {NAMES.get(v, v)} | {mean(v, 'brier_correctness'):.3f} | {mean(v, 'correctness_auroc'):.3f} | {100 * mean(v, 'selective_accuracy_50'):.2f}% |\n"
    task_table = "| Task | Frozen model | One-pass head | Recurrence | Memory |\n|---|---:|---:|---:|---:|\n"
    for task in select("baseline")[0]["per_task"]:
        task_table += (
            f"| {task} | "
            + " | ".join(
                f"{100 * np.mean([r['per_task'][task]['accuracy'] for r in select(v)]):.2f}%"
                for v in [
                    "baseline",
                    "feedforward",
                    "recurrence_selected",
                    "memory_selected",
                ]
            )
            + " |\n"
        )
    parameter_table = "| Trained module | Parameters | Total fit seconds across seeds | Best epochs (0-based) |\n|---|---:|---:|---|\n"
    for name in inputs[0]["training"]:
        infos = [d["training"][name] for d in inputs]
        parameter_table += f"| {name} | {infos[0]['trainable_parameters']:,} | {sum(i['training_seconds'] for i in infos):.2f} | {', '.join(str(i['best_epoch']) for i in infos)} |\n"
    cost_table = "| Variant | Extra matrix FLOPs, estimate | Head latency, mean seed median |\n|---|---:|---:|\n"
    for v in [
        "confidence_only",
        "feedforward",
        "wide_feedforward",
        "recurrence_selected",
        "memory_selected",
        "recurrence_adaptive",
        "memory_adaptive",
    ]:
        ts = []
        for d in inputs:
            if v == "confidence_only":
                key = "confidence"
            elif v.endswith("_selected"):
                name = v.removesuffix("_selected")
                key = f"{name}_d{d['selections'][name + '_fixed_depth']}"
            elif v in ("feedforward", "wide_feedforward"):
                key = v + "_d1"
            else:
                key = v
            ts.append(d["timings"][key]["median_ms"])
        cost_table += (
            f"| {NAMES[v]} | {costs[v] / 1e6:.2f} M | {np.mean(ts) * 1000:.1f} µs |\n"
            if v in costs
            else f"| {NAMES[v]} | variable; head and stopping overhead | {np.mean(ts) * 1000:.1f} µs |\n"
        )
    best = max(MAIN, key=lambda v: mean(v))
    ff = comparisons["feedforward_vs_baseline"]["test"]
    rec = comparisons["recurrence_selected_vs_feedforward"]["test"]
    peak = max(
        [backbone["peak_mlx_bytes"]]
        + [d["training"][n]["peak_mlx_bytes"] for d in inputs for n in d["training"]]
        + ([latency["peak_mlx_bytes"]] if latency else [])
    )
    text = f"""# Confidence-guided latent refinement: research summary

## Executive summary

We tested whether lightweight heads can recover useful decisions from frozen Qwen3-0.6B hidden states, and whether recurrence, memory, or learned continuation provides an advantage over one pass. This is an independent synthetic-task pilot, not an official RecurTrace reproduction.

The best primary test mean was **{NAMES[best]}: {100 * mean(best):.2f}%**, compared with **{100 * mean("baseline"):.2f}%** for the frozen conditional baseline. A one-pass width-{width} head improved accuracy by {100 * ff["mean_delta"]:.2f} percentage points; its paired item-bootstrap 95% interval was [{100 * ff["paired_item_bootstrap95"][0]:.2f}, {100 * ff["paired_item_bootstrap95"][1]:.2f}] points. Recurrence versus that head changed accuracy by {100 * rec["mean_delta"]:.2f} points, interval [{100 * rec["paired_item_bootstrap95"][0]:.2f}, {100 * rec["paired_item_bootstrap95"][1]:.2f}]. This does **not** establish a benefit from recurrence. Memory and learned stopping did not establish an accuracy advantage either.

These intervals describe this pilot. Only {len(seeds)} training seed(s), multiple comparisons, reused templates, synthetic tasks, and an adjusted prompt format limit confirmatory interpretation. The held-out set is {config["sizes"]["test"]} unique examples; shared seed evaluations do not multiply the number of independent test items.

## Hardware and execution

{backbone["device"]["device_name"]}, {backbone["device"]["memory_size"] / 1024**3:.0f} GiB unified memory; MLX Metal; FP32 backbone and heads. Peak recorded MLX allocation: **{peak / 1024**3:.2f} GiB** (allocator accounting, not total system memory). Versions: `{json.dumps(backbone["versions"])}`. Configured memory budget: {backbone["memory_budget_gb"]} GiB. Primary extraction took {backbone["extraction_seconds"]:.2f} seconds. Auxiliary fit times are below; they exclude extraction, oracle trajectory construction, evaluation, and checkpoint/report I/O. Command wall times are in `experiments/last_*.json`.

The original BF16 parity test failed and is retained in `experiments/failure_audit.json`. FP32 interception max absolute logit error was {backbone["hidden_parity_max_abs"]:.8f}; the repeated-middle-block causal-prefix error was {backbone["middle_prefix_causal_max_abs"]:.8f}. No unrelated service, checkpoint, or workstation was modified.

## Methodology

Six deterministic tasks cover comparison, arithmetic, register state, pointer chasing, registry-based tool routing, and conjunction-based candidate classification. Counts: {config["sizes"]["train"]} training, {config["sizes"]["val"]} validation, {config["sizes"]["test"]} test, {config["sizes"]["ood"]} OOD; approximately equal task counts. Training/test use steps 1–4; OOD uses steps 5–8 for the three sequential tasks, larger numeric range, and separate symbols. Easy comparison, routing, and classification remain one-step tasks in OOD, so the aggregate mixes several shifts. Independent solvers validate labels. Exact instance hashes and a token 13-gram near-duplicate audit found no leaked instances in the primary dataset; templates are shared. Difficulty and per-task results are recorded for every variant.

The pretrained backbone is frozen, unquantized, revision `{backbone["revision"]}`. We extract the last prompt hidden state and score the verified single-token letters A–D. A fixed assistant prefix `Answer: (` was chosen using 60 training examples after the unprefixed pilot exposed a severe format artifact. No reasoning text is generated. All models see the same prompt and candidates. Primary accuracy and NLL are **conditional on four answer labels**; they are not free-generation accuracy or full language-model perplexity.

The confidence head uses normalized hidden state, candidate label embedding, candidate conditional log probability, original vocabulary log probability, vocabulary entropy, option margin, and option entropy. It learns binary candidate correctness. Calibration is evaluated with unchanged greedy decisions. Confidence-only selection and validation-selected fusion are separate decision experiments; top-2 coverage is recorded, while four-option coverage is 100% by construction.

The refiner learns a closed-gate residual MLP on the final hidden vector, reinjecting the original context. The same weights train at uniformly sampled depths 1–4. Memory reads a bounded three-state history along loop time for this final-position vector, with learned normalized query/key projections and a signed distance bias. This is not per-layer, per-token RecurTrace LMA. A learned continuation head reads current state, current probabilities, and depth; oracle future-loss improvements over a margin of 0.01 supervise it on training trajectories only. At deployment it never receives labels or future states. Validation selects checkpoints, fixed depths, fusion, and stopping thresholds; the stopping budget is four. Eight-loop fixed evaluation tests extrapolation.

All auxiliary fits use AdamW, learning rate {config["learning_rate"]}, weight decay 0.01, gradient norm clipping at 1, batch {config["train_batch"]}, {config["epochs"]} epochs, seeds {seeds}. Every fit saves last weights, best weights, optimizer state, and epoch metadata. Checkpoints and cached features stay outside git. Confidence correctness BCE, candidate CE, and continuation BCE are distinct objectives, explicitly recorded.

## Results

{table}

Mean ± sample SD is across training seeds; test/OOD examples are shared. NLL and multiclass Brier describe the option distribution. Correctness ECE/Brier can instead use an independently calibrated probability, so the unchanged-decision confidence row retains the original option NLL. Temperature scaling improves calibration without changing decisions. Confidence fusion worsens NLL even when its argmax improves slightly; higher certainty is not evidence of better reasoning.

{format_text}

{confidence_table}

The learned confidence head ranks correctness better than the original probability on this pilot, but its ECE remains worse than simple temperature/entropy calibration. A constant prior can achieve low ECE while having no ranking ability; ECE alone is insufficient. The held-out prediction-risk metrics support limited usefulness as an uncertainty readout, not a correctness guarantee.

{task_table}

Arithmetic, pointer chasing, and state tracking remain difficult and often near chance. Aggregate gains do not establish reliable multi-step reasoning, and OOD aggregates mix easy and harder task families.

![Accuracy by depth](../results/figures/accuracy_depth.png)

![Calibration](../results/figures/calibration.png)

![Memory](../results/figures/memory_effect.png)

Memory removal after training damages a memory-dependent model, but the stronger comparison is separately trained no-memory recurrence. That comparison provides no consistent memory advantage. Learned stopping reduces mean loops, yet does not convincingly beat validation-selected fixed recurrence in accuracy. Oracle stopping is a label-informed diagnostic only; it is excluded from the deployable comparison table. Risk-coverage, Brier, AUROC, state cosine, magnitude changes, and stopping errors are in the public metric files.

![Halting](../results/figures/halting_depth.png)

## Failure analysis

Eight loops reduce no-memory recurrence to {100 * mean("recurrence_d8"):.2f}% and memory recurrence to {100 * mean("memory_d8"):.2f}% on test data. Extra steps beyond training induce harmful prediction changes and increasing state drift. Learned stopping is a prediction of improvement opportunity, not a guarantee that continuing corrects an answer. The cheap one-pass head is competitive with recurrence and substantially cheaper in added matrix operations.

The unprefixed initial experiment remains in `results/initial/metrics/`: frozen accuracy was 28.83%, dominated by answer-letter bias because the model wanted to start with prose. Its apparent learned gains cannot be interpreted as improved reasoning. The prompt-format audit and all negative controls are retained. Frozen closed-gate recurrence equals the baseline; randomized open-gate updates, memory-disabled inference, fixed depths, confidence stopping, and oracle ceilings are also retained.

Concrete corrected, degraded, and misleading-confidence examples are saved in `results/reports/failure_cases.json`. Such examples illustrate measured transitions; they do not establish a mechanistic account of internal reasoning.

![Corrected and degraded decisions](../results/figures/prediction_changes.png)

## Compute analysis

{parameter_table}

{cost_table}

Estimated extra matrix FLOPs count multiply-add as two operations. They omit nonlinear functions, normalization, host synchronization, and some attention overhead; they are not profiler measurements. Width 256 is approximately matrix-compute-matched to four width-64 refinement loops but has four times the update parameters. The width-64 one-pass head provides a parameter-matched comparator for no-memory recurrence. These are capacity/control comparisons, not exact end-to-end hardware-budget equivalence. No larger-backbone comparison was run.

Fixed-head timing uses the same batch-one cached validation hidden state, three warmups, synchronized execution, and {config.get("latency_repeats", 20)} repetitions. Adaptive timing executes real label-free early stopping over 12 validation examples, also batch one; it includes host decision synchronization. Prompt baseline timing uses those 12 examples separately through the backbone with a full-vocabulary final projection. These head measurements exclude backbone extraction and must not be presented as end-to-end speedups. {baseline_timing_text}

![Accuracy and estimated incremental compute](../results/figures/accuracy_cost.png)

## Middle-layer experiment

We also repeated original layers 12–14 (zero-based) with shared frozen weights and causal masks. A minimal trained variant adds three input-injection scalars, trained at depth two on 240 training examples for five epochs and selected on 60 validation examples. It has no LMA or learned stopping. This is an architecture-2 feasibility probe, distinct from the post-backbone head experiments.

"""
    if middle:
        text += (
            "| Middle-layer variant | Test accuracy | OOD accuracy |\n|---|---:|---:|\n"
        )
        for v in dict.fromkeys(r["variant"] for r in middle):
            test = next(
                (r for r in middle if r["variant"] == v and r["split"] == "test"), None
            )
            ood = next(
                (r for r in middle if r["variant"] == v and r["split"] == "ood"), None
            )
            if test and ood:
                text += f"| {v} | {test['accuracy'] * 100:.2f}% | {ood['accuracy'] * 100:.2f}% |\n"
    text += """
The middle-layer experiment validates interception and tied recurrence, but three scalars and a tiny training subset cannot reproduce RecurTrace's memory training protocol. Depth-one behavior remains unchanged after injection training. The plain repeated-block results are untrained controls; do not attribute them to learning. Depths beyond two are diagnostic and were not used to select the checkpoint.

## Conclusions and next steps

Small supervised heads recovered some useful decision information from the frozen representation. This supports a limited readout hypothesis, not a demonstration of new latent reasoning. Recurrence, memory, and learned stopping remain unproven advantages over a one-pass head at comparable extra compute. More recurrence is demonstrably unsafe as an accuracy assumption in this pilot.

Next, enlarge the held-out set with fresh generators and compare a one-pass readout against a trained middle-layer module under fixed training and compute budgets. Separate format-following failures from task-solving failures, inspect per-task results, and pre-register the comparison before measuring the new test set. Structured tool routing is a plausible target because it has checkable labels, but this pilot only tests synthetic registry lookup. A larger model could improve the representation or saturate easy tasks; neither outcome is established here. Abandon unconditional extra-loop deployment and confidence-only claims of reasoning improvement.

## Reproduction

On Apple Silicon with the optional dependencies installed:

```sh
python -m latent_loop.cli download --config configs/pilot.yaml
python -m latent_loop.cli run --config configs/pilot.yaml --run-dir artifacts/reproduction
python -m latent_loop.cli middle --config configs/pilot.yaml --run-dir artifacts/reproduction
python -m latent_loop.cli report --config configs/pilot.yaml --run-dir artifacts/reproduction
```

The `run` command includes confidence, reranking, no-memory recurrence, memory, both continuation heads, and the two one-pass controls for all three seeds. It also reproduces the key eight-loop negative result. `evaluate` reloads completed checkpoints and recomputes all diagnostics; no fitting is performed for complete stages. Interruptions stop at checkpoint boundaries, then rerunning resumes. Use a new run directory when changing model, dataset, or prompt. Basic CPU tests and public report regeneration require no model download or accelerator.

For the initial format failure, use `configs/initial.yaml` with a fresh run directory. The historical initial stopping policy's no-eligible-threshold fallback was later corrected to minimize validation NLL; initial measurements are archived rather than silently overwritten. This policy implementation detail means the initial adaptive rows are historical exploratory evidence, not an exact current-code target.

Original results are this repository's recorded synthetic experiments. Prior work appears only in `docs/literature_review.md`; its benchmark results are not included in our plots. See the literature review for [RecurTrace](https://arxiv.org/html/2609.03379v2), Universal Transformers, ACT, PonderNet, Encode-Think-Decode, Think-at-Hard, Semantic Entropy Probes, and process supervision. No novelty claim is made for combining these techniques.
"""
    # Selected failure examples are deterministic, not selected for flattering outcomes.
    p = predictions[0]["test"]
    truth = np.array(p["labels"])
    base = np.array(p["baseline"]["logits"])
    head = np.array(p["feedforward"]["logits"])
    cb = base.argmax(-1) == truth
    ch = head.argmax(-1) == truth
    cases = []
    dataset = load("data", "test") if Path("data/test.jsonl").exists() else None
    old_cases = (
        {
            c["id"]: c
            for c in json.loads(Path("results/reports/failure_cases.json").read_text())
        }
        if Path("results/reports/failure_cases.json").exists()
        else {}
    )
    for kind, mask in [
        ("corrected", ~cb & ch),
        ("degraded", cb & ~ch),
        ("confident_error", ~ch & (np.max(softmax(head, axis=-1), axis=-1) > 0.7)),
    ]:
        for i in np.flatnonzero(mask)[:3]:
            case = {
                "category": kind,
                "id": p["ids"][i],
                "label": int(truth[i]),
                "baseline_logits": base[i].tolist(),
                "one_pass_logits": head[i].tolist(),
                "seed": seeds[0],
            }
            if dataset:
                case.update(
                    {
                        k: dataset[i][k]
                        for k in ["task", "prompt", "choices", "difficulty"]
                    }
                )
            elif case["id"] in old_cases:
                case.update(
                    {
                        k: old_cases[case["id"]][k]
                        for k in ["task", "prompt", "choices", "difficulty"]
                        if k in old_cases[case["id"]]
                    }
                )
            cases.append(case)
    atomic_json("results/reports/failure_cases.json", cases)
    summary = {
        "best_variant": best,
        "test_accuracy": mean(best),
        "baseline_accuracy": mean("baseline"),
        "peak_mlx_bytes": peak,
        "primary_auxiliary_fits": len(inputs) * len(inputs[0]["training"]),
        "primary_evaluation_rows": len(rows),
        "seeds": seeds,
        "paired_comparisons": comparisons,
        "extraction_seconds": backbone["extraction_seconds"],
        "auxiliary_fit_seconds": sum(
            d["training"][n]["training_seconds"] for d in inputs for n in d["training"]
        ),
        "middle_evaluation_rows": len(middle),
    }
    atomic_json("results/reports/summary.json", summary)
    Path("reports").mkdir(exist_ok=True)
    Path("reports/research_summary.md").write_text(text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    for path in out.glob("*.png"):
        body = body.replace(
            f"../results/figures/{path.name}",
            "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(),
        )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Latent refinement research pilot</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#18252d}h1,h2{line-height:1.2}table{border-collapse:collapse;font-size:13px;width:100%;display:block;overflow:auto}td,th{border:1px solid #ccd4d9;padding:8px}img{max-width:100%;height:auto}pre{background:#f1f4f6;padding:16px;overflow:auto}a{color:#175795}</style><body>'
        + body
        + "</body></html>"
    )
    Path("reports/research_summary.html").write_text(html)
    print(
        f"Report generated: {len(rows)} primary evaluation rows, {len(inputs)} seeds",
        flush=True,
    )
