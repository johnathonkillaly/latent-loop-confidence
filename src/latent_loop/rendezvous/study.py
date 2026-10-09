"""Training, prospective selection, and actual candidate-choice evaluation."""

import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np

from .data import ROOT, digest, read_rows, write_rows
from .features import CACHE, load_features

ART = Path("artifacts/experiment_05e")
CONDITIONS = ("correctness", "ranking", "c1", "c2", "no_logits", "logits_only", "description_only", "residual")


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def softmax(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def choose(z):
    # Semantic candidate index is independent of display labels/order. No tolerance changes.
    return np.argmax(np.asarray(z, dtype=np.float64), axis=-1)


def normalized(q):
    q = np.asarray(q, dtype=np.float64)
    q = q - q.mean(axis=-1, keepdims=True)
    return q / np.maximum(np.sqrt(np.mean(q*q, axis=-1, keepdims=True)), 1)


def accuracy(z, y):
    return float(y[np.arange(len(y)), choose(z)].mean())


def nll(z, y):
    return float(-np.log(np.maximum((softmax(z)*y).sum(axis=-1), 1e-15)).mean())


def checkpoint(condition, seed):
    return ART / "checkpoints" / f"{condition}_{seed}.safetensors"


def predictions(condition, seed, x):
    import mlx.core as mx
    from .models import CandidateHead, view
    model = CandidateHead(x.shape[-1])
    model.load_weights(str(checkpoint(condition, seed)))
    model.eval()
    output = []
    for start in range(0, len(x), 256):
        z = model(view(mx.array(x[start:start+256]), condition))
        mx.eval(z); output.append(np.array(z))
    return np.concatenate(output)


def train(config):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from .models import CandidateHead, objective, view

    if (ROOT / "final_openings.json").exists():
        raise RuntimeError("Final tests opened: training is locked. Use a new protocol for further research.")
    _, train = load_features("train"); _, validation = load_features("validation")
    # First validation half selects checkpoints; second half selects fusion/temperature.
    v = {k: a[:len(validation["y"])//2] for k, a in validation.items() if isinstance(a, np.ndarray)}
    X, Y, Z = mx.array(train["x"]), mx.array(train["y"]), mx.array(train["logits"])
    VX, VY, VZ = mx.array(v["x"]), mx.array(v["y"]), mx.array(v["logits"])
    directory = ART / "checkpoints"; directory.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        for seed in config["training_seeds"]:
            summary_path = ROOT / "training" / f"{condition}_{seed}.json"
            if summary_path.exists() and json.loads(summary_path.read_text()).get("complete"):
                continue
            mx.random.seed(seed)
            model = CandidateHead(X.shape[-1], config["hidden_width"])
            optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
            mx.eval(model.parameters())
            fit_signature = digest({"config": config, "condition": condition, "feature_train": json.loads((CACHE / "train/signature.json").read_text())})
            state_path = directory / f"{condition}_{seed}_resume.npz"
            history, first_epoch, best = [], 0, (-1, -float("inf"))
            if summary_path.exists():
                previous = json.loads(summary_path.read_text())
                if previous["signature"] != fit_signature:
                    raise RuntimeError("Fit signature changed")
                history = previous["epochs"]; first_epoch = len(history)
                best = tuple(previous["best_metric"])
                model.load_weights(str(directory / f"{condition}_{seed}_latest.safetensors"))
                from mlx.utils import tree_unflatten
                optimizer.state = tree_unflatten(list(mx.load(str(state_path)).items()))
            def loss_fn(m, xx, yy, zz):
                return objective(m(view(xx, condition)), yy, condition, zz)
            gradient_fn = nn.value_and_grad(model, loss_fn)
            def step(xx, yy, zz):
                value, grads = gradient_fn(model, xx, yy, zz)
                grad_norm = mx.sqrt(sum(mx.sum(g*g) for _, g in tree_flatten(grads)))
                grads = optim.clip_grad_norm(grads, 1.0)[0]
                optimizer.update(model, grads)
                return value, grad_norm
            tic = time.perf_counter()
            for epoch in range(first_epoch, config["epochs"]):
                rng = np.random.default_rng(seed * 1000 + epoch)
                order = rng.permutation(len(Y))
                values, norms = [], []
                for start in range(0, len(Y), config["batch_size"]):
                    idx = mx.array(order[start:start+config["batch_size"]])
                    loss, norm = step(X[idx], Y[idx], Z[idx])
                    mx.eval(model.parameters(), optimizer.state, loss, norm)
                    values.append(float(loss.item())); norms.append(float(norm.item()))
                q = model(view(VX, condition))
                if condition == "residual":
                    q = q + VZ
                mx.eval(q); z = np.array(q)
                acc, vnll = accuracy(z, v["y"]), nll(z, v["y"])
                if (acc, -vnll) > best:
                    best = (acc, -vnll)
                    model.save_weights(str(checkpoint(condition, seed)))
                history.append({"epoch": epoch+1, "loss": float(np.mean(values)), "validation_accuracy": acc, "validation_nll": vnll, "gradient_norm": float(np.mean(norms)), "cumulative_this_invocation_seconds": time.perf_counter()-tic})
                model.save_weights(str(directory / f"{condition}_{seed}_latest.safetensors"))
                mx.savez(str(state_path), **dict(tree_flatten(optimizer.state)))
                summary = {"condition": condition, "seed": seed, "signature": fit_signature, "parameters": sum(p.size for _, p in tree_flatten(model.parameters())), "epochs": history, "best_metric": best, "complete": epoch+1 == config["epochs"], "fit_seconds": time.perf_counter()-tic, "peak_mlx_gib": mx.get_peak_memory()/1024**3, "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**3, "checkpoint_sha256": hashlib.sha256(checkpoint(condition, seed).read_bytes()).hexdigest()}
                save(summary_path, summary)
            print(f"trained {condition} seed={seed} best={best[0]:.4f} seconds={summary['fit_seconds']:.1f}", flush=True)


def freeze(config):
    if (ROOT / "final_openings.json").exists():
        raise RuntimeError("Cannot change the protocol after final opening")
    rows, f = load_features("validation")
    cut = len(rows)//2; x, y, base = f["x"][cut:], f["y"][cut:], f["logits"][cut:]
    candidates = []
    for objective in config["primary_objective_candidates"]:
        qs = [predictions(objective, seed, x) for seed in config["training_seeds"]]
        for lam in config["lambda_grid"]:
            accs = [accuracy(base+lam*normalized(q), y) for q in qs]
            candidates.append({"objective": objective, "lambda": lam, "accuracy": float(np.mean(accs)), "seed_accuracies": accs})
    selected = max(candidates, key=lambda r: (r["accuracy"], -r["lambda"], r["objective"] == "correctness"))
    temperatures = [{"temperature": float(t), "nll": nll(base/t, y)} for t in np.geomspace(.25, 4, 81)]
    temperature = min(temperatures, key=lambda r: r["nll"])
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("src/latent_loop/rendezvous").glob("*.py"))}
    ckpts = {str(checkpoint(c, s)): hashlib.sha256(checkpoint(c, s).read_bytes()).hexdigest() for c in CONDITIONS for s in config["training_seeds"]}
    result = {"selected": selected, "all_validation_fusions": candidates, "temperature": temperature, "primary_population": "structure: 400 routing deny-override, 400 copy/multiply register programs, 400 directed DAG routes", "primary_comparison": "same frozen BaseLM versus BaseLM + selected candidate Q; three seeds averaged; paired graph/task-instance and seed bootstrap 95% interval", "strength_check": "selected fusion versus same-feature, same-parameter C2 set-aware direct classifier", "other_tests": "secondary/exploratory; no final selection", "config": config, "source_sha256": sources, "checkpoint_sha256": ckpts, "dataset_manifest_sha256": hashlib.sha256((ROOT / "dataset_manifest.json").read_bytes()).hexdigest(), "permutation_rule": "first 120 ID and first 120 structural instances; two independent presentation/identifier permutations each; cluster by original instance", "latency_rule": "validation prompts only; interleaved base/Q/C2, synchronized full prefix, tokenization and final selection included; conditional four-label projection", "prospective_threshold": "primary 95% paired interval lower bound > 0; classifier superiority assessed separately"}
    save(ROOT / "protocol_freeze.json", result)
    print(json.dumps(selected, indent=2), flush=True)


def paired_ci(difference, seed=5454, repetitions=10000):
    """Two-way paired bootstrap: independent examples and fitted seeds, not 3N observations."""
    a = np.asarray(difference, dtype=float)
    if a.ndim == 1:
        a = a[None]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(repetitions):
        s = rng.integers(a.shape[0], size=a.shape[0]); i = rng.integers(a.shape[1], size=a.shape[1])
        draws.append(a[s][:, i].mean())
    return {"difference": float(a.mean()), "ci95": np.quantile(draws, [.025, .975]).tolist(), "seed_differences": a.mean(axis=1).tolist(), "independent_instances": a.shape[1], "training_seeds": a.shape[0]}


def evaluate(config):
    frozen = json.loads((ROOT / "protocol_freeze.json").read_text())
    for file, sha in frozen["checkpoint_sha256"].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != sha:
            raise RuntimeError("Frozen checkpoint changed")
    openings_path = ROOT / "final_openings.json"
    openings = json.loads(openings_path.read_text()) if openings_path.exists() else []
    selected = frozen["selected"]
    for split in ("id", "wording", "format", "structure", "harder", "permutations"):
        target = ROOT / "decisions" / f"{split}.jsonl.gz"
        if target.exists():
            continue
        if not (ROOT / "datasets" / f"{split}.jsonl.gz").exists():
            continue
        openings.append({"split": split, "opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "purpose": "fixed final evaluation"})
        save(openings_path, openings)
        rows, f = load_features(split); x, y, base = f["x"], f["y"], f["logits"]
        all_q = {c: [predictions(c, s, x) for s in config["training_seeds"]] for c in CONDITIONS}
        import mlx.core as mx
        from .models import CandidateHead
        random_heads=[]
        for seed in config["training_seeds"]:
            mx.random.seed(seed+91000)
            untrained=CandidateHead(x.shape[-1],config["hidden_width"])
            z=untrained(mx.array(x));mx.eval(z);random_heads.append(np.array(z))
        decisions = []
        rng = np.random.default_rng(90901)
        label_prior_rows = read_rows(ROOT / "datasets/train.jsonl.gz")
        prevalence = {l: sum(l in [r["labels"][j] for j in r["optimal"]] for r in label_prior_rows)/len(label_prior_rows) for l in "ABCD"}
        label_bias = np.array([[np.log(prevalence[l]) for l in r["labels"]] for r in rows])
        for i, row in enumerate(rows):
            record = {"id": row["id"], "family": row["family"], "difficulty": row["difficulty"], "optimal": row["optimal"], "labels": row["labels"], "base_logits": base[i].tolist(), "base_choice": int(choose(base[i])), "base_probabilities": softmax(base[i]).tolist(), "token_length": int(f["lengths"][i]), "unrestricted_token": int(f["greedy"][i]), "seeds": []}
            if "parent_id" in row:
                record["parent_id"] = row["parent_id"]
            for j, seed in enumerate(config["training_seeds"]):
                qs = {c: all_q[c][j][i] for c in CONDITIONS}
                q = qs[selected["objective"]]
                fused = base[i] + selected["lambda"]*normalized(q)
                shuffled = q[rng.permutation(4)]
                random = rng.normal(size=4)
                prior = np.zeros(4); prior[3 if row["family"] != "state" else 0] = 1
                logits = {"base": base[i], "fusion": fused, "q_only": q, "temperature": base[i]/frozen["temperature"]["temperature"], "constant": base[i], "random": base[i]+selected["lambda"]*normalized(random), "random_shared_head":base[i]+selected["lambda"]*normalized(random_heads[j][i]), "random_q_only":random_heads[j][i], "shuffled": base[i]+selected["lambda"]*normalized(shuffled), "label_bias": base[i]+label_bias[i], "abstain_prior": prior, **{c: z + (base[i] if c == "residual" else 0) for c, z in qs.items()}}
                record["seeds"].append({"seed": seed, "choices": {name: int(choose(z)) for name, z in logits.items()}, "scores": {name: np.asarray(z, dtype=float).tolist() for name, z in logits.items()}})
            decisions.append(record)
        write_rows(target, decisions)
        print(f"evaluated {split}: base={accuracy(base,y):.4f} fusion={np.mean([r['seeds'][j]['choices']['fusion'] in r['optimal'] for r in decisions for j in range(3)]):.4f}", flush=True)
    from .report import summarize
    summarize(config)


def run(command, config):
    if command == "train":
        train(config)
    elif command == "freeze":
        freeze(config)
    elif command == "evaluate":
        evaluate(config)
    elif command == "timing":
        from .timing import measure
        measure(config)
    elif command == "report":
        from .report import report
        report(config)
    elif command == "audit":
        from .audit import audit
        audit(config)
