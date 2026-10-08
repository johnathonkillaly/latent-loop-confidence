"""Deterministic public synthetic tasks; labels never enter feature extraction."""

import hashlib
import json
import random
from pathlib import Path

VERSION = "synthetic-v1"
TASKS = ("comparison", "arithmetic", "state", "pointer", "routing", "classification")
LABELS = ("A", "B", "C", "D")


def reference(task, spec):
    """Independent reference implementations, operating only on structured inputs."""
    if task == "comparison":
        return spec["names"][max(range(4), key=lambda i: spec["values"][i])]
    if task == "arithmetic":
        # Compose affine transforms independently of the generator's running state.
        a, b = 1, 0
        for op, n in spec["ops"]:
            if op == "add":
                b += n
            elif op == "subtract":
                b -= n
            else:
                a, b = a * n, b * n
        return str(a * spec["start"] + b)
    if task == "state":
        history = [dict(spec["initial"])]
        for op, x, y in spec["ops"]:
            s = dict(history[-1])
            if op == "swap":
                s[x], s[y] = s[y], s[x]
            elif op == "copy":
                s[x] = s[y]
            else:
                s[x] = (s[x] + y) % 10
            history.append(s)
        return str(history[-1][spec["query"]])
    if task == "pointer":
        chain = [spec["start"]]
        for _ in range(spec["steps"]):
            chain.append(spec["mapping"][chain[-1]])
        return chain[-1]
    if task == "routing":
        return spec["registry"][spec["verb"]]
    if task == "classification":
        valid = [name for name, flags in spec["rows"] if all(flags)]
        assert len(valid) == 1
        return valid[0]
    raise ValueError(task)


def generate_one(rng, task, split, serial):
    bucket = {"train": 0, "val": 1, "test": 2, "ood": 3}[split]
    difficulty = rng.choice([1, 2, 3, 4] if split != "ood" else [5, 6, 7, 8])
    # Split-exclusive symbols and numeric residues; templates intentionally shared.
    names = [
        f"{['K', 'L', 'M', 'Z'][bucket]}{rng.randrange(100, 999)}{j}" for j in range(4)
    ]
    number = lambda: rng.randrange(10, 120 if split != "ood" else 500) * 10 + bucket
    if task == "comparison":
        vals = rng.sample(range(10, 120 if split != "ood" else 500), 4)
        vals = [v * 10 + bucket for v in vals]
        spec = {"names": names, "values": vals}
        answer = names[vals.index(max(vals))]
        question = "Which host has the most free memory? " + "; ".join(
            f"{n}: {v} MB" for n, v in zip(names, vals)
        )
        choices = names[:]
        difficulty = 1
    elif task == "arithmetic":
        start, value = number(), None
        value = start
        ops = []
        for _ in range(difficulty):
            op, n = rng.choice(["add", "subtract", "multiply"]), rng.randrange(2, 10)
            ops.append([op, n])
            value = (
                value + n
                if op == "add"
                else value - n
                if op == "subtract"
                else value * n
            )
        spec = {"start": start, "ops": ops}
        answer = str(value)
        choices = [answer] + [
            str(value + x) for x in rng.sample([-21, -11, -3, -1, 1, 3, 11, 21], 3)
        ]
        question = (
            f"Start with {start}. Apply in order: "
            + "; then ".join(f"{op} {n}" for op, n in ops)
            + ". What is the final number?"
        )
    elif task == "state":
        initial = dict(zip(names, [rng.randrange(10) for _ in names]))
        s, ops = dict(initial), []
        for _ in range(difficulty):
            op = rng.choice(["swap", "copy", "increment"])
            x, y = rng.sample(names, 2)
            if op == "increment":
                y = rng.randrange(1, 5)
            ops.append([op, x, y])
            if op == "swap":
                s[x], s[y] = s[y], s[x]
            elif op == "copy":
                s[x] = s[y]
            else:
                s[x] = (s[x] + y) % 10
        query = rng.choice(names)
        spec = {"initial": initial, "ops": ops, "query": query}
        answer = str(s[query])
        choices = [answer] + list(
            map(str, rng.sample([v for v in range(10) if str(v) != answer], 3))
        )
        question = (
            "Registers: "
            + "; ".join(f"{k}={v}" for k, v in initial.items())
            + ". Apply in order: "
        )
        question += "; then ".join(
            f"swap {x} and {y}"
            if op == "swap"
            else f"copy {y} into {x}"
            if op == "copy"
            else f"increment {x} by {y} modulo 10"
            for op, x, y in ops
        )
        question += f". Final value of {query}?"
    elif task == "pointer":
        nodes = [
            f"{['K', 'L', 'M', 'Z'][bucket]}{rng.randrange(100, 999)}{j}"
            for j in range(8)
        ]
        targets = rng.sample(nodes, len(nodes))
        mapping = dict(zip(nodes, targets))
        start = rng.choice(nodes)
        current = start
        for _ in range(difficulty):
            current = mapping[current]
        spec = {"mapping": mapping, "start": start, "steps": difficulty}
        answer = current
        choices = [answer] + rng.sample([n for n in nodes if n != answer], 3)
        question = "Mapping: " + "; ".join(f"{k}->{v}" for k, v in mapping.items())
        question += (
            f". Start at {start}; follow exactly {difficulty} links. Where do you end?"
        )
    elif task == "routing":
        verbs = ["read", "write", "calculate", "search"]
        registry = dict(zip(verbs, rng.sample(names, 4)))
        verb = rng.choice(verbs)
        spec = {"registry": registry, "verb": verb}
        answer, choices = registry[verb], names[:]
        question = (
            "Tool registry: "
            + "; ".join(f"{v} requests use {n}" for v, n in registry.items())
            + f". Route this request: {verb}."
        )
        difficulty = 1
    else:
        winner = rng.randrange(4)
        rows = [
            [
                n,
                [True, True]
                if j == winner
                else rng.choice([[False, True], [True, False], [False, False]]),
            ]
            for j, n in enumerate(names)
        ]
        spec = {"rows": rows}
        answer, choices = names[winner], names[:]
        question = "Select the item which is both ready and authorized: " + "; ".join(
            f"{n}: ready={a}, authorized={b}" for n, (a, b) in rows
        )
        difficulty = 1
    assert answer == reference(task, spec)
    rng.shuffle(choices)
    label = choices.index(answer)
    prompt = (
        question
        + "\n"
        + "\n".join(f"{l}. {c}" for l, c in zip(LABELS, choices))
        + "\nAnswer with only the correct option letter."
    )
    canonical = json.dumps([task, spec], sort_keys=True, separators=(",", ":"))
    return {
        "id": hashlib.sha256(canonical.encode()).hexdigest(),
        "task": task,
        "split": split,
        "difficulty": difficulty,
        "prompt": prompt,
        "choices": choices,
        "label": label,
        "spec": spec,
    }


def generate(root, sizes, seed=1701):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    seen = set()
    manifest = {"version": VERSION, "seed": seed, "splits": {}}
    for si, (split, size) in enumerate(sizes.items()):
        rng = random.Random(seed + 100000 * si)
        rows = []
        while len(rows) < size:
            row = generate_one(rng, TASKS[len(rows) % len(TASKS)], split, len(rows))
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            rows.append(row)
        content = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
        (root / f"{split}.jsonl").write_text(content)
        manifest["splits"][split] = {
            "size": size,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(root, split):
    return [
        json.loads(line)
        for line in (Path(root) / f"{split}.jsonl").read_text().splitlines()
    ]
