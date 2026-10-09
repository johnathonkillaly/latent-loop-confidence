"""Deterministic public tasks. Solvers are used for labels, never policy inputs."""

import gzip
import hashlib
import itertools
import json
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path("results/experiment_05e")
OPERATIONS = ["inspect", "copy", "restart", "erase", "compress", "verify"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        with gzip.GzipFile(filename="", mode="wb", fileobj=f, mtime=0) as z:
            z.write("".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows).encode())


def read_rows(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def graph_distances(n, edges, goal, directed=False):
    reverse = [[] for _ in range(n)]
    for a, b in edges:
        reverse[b].append(a)
        if not directed:
            reverse[a].append(b)
    d = [n + 1] * n
    d[goal] = 0
    q = deque([goal])
    while q:
        u = q.popleft()
        for v in reverse[u]:
            if d[v] > d[u] + 1:
                d[v] = d[u] + 1
                q.append(v)
    return d


def graph_fingerprint(n, edges, directed):
    """Unrooted WL invariant: equal hashes conservatively rejected, not proof of isomorphism."""
    adj = [set() for _ in range(n)]
    rev = [set() for _ in range(n)]
    for a, b in edges:
        adj[a].add(b)
        rev[b].add(a)
        if not directed:
            adj[b].add(a)
            rev[a].add(b)
    colors = [digest([len(adj[v]), len(rev[v])]) for v in range(n)]
    for _ in range(n):
        colors = [digest([colors[v], sorted(colors[u] for u in adj[v]), sorted(colors[u] for u in rev[v])]) for v in range(n)]
    return digest([n, directed, sorted(colors)])


def solve(row, independent=False):
    r = row["raw"]
    if row["family"] == "routing":
        valid = []
        for i, t in enumerate(r["tools"]):
            if independent:
                caps = sum(1 << OPERATIONS.index(o) for o in t["ops"])
                ok = bool(caps & (1 << OPERATIONS.index(r["request"])))
                ok &= t["online"] and t["clearance"] >= r["required"] and t["cost"] <= r["budget"]
                ok &= not (r["deny_unsafe"] and t["unsafe"])
            else:
                ok = (r["request"] in t["ops"] and t["online"] and t["clearance"] >= r["required"] and t["cost"] <= r["budget"] and not (r["deny_unsafe"] and t["unsafe"]))
            if ok:
                valid.append(i)
        return valid or [3]
    if row["family"] == "state":
        state = list(r["initial"])
        if independent:
            # A functional state transition interpreter, separate from the mutable one.
            for op, a, b in r["program"]:
                old = tuple(state)
                if op == "swap":
                    state = [old[1], old[0]]
                else:
                    fn = {"add": lambda: old[a] + b, "sub": lambda: old[a] - b,
                          "set": lambda: b, "copy": lambda: old[b], "mul": lambda: old[a] * b}[op]
                    state = [fn() if j == a else old[j] for j in range(2)]
        else:
            for op, a, b in r["program"]:
                if op == "swap":
                    state.reverse()
                elif op == "add":
                    state[a] += b
                elif op == "sub":
                    state[a] -= b
                elif op == "set":
                    state[a] = b
                elif op == "copy":
                    state[a] = state[b]
                elif op == "mul":
                    state[a] *= b
        return [r["values"].index(state[r["query"]])]
    n, edges, goal = r["n"], r["edges"], r["goal"]
    if independent:
        d = np.full((n, n), n + 1, dtype=int)
        np.fill_diagonal(d, 0)
        for a, b in edges:
            d[a, b] = 1
            if not r["directed"]:
                d[b, a] = 1
        for k in range(n):
            d = np.minimum(d, d[:, k, None] + d[None, k, :])
        distances = d[:, goal].tolist()
    else:
        distances = graph_distances(n, edges, goal, r["directed"])
    if distances[0] >= n + 1:
        return [3]
    return [i for i, c in enumerate(r["neighbors"]) if distances[c] + 1 == distances[0]]


def generate_raw(family, rng, split):
    structural = split == "structure"
    hard = split == "harder"
    if family == "routing":
        request = str(rng.choice(OPERATIONS))
        required, budget = int(rng.integers(1, 4)), int(rng.integers(3, 10))
        tools = []
        for _ in range(3):
            ops = sorted(rng.choice(OPERATIONS, size=int(rng.integers(2, 6)), replace=False).tolist())
            tools.append({"ops": ops, "online": bool(rng.random() < .85), "clearance": int(rng.integers(1, 5)), "cost": int(rng.integers(1, 10)), "unsafe": bool(rng.random() < .35)})
        raw = {"request": request, "required": required, "budget": budget, "deny_unsafe": structural or hard, "tools": tools}
        fingerprint = min(digest({**raw, "tools": list(p)}) for p in itertools.permutations(tools))
        return raw, fingerprint, {"difficulty": required + (structural or hard), "generator": "deny_override" if structural else "capability_constraints"}
    if family == "state":
        initial = rng.integers(-5, 10, size=2).tolist()
        steps = int(rng.integers(8, 13) if hard else rng.integers(1, 4))
        allowed = ["add", "sub", "set", "swap"] + (["copy", "mul"] if structural else [])
        program = []
        state = initial.copy()
        for _ in range(steps):
            op, a, b = str(rng.choice(allowed)), int(rng.integers(2)), int(rng.integers(1, 6))
            if op == "copy":
                b = 1 - a
                state[a] = state[b]
            elif op == "mul":
                b = int(rng.integers(2, 4)); state[a] *= b
            elif op == "swap":
                a, b = 0, 1; state.reverse()
            elif op == "set":
                state[a] = b
            elif op == "add":
                state[a] += b
            else:
                state[a] -= b
            program.append([op, a, b])
        query = int(rng.integers(2)); answer = state[query]
        distractors = set()
        while len(distractors) < 3:
            v = answer + int(rng.integers(-9, 10))
            if v != answer:
                distractors.add(v)
        values = [answer] + sorted(distractors)
        rng.shuffle(values)
        raw = {"initial": initial, "program": program, "query": query, "values": values}
        # Alpha-renaming the two registers gives the same canonical instance.
        renamed = {"initial": initial[::-1], "query": 1 - query, "program": [[op, 1-a if op != "swap" else 0, 1-b if op == "copy" else b] for op, a, b in program]}
        plain = {k: raw[k] for k in ("initial", "program", "query")}
        return raw, min(digest(plain), digest(renamed)), {"difficulty": steps, "generator": "copy_multiply" if structural else "register_updates"}
    n = int(rng.integers(16, 21) if hard else rng.integers(9, 13))
    directed = structural
    neighbors = rng.choice(np.arange(1, n), size=3, replace=False).tolist()
    edges = {(0, v) for v in neighbors}
    for a in range(1, n):
        for b in range(a + 1, n):
            if rng.random() < (.13 if hard else .18):
                edges.add((a, b))
    # Random latent order for directed DAGs; presentation identifiers are independently shuffled.
    edges = sorted(edges)
    goal = int(rng.integers(1, n))
    distances = graph_distances(n, edges, goal, directed)
    if hard and not (5 <= distances[0] <= n):
        return None
    raw = {"n": n, "edges": edges, "goal": goal, "neighbors": neighbors, "directed": directed}
    return raw, graph_fingerprint(n, edges, directed), {"difficulty": distances[0] if distances[0] <= n else -1, "generator": "directed_DAG" if directed else "random_undirected", "nodes": n, "edges": len(edges)}


def render(row, rng, style="natural"):
    """Only raw observations and candidate descriptions enter the prompt."""
    r = row["raw"]
    labels = list("ABCD"); rng.shuffle(labels)
    order = rng.permutation(4).tolist() if style == "format" else sorted(range(4), key=lambda i: labels[i])
    if row["family"] == "routing":
        names = rng.choice(["Luma", "Nori", "Vela", "Pavo", "Tavi", "Zeno", "Kiri", "Miro"], 3, replace=False).tolist()
        descriptions = [f"use {name}" for name in names] + ["abstain: no permitted tool"]
        lines = [f"{name}: operations={','.join(t['ops'])}; online={str(t['online']).lower()}; clearance={t['clearance']}; cost={t['cost']}; unsafe={str(t['unsafe']).lower()}" for name, t in zip(names, r["tools"])]
        rng.shuffle(lines)
        rule = "Unsafe tools are forbidden even if all other conditions hold. " if r["deny_unsafe"] else ""
        body = (f"Request: {r['request']}. Required clearance >= {r['required']}; cost <= {r['budget']}. Only online tools supporting the requested operation may be used. {rule}Choose any permitted tool, or abstain if none exists.\n" + "\n".join(lines))
    elif row["family"] == "state":
        names = rng.choice(["p", "q", "u", "v", "m", "n", "r", "s"], 2, replace=False).tolist()
        lines = []
        for op, a, b in r["program"]:
            if op == "swap":
                lines.append(f"swap {names[0]} and {names[1]}")
            elif op == "copy":
                lines.append(f"{names[a]} := {names[b]}")
            else:
                symbol = {"add": "+", "sub": "-", "set": "", "mul": "*"}[op]
                lines.append(f"{names[a]} := " + (f"{names[a]} {symbol} " if symbol else "") + str(b))
        body = f"Integer registers initially: {names[0]}={r['initial'][0]}, {names[1]}={r['initial'][1]}. Execute every instruction in order:\n" + "; ".join(lines) + f".\nWhat is the final value of {names[r['query']]}?"
        descriptions = [str(v) for v in r["values"]]
    else:
        names = [f"N{v}" for v in rng.permutation(np.arange(10, 10+r["n"])).tolist()]
        edges = list(r["edges"]); rng.shuffle(edges)
        sep = "->" if r["directed"] else "--"
        body = (f"Graph edges ({'directed; follow arrows only' if r['directed'] else 'undirected; travel either way'}): " + ", ".join(names[a]+sep+names[b] for a, b in edges) + f".\nStart={names[0]}; goal={names[r['goal']]}. Choose a first move on any shortest route to the goal. Abstain only if no route exists.")
        descriptions = [f"move to {names[v]}" for v in r["neighbors"]] + ["abstain: no route"]
    options = [f"({labels[i]}) {descriptions[i]}" for i in order]
    if style == "wording":
        intro = "Determine the permitted answer from the facts below. Several answers may be valid; choose any valid one.\n"
        body = body.replace("Choose", "Select").replace("Request:", "Requested operation:")
    else:
        intro = "Solve the following deterministic task.\n"
    if style == "format":
        options = [f"label={labels[i]} | candidate={descriptions[i]}" for i in order]
    prompt = intro + body + "\nCandidates:\n" + "\n".join(options) + "\nRespond with one candidate label only."
    return {**row, "prompt": prompt, "labels": labels, "candidate_descriptions": descriptions, "presentation_order": order, "style": style}


def generate_split(split, size, seed, seen):
    rng = np.random.default_rng(seed)
    rows, attempts = [], 0
    while len(rows) < size:
        family = ["routing", "state", "graph"][len(rows) % 3]
        result = generate_raw(family, rng, split)
        attempts += 1
        if attempts > size * 1000:
            raise RuntimeError("Generator exhausted prospective rejection budget")
        if result is None:
            continue
        raw, key, metadata = result
        key = family + ":" + key
        if key in seen:
            continue
        seen.add(key)
        row = {"id": digest([split, key]), "semantic_hash": key, "family": family, "raw": raw, **metadata}
        row["optimal"] = solve(row)
        if row["optimal"] != solve(row, independent=True):
            raise AssertionError("Independent solver disagreement")
        style = split if split in ("wording", "format") else "natural"
        rows.append(render(row, rng, style))
    return rows, attempts


def generate(config, pilot=False):
    sizes = {"pilot": config["pilot_size"]} if pilot else config["sizes"]
    seen = set()
    manifest = {"seed": config["dataset_seed"], "splits": {}, "deduplication": "routing: tool permutation; state: register alpha-renaming; graph: conservative unrooted WL invariant. Not complete program equivalence or graph isomorphism."}
    for j, (split, count) in enumerate(sizes.items()):
        rows, attempts = generate_split(split, count, config["dataset_seed"] + j * 1009 + (909 if pilot else 0), seen)
        path = ROOT / "datasets" / f"{split}.jsonl.gz"
        write_rows(path, rows)
        manifest["splits"][split] = {"count": len(rows), "attempts": attempts, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "family_counts": {f: sum(r["family"] == f for r in rows) for f in ("routing", "state", "graph")}, "optimal_label_prevalence": {l: sum(l in [r["labels"][i] for i in r["optimal"]] for r in rows) / len(rows) for l in "ABCD"}, "multiple_optimal": sum(len(r["optimal"]) > 1 for r in rows), "abstain": sum(r["family"] != "state" and r["optimal"] == [3] for r in rows)}
    if not pilot:
        permutations = []
        for split in ("id", "structure"):
            originals = read_rows(ROOT / "datasets" / f"{split}.jsonl.gz")[:120]
            for row in originals:
                for view in range(2):
                    rng = np.random.default_rng(int(row["id"][:12],16)+view)
                    changed = render(row, rng)
                    if view == 0:
                        changed["candidate_descriptions"] = row["candidate_descriptions"]
                        changed["prompt"] = row["prompt"].split("\nCandidates:\n")[0] + "\nCandidates:\n" + "\n".join(f"({changed['labels'][j]}) {row['candidate_descriptions'][j]}" for j in changed["presentation_order"]) + "\nRespond with one candidate label only."
                    changed.update({"id": digest([row["id"],view]), "parent_id": row["id"], "parent_split": split, "permutation_kind": "candidate_relabel" if view == 0 else "symbol_and_candidate_relabel"})
                    permutations.append(changed)
        path = ROOT / "datasets/permutations.jsonl.gz"
        write_rows(path, permutations)
        manifest["splits"]["permutations"] = {"count":len(permutations), "independent_parents":240, "sha256":hashlib.sha256(path.read_bytes()).hexdigest(), "intentional_duplicates": "paired presentation views of ID/structural examples, never training data"}
    (ROOT / ("pilot_dataset_manifest.json" if pilot else "dataset_manifest.json")).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
