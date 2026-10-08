"""Independent semantic generators, solvers, and a prospective sealed dataset."""

import gzip
import hashlib
import itertools
import json
import random
from fractions import Fraction
from pathlib import Path

from latent_loop.training import atomic_json

EXP = Path("experiments/experiment_04a")
OUT = Path("results/experiment_04a")
CACHE = Path("artifacts/experiment_04a")
TASKS = ("arithmetic", "comparison", "state", "pointer", "classification", "routing")
FAMILIES = {
    "arithmetic": ("accumulator", "expression_tree", "parallel_ledgers"),
    "comparison": ("mixed_units", "resource_filter", "value_per_cost"),
    "state": ("register_program", "inventory_transfers", "boolean_program"),
    "pointer": ("functional_graph", "alternating_maps", "symbolic_modular_walk"),
    "classification": (
        "numeric_conjunction",
        "disjunction_negation",
        "weighted_conditions",
    ),
    "routing": ("capability_constraints", "incomplete_request", "ordered_rules"),
}
FINAL = ("fresh_id", "new_template", "structural_shift", "harder_depth")
VERSION = "foundation-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(family, s):
    """Independent declarative/algebraic solvers; generators use other evaluators."""
    if family == "accumulator":
        a, b = 1, 0
        for op, v in s["ops"]:
            if op == "multiply":
                a, b = a * v, b * v
            else:
                b += v if op == "add" else -v
        return str(a * s["initial"] + b)
    if family == "expression_tree":

        def expression(node):
            return (
                str(node)
                if isinstance(node, int)
                else "("
                + expression(node[1])
                + {"add": "+", "subtract": "-", "multiply": "*"}[node[0]]
                + expression(node[2])
                + ")"
            )

        return str(eval(expression(s["tree"]), {"__builtins__": {}}, {}))
    if family == "parallel_ledgers":
        query = s["query"]
        value = s["initial"][query]
        for op, index, amount in s["ops"]:
            value += (
                amount * (1 if index == query else -1)
                if op == "transfer"
                else (amount if index == query else 0)
            )
        return str(value)
    if family in ("mixed_units", "resource_filter", "value_per_cost"):
        if family == "mixed_units":
            scores = [
                v * {"KB": 1024, "MB": 1024**2, "GB": 1024**3}[unit]
                for v, unit in s["items"]
            ]
            return max(range(4), key=lambda i: scores[i])
        if family == "resource_filter":
            eligible = [
                i for i, (_, cpu) in enumerate(s["items"]) if cpu >= s["minimum_cpu"]
            ]
            return max(eligible, key=lambda i: s["items"][i][0])
        eligible = [
            i for i, (ram, _) in enumerate(s["items"]) if ram >= s["minimum_ram"]
        ]
        return max(eligible, key=lambda i: Fraction(*s["items"][i]))
    if family in ("register_program", "inventory_transfers"):
        # Execute affine transforms by a homogeneous matrix, not generator state updates.
        values = s["initial"] + [1]
        for op, i, j in s["ops"]:
            matrix = [[int(a == b) for b in range(4)] for a in range(4)]
            if op == "add":
                matrix[i][3] += j
            elif op == "reset":
                matrix[i] = [0, 0, 0, j]
            elif op == "swap":
                matrix[i], matrix[j] = matrix[j], matrix[i]
            else:
                matrix[i][3] -= j
                matrix[(i + 1) % 3][3] += j
            values = [sum(v * w for v, w in zip(row, values)) for row in matrix]
        return str(values[s["query"]])
    if family == "boolean_program":
        # Bit-mask operations provide an independent representation of flag updates.
        bits = sum(int(v) << i for i, v in enumerate(s["initial"]))
        for op, i, v in s["ops"]:
            if op == "toggle":
                bits ^= 1 << i
            elif op == "set":
                bits = bits | (1 << i) if v else bits & ~(1 << i)
            else:
                if (bits >> v) & 1:
                    bits ^= 1 << i
        return "".join(str((bits >> i) & 1) for i in s["query"])
    if family in ("functional_graph", "alternating_maps"):
        current = s["start"]
        # Binary matrix/vector transitions instead of following stored pointers directly.
        vector = [int(i == current) for i in range(len(s["maps"][0]))]
        for k in range(s["steps"]):
            mapping = s["maps"][k % len(s["maps"])]
            vector = [
                sum(vector[i] for i, target in enumerate(mapping) if target == j)
                for j in range(len(vector))
            ]
        return vector.index(1)
    if family == "symbolic_modular_walk":
        value = s["start"]
        for i in range(s["steps"]):
            a, b = s["rules"][i % len(s["rules"])]
            value = (a * value + b) % s["modulus"]
        return str(value)
    if family in ("numeric_conjunction", "disjunction_negation", "weighted_conditions"):
        valid = []
        for i, (a, b, c) in enumerate(s["items"]):
            if family == "numeric_conjunction":
                ok = a >= s["thresholds"][0] and b < s["thresholds"][1] and c == 1
            elif family == "disjunction_negation":
                ok = (a >= s["thresholds"][0] or b >= s["thresholds"][1]) and not c
            else:
                ok = (
                    sum(
                        int(x >= t) * w
                        for x, t, w in zip([a, b, c], s["thresholds"], s["weights"])
                    )
                    >= s["target"]
                )
            if ok:
                valid.append(i)
        assert len(valid) == 1
        return valid[0]
    if family in ("capability_constraints", "incomplete_request"):
        q = s["request"]
        if family == "incomplete_request":
            if q["jobs"] is None:
                return "Request more information"
            if q["write"] and q["read_only"]:
                return "No valid tool"
        valid = [
            i
            for i, t in enumerate(s["tools"])
            if set(q["verbs"]).issubset(t["verbs"])
            and t["capacity"] >= q["jobs"]
            and (not q["write"] or t["write"])
        ]
        if not valid:
            return "No valid tool"
        return min(valid, key=lambda i: s["tools"][i]["priority"])
    if family == "ordered_rules":
        q = s["request"]
        for i, (minimum, encrypted, writable, action) in enumerate(s["rules"]):
            if (
                q["urgency"] >= minimum
                and (encrypted == -1 or q["encrypted"] == encrypted)
                and (writable == -1 or q["write"] == writable)
            ):
                return action
        return s["default"]
    raise ValueError(family)


def graph_fingerprint(maps, root, steps):
    """Conservative WL invariant: isomorphic renamed graphs always collide/reject."""
    n = len(maps[0])
    colors = [digest(["root" if i == root else "node"]) for i in range(n)]
    for _ in range(n):
        colors = [
            digest(
                [
                    colors[i],
                    [(k, colors[m[i]]) for k, m in enumerate(maps)],
                    sorted(
                        (k, colors[j])
                        for k, m in enumerate(maps)
                        for j, t in enumerate(m)
                        if t == i
                    ),
                ]
            )
            for i in range(n)
        ]
    return digest([steps, sorted(colors), colors[root]])


def structure(family, s):
    if family in ("functional_graph", "alternating_maps"):
        return digest([family, graph_fingerprint(s["maps"], s["start"], s["steps"])])
    if family in (
        "register_program",
        "inventory_transfers",
        "parallel_ledgers",
        "boolean_program",
    ):
        candidates = []
        for perm in itertools.permutations(range(len(s["initial"]))):
            initial = [0] * len(perm)
            for i, v in enumerate(s["initial"]):
                initial[perm[i]] = v
            ops = [
                [op, perm[i], perm[j] if op in ("swap", "xor") else j]
                for op, i, j in s["ops"]
            ]
            if family == "inventory_transfers":
                ops = [
                    [op, perm[i], perm[(i + 1) % 3], amount]
                    if op == "transfer"
                    else [op, perm[i], amount]
                    for op, i, amount in s["ops"]
                ]
            q = (
                [perm[i] for i in s["query"]]
                if isinstance(s["query"], list)
                else perm[s["query"]]
            )
            candidates.append(
                canonical({**s, "initial": initial, "ops": ops, "query": q})
            )
        return digest([family, min(candidates)])
    if family in (
        "mixed_units",
        "resource_filter",
        "value_per_cost",
        "numeric_conjunction",
        "disjunction_negation",
        "weighted_conditions",
    ):
        s = {**s, "items": sorted(s["items"])}
    if family in ("capability_constraints", "incomplete_request"):
        s = {
            **s,
            "tools": sorted(
                [{**t, "verbs": sorted(t["verbs"])} for t in s["tools"]], key=canonical
            ),
            "request": {**s["request"], "verbs": sorted(s["request"]["verbs"])},
        }
    return digest([family, s])


def make_spec(rng, task, approach, operations, stratum):
    family = FAMILIES[task][approach]
    if family == "accumulator":
        initial = rng.randint(-30, 70)
        value = initial
        ops = []
        for _ in range(operations):
            op = rng.choice(["add", "subtract", "multiply"])
            v = rng.randint(2, 5) if op == "multiply" else rng.randint(1, 29)
            ops.append([op, v])
            value = (
                value * v
                if op == "multiply"
                else value + v
                if op == "add"
                else value - v
            )
        return {"initial": initial, "ops": ops}, str(value)
    if family == "expression_tree":
        depth = {"easy": 1, "moderate": 2, "hard": 3, "extrapolation": 4}[stratum]

        def build(level):
            if level == 0:
                v = rng.randint(-9, 19)
                return v, v
            a, av = build(level - 1)
            b, bv = build(level - 1)
            op = rng.choice(["add", "subtract", "multiply"])
            return [
                op,
                a,
                b,
            ], av + bv if op == "add" else av - bv if op == "subtract" else av * bv

        tree, value = build(depth)
        return {"tree": tree}, str(value)
    if family == "parallel_ledgers":
        initial = [rng.randint(15, 95), rng.randint(15, 95)]
        v = initial[:]
        ops = []
        for _ in range(operations):
            op = rng.choice(["credit", "transfer"])
            i = rng.randrange(2)
            amount = rng.randint(1, 19)
            ops.append([op, i, amount])
            v[i] += amount
            if op == "transfer":
                v[1 - i] -= amount
        # Transfer opcode denotes transfer INTO account i.
        q = rng.randrange(2)
        return {"initial": initial, "ops": ops, "query": q}, str(v[q])
    if family in ("mixed_units", "resource_filter", "value_per_cost"):
        items = (
            [[rng.randint(2, 250), rng.choice(["KB", "MB", "GB"])] for _ in range(4)]
            if family == "mixed_units"
            else [[rng.randint(8, 240), rng.randint(1, 32)] for _ in range(4)]
        )
        s = {"items": items}
        if family == "mixed_units":
            scores = [v * 1024 ** (["KB", "MB", "GB"].index(u) + 1) for v, u in items]
            valid = list(range(4))
        elif family == "resource_filter":
            s["minimum_cpu"] = rng.randint(2, 24)
            valid = [i for i, (_, cpu) in enumerate(items) if cpu >= s["minimum_cpu"]]
            scores = [v for v, _ in items]
        else:
            s["minimum_ram"] = rng.randint(10, 180)
            valid = [i for i, (ram, _) in enumerate(items) if ram >= s["minimum_ram"]]
            scores = [ram / cost for ram, cost in items]
        if not valid:
            return None, None
        best = max(valid, key=lambda i: scores[i])
        if sum(scores[i] == scores[best] for i in valid) != 1:
            return None, None
        return s, best
    if family in ("register_program", "inventory_transfers"):
        initial = [rng.randint(-5, 40) for _ in range(3)]
        values = initial[:]
        ops = []
        for _ in range(operations):
            op = rng.choice(
                ["add", "reset", "swap"]
                if family == "register_program"
                else ["add", "transfer"]
            )
            i = rng.randrange(3)
            j = rng.randrange(3) if op == "swap" else rng.randint(1, 15)
            ops.append([op, i, j])
            if op == "add":
                values[i] += j
            elif op == "reset":
                values[i] = j
            elif op == "swap":
                values[i], values[j] = values[j], values[i]
            else:
                values[i] -= j
                values[(i + 1) % 3] += j
        q = rng.randrange(3)
        return {"initial": initial, "ops": ops, "query": q}, str(values[q])
    if family == "boolean_program":
        initial = [rng.randrange(2) for _ in range(6)]
        values = initial[:]
        ops = []
        for _ in range(operations):
            op = rng.choice(["toggle", "set", "xor"])
            i = rng.randrange(6)
            v = rng.randrange(6) if op == "xor" else rng.randrange(2)
            ops.append([op, i, v])
            values[i] = (
                1 - values[i]
                if op == "toggle"
                else v
                if op == "set"
                else values[i] ^ values[v]
            )
        q = rng.sample(range(6), 2)
        return {"initial": initial, "ops": ops, "query": q}, "".join(
            str(values[i]) for i in q
        )
    if family in ("functional_graph", "alternating_maps"):
        n = {"easy": 8, "moderate": 10, "hard": 12, "extrapolation": 16}[stratum]
        maps = [
            [rng.randrange(n) for _ in range(n)]
            for _ in range(1 if family == "functional_graph" else 2)
        ]
        start = rng.randrange(n)
        value = start
        for i in range(operations):
            value = maps[i % len(maps)][value]
        return {"maps": maps, "start": start, "steps": operations}, value
    if family == "symbolic_modular_walk":
        modulus = rng.choice([7, 11, 13, 17, 19, 23])
        start = rng.randrange(modulus)
        rules = [[rng.randint(1, 4), rng.randint(0, 8)] for _ in range(2)]
        value = start
        for i in range(operations):
            a, b = rules[i % 2]
            value = (value * a + b) % modulus
        return {
            "modulus": modulus,
            "start": start,
            "rules": rules,
            "steps": operations,
        }, str(value)
    if family in ("numeric_conjunction", "disjunction_negation", "weighted_conditions"):
        thresholds = [
            rng.randint(4, 20)
            for _ in range(3 if family == "weighted_conditions" else 2)
        ]
        items = [
            [
                rng.randint(0, 30),
                rng.randint(0, 30),
                rng.randrange(2)
                if family != "weighted_conditions"
                else rng.randint(0, 30),
            ]
            for _ in range(4)
        ]
        s = {"items": items, "thresholds": thresholds}
        if family == "numeric_conjunction":
            ok = lambda v: (
                not (v[0] < thresholds[0] or v[1] >= thresholds[1] or v[2] != 1)
            )
        elif family == "disjunction_negation":
            ok = lambda v: (
                (v[0] >= thresholds[0] or v[1] >= thresholds[1]) and v[2] == 0
            )
        else:
            weights = [rng.randint(1, 3) for _ in range(3)]
            target = rng.randint(2, sum(weights))
            s.update(weights=weights, target=target)
            ok = lambda v: (
                sum(w for x, t, w in zip(v, thresholds, weights) if x >= t) >= target
            )
        valid = [i for i, v in enumerate(items) if ok(v)]
        return (s, valid[0]) if len(valid) == 1 else (None, None)
    if family in ("capability_constraints", "incomplete_request"):
        verbs = ["read", "write", "search", "compute", "inspect", "archive"]
        q = {
            "verbs": rng.sample(verbs, min(3, max(1, operations // 2))),
            "jobs": rng.randint(1, 35),
            "write": bool(rng.randrange(2)),
            "read_only": False,
        }
        count = 4 if family == "capability_constraints" else 2
        tools = [
            {
                "verbs": rng.sample(verbs, rng.randint(2, 6)),
                "capacity": rng.randint(5, 50),
                "write": bool(rng.randrange(2)),
                "priority": p,
            }
            for p in rng.sample(range(1, 100), count)
        ]
        if family == "incomplete_request":
            mode = rng.randrange(4)
            if mode == 0:
                q["jobs"] = None
            elif mode == 1:
                q["write"], q["read_only"] = True, True
        if q["jobs"] is None:
            answer = "Request more information"
        elif q["write"] and q["read_only"]:
            answer = "No valid tool"
        else:
            valid = [
                i
                for i, t in enumerate(tools)
                if all(v in t["verbs"] for v in q["verbs"])
                and q["jobs"] <= t["capacity"]
                and (not q["write"] or t["write"])
            ]
            answer = (
                min(valid, key=lambda i: tools[i]["priority"])
                if valid
                else "No valid tool"
            )
        if family == "capability_constraints" and isinstance(answer, str):
            return None, None
        return {"tools": tools, "request": q}, answer
    if family == "ordered_rules":
        actions = [
            "Inspect system",
            "Archive data",
            "Request more information",
            "No valid tool",
        ]
        q = {
            "urgency": rng.randint(0, 99),
            "encrypted": rng.randrange(2),
            "write": rng.randrange(2),
        }
        rules = [
            [
                rng.randint(0, 90),
                rng.choice([-1, 0, 1]),
                rng.choice([-1, 0, 1]),
                rng.choice(actions),
            ]
            for _ in range(operations)
        ]
        default = rng.choice(actions)
        answer = default
        for minimum, encrypted, writable, action in rules:
            if (
                q["urgency"] >= minimum
                and (encrypted == -1 or q["encrypted"] == encrypted)
                and (writable == -1 or q["write"] == writable)
            ):
                answer = action
                break
        return {"request": q, "rules": rules, "default": default}, answer
    raise ValueError(family)


def render(family, s, names, template):
    # Templates 0/1 are development wording; 2/3 are withheld lexical presentations.
    intro = [
        "Solve the following deterministic problem.",
        "Compute the requested result from the supplied data.",
        "Work item: apply the specification below exactly.",
        "Use this record to select the uniquely correct outcome.",
    ][template]
    if family == "accumulator":
        q = (
            f"Counter initially {s['initial']}. Execute in order: "
            + "; ".join(f"{op} {v}" for op, v in s["ops"])
            + ". Give the final integer."
        )
    elif family == "expression_tree":

        def expr(n):
            return (
                str(n)
                if isinstance(n, int)
                else f"({expr(n[1])} {dict(add='+', subtract='-', multiply='*')[n[0]]} {expr(n[2])})"
            )

        q = "Evaluate this fully parenthesized integer expression: " + expr(s["tree"])
    elif family == "parallel_ledgers":
        q = (
            "Two account balances: "
            + ", ".join(f"{names[i]}={v}" for i, v in enumerate(s["initial"]))
            + ". Negative balances are allowed. In order: "
            + "; ".join(
                f"credit {v} to {names[i]}"
                if op == "credit"
                else f"transfer {v} from {names[1 - i]} into {names[i]}"
                for op, i, v in s["ops"]
            )
            + f". Final balance of {names[s['query']]}?"
        )
    elif family == "mixed_units":
        q = (
            "Storage capacities: "
            + "; ".join(f"{n}: {v} {u}" for n, (v, u) in zip(names, s["items"]))
            + ". Here 1 MB=1024 KB and 1 GB=1024 MB. Which capacity is largest?"
        )
    elif family == "resource_filter":
        q = (
            "Resource table (free memory MB, free CPU units): "
            + "; ".join(f"{n}: {a}, {b}" for n, (a, b) in zip(names, s["items"]))
            + f". Keep only machines with at least {s['minimum_cpu']} free CPU units. Among them select the largest free memory."
        )
    elif family == "value_per_cost":
        q = (
            "Offers (RAM units, cost): "
            + "; ".join(f"{n}: {a}, {b}" for n, (a, b) in zip(names, s["items"]))
            + f". Eligible RAM must be at least {s['minimum_ram']}. Choose the highest RAM/cost ratio among eligible offers."
        )
    elif family in ("register_program", "inventory_transfers"):
        q = (
            (
                "Registers: "
                if family == "register_program"
                else "Inventory counts (negative means backorders): "
            )
            + ", ".join(f"{names[i]}={v}" for i, v in enumerate(s["initial"]))
            + ". In order: "
        )
        q += (
            "; ".join(
                f"add {j} to {names[i]}"
                if op == "add"
                else f"set {names[i]}={j}"
                if op == "reset"
                else f"swap {names[i]} and {names[j]}"
                if op == "swap"
                else f"move {j} from {names[i]} to {names[(i + 1) % 3]}"
                for op, i, j in s["ops"]
            )
            + f". Final value of {names[s['query']]}?"
        )
    elif family == "boolean_program":
        q = (
            "Binary flags: "
            + ", ".join(f"{names[i]}={v}" for i, v in enumerate(s["initial"]))
            + ". Apply sequentially: "
            + "; ".join(
                f"toggle {names[i]}"
                if op == "toggle"
                else f"set {names[i]}={v}"
                if op == "set"
                else f"set {names[i]} to {names[i]} XOR {names[v]}"
                for op, i, v in s["ops"]
            )
            + f". Return the two bits for {names[s['query'][0]]} then {names[s['query'][1]]}, in that order."
        )
    elif family in ("functional_graph", "alternating_maps"):
        q = (
            "Mapping table: "
            + " | ".join(
                f"map {k + 1}: "
                + ", ".join(f"{names[i]} -> {names[t]}" for i, t in enumerate(m))
                for k, m in enumerate(s["maps"])
            )
            + f". Begin at {names[s['start']]}; take exactly {s['steps']} steps. "
        )
        q += (
            "Use map 1 each step. "
            if len(s["maps"]) == 1
            else "Alternate map 1 then map 2, repeating. "
        ) + "Return the endpoint."
    elif family == "symbolic_modular_walk":
        q = (
            f"Integer state initially {s['start']}, modulo {s['modulus']}. Alternate transformations "
            + ", then ".join(
                f"x <- ({a}*x + {b}) modulo {s['modulus']}" for a, b in s["rules"]
            )
            + f". Apply exactly {s['steps']} transformations, starting with the first. Final state?"
        )
    elif family in (
        "numeric_conjunction",
        "disjunction_negation",
        "weighted_conditions",
    ):
        a, b = s["thresholds"][:2]
        rule = (
            f"x >= {a} AND y < {b} AND flag = 1"
            if family == "numeric_conjunction"
            else f"(x >= {a} OR y >= {b}) AND NOT flag"
            if family == "disjunction_negation"
            else "score = "
            + " + ".join(
                f"{w} if {p}>={t}, otherwise 0"
                for p, t, w in zip("xyz", s["thresholds"], s["weights"])
            )
            + f"; score must be >= {s['target']}"
        )
        q = (
            "Rule: "
            + rule
            + ". Candidate records ("
            + ("x,y,z" if family == "weighted_conditions" else "x,y,flag")
            + "): "
            + "; ".join(
                f"{n}: " + ",".join(map(str, v)) for n, v in zip(names, s["items"])
            )
            + ". Select the unique eligible record."
        )
    elif family in ("capability_constraints", "incomplete_request"):
        q = (
            "Synthetic tool registry: "
            + "; ".join(
                f"{names[i]} supports {','.join(t['verbs'])}, capacity={t['capacity']}, write_allowed={t['write']}, priority={t['priority']}"
                for i, t in enumerate(s["tools"])
            )
            + ". Request: "
            + canonical(s["request"])
            + ". All requested verbs and capacity and write permission must match. Select lowest numeric priority among compatible tools."
        )
        if family == "incomplete_request":
            q += " If jobs is unspecified, request more information first. Otherwise write plus read_only is invalid. If no tool matches, choose No valid tool."
    else:
        q = (
            "Apply first matching rule in this ordered policy. Each rule is (minimum urgency, encryption requirement, write requirement, action); -1 means either flag value. Rules: "
            + canonical(s["rules"])
            + ". Request: "
            + canonical(s["request"])
            + ". Default action if no match: "
            + s["default"]
            + ". Which action follows?"
        )
    if template in (1, 3):
        q = q.replace("; ", "\n").replace(". In order: ", "\nOrdered instructions:\n")
    if template >= 2:
        # Withheld formal-record presentation uses the same data once, with a
        # semantic glossary and symbol-index legend instead of the prose table.
        glossary = {
            "accumulator": "Start at initial; apply ops in listed order. add/subtract/multiply have their usual integer meanings. Return the final integer.",
            "expression_tree": "A node [operation,left,right] applies that operation to its two child values. Integer leaves are constants. Return the root value.",
            "register_program": "initial lists register values. Ops [operation,index,value] execute in order; add increments, reset assigns, swap exchanges with register index value. Return register query.",
            "inventory_transfers": "initial lists inventory counts, including backorders. add increases index by amount. transfer moves amount from index to the next cyclic index (0 to 1, 1 to 2, 2 to 0). Return count at query.",
            "functional_graph": "maps[0][i] is successor of symbol i. Begin at start and follow steps successors. Return endpoint symbol.",
            "alternating_maps": "maps[0] and maps[1] are successor arrays. Begin at start, alternate first then second map for steps transitions. Return endpoint symbol.",
            "mixed_units": "items[i] is [storage,unit] for symbol i. KB/MB/GB scale by 1024 per unit. Return symbol with largest storage.",
            "resource_filter": "items[i] is [free memory,free CPU]. Retain CPU >= minimum_cpu; select symbol with greatest memory among retained rows.",
            "numeric_conjunction": "items[i] is [x,y,flag]. Eligible means x >= thresholds[0], y < thresholds[1], and flag=1. Return the unique eligible symbol.",
            "disjunction_negation": "items[i] is [x,y,flag]. Eligible means (x >= thresholds[0] OR y >= thresholds[1]) AND flag=0. Return the unique eligible symbol.",
            "capability_constraints": "Tool i has supported verbs, capacity, write permission and numeric priority. All request verbs, jobs <= capacity and requested write permission must match. Return matching tool of lowest priority.",
            "incomplete_request": "Missing request jobs takes precedence: Request more information. Otherwise write plus read_only is invalid: No valid tool. Match all request verbs, capacity and write permission; select lowest-priority compatible tool, else No valid tool.",
        }
        if family in glossary:
            q = (
                "Semantics: "
                + glossary[family]
                + "\nSymbol-index legend: "
                + ", ".join(f"{i}={n}" for i, n in enumerate(names))
                + "\nData record: "
                + canonical(s)
            )
        else:
            q = (
                "Specification:\n"
                + q
                + "\nDecision required: the outcome specified above."
            )
    return intro + "\n" + q


def row(rng, task, approach, stratum, template, split):
    operations = {"easy": 1, "moderate": 3, "hard": 5, "extrapolation": 8}[stratum]
    family = FAMILIES[task][approach]
    if family == "boolean_program":
        operations = max(2, operations)
    spec, answer = make_spec(rng, task, approach, operations, stratum)
    if spec is None:
        return None
    assert answer == reference(family, spec), (family, spec, answer)
    count = (
        len(spec["maps"][0])
        if "maps" in spec
        else 6
        if family == "boolean_program"
        else 4
    )
    names = [f"N{rng.randrange(1000, 9999)}_{i}" for i in range(count)]
    if isinstance(answer, int):
        choices = (
            names[:4]
            if task in ("comparison", "classification")
            or family == "capability_constraints"
            else names[:2] + ["Request more information", "No valid tool"]
            if family == "incomplete_request"
            else [names[answer]]
            + rng.sample([n for i, n in enumerate(names) if i != answer], 3)
        )
        answer_text = names[answer]
    elif family == "incomplete_request":
        choices = names[:2] + ["Request more information", "No valid tool"]
        answer_text = answer
    elif family == "ordered_rules":
        choices = [
            "Inspect system",
            "Archive data",
            "Request more information",
            "No valid tool",
        ]
        answer_text = answer
    elif family == "boolean_program":
        choices = ["00", "01", "10", "11"]
        answer_text = answer
    else:
        value = int(answer)
        choices = [answer] + [
            str(value + v) for v in rng.sample([-31, -11, -3, -1, 1, 3, 11, 31], 3)
        ]
        answer_text = answer
    rng.shuffle(choices)
    question = render(family, spec, names, template)
    prompt = (
        question
        + "\n"
        + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", choices))
        + "\nAnswer with only the correct option letter."
    )
    numbers = []

    def collect(v):
        if isinstance(v, (int, float)):
            numbers.append(abs(v))
        elif isinstance(v, list):
            for item in v:
                collect(item)
        elif isinstance(v, dict):
            for item in v.values():
                collect(item)

    collect(spec)
    actual_ops = (
        2 ** {"easy": 1, "moderate": 2, "hard": 3, "extrapolation": 4}[stratum] - 1
        if family == "expression_tree"
        else operations
        if task in ("arithmetic", "state", "pointer") or family == "ordered_rules"
        else len(spec.get("request", {}).get("verbs", [])) + 2
        if task == "routing"
        else 3
        if task == "classification"
        else 2
        if family != "mixed_units"
        else 1
    )
    return {
        "id": digest([family, spec]),
        "structural_hash": structure(family, spec),
        "task": task,
        "generator_id": family,
        "split": split,
        "template_id": template,
        "difficulty_stratum": stratum,
        "operations": actual_ops,
        "reasoning_depth": operations
        if task in ("arithmetic", "state", "pointer")
        else actual_ops,
        "numeric_max": max(numbers, default=0),
        "prompt": prompt,
        "prompt_characters": len(prompt),
        "choices": choices,
        "label": choices.index(answer_text),
        "spec": spec,
        "entities": names,
    }


def build(config):
    CACHE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    EXP.mkdir(parents=True, exist_ok=True)
    seen, structures, prompts = set(), set(), set()
    exclusion_path = EXP / "historical_structure_exclusions.json"
    exclusions = set(json.loads(exclusion_path.read_text())["fingerprints"])
    manifest = {
        "version": VERSION,
        "split_seeds": config["split_seeds"],
        "splits": {},
        "integrity": {},
        "withheld_generator_ids": {
            task: families[2] for task, families in FAMILIES.items()
        },
        "fingerprint_definition": "Constants, operations, and queries retained; presentation/entity names/candidate order removed. Symbolic register renamings minimized; graphs use a conservative directed WL invariant. Algorithm archetypes intentionally shared in ID/lexical splits.",
        "semantic_solver": "Independent deterministic algebraic, matrix, bitmask, or declarative solvers; no language-model labels.",
        "historical_exclusions_sha256": sha(exclusion_path),
    }
    assert len(set(config["split_seeds"].values())) == len(config["split_seeds"])
    rejection = {
        "solver_or_uniqueness": 0,
        "exact_instance": 0,
        "renamed_structure": 0,
        "prompt": 0,
        "historical_renamed_structure": 0,
    }
    for split, size in config["sizes"].items():
        rng = random.Random(config["split_seeds"][split])
        approaches = [2] if split == "structural_shift" else [0, 1]
        strata = (
            ["extrapolation"]
            if split == "harder_depth"
            else ["easy", "moderate", "hard"]
        )
        cells = list(itertools.product(TASKS, approaches, strata))
        assert size % len(cells) == 0, (split, size, len(cells))
        rows = []
        attempts = 0
        while len(rows) < size:
            attempts += 1
            if attempts > size * 100:
                raise RuntimeError("Insufficient independent generator structures")
            task, approach, stratum = cells[len(rows) % len(cells)]
            template = rng.choice(
                config["template_test_ids"]
                if split == "new_template"
                else config["template_train_ids"]
            )
            item = row(rng, task, approach, stratum, template, split)
            if item is None:
                rejection["solver_or_uniqueness"] += 1
                continue
            historical_key = (
                digest(
                    [
                        "normalized_memory_bytes",
                        sorted(
                            v * 1024 ** (["KB", "MB", "GB"].index(u) + 1)
                            for v, u in item["spec"]["items"]
                        ),
                    ]
                )
                if item["generator_id"] == "mixed_units"
                else item["structural_hash"]
            )
            if historical_key in exclusions:
                rejection["historical_renamed_structure"] += 1
                continue
            if item["id"] in seen:
                rejection["exact_instance"] += 1
                continue
            if item["structural_hash"] in structures:
                rejection["renamed_structure"] += 1
                continue
            if item["prompt"] in prompts:
                rejection["prompt"] += 1
                continue
            seen.add(item["id"])
            structures.add(item["structural_hash"])
            prompts.add(item["prompt"])
            rows.append(item)
        path = (
            CACHE
            / "datasets"
            / ("sealed" if split in FINAL else "development")
            / (split + ".jsonl.gz")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(canonical(r) + "\n" for r in rows).encode()
        if path.exists():
            assert gzip.open(path, "rb").read() == content, "Dataset recipe changed"
        else:
            with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as f:
                f.write(content)
        manifest["splits"][split] = {
            "n": size,
            "sha256": sha(path),
            "content_sha256": hashlib.sha256(content).hexdigest(),
            "seed": config["split_seeds"][split],
            "generator_ids": sorted({r["generator_id"] for r in rows}),
            "difficulty_strata": strata,
            "balanced_task_count": size // 6,
            "path": path.as_posix(),
            "sealed": split in FINAL,
        }
    manifest["integrity"] = {
        "exact_instance_overlap": 0,
        "alpha_renamed_instance_overlap": 0,
        "exact_prompt_overlap": 0,
        "compatible_historical_structure_overlap": 0,
        "unique_instances": len(seen),
        "unique_structural_fingerprints": len(structures),
        "generation_rejections": rejection,
        "topology_only_overlap": "Allowed for ID and lexical shift; withheld generator families define structural shift. Literal-free archetype independence is not claimed.",
    }
    atomic_json(EXP / "dataset_manifest.json", manifest)
    print(
        json.dumps(
            {
                "version": VERSION,
                "counts": config["sizes"],
                "integrity": manifest["integrity"],
            },
            indent=2,
        ),
        flush=True,
    )
    return manifest


def load(split, allow_final=False):
    if split in FINAL and not allow_final:
        raise PermissionError(
            "Final lockbox is sealed; freeze protocol and explicitly open it"
        )
    path = (
        CACHE
        / "datasets"
        / ("sealed" if split in FINAL else "development")
        / (split + ".jsonl.gz")
    )
    manifest = json.loads((EXP / "dataset_manifest.json").read_text())
    assert sha(path) == manifest["splits"][split]["sha256"]
    return [json.loads(line) for line in gzip.open(path, "rt")]
