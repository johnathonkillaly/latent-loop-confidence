"""Small cardinal-action mazes, canonical layouts, and deterministic generation."""
from dataclasses import dataclass, replace
from collections import deque
import hashlib
import numpy as np

ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NAMES = ("N", "E", "S", "W")

@dataclass(frozen=True)
class Maze:
    walls: tuple
    agent: tuple
    goal: tuple

    def __post_init__(self):
        object.__setattr__(self,"walls",tuple(tuple(int(v) for v in row) for row in self.walls))
        object.__setattr__(self,"agent",tuple(int(v) for v in self.agent))
        object.__setattr__(self,"goal",tuple(int(v) for v in self.goal))
        n = len(self.walls)
        if n < 2 or any(len(row) != n for row in self.walls):
            raise ValueError("Require a square board")
        for p in (self.agent, self.goal):
            if not self.free(p):
                raise ValueError("Agent and goal must be on traversable cells")

    @property
    def size(self):
        return len(self.walls)

    def free(self, p):
        r, c = p
        return 0 <= r < self.size and 0 <= c < self.size and not self.walls[r][c]

    def neighbor(self, p, action):
        dr, dc = ACTIONS[action]
        return (p[0] + dr, p[1] + dc)

    def step(self, action):
        if action not in range(4):
            raise ValueError("Action must be N/E/S/W indexed 0..3")
        if self.agent == self.goal:
            return self, 0.0, True, {"invalid": False}
        p = self.neighbor(self.agent, action)
        invalid = not self.free(p)
        out = self if invalid else replace(self, agent=p)
        done = out.agent == out.goal
        return out, (1.0 if done else -1.0 if invalid else -0.01), done, {"invalid": invalid}

    def observation(self):
        """Only walls/free/agent/goal, NHWC when subsequently batched."""
        w = np.asarray(self.walls, dtype=np.float32)
        x = np.stack([w, 1-w, np.zeros_like(w), np.zeros_like(w)], axis=-1)
        x[self.agent + (2,)] = 1
        x[self.goal + (3,)] = 1
        return x

    def to_dict(self):
        return {"walls": [[int(v) for v in r] for r in self.walls], "agent": [int(v) for v in self.agent], "goal": [int(v) for v in self.goal]}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple(tuple(int(v) for v in r) for r in d["walls"]), tuple(d["agent"]), tuple(d["goal"]))

    def text(self):
        chars = np.where(np.asarray(self.walls), "#", ".")
        chars[self.agent] = "A"
        chars[self.goal] = "G"
        return "\n".join("".join(r) for r in chars)

    def canonical_hash(self, layout_only=True):
        """D4 canonical wall layout excludes ALL start/goal copies across splits."""
        x = np.asarray(self.walls, dtype=np.uint8).copy()
        if not layout_only:
            x[self.agent] = 2
            x[self.goal] = 3
        variants = [np.rot90(y, k).tobytes() for y in (x, np.fliplr(x)) for k in range(4)]
        return hashlib.sha256(bytes([self.size]) + min(variants)).hexdigest()


def distances(maze, source=None, blocked=None):
    """Queue-based exact reference; -1 denotes unreachable/wall."""
    d = np.full((maze.size, maze.size), -1, dtype=np.int32)
    source = maze.goal if source is None else source
    if source == blocked or not maze.free(source):
        return d
    d[source] = 0
    q = deque([source])
    while q:
        p = q.popleft()
        for a in range(4):
            u = maze.neighbor(p, a)
            if u != blocked and maze.free(u) and d[u] < 0:
                d[u] = d[p] + 1
                q.append(u)
    return d


def labels(maze):
    d = distances(maze)
    length = int(d[maze.agent])
    if length <= 0:
        raise ValueError("Training examples must be solvable and nonterminal")
    legal, after, traps, regret = [], [], [], []
    without_agent = distances(maze, blocked=maze.agent)
    for a in range(4):
        u = maze.neighbor(maze.agent, a)
        ok = maze.free(u)
        legal.append(ok)
        after.append(int(d[u]) if ok else length)
        traps.append(bool(ok and without_agent[u] < 0))
        regret.append(int(1 + (d[u] if ok else length) - length))
    optimal = [legal[a] and after[a] == length-1 for a in range(4)]
    manhattan = abs(maze.agent[0]-maze.goal[0])+abs(maze.agent[1]-maze.goal[1])
    proximity = [abs(maze.neighbor(maze.agent,a)[0]-maze.goal[0])+abs(maze.neighbor(maze.agent,a)[1]-maze.goal[1]) if legal[a] else 999 for a in range(4)]
    greedy = [legal[a] and proximity[a] == min(proximity) for a in range(4)]
    reachable = int((d >= 0).sum())
    edges = sum(maze.free(maze.neighbor((r,c),a)) for r,c in zip(*np.where(d>=0)) for a in range(4))//2
    return {"optimal": optimal, "legal": legal, "after_distance": after,
            "dead_end": traps, "regret": regret, "distance": length,
            "manhattan": manhattan, "detour_ratio": length/manhattan,
            "cycle_rank": int(edges-reachable+1), "reachable": reachable,
            "greedy": greedy, "initial_away": all(proximity[a] > manhattan for a in range(4) if optimal[a])}


def family_matches(maze, y, family):
    if family == 0:
        return (maze.agent[0] == maze.goal[0] or maze.agent[1] == maze.goal[1]) and y["distance"] == y["manhattan"]
    if family == 1:
        # Explicit one/two-turn shortest path, rather than just small grid size.
        if maze.agent[0] == maze.goal[0] or maze.agent[1] == maze.goal[1]:
            return False
        if y["distance"] != y["manhattan"]:
            return False
        ar, ac = maze.agent; gr, gc = maze.goal
        for rr in range(min(ar,gr),max(ar,gr)+1):
            route = [(r,ac) for r in range(min(ar,rr),max(ar,rr)+1)] + [(rr,c) for c in range(min(ac,gc),max(ac,gc)+1)] + [(r,gc) for r in range(min(rr,gr),max(rr,gr)+1)]
            if all(maze.free(p) for p in route):
                return True
        return False
    if family == 2:
        return not any(o and g for o,g in zip(y["optimal"],y["greedy"]))
    if family == 3:
        return any(t and g for t,g in zip(y["dead_end"],y["greedy"]))
    if family == 4:
        return y["cycle_rank"] > 0 and sum(l and not t for l,t in zip(y["legal"],y["dead_end"])) >= 2
    if family == 5:
        return y["initial_away"] and y["distance"] >= 6 and y["detour_ratio"] >= 1.5
    raise ValueError("Unknown family")


def candidate(rng, size=5, algorithm="occupancy", density=.30):
    if algorithm == "occupancy":
        walls = rng.random((size,size)) < density
    elif algorithm == "growth":
        # Held-out algorithm: connected frontier growth, rather than independent walls.
        walls = np.ones((size,size), dtype=bool)
        start = tuple(rng.integers(size, size=2)); walls[start] = False
        frontier = set()
        for _ in range(int(size*size*(1-density))-1):
            for r,c in zip(*np.where(~walls)):
                for dr,dc in ACTIONS:
                    u=(r+dr,c+dc)
                    if 0<=u[0]<size and 0<=u[1]<size and walls[u]: frontier.add(u)
            if not frontier: break
            u=sorted(frontier)[int(rng.integers(len(frontier)))]; frontier.remove(u); walls[u]=False
    else:
        raise ValueError("Unknown maze algorithm")
    free = np.argwhere(~walls)
    if len(free)<2: return None
    picks = rng.choice(len(free), 2, replace=False)
    return Maze(tuple(tuple(int(v) for v in row) for row in walls), tuple(free[picks[0]]), tuple(free[picks[1]]))
