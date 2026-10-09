"""Classical references, never inference inputs to learned policies."""
import numpy as np
from latent_loop.environments.maze import distances, labels

def propagation(maze, steps):
    d = np.full((maze.size,maze.size), np.inf)
    d[maze.goal] = 0
    trace = [d.copy()]
    for _ in range(steps):
        out = d.copy()
        for r in range(maze.size):
            for c in range(maze.size):
                if maze.free((r,c)):
                    out[r,c] = min([d[r,c]]+[1+d[maze.neighbor((r,c),a)] for a in range(4) if maze.free(maze.neighbor((r,c),a))])
        d = out; trace.append(d.copy())
    return trace

def bfs_action(maze):
    return labels(maze)["optimal"].index(True)

def greedy_action(maze):
    # Deployment baseline uses geometry and legality only; it never calls BFS.
    choices=[]
    for a in range(4):
        p=maze.neighbor(maze.agent,a)
        if maze.free(p): choices.append((abs(p[0]-maze.goal[0])+abs(p[1]-maze.goal[1]),a))
    return min(choices)[1]

def shortest_path(maze):
    d=distances(maze); p=maze.agent; out=[p]
    while p != maze.goal:
        choices=[maze.neighbor(p,a) for a in range(4) if maze.free(maze.neighbor(p,a)) and d[maze.neighbor(p,a)]==d[p]-1]
        if not choices: raise ValueError("Unsolvable maze")
        p=choices[0]; out.append(p)
    return out
