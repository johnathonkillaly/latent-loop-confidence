"""One observation -> internal computation -> exactly one environment action."""
import time
from latent_loop.environments.maze import labels

def navigate(initial,policy,step_budget):
    """policy receives ONLY the current Maze; no solver targets or histories.

    Model policies construct fresh per-decision hidden states. Oracle labels below
    are computed exclusively AFTER the committed decision, for scoring.
    """
    maze=initial; positions=[list(maze.agent)]; actions=[]; invalid=0; inference=0.; regrets=[]
    for _ in range(step_budget):
        t=time.perf_counter(); action=int(policy(maze)); inference+=time.perf_counter()-t
        y=labels(maze)
        regrets.append(y["regret"][action])
        maze,_,done,info=maze.step(action)
        actions.append(action); positions.append(list(maze.agent)); invalid+=int(info["invalid"])
        if done: break
    optimal=labels(initial)["distance"]
    success=maze.agent==maze.goal
    return {"success":success,"steps":len(actions),"optimal_steps":optimal,"steps_over_optimal":len(actions)/optimal,
            "invalid_moves":invalid,"repeated_positions":len(positions)-len(set(map(tuple,positions))),
            "inference_seconds":inference,"mean_regret":sum(regrets)/len(regrets),"positions":positions,"actions":actions}
