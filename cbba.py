
"""Very small ‑‑ and not fully optimal ‑‑ CBBA‑like single‑round allocation for demo purposes."""
import numpy as np

def allocate(agents):
    # Collect all frontier centroids
    all_frontiers = []
    for agent in agents:
        for f in agent.frontiers:
            all_frontiers.append(tuple(f))
    all_frontiers = np.unique(np.array(all_frontiers), axis=0)

    # Each agent bids = negative euclidean distance (closer is higher bid)
    assigned = {}
    for frontier in all_frontiers:
        best_agent = None
        best_bid = -np.inf
        for agent in agents:
            dist = np.linalg.norm(agent.pose[:2] - frontier)
            bid = -dist
            if bid > best_bid:
                best_bid = bid
                best_agent = agent
        assigned.setdefault(best_agent.id, []).append(frontier)

    # Write allocations back
    for agent in agents:
        agent.goals = assigned.get(agent.id, [])
    # print(f"[CBBA] agent {agent.id} goals = {agent.goals}")
