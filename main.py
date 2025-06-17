
"""Entry point for multi‑robot exploration demo."""
import numpy as np
from explorer_world import ExplorerWorld
np.set_printoptions(threshold=np.inf)
if __name__ == "__main__":
    world = ExplorerWorld(num_agents=3,map_path=".\\map\\maze.xml")
    world.run(max_steps=1000)
