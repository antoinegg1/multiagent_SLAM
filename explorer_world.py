
from dm_control import mjcf, mujoco
import numpy as np
from mapping import GlobalMap
from agent import Agent
import cbba

class ExplorerWorld:
    """Container that holds MuJoCo physics + all robots and executes the explore loop."""

    def __init__(self, num_agents: int = 3, map_size: float = 10.0, dt: float = 0.05):
        # Load MJCF
        self.mj_model = mjcf.from_path("maze.xml")
        self.physics = mujoco.Physics.from_mjcf_model(self.mj_model)
        self.dt = dt

        # Shared global map
        self.global_map = GlobalMap(size=map_size, resolution=0.1)

        # Agents
        self.agents = [
            Agent(i, self.physics, self.global_map)
            for i in range(num_agents)
        ]
        self.step_count = 0

    def step(self):
        """One closed‑loop iteration."""
        # Sense → local map update
        for agent in self.agents:
            agent.sense_and_update()

        # Frontier detection happens inside each agent
        # CBBA allocation across all detected frontiers
        cbba.allocate(self.agents)

        # Move the agents
        for agent in self.agents:
            agent.control_step(self.dt)

        # Physics step
        self.physics.step()

        self.step_count += 1

    def run(self, max_steps: int = 1000):
        from tqdm import tqdm
        for _ in tqdm(range(max_steps), desc="Exploring"):
            self.step()
