
from dm_control import mjcf, mujoco
import numpy as np
from mapping import GlobalMap
from agent import Agent
import cbba, slam_graph as sg

class ExplorerWorld:
    def __init__(self, num_agents: int = 3, map_size: float = 10.0, dt: float = 0.1):
        self.mj_model = mjcf.from_path("maze.xml")
        self.physics = mujoco.Physics.from_mjcf_model(self.mj_model)
        self.dt = dt

        self.global_map = GlobalMap(size=map_size, resolution=0.1)

        self.graph = sg.PoseGraph()
        self.agents = [
            Agent(i, self.physics, self.global_map, self.graph)
            for i in range(num_agents)
        ]
        self.step_count = 0

    def step(self):
        # SLAM sense + mapping
        for agent in self.agents:
            agent.sense_and_update(self.dt, self.agents)

        # CBBA allocation
        cbba.allocate(self.agents)

        # Control
        for agent in self.agents:
            agent.control_step(self.dt)

        # Physics step
        self.physics.step()
        self.step_count += 1

        # Optimise pose‑graph every 20 steps
        if self.step_count % 20 == 0:
            self.graph.optimise()

    def run(self, max_steps=800):
        from tqdm import tqdm
        for _ in tqdm(range(max_steps), desc="Full‑SLAM explore"):
            self.step()
