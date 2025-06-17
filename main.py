
"""Entry point for multi‑robot exploration demo."""

from explorer_world import ExplorerWorld

if __name__ == "__main__":
    world = ExplorerWorld(num_agents=3)
    world.run(max_steps=100)
