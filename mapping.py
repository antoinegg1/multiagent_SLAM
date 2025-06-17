
import numpy as np

LOG_ODDS_FREE  = -1.0
LOG_ODDS_OCC   =  1.0
LOG_ODDS_CLAMP = 10.0

class GlobalMap:
    """Shared occupancy grid in log‑odds representation."""

    def __init__(self, size: float = 10.0, resolution: float = 0.1):
        self.size = size
        self.resolution = resolution
        n = int(size / resolution)
        self.l0 = 0.0
        self.logodds = np.full((n, n), self.l0, dtype=np.float32)
        self.center = n // 2

    def _xy_to_idx(self, x: float, y: float):
        gx = int(x / self.resolution) + self.center
        gy = int(y / self.resolution) + self.center
        return gx, gy

    def update_cell(self, x, y, is_hit: bool):
        """Update one grid cell at world coord (x,y)."""
        gx, gy = self._xy_to_idx(x, y)
        if 0 <= gx < self.logodds.shape[0] and 0 <= gy < self.logodds.shape[1]:
            delta = LOG_ODDS_OCC if is_hit else LOG_ODDS_FREE
            self.logodds[gx, gy] = np.clip(self.logodds[gx, gy] + delta, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)

    def probability_map(self):
        """Return occupancy probability (0‑1)."""
        return 1 - 1 / (1 + np.exp(self.logodds))
