
import numpy as np

LOG_ODDS_FREE  = -0.30   # 单次 free 积分较小
LOG_ODDS_OCC   =  1.00   # 障碍一次就能显现

LOG_ODDS_CLAMP = 10.0
LIADR_RANGE=1.0
class GlobalMap:
    """Shared occupancy grid in log‑odds representation."""

    def __init__(self, size: float = 10.0, resolution: float = 0.1):
        self.size = size
        self.resolution = resolution
        n = int(size / resolution)
        self.l0 = 0.0
        self.logodds = np.full((n, n), self.l0, dtype=np.float32)
        self.center = n // 2
        self.max_cells = int(LIADR_RANGE/ resolution)

    def _xy_to_idx(self, x: float, y: float):
        row = int(round(y / self.resolution)) + self.center   # 行 = Y
        col = int(round(x / self.resolution)) + self.center   # 列 = X
        nrows, ncols = self.logodds.shape
        row = max(0, min(row, nrows - 1))
        col = max(0, min(col, ncols - 1))
        return row, col


    def update_cell(self, x, y, is_hit: bool):
        row, col = self._xy_to_idx(x, y)
        if 0 <= row < self.logodds.shape[0] and 0 <= col < self.logodds.shape[1]:
            delta = LOG_ODDS_OCC if is_hit else LOG_ODDS_FREE
            self.logodds[row, col] = np.clip(
                self.logodds[row, col] + delta,
                -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)

    def probability_map(self):
        """Return occupancy probability (0‑1)."""
        return 1 - 1 / (1 + np.exp(self.logodds))
    def coverage(self):
        occ = self.probability_map()
        known = (occ != 0.5)        # 未知≈0.5
        free  = (occ < 0.2) & known
        return free.sum() / known.sum() if known.any() else 0.0