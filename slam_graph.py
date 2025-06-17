
"""Tiny 2‑D pose‑graph (x, y, θ in rad) with Gauss‑Newton optimisation using SciPy."""
import numpy as np
from scipy.optimize import least_squares

def pose_to_mat(p):
    x, y, th = p
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]])

def mat_to_pose(M):
    x, y = M[0, 2], M[1, 2]
    th = np.arctan2(M[1, 0], M[0, 0])
    return np.array([x, y, th])

def relative_pose(p_i, p_j):
    """Return p_ij that transforms i→j."""
    T_i = pose_to_mat(p_i)
    T_j = pose_to_mat(p_j)
    T_ij = np.linalg.inv(T_i) @ T_j
    return mat_to_pose(T_ij)

class PoseGraph:
    def __init__(self):
        self.nodes = {}           # id -> pose (3,)
        self.edges = []           # (id_i, id_j, rel_pose, information(3×3))

    def add_node(self, node_id, pose):
        self.nodes.setdefault(node_id, pose.copy())

    def add_edge(self, i, j, rel, info=np.eye(3)):
        self.edges.append((i, j, rel.copy(), info.copy()))

    # --------------------------------------------------------------------- #
    def _pack(self):
        """Return vectorised parameters in the order of sorted node ids."""
        ids = sorted(self.nodes)
        x = np.concatenate([self.nodes[i] for i in ids])
        return x, ids

    def _unpack(self, x, ids):
        for k, i in enumerate(ids):
            self.nodes[i] = x[3*k:3*k+3]

    def optimise(self, fixed_id=None, iterations=10):
        """Gauss‑Newton (via least_squares). Keep node ``fixed_id`` fixed."""
        x0, ids = self._pack()
        if not ids:
            return

        # Default to the first node if ``fixed_id`` is not provided
        if fixed_id is None:
            fixed_id = ids[0]
        if fixed_id not in ids:
            raise KeyError(f"Node {fixed_id!r} not found in graph")

        fixed_idx = ids.index(fixed_id)

        mask = np.ones_like(x0, dtype=bool)
        mask[3*fixed_idx:3*fixed_idx+3] = False
        x_free = x0[mask]

        def residuals(x_var):
            x = x0.copy()
            x[mask] = x_var
            self._unpack(x, ids)
            res = []
            for i, j, rel, info in self.edges:
                e = relative_pose(self.nodes[i], self.nodes[j]) - rel
                res.extend(info @ e)
            return np.array(res)

        ls = least_squares(
            residuals,
            x_free,
            jac="2-point",
            max_nfev=iterations,
            verbose=0,
            x_scale="jac",
            diff_step=1e-3,
            tr_solver="lsmr",
            loss="linear",
            ftol=1e-6,
            xtol=1e-6,
        )

        x0[mask] = ls.x
        self._unpack(x0, ids)
