
import numpy as np
from utils.lidar import ray_cast_2d
from utils.bresenham import bresenham
from mapping import LOG_ODDS_FREE, LOG_ODDS_OCC
from image_pose import ImagePoseEstimator
import control

TOP_CAM = dict(camera_id="top", width=128, height=128)

class Agent:
    def __init__(self, idx, physics, global_map):
        self.id = idx
        self.physics = physics
        self.body_id = idx  # assumes bodies ordered agent0, agent1…
        self.global_map = global_map
        self.frontiers = []
        self.goals = []
        self.pose = np.array([0.0, 0.0, 0.0])  # x, y, yaw

        self.pose_estimator = ImagePoseEstimator()

    # -------------------------------- Sensors -------------------------------- #
    def _update_true_pose(self):
        qpos = self.physics.data.qpos[3*self.id : 3*self.id+3]
        self.pose = qpos.copy()

    def lidar(self):
        endpoints = ray_cast_2d(self.physics, self.body_id, n_beams=360, max_range=5.0)
        return endpoints

    def capture_top_image(self):
        img = self.physics.render(**TOP_CAM)
        return img

    # ----------------------------- SLAM / Mapping ---------------------------- #
    def sense_and_update(self):
        self._update_true_pose()
        scan = self.lidar()
        x0, y0, _ = self.pose
        # Update occupancy along rays
        for (x, y) in scan:
            # Bresenham in grid coordinates
            pts = bresenham(int(x0*10), int(y0*10), int(x*10), int(y*10))
            # Free cells
            for gx, gy in pts[:-1]:
                self.global_map.update_cell(gx*0.1, gy*0.1, is_hit=False)
            # Hit cell
            hx, hy = pts[-1]
            self.global_map.update_cell(hx*0.1, hy*0.1, is_hit=True)

        # Frontier detection: unknown cells (l=0) touching free cells
        occ = self.global_map.logodds
        frontiers = []
        it = np.nditer(occ, flags=['multi_index'])
        while not it.finished:
            gx, gy = it.multi_index
            if occ[gx, gy] == 0:
                neighbors = occ[max(gx-1,0):gx+2, max(gy-1,0):gy+2]
                if np.any(neighbors < 0):
                    wx = (gx - self.global_map.center) * self.global_map.resolution
                    wy = (gy - self.global_map.center) * self.global_map.resolution
                    frontiers.append((wx, wy))
            it.iternext()
        self.frontiers = frontiers

    # ------------------------------- Control --------------------------------- #
    def control_step(self, dt):
        if not self.goals:
            return
        goal = np.array(self.goals[0])
        vxy = control.p_controller(self.pose[:2], goal, kp=0.5, vmax=1.0)
        # Apply velocity to qvel directly (simple)
        qvel_idx = self.physics.model.jnt_qveladr[3*self.id : 3*self.id+2]
        self.physics.data.qvel[qvel_idx] = vxy
        # Simple goal reached check
        if np.linalg.norm(self.pose[:2] - goal) < 0.2:
            self.goals.pop(0)
