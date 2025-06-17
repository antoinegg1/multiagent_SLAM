import numpy as np
from utils.lidar import ray_cast_2d
from utils.bresenham import bresenham
from mapping import LOG_ODDS_FREE, LOG_ODDS_OCC
from image_pose import ImagePoseEstimator
import control
import slam_graph as sg
import itertools, math

TOP_CAM = dict(camera_id="top", width=128, height=128)
SIGMA_ODOM_NOISE = 0.01    # m per step
SIGMA_YAW_NOISE  = 0.01    # rad per step

class Agent:
    _id_iter = itertools.count()

    def __init__(self, idx, physics, global_map, graph: sg.PoseGraph):
        self.id = idx
        self.physics = physics
        self.body_id = idx
        self.global_map = global_map
        self.frontiers = []
        self.goals = []

        # SLAM state
        self.pose = np.zeros(3)               # current estimate
        self.last_control = np.zeros(3)       # vx, vy, w
        self.node_idx = 0                     # incremental node id inside graph
        self.graph = graph
        self.graph.add_node(self._node_id(), self.pose)

        self.last_image = None
        self.pose_estimator = ImagePoseEstimator()

    # Helper
    def _node_id(self):
        return f"A{self.id}_{self.node_idx}"

    # -------------------------------- Sensors -------------------------------- #
    def odom_predict(self, dt):
        vx, vy = self.last_control[:2]
        w = self.last_control[2]
        # Integrate in body frame
        dx = vx * dt + np.random.randn()*SIGMA_ODOM_NOISE
        dy = vy * dt + np.random.randn()*SIGMA_ODOM_NOISE
        dth = w * dt + np.random.randn()*SIGMA_YAW_NOISE
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        self.pose[0] += c*dx - s*dy
        self.pose[1] += s*dx + c*dy
        self.pose[2] = (self.pose[2] + dth + np.pi) % (2*np.pi) - np.pi

    def lidar(self):
        endpoints = ray_cast_2d(self.physics, self.body_id, n_beams=360, max_range=5.0)
        return endpoints

    def capture_top_image(self):
        img = self.physics.render(**TOP_CAM)
        return img

    # ----------------------------- SLAM / Mapping ---------------------------- #
    def sense_and_update(self, dt, agents):
        # 0. motion prediction by odometry
        self.odom_predict(dt)

        # 1. add odometry edge to graph
        rel = np.array([self.last_control[0]*dt, self.last_control[1]*dt, self.last_control[2]*dt])
        if self.node_idx > 0:
            prev = f"A{self.id}_{self.node_idx-1}"
            curr = self._node_id()
            self.graph.add_node(curr, self.pose)
            info = np.diag([1/(SIGMA_ODOM_NOISE**2)]*2 + [1/(SIGMA_YAW_NOISE**2)])
            self.graph.add_edge(prev, curr, rel, info)
        # 2. image capture & loop closure with previous image
        img = self.capture_top_image()
        if self.last_image is not None:
            H = self.pose_estimator.estimate(self.last_image, img)
            if H is not None:
                dx, dy = H[0,2], H[1,2]
                dth = math.atan2(H[1,0], H[0,0])
                rel_icp = np.array([dx, dy, dth])
                prev = f"A{self.id}_{self.node_idx-1}"
                curr = self._node_id()
                info = np.diag([50, 50, 50])  # higher confidence
                self.graph.add_edge(prev, curr, rel_icp, info)
        self.last_image = img.copy()

        # 3. Mapping with current pose
        scan = self.lidar()
        x0, y0, _ = self.pose
        for (x, y) in scan:
            pts = bresenham(int(x0*10), int(y0*10), int(x*10), int(y*10))
            for gx, gy in pts[:-1]:
                self.global_map.update_cell(gx*0.1, gy*0.1, is_hit=False)
            hx, hy = pts[-1]
            self.global_map.update_cell(hx*0.1, hy*0.1, is_hit=True)

        # 4. Frontier detection
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

        self.node_idx += 1

    # ------------------------------- Control --------------------------------- #
    def control_step(self, dt):
        if not self.goals:
            self.last_control[:] = 0.0
            # 也要清零 qvel，防止滑动
            self._set_qvel([0, 0, 0])
            return

        # ---- 1. 计算期望线速度 / 角速度 ----
        goal = np.array(self.goals[0])
        vxy  = control.p_controller(self.pose[:2], goal, kp=0.8, vmax=1.5)
        # 简单面朝目标
        desired_yaw = math.atan2(goal[1] - self.pose[1], goal[0] - self.pose[0])
        yaw_err     = (desired_yaw - self.pose[2] + np.pi) % (2*np.pi) - np.pi
        w = np.clip(2.0 * yaw_err, -2.0, 2.0)         # P 控制转速

        self.last_control[:] = [vxy[0], vxy[1], w]

        # ---- 2. 把速度写进 MuJoCo ----
        self._set_qvel([vxy[0], vxy[1], w])

        # ---- 3. 到点就换下一个 frontier ----
        if np.linalg.norm(self.pose[:2] - goal) < 0.3:
            self.goals.pop(0)

    # --------- 工具函数 ----------
    def _set_qvel(self, vel_xyz):
        model = self.physics.model
        qvel  = self.physics.data.qvel

        # 关节 → 对应 DOF 在 qvel 中的起始下标
        jnt_x   = model.name2id(f"agent{self.id}_x",   "joint")
        jnt_y   = model.name2id(f"agent{self.id}_y",   "joint")
        jnt_yaw = model.name2id(f"agent{self.id}_yaw", "joint")

        idx_x   = model.jnt_dofadr[jnt_x]
        idx_y   = model.jnt_dofadr[jnt_y]
        idx_yaw = model.jnt_dofadr[jnt_yaw]

        qvel[[idx_x, idx_y, idx_yaw]] = vel_xyz


