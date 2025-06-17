from matplotlib.pylab import f
import numpy as np
from utils.lidar import ray_cast_2d
from utils.bresenham import bresenham
from mapping import LOG_ODDS_FREE, LOG_ODDS_OCC,LOG_ODDS_CLAMP,LIADR_RANGE
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
        self.body_id = self.physics.model.name2id(f"agent{self.id}", "body")
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
        endpoints = ray_cast_2d(self.physics, self.body_id, n_beams=360, max_range=LIADR_RANGE)
        return endpoints

    def capture_top_image(self):
        img = self.physics.render(**TOP_CAM)
        return img

    # ----------------------------- SLAM / Mapping ---------------------------- #
    def sense_and_update(self, dt, agents):
        # 0. motion prediction by odometry
        # self.odom_predict(dt)
        self.update_pose_from_mujoco()  # 从 MuJoCo 中更新 pose
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

        scan, hit_mask = self.lidar()
        row_r, col_r = self.global_map._xy_to_idx(*self.pose[:2])
        max_cells    = self.global_map.max_cells        # 例如 20 格

        for (wx, wy), hit in zip(scan, hit_mask):
            row_t, col_t = self.global_map._xy_to_idx(wx, wy)

            # ① 生成 Bresenham 路径
            pts = bresenham(row_r, col_r, row_t, col_t)
            truncated = False
            if len(pts) > max_cells + 1:                # ★ 发生截断
                pts = pts[:max_cells + 1]
                truncated = True

            # ② 中途 free（可选；若不要中途 free 就删掉此循环）
            for r, c in pts[1:-1]:
                self.global_map.logodds[r, c] = np.clip(
                    self.global_map.logodds[r, c] + LOG_ODDS_FREE,
                    -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)

            # ③ 末端格：根据“是否截断”判定 free / occ
            r_end, c_end = pts[-1]
            if truncated:
                delta = LOG_ODDS_FREE                  # 截断 → free
            else:
                delta = LOG_ODDS_OCC if hit else LOG_ODDS_FREE
            self.global_map.logodds[r_end, c_end] = np.clip(
                self.global_map.logodds[r_end, c_end] + delta,
                -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP)


        

        frontiers = []
        occ = self.global_map.logodds
        it = np.nditer(occ, flags=['multi_index'])
        #find where occ<0
        # x,y= np.where(occ < 0)
        
        # breakpoint()
        while not it.finished:
            r, c = it.multi_index          # 行, 列
            
            if abs(occ[r, c]) < 1e-3:      # 未知
                nb = occ[max(r-1,0):r+2, max(c-1,0):c+2]
                
                if np.any(nb < -1.0):         # 邻接 free

                    wx = (c - self.global_map.center) * self.global_map.resolution
                    wy = (r - self.global_map.center) * self.global_map.resolution
                    frontiers.append((wx, wy))
            it.iternext()
        self.frontiers = frontiers

        # breakpoint()
        self.node_idx += 1
        free_cnt = (self.global_map.logodds < 0).sum()
        # print(f"[Agent {self.id}] free cells: {free_cnt}, frontiers: {len(frontiers)}")

    # ------------------------------- Control --------------------------------- #
    def control_step(self, dt):
        # breakpoint()
        if not self.goals:
            self.last_control[:] = 0.0
            # 也要清零 qvel，防止滑动
            self._set_qvel([0, 0, 0])
            return

        # ---- 1. 计算期望线速度 / 角速度 ----
        goal = np.array(self.goals[0])
        vxy  = control.p_controller(self.pose[:2], goal, kp=0.8, vmax=10.0)
        # 简单面朝目标
        desired_yaw = math.atan2(goal[1] - self.pose[1], goal[0] - self.pose[0])
        yaw_err     = (desired_yaw - self.pose[2] + np.pi) % (2*np.pi) - np.pi
        w = np.clip(4.0 * yaw_err, -2.0, 2.0)         # P 控制转速

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

    def update_pose_from_mujoco(self):
        pos = self.physics.data.xpos[self.body_id]
        xmat = self.physics.data.xmat[self.body_id].reshape(3, 3)
        theta = np.arctan2(xmat[1, 0], xmat[0, 0])   # 从旋转矩阵提取 yaw
        self.pose[:] = [pos[0], pos[1], theta]
