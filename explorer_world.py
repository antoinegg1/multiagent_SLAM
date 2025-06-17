
from dm_control import mjcf
import numpy as np
from mapping import GlobalMap
from agent import Agent
import cbba, slam_graph as sg
import pose_graph_torch as pg_torch
import mujoco
import mujoco.viewer 
# import matplotlib.pyplot as plt
import cv2
# plt.ion()

class ExplorerWorld:
    def __init__(self, num_agents: int = 3, map_size: float = 10.0, dt: float = 0.1,map_path:str=".\\map\\box.xml"):
        self.mj_model = mjcf.from_path(map_path)
        self.physics = mjcf.Physics.from_mjcf_model(self.mj_model)
        self.physics.forward() 
        # self.fig, self.ax = plt.subplots(figsize=(4,4))
        self.global_map = GlobalMap(size=map_size, resolution=0.1)
        # self.img = self.ax.imshow(self.global_map.probability_map(),
        #                         cmap="gray_r", vmin=0, vmax=1, origin="lower")
        # self.ax.set_title("Occupancy grid")
        self.viewer = mujoco.viewer.launch_passive(
            self.physics.model.ptr,   # MjModel  (dm_control.Physics → .model.ptr)
            self.physics.data.ptr     # MjData   (dm_control.Physics → .data.ptr)
        )
        self.dt = dt
        # breakpoint()
        

        self.graph = pg_torch.TorchPoseGraph()  # Torch 优化的图优化器
        self.agents = [
            Agent(i, self.physics, self.global_map, self.graph)
            for i in range(num_agents)
        ]
        self.step_count = 0
        
        # ---------- 纹理基础信息 ----------
        self.tex_id  = self.physics.model.name2id("explore_tex", "texture")
        self.W           = int(self.physics.model.tex_width [self.tex_id])
        self.H           = int(self.physics.model.tex_height[self.tex_id])
        self.C           = int(self.physics.model.tex_nchannel[self.tex_id])   # 3 或 4
        self.offset      = int(self.physics.model.tex_adr[self.tex_id])        # 起始索引
        self.length      = self.W * self.H * self.C

        # ---------- 生成纯白像素 ----------
        self.tex_img = np.full((self.H, self.W, self.C), 255, np.uint8)      # 255=白 (RGBA or RGB)

        # 写入 tex_data
        self.physics.model.tex_data[self.offset: self.offset+self.length] = self.tex_img.ravel()

        # 通知 GPU
        self.viewer.update_texture(self.tex_id)               # 只传 texid






    def step(self):
        # SLAM sense + mapping
        for agent in self.agents:
            agent.sense_and_update(self.dt, self.agents)

        # CBBA allocation
        cbba.allocate(self.agents)
        # breakpoint()
        # if self.step_count == 1:
        #     self.agents[0].goals = [(1.5, 1.5)]
        #     self.agents[1].goals = [(-1.5, -1.5)]
        #     self.agents[2].goals = [(2.0, -1.0)]
        # Control
        for agent in self.agents:
            agent.control_step(self.dt)

        # Physics step
        if self.viewer.is_running():
            self.physics.step()
            self.viewer.sync()  
            if self.step_count % 10 == 0:
                
                # 1) 获取概率地图并缩放为 512x512
                prob     = self.global_map.probability_map()                    # (h0, w0)
                gray255  = (prob * 255).astype(np.uint8)                        # 0–255
                scaled = cv2.resize(gray255[::-1, :], (self.W, self.H), interpolation=cv2.INTER_NEAREST)

                # 2) 构造 RGB(A) 图像（黑 = 已探测，白 = 未知）
                img =255-scaled                                               # 黑白翻转
                if self.C == 4:                                                  # RGBA
                    tex_px = np.repeat(img[..., None], 4, axis=-1)
                    tex_px[..., 3] = 255
                else:                                                            # RGB
                    tex_px = np.repeat(img[..., None], 3, axis=-1)

                # 3) 写入 tex_data 并上传纹理
                # breakpoint()
                self.physics.model.tex_data[self.offset : self.offset + self.length] = tex_px.ravel()
                self.viewer.update_texture(self.tex_id)


        else:
            return 
        self.step_count += 1

        # Optimise pose‑graph every 20 steps
        # if self.step_count % 20 == 0:
        #     self.graph.optimise()
        # if self.step_count % 5 == 0:
        #     self.img.set_data(self.global_map.probability_map())
        #     self.fig.canvas.draw_idle(); self.fig.canvas.flush_events()
            
    def run(self, max_steps=800):
        from tqdm import tqdm
        for _ in tqdm(range(max_steps), desc="Full‑SLAM explore"):
            self.step()
        print(f"Coverage = {self.global_map.coverage()*100:.1f}%")
