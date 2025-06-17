# pose_graph_torch.py  (示例)
import torch

class TorchPoseGraph:
    def __init__(self, device='cuda'):
        self.nodes   = {}      # id -> (x,y,theta)
        self.edges   = []      # (id_i, id_j, rel(3,), sqrt_info(3,3))
        self.device  = torch.device(device)

    def add_node(self, nid, pose):
        self.nodes[nid] = torch.tensor(pose, dtype=torch.float32,
                                       device=self.device)

    def add_edge(self, i, j, rel, info_sqrt):
        self.edges.append((i, j,
                           torch.tensor(rel       , dtype=torch.float32 , device=self.device),
                           torch.tensor(info_sqrt  , dtype=torch.float32, device=self.device)))

    # ---------- GPU Gauss-Newton ----------
    def optimise(self, fixed_id=None, iters=10):
        if fixed_id is None:
            fixed_id = list(self.nodes)[0]

        # 参数向量：排除固定节点
        ids   = list(self.nodes)
        idx_f = ids.index(fixed_id)
        poses = torch.stack([self.nodes[k] for k in ids])   # (N,3)
        poses = poses.clone().detach()
        poses = poses.to(self.device)

        mask  = torch.ones_like(poses, dtype=torch.bool)
        mask[idx_f] = False

        # 迭代
        for _ in range(iters):
            J_list, r_list = [], []
            for i, j, rel, info in self.edges:
                idx_i, idx_j = ids.index(i), ids.index(j)
                xi, xj = poses[idx_i], poses[idx_j]

                # 相对位姿预测
                ci, si = torch.cos(xi[2]), torch.sin(xi[2])
                R_iT   = torch.stack([ torch.stack([ ci, si]),
                                       torch.stack([-si, ci]) ])
                dz     = R_iT @ (xj[:2]-xi[:2])
                dth    = (xj[2]-xi[2]+torch.pi) % (2*torch.pi) - torch.pi
                pred   = torch.cat([dz, dth.unsqueeze(0)])

                e      = pred - rel                       # (3,)

                # 雅可比近似：对 xi 取负、对 xj 取正的恒等块
                Ji = torch.eye(3, device=self.device) * -1
                Jj = torch.eye(3, device=self.device)

                # 加权
                Ji = info @ Ji
                Jj = info @ Jj
                e  = info @ e

                # 拼进大稀疏块（这里简单展开为稠密，演示用）
                row = torch.zeros((3, poses.numel()),
                                  device=self.device)
                row[:, 3*idx_i:3*idx_i+3] = Ji
                row[:, 3*idx_j:3*idx_j+3] = Jj

                J_list.append(row)
                r_list.append(e)

            J = torch.cat(J_list)             # (3|E|, 3N)
            r = torch.cat(r_list)             # (3|E|,)

            # 去掉固定节点列
            J_free = J[:, mask.view(-1)].contiguous()
            # 高斯-牛顿步长： δ = (JᵀJ)⁻¹ Jᵀ r
            H = J_free.T @ J_free
            b = -J_free.T @ r
            lm = 1e-4 * torch.mean(torch.diag(H))
            H_reg = H + lm * torch.eye(H.shape[0], device=self.device)

            delta = torch.linalg.solve(H_reg, b)

            # 更新
            poses.view(-1)[mask.view(-1)] += delta

            # 收敛检测
            if delta.norm() < 1e-4:
                break

        # 写回
        for k, p in zip(ids, poses):
            self.nodes[k] = p
