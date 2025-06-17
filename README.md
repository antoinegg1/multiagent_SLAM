
# Explorer Project – **Full‑SLAM Edition**

This variant removes all access to MuJoCo “ground‑truth” poses.  
Every robot now estimates its own trajectory with **noisy wheel‑odometry ➜ pose‑graph ➜ ORB + RANSAC loop‑closure**, then optimises the graph (Gauss‑Newton via SciPy).

Run on Windows 11 + Conda or Linux exactly as before:

```bash
conda create -n explorer_slam python=3.10 -y
conda activate explorer_slam

# Tsinghua mirrors
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes

pip install mujoco==3.2.2 dm_control opencv-python numpy scipy matplotlib tqdm
```

> **No external SLAM library needed** – a 40‑line Gauss‑Newton optimiser in `slam_graph.py` handles 2‑D pose‑graphs.

```bash
python main.py
```

---

## Top‑level Changes vs “mapping only” version

| Module | New / Updated | What changed |
|--------|---------------|--------------|
| `slam_graph.py` | **NEW** | Minimal pose‑graph container + Gauss‑Newton optimiser |
| `agent.py` | **UPDATED** | • Integrates **commanded v + Gaussian noise** for odometry<br>• Adds odometry & inter‑robot loop‑closure edges to `slam_graph`<br>• Uses optimised pose to update mapping & control |
| `explorer_world.py` | updated | Orchestrates global pose‑graph optimisation every N steps |

Everything else (mapping, CBBA, control) stays the same.
