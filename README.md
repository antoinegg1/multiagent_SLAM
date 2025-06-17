
# Explorer Project (MuJoCo Multi‑Robot 2‑D Exploration)

A minimal, self‑contained prototype of 2‑D frontier exploration with **three robots**, simulated in MuJoCo 3.2.2.
It demonstrates:

* 360° lidar via MuJoCo ray‑casting  
* Bayesian occupancy grid mapping (local + global)  
* Frontier detection + **CBBA** task allocation  
* Simple P controller with obstacle avoidance  
* **Image‑based pose alignment** using **ORB + RANSAC** (OpenCV) to mimic real multi‑robot SLAM data fusion  

Run on **Windows 11 (Conda, Python 3.10)** or Linux.  
GPU is *not* required; real‑time speed on a single CPU core is common.

```bash
conda create -n explorer python=3.10 -y
conda activate explorer

# Tsinghua mirrors
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --set show_channel_urls yes

pip install mujoco==3.2.2 dm_control opencv-python numpy scipy matplotlib tqdm
```

```bash
python main.py                 # Launch the demo
```

---

## File Tree (generated)

```
explorer_project/
├── main.py
├── maze.xml
├── explorer_world.py
├── agent.py
├── mapping.py
├── cbba.py
├── control.py
├── image_pose.py
├── utils/
│   ├── __init__.py
│   ├── bresenham.py
│   └── lidar.py
└── README.md
```
