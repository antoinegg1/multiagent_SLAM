# utils/lidar.py
import numpy as np
from math import cos, sin, pi
import mujoco

def ray_cast_2d(physics, body_id, n_beams=360, max_range=5.0):
    """
    Returns an (n_beams, 2) array of ray-endpoints in world X-Y using mujoco.mj_ray.
    Works with MuJoCo 3.x + dm_control 1.x.
    """
    model, data = physics.model.ptr, physics.data.ptr
    origin = physics.data.xpos[body_id].copy()      # 3-D world pos

    endpoints = []
    geomgroup   = np.zeros(6, dtype=np.uint8)       # enable all geom groups
    flg_static  = 0                                 # collide with static + dynamic
    bodyexclude = -1                                # don't exclude any body
    geomid_buf  = np.zeros(1, dtype=np.int32)       # will store hit geom id

    for a in np.linspace(0, 2*pi, n_beams, endpoint=False):
        # direction vector (not end-point)
        vec = max_range * np.array([cos(a), sin(a), 0.0], dtype=np.float64)

        # mj_ray returns fraction [0,1]; -1 if no hit
        frac = mujoco.mj_ray(
            model, data,
            origin.astype(np.float64), vec,
            geomgroup, flg_static, bodyexclude, geomid_buf
        )
        geom_id = int(geomid_buf[0])

        hit_pos = origin + frac * vec if geom_id != -1 else origin + vec
        endpoints.append(hit_pos[:2])

    return np.asarray(endpoints, dtype=np.float32)
