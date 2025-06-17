
"""Wrapper for MuJoCo ray casting producing 360° lidar endpoints."""
import numpy as np
from math import cos, sin, pi

def ray_cast_2d(physics, body_id, n_beams=360, max_range=5.0):
    origin = physics.bind(physics.model.body_xpos[body_id]).copy()
    angles = np.linspace(0, 2*pi, n_beams, endpoint=False)
    endpoints = []

    for a in angles:
        dx, dy = cos(a), sin(a)
        geom_id, ray_end, _ = physics.ray_fast(origin[:3],
                                               origin[:3] + max_range * np.array([dx, dy, 0.0]))
        endpoints.append(ray_end[:2])
    return np.array(endpoints)
