
import numpy as np

def p_controller(current, target, kp=1.0, vmax=1.0):
    """Return (vx, vy) toward target with proportional gain."""
    err = np.clip(target - current, -vmax, vmax)
    return kp * err
