import numpy as np, mujoco, math

def ray_cast_2d(physics, body_id, n_beams=360, max_range=1.0):
    """
    returns  endpoints (n,2)  and  hit_mask (n,) bool
    - 忽略本机器人自身
    - 忽略地面 (group=1)
    - 检测所有静态+动态障碍 (group=0、flg_static=1)
    """
    model, data = physics.model.ptr, physics.data.ptr
    # print(f"Ray casting from body {body_id} with {n_beams} beams, max range {max_range:.2f} m")
    origin = physics.data.xpos[body_id].copy()
    origin[2] += 0.05                         # 起点抬高 5 cm，避免落在 plane 共面

    endpoints, hits = [], []
    geomgroup   = np.ones(6, dtype=np.uint8)
    geomgroup[1] = 0                          # ★ 关闭组 1 → 地面被忽略
    flg_static  = 1                           # ★ 同时检测静态几何
    bodyexclude = body_id                     # ★ 排除自身
    buf = np.empty(1, dtype=np.int32)
    angles = np.linspace(0, 2*math.pi, n_beams, endpoint=False)
    angles = angles[::6] 
    for a in angles:
        buf[0] = -1                           # 每束光线前清零
        vec = max_range * np.array([math.cos(a), math.sin(a), 0.0])
        frac = mujoco.mj_ray(model, data, origin, vec,
                             geomgroup, flg_static, bodyexclude, buf)

        hit  = (buf[0] != -1)
        point = origin + frac * vec if hit else origin + vec
        endpoints.append(point[:2])
        hits.append(hit)

    return np.asarray(endpoints, np.float32), np.asarray(hits, bool)
