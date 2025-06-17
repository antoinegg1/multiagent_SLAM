
"""Integer Bresenham ray‑tracing for grid occupancy update."""


def bresenham_limited(r0, c0, r1, c1, max_cells):
    """返回从 (r0,c0) 出发、最长 max_cells+1 个格的 Bresenham 路径"""
    pts = bresenham(r0, c0, r1, c1)
    if len(pts) > max_cells + 1:
        return pts[:max_cells + 1]
    return pts
def bresenham(x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    pts = []
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return pts
