"""Nowcast: measure how the clouds moved over the last half hour and carry them on.

The motion is found by optical flow (OpenCV's Farneback method) between two infrared pictures,
smoothed into a steady drift field, and the latest cloud and rain are moved along it in
15-minute steps (semi-Lagrangian advection).
"""
import cv2
import numpy as np
from scipy import ndimage

from . import grid

STEP_MIN = 15


def motion(bt_old, bt_new, minutes):
    """Drift in pixels per 15 minutes, as (u, v) arrays on our grid; plus a quality flag."""
    def prep(bt):
        a = np.nan_to_num(bt, nan=290.0)
        a = np.clip((300.0 - a) * 2.0, 0, 255).astype(np.uint8)   # colder (cloud) = brighter
        return cv2.resize(a, (grid.W // 2, grid.H // 2), interpolation=cv2.INTER_AREA)

    a, b = prep(bt_old), prep(bt_new)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, pyr_scale=0.5, levels=5, winsize=31,
                                        iterations=4, poly_n=7, poly_sigma=1.5, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    u, v = flow[..., 0], flow[..., 1]
    # weight by texture: flow over blank areas is meaningless, so fill it from the surroundings
    tex = cv2.GaussianBlur(np.abs(cv2.Laplacian(b.astype(np.float32), cv2.CV_32F)), (0, 0), 3)
    w = np.clip(tex / (np.percentile(tex, 75) + 1e-6), 0, 1)
    s = 12
    den = cv2.GaussianBlur(w, (0, 0), s) + 1e-6
    u = cv2.GaussianBlur(u * w, (0, 0), s) / den
    v = cv2.GaussianBlur(v * w, (0, 0), s) / den
    scale = 2.0 * STEP_MIN / float(minutes)          # back to full pixels, per 15 minutes
    u = cv2.resize(u, (grid.W, grid.H), interpolation=cv2.INTER_LINEAR) * scale
    v = cv2.resize(v, (grid.W, grid.H), interpolation=cv2.INTER_LINEAR) * scale
    # no more than about 150 km/h
    lim = 150.0 * STEP_MIN / 60.0 / 2.3
    mag = np.hypot(u, v)
    f = np.minimum(1.0, lim / (mag + 1e-6))
    return (u * f).astype(np.float32), (v * f).astype(np.float32)


def advect(field, u, v, steps, fill):
    """The field carried along (u, v) for 1..steps steps. Returns a list of arrays."""
    h, w = field.shape
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = X - u, Y - v
    out, cur = [], field.astype(np.float32)
    for _ in range(steps):
        cur = cv2.remap(cur, mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        out.append(cur.copy())
    return out


def neighbourhood(p, sigma_km, lat_mid=55.0):
    """Spread a cloud field over its surroundings: the chance of cloud within a few km."""
    s = max(0.5, sigma_km / grid.km_per_pixel(lat_mid))
    return cv2.GaussianBlur(p.astype(np.float32), (0, 0), s)


def motion_kmh(u, v, step=64):
    """Coarse drift field in km/h for the map's arrow (east and north positive)."""
    lat = grid.y2lat(grid.Y0 + np.arange(grid.H) + 0.5)
    kmpx = np.array([grid.km_per_pixel(x) for x in lat], dtype=np.float32)[:, None]
    ue = u * kmpx * (60.0 / STEP_MIN)
    vn = -v * kmpx * (60.0 / STEP_MIN)
    return grid.downsample(ue, step), grid.downsample(vn, step)
