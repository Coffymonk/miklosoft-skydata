"""The common map grid: Web Mercator (EPSG:3857) at zoom 5, cut into 128-pixel tiles.

Everything (satellite, radar, forecasts) is resampled onto this one grid so a map in the
browser can lay the pictures over its own tiles without any further warping.
"""
import math
import numpy as np

ZOOM = 5
WORLD = 256 * 2 ** ZOOM          # pixels round the world at this zoom (8192)
TILE = 128                        # pixels per data tile
R = 6378137.0                     # Web Mercator sphere radius (m)

# Europe and its surroundings, snapped to whole tiles
TX0, TX1 = 27, 40                 # tile columns (x 3456 ... 5120)
TY0, TY1 = 13, 26                 # tile rows    (y 1664 ... 3328)
X0, X1 = TX0 * TILE, TX1 * TILE
Y0, Y1 = TY0 * TILE, TY1 * TILE
W, H = X1 - X0, Y1 - Y0           # 1664 x 1664


def lon2x(lon):
    return (np.asarray(lon, dtype=float) + 180.0) / 360.0 * WORLD


def lat2y(lat):
    s = np.sin(np.radians(np.asarray(lat, dtype=float)))
    return (0.5 - np.log((1 + s) / (1 - s)) / (4 * math.pi)) * WORLD


def x2lon(x):
    return np.asarray(x, dtype=float) / WORLD * 360.0 - 180.0


def y2lat(y):
    n = math.pi - 2 * math.pi * np.asarray(y, dtype=float) / WORLD
    return np.degrees(np.arctan(np.sinh(n)))


def pixel_lonlat():
    """Longitude and latitude of every pixel centre, as two (H, W) arrays."""
    xs = X0 + np.arange(W) + 0.5
    ys = Y0 + np.arange(H) + 0.5
    lon = x2lon(xs)
    lat = y2lat(ys)
    return np.broadcast_to(lon[None, :], (H, W)), np.broadcast_to(lat[:, None], (H, W))


def bbox_3857():
    """minx, miny, maxx, maxy of the grid in EPSG:3857 metres (for a WMS request)."""
    m = 2 * math.pi * R / WORLD
    minx = X0 * m - math.pi * R
    maxx = X1 * m - math.pi * R
    maxy = math.pi * R - Y0 * m
    miny = math.pi * R - Y1 * m
    return minx, miny, maxx, maxy


def bounds_lonlat():
    return dict(west=float(x2lon(X0)), east=float(x2lon(X1)), north=float(y2lat(Y0)), south=float(y2lat(Y1)))


def km_per_pixel(lat):
    return 2 * math.pi * R * math.cos(math.radians(lat)) / WORLD / 1000.0


def downsample(a, f, how="mean"):
    """Shrink an (H, W) array by an integer factor."""
    h, w = a.shape
    b = a[: h - h % f, : w - w % f].reshape(h // f, f, w // f, f)
    return np.nanmean(b, axis=(1, 3)) if how == "mean" else np.nanmax(b, axis=(1, 3))


def fill_nearest(a):
    """Fill NaN pixels with the nearest valid value (for edges of a model's area)."""
    from scipy import ndimage
    bad = np.isnan(a)
    if not bad.any() or bad.all():
        return a
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]
