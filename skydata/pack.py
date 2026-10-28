"""Pack the frames into small tiles for the website.

Each pixel of each frame is one byte: the high four bits are cloud (0..10 = 0 % ... 100 %
chance of cloud, 15 = no data), the low four bits rain (0 = none, 1..11 = heavier and heavier,
15 = not known there). A tile file holds all frames of a 128 x 128 square, gzipped.
"""
import gzip
import json
import os

import numpy as np

from . import grid

RAIN_EDGES = [0.1, 0.3, 0.6, 1, 2, 4, 7, 12, 20, 35, 60]     # mm/h


def rain_class(mmh):
    c = np.searchsorted(np.array(RAIN_EDGES, dtype=np.float32), np.nan_to_num(mmh, nan=0.0), side="right")
    c = c.astype(np.uint8)
    c[np.isnan(mmh)] = 15
    return c


def cloud_level(p):
    c = np.round(np.clip(np.nan_to_num(p, nan=0.0), 0, 1) * 10).astype(np.uint8)
    c[np.isnan(p)] = 15
    return c


def encode(cloud_p, rain_mmh):
    return (cloud_level(cloud_p) << 4) | rain_class(rain_mmh)


def write_tiles(out_dir, frames, tag):
    """frames: list of (H, W) uint8 arrays. Writes tiles/{tag}/{tx}_{ty}.bin.gz; returns tile names."""
    d = os.path.join(out_dir, "tiles", tag)
    os.makedirs(d, exist_ok=True)
    stack = np.stack(frames)
    names = []
    total = 0
    for ty in range(grid.TY0, grid.TY1):
        for tx in range(grid.TX0, grid.TX1):
            y = (ty - grid.TY0) * grid.TILE
            x = (tx - grid.TX0) * grid.TILE
            block = np.ascontiguousarray(stack[:, y:y + grid.TILE, x:x + grid.TILE])
            if (block == 0xFF).all():
                continue
            name = f"{tx}_{ty}"
            data = gzip.compress(block.tobytes(), compresslevel=9, mtime=0)
            with open(os.path.join(d, name + ".bin.gz"), "wb") as f:
                f.write(data)
            total += len(data)
            names.append(name)
    return names, total


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
