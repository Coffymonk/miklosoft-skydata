"""Forecast cloud and rain from DWD's ICON-EU model (open data, credit: Deutscher Wetterdienst).

Total cloud cover (CLCT, %) and total precipitation (TOT_PREC, mm since the start of the run)
on the 0.0625-degree European grid, hourly. Runs every 3 hours.
"""
import bz2
import os
import shutil
import datetime as dt

import numpy as np
from scipy import ndimage

from .. import grid, net
from . import gribio

BASE = "https://opendata.dwd.de/weather/nwp/icon-eu/grib/{hh:02d}/{var}/"
FILE = "icon-eu_europe_regular-lat-lon_single-level_{run}_{step:03d}_{VAR}.grib2.bz2"
VARS = ("clct", "tot_prec")


def url(run, var, step):
    return BASE.format(hh=run.hour, var=var) + FILE.format(run=run.strftime("%Y%m%d%H"), step=step, VAR=var.upper())


def latest_run(now, need_step, log):
    """Newest run whose files reach `need_step` hours (tries the last few runs)."""
    base = now.replace(minute=0, second=0, microsecond=0)
    base -= dt.timedelta(hours=base.hour % 3)
    for k in range(0, 6):
        run = base - dt.timedelta(hours=3 * k)
        step = need_step + int((now - run).total_seconds() // 3600)
        if step > 78:
            continue
        if net.head_ok(url(run, "clct", step)) and net.head_ok(url(run, "tot_prec", step)):
            log["icon_run"] = run.isoformat()
            return run
    return None


def fetch(now, hours, state_dir, log):
    """Download (or reuse) the steps covering now ... now+hours. Returns (run, {step: {var: path}})."""
    run = latest_run(now, hours, log)
    if run is None:
        log["icon_error"] = "no recent ICON-EU run found"
        return None, {}
    d = os.path.join(state_dir, "icon", run.strftime("%Y%m%d%H"))
    os.makedirs(d, exist_ok=True)
    # drop older runs from the state
    root = os.path.join(state_dir, "icon")
    for old in os.listdir(root):
        if old != os.path.basename(d):
            shutil.rmtree(os.path.join(root, old), ignore_errors=True)
    first = max(0, int((now - run).total_seconds() // 3600))
    files = {}
    for step in range(first, first + hours + 2):
        if step > 78:
            break
        files[step] = {}
        for var in VARS + (("t_g",) if step <= first + 1 else ()):
            fn = os.path.join(d, f"{var}_{step:03d}.grib2")
            if not os.path.exists(fn):
                try:
                    raw = net.get(url(run, var, step), timeout=60).content
                    with open(fn + ".tmp", "wb") as f:
                        f.write(bz2.decompress(raw))
                    os.replace(fn + ".tmp", fn)
                    log["icon_downloads"] = log.get("icon_downloads", 0) + 1
                except Exception as e:
                    log.setdefault("icon_errors", []).append(f"{var} {step}: {e}"[:200])
                    continue
            files[step][var] = fn
    return run, files


_coords = {}


def to_grid(fn):
    """Bilinear resample of one regular lat-lon GRIB field onto our grid (NaN outside)."""
    vals, g = gribio.read_latlon(fn)
    key = (g["lat0"], g["lon0"], g["dlat"], g["dlon"], vals.shape)
    if key not in _coords:
        lon, lat = grid.pixel_lonlat()
        r = (lat - g["lat0"]) / g["dlat"]
        c = (lon - g["lon0"]) / g["dlon"]
        _coords[key] = np.array([r, c])
    rc = _coords[key]
    return ndimage.map_coordinates(vals, rc, order=1, mode="constant", cval=np.nan).astype(np.float32)


def fields(run, files, times, log):
    """{time: (cloud 0..1, rain mm/h)} at the wanted times (hourly steps, linear in time)."""
    cache = {}

    def at(step):
        if step in cache:
            return cache[step]
        f = files.get(step, {})
        if "clct" not in f or "tot_prec" not in f:
            cache[step] = None
            return None
        cl = to_grid(f["clct"]) / 100.0
        tp = to_grid(f["tot_prec"])
        cache[step] = (np.clip(cl, 0, 1), tp)
        return cache[step]

    out = {}
    for t in times:
        h = (t - run).total_seconds() / 3600.0
        s0 = int(np.floor(h))
        a, b = at(s0), at(s0 + 1)
        if a is None or b is None:
            continue
        w = h - s0
        cloud = a[0] * (1 - w) + b[0] * w
        rain = np.maximum(b[1] - a[1], 0)  # mm in that hour = mm/h
        out[t] = (cloud.astype(np.float32), rain.astype(np.float32))
    log["icon_frames"] = len(out)
    return out


def surface_temperature(run, files, t):
    """Ground temperature (K) near time t, for the infrared cloud test."""
    step = int(round((t - run).total_seconds() / 3600.0))
    for s in (step, step + 1, step - 1):
        f = files.get(s, {})
        if "t_g" in f:
            return to_grid(f["t_g"])
    return None
