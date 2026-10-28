"""The ECMWF ensemble (open data, CC BY 4.0): 50 slightly different forecasts of total cloud cover.

The share of them that bring cloud (50 % or more) to a place is a real chance of cloud.
Only the cloud cover of each member is downloaded (byte ranges from the index file), every
3 hours of forecast, and kept as a small probability map on our grid until the next run.
"""
import json
import os
import shutil
import tempfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import ndimage

from .. import grid, net
from . import gribio

BASE = "https://data.ecmwf.int/forecasts/{d:%Y%m%d}/{d:%H}z/ifs/0p25/enfo/{d:%Y%m%d%H}0000-{s}h-enfo-ef"


def _index(run, step):
    r = net.get(BASE.format(d=run, s=step) + ".index", timeout=60)
    return [json.loads(l) for l in r.text.splitlines() if l.strip()]


def latest_run(now, last_step, log):
    base = now.replace(minute=0, second=0, microsecond=0)
    base -= dt.timedelta(hours=base.hour % 6)
    for k in range(0, 4):
        run = base - dt.timedelta(hours=6 * k)
        need = int((now - run).total_seconds() // 3600) + last_step
        need = int(np.ceil(need / 3.0) * 3)
        if need > 144:
            continue
        try:
            _index(run, need)
            log["ens_run"] = run.isoformat()
            return run
        except Exception:
            continue
    return None


_coords = {}


def _to_grid(vals, g):
    key = (g["lat0"], g["lon0"], g["dlat"], g["dlon"], vals.shape)
    if key not in _coords:
        lon, lat = grid.pixel_lonlat()
        r = (lat - g["lat0"]) / g["dlat"]
        c = ((lon - g["lon0"]) % 360) / g["dlon"]
        _coords[key] = np.array([r, c])
    return ndimage.map_coordinates(vals, _coords[key], order=1, mode="nearest")


def _member(url, item, tmpdir):
    off, ln = item["_offset"], item["_length"]
    r = net.get(url, headers={"Range": f"bytes={off}-{off + ln - 1}"}, timeout=90, ok=(206, 200))
    fn = os.path.join(tmpdir, f"m{item.get('number', 'x')}_{off}.grib2")
    with open(fn, "wb") as f:
        f.write(r.content)
    v, g = gribio.read_latlon(fn)
    os.remove(fn)
    return v, g


def step_probability(run, step, log):
    """(chance of cloud >= 50 %, mean cloud) as 0..1 arrays on our grid."""
    items = [i for i in _index(run, step) if i.get("param") == "tcc" and i.get("type") in ("pf", "cf")]
    if not items:
        raise ValueError("no tcc in index")
    url = BASE.format(d=run, s=step) + ".grib2"
    tmp = tempfile.mkdtemp()
    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            res = list(ex.map(lambda it: _member(url, it, tmp), items))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    g = res[0][1]
    stack = np.stack([v for v, _ in res])
    p = (stack >= 0.5).mean(axis=0).astype(np.float32)
    m = stack.mean(axis=0).astype(np.float32)
    log["ens_members"] = len(res)
    return np.clip(_to_grid(p, g), 0, 1), np.clip(_to_grid(m, g), 0, 1)


def fetch(now, hours, state_dir, log):
    """{valid time: chance of cloud} for the ensemble steps covering now-3h ... now+hours+3h."""
    run = latest_run(now, hours, log)
    if run is None:
        log["ens_error"] = "no recent ensemble run found"
        return {}
    d = os.path.join(state_dir, "ens")
    os.makedirs(d, exist_ok=True)
    tag = run.strftime("%Y%m%d%H")
    for old in os.listdir(d):
        if not old.startswith(tag):
            os.remove(os.path.join(d, old))
    first = int((now - run).total_seconds() // 3600) - 3
    out = {}
    for step in range(max(0, first // 3 * 3), first + hours + 7, 3):
        if step > 144:
            break
        fn = os.path.join(d, f"{tag}_{step:03d}.npz")
        if not os.path.exists(fn):
            try:
                p, m = step_probability(run, step, log)
                np.savez_compressed(fn, p=np.round(p * 250).astype(np.uint8))
                log["ens_downloads"] = log.get("ens_downloads", 0) + 1
            except Exception as e:
                log.setdefault("ens_errors", []).append(f"{step}: {e}"[:200])
                continue
        z = np.load(fn)
        out[run + dt.timedelta(hours=step)] = z["p"].astype(np.float32) / 250.0
    return out


def at(fields, t):
    """Chance of cloud at time t, linear between the 3-hourly steps."""
    ts = sorted(fields)
    for a, b in zip(ts, ts[1:]):
        if a <= t <= b:
            w = (t - a).total_seconds() / (b - a).total_seconds()
            return fields[a] * (1 - w) + fields[b] * w
    return None
