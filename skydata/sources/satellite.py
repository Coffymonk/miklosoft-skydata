"""Clouds seen from space, through EUMETSAT's EUMETView map service (EUMETSAT Core data).

* The Meteosat (MSG, 0 degree) Cloud Mask: cloud or clear, from all the satellite's channels,
  every 15 minutes. It stops short of the far north, where the satellite looks too slantwise.
* The Meteosat Third Generation (MTG) infrared 10.5 um picture, every 10 minutes, for the
  far north (clouds colder than the ground) and to measure how the clouds move.

The pictures are asked for directly in our map grid, so they need no warping.
"""
import io
import os
import re
import time
import datetime as dt
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .. import grid, net

WMS = "https://view.eumetsat.int/geoserver/ows"
CLM = "msg_fes:clm"
IR = "mtg_fd:ir105_hrfi"
LAND = "osmgray:simplified_land_polygons"
WMSNS = "{http://www.opengis.net/wms}"


def capabilities(state_dir, log, max_age=600):
    """Latest time of each layer we use, from the (cached) GetCapabilities document."""
    fn = os.path.join(state_dir, "wms_caps.xml")
    if not (os.path.exists(fn) and time.time() - os.path.getmtime(fn) < max_age):
        r = net.get(WMS, params=dict(service="WMS", version="1.3.0", request="GetCapabilities"), timeout=60)
        with open(fn, "wb") as f:
            f.write(r.content)
    root = ET.parse(fn).getroot()
    out = {}
    for L in root.iter(WMSNS + "Layer"):
        n = L.find(WMSNS + "Name")
        if n is None or n.text not in (CLM, IR):
            continue
        d = L.find(WMSNS + "Dimension")
        if d is None:
            continue
        m = re.search(r"/(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.\d+)?Z/PT(\d+)M\s*$", (d.text or "").strip())
        if m:
            out[n.text] = (dt.datetime.fromisoformat(m.group(1)).replace(tzinfo=dt.timezone.utc), int(m.group(2)))
    log["sat_latest"] = {k: v[0].isoformat() for k, v in out.items()}
    return out


def getmap(layer, t=None, timeout=120):
    minx, miny, maxx, maxy = grid.bbox_3857()
    p = dict(service="WMS", version="1.3.0", request="GetMap", layers=layer, styles="", crs="EPSG:3857",
             bbox=f"{minx},{miny},{maxx},{maxy}", width=grid.W, height=grid.H, format="image/png", transparent="true")
    if t is not None:
        p["time"] = t.strftime("%Y-%m-%dT%H:%M:%SZ")
    r = net.get(WMS, params=p, timeout=timeout)
    if not r.headers.get("Content-Type", "").startswith("image/"):
        raise net.FetchError(f"{layer}: not an image ({r.text[:160]})")
    return np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))


def decode_clm(a):
    """0 clear, 1 cloud, 255 no data. White = cloud, green = clear land, blue = clear water."""
    out = np.full(a.shape[:2], 255, np.uint8)
    vis = a[..., 3] > 127
    r, g, b = a[..., 0].astype(int), a[..., 1].astype(int), a[..., 2].astype(int)
    cloud = vis & (r > 200) & (g > 200) & (b > 200)
    clear = vis & ~cloud & ((g > 120) | (b > 120))
    out[cloud] = 1
    out[clear] = 0
    return out


def decode_ir(a):
    """Brightness temperature (K) from the grey scale: white -80 C ... black +30 C. NaN = no data."""
    g = a[..., 0].astype(np.float32)
    bt = 273.15 - 80.0 + (255.0 - g) / 255.0 * 110.0
    bt[a[..., 3] < 128] = np.nan
    return bt


def land_mask(state_dir):
    fn = os.path.join(state_dir, "landmask.npy")
    if os.path.exists(fn):
        return np.load(fn)
    a = getmap(LAND, None)
    m = a[..., 3] > 127
    np.save(fn, m)
    return m


def _cache(state_dir, kind, t):
    d = os.path.join(state_dir, "sat")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{kind}_{t:%Y%m%d%H%M}.npz")


def prune(state_dir, keep_after):
    d = os.path.join(state_dir, "sat")
    if not os.path.isdir(d):
        return
    for fn in os.listdir(d):
        m = re.search(r"_(\d{12})\.npz$", fn)
        if m and dt.datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc) < keep_after:
            os.remove(os.path.join(d, fn))


def clm(t, state_dir, log):
    fn = _cache(state_dir, "clm", t)
    if os.path.exists(fn):
        return np.load(fn)["v"]
    try:
        v = decode_clm(getmap(CLM, t))
    except Exception as e:
        log.setdefault("sat_errors", []).append(f"clm {t:%H%M}: {e}"[:200])
        return None
    if (v == 255).all():
        return None
    np.savez_compressed(fn, v=v)
    log["sat_downloads"] = log.get("sat_downloads", 0) + 1
    return v


def ir(t, state_dir, log):
    fn = _cache(state_dir, "ir", t)
    if os.path.exists(fn):
        g = np.load(fn)["g"]
    else:
        try:
            a = getmap(IR, t, timeout=180)
        except Exception as e:
            log.setdefault("sat_errors", []).append(f"ir {t:%H%M}: {e}"[:200])
            return None
        g = np.where(a[..., 3] < 128, 0, np.maximum(a[..., 0], 1)).astype(np.uint8)  # 0 = no data
        np.savez_compressed(fn, g=g)
        log["sat_downloads"] = log.get("sat_downloads", 0) + 1
    bt = 273.15 - 80.0 + (255.0 - g.astype(np.float32)) / 255.0 * 110.0
    bt[g == 0] = np.nan
    return bt


def ir_cloud(bt, tsurf):
    """Our own simple cloud test where the Cloud Mask has no data: clouds are colder than the ground."""
    out = np.full(bt.shape, 255, np.uint8)
    ok = np.isfinite(bt)
    cold = bt < 248.0
    if tsurf is not None:
        cold |= np.isfinite(tsurf) & (bt < tsurf - 9.0)
    out[ok] = 0
    out[ok & cold] = 1
    return out
