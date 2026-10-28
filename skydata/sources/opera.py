"""Rain from the European weather radars: the EUMETNET OPERA composite (open data, CC BY 4.0).

The newest composites sit in a public 24-hour bucket. We list it, take the maximum-reflectivity
(DBZH) composite nearest each wanted time and resample it onto our grid.
"""
import io
import os
import re
import datetime as dt
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from pyproj import CRS, Transformer
from scipy import ndimage

from .. import grid, net

BUCKET = "https://s3.waw3-1.cloudferro.com/openradar-24h/"
NAME = re.compile(r"OPERA@(\d{8}T\d{4})@0@DBZH\.h5$")


def list_files(day, after=None):
    """{datetime: key} of the DBZH composites of one UTC day."""
    out, token = {}, None
    prefix = day.strftime("%Y/%m/%d/OPERA/COMP/")
    for _ in range(10):
        p = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if after:
            p["start-after"] = prefix + after
        if token:
            p["continuation-token"] = token
        r = net.get(BUCKET, params=p, timeout=30)
        root = ET.fromstring(r.content)
        ns = {"s": root.tag.split("}")[0].strip("{")} if root.tag.startswith("{") else {}
        q = (lambda n: "s:" + n) if ns else (lambda n: n)
        for c in root.findall(q("Contents"), ns):
            k = c.find(q("Key"), ns).text
            m = NAME.search(k)
            if m:
                out[dt.datetime.strptime(m.group(1), "%Y%m%dT%H%M").replace(tzinfo=dt.timezone.utc)] = k
        t = root.find(q("NextContinuationToken"), ns)
        if t is None or not t.text:
            break
        token = t.text
    return out


def available(now):
    files = {}
    start = now - dt.timedelta(hours=3)
    for d in sorted({start.date(), now.date()}):
        day = dt.datetime.combine(d, dt.time(), dt.timezone.utc)
        after = start.strftime("OPERA@%Y%m%dT%H%M") if d == start.date() else None
        try:
            files.update(list_files(day, after))
        except net.FetchError:
            pass
    return files


def nearest(files, t, tol_min=6):
    best = None
    for ft, k in files.items():
        d = abs((ft - t).total_seconds())
        if d <= tol_min * 60 and (best is None or d < best[0]):
            best = (d, ft, k)
    return best and (best[1], best[2])


def _attrs(node):
    return {k: (v.decode() if isinstance(v, bytes) else v) for k, v in node.attrs.items()}


def read(raw):
    """dBZ array (NaN outside radar coverage, -inf where nothing was detected) and its geometry."""
    with h5py.File(io.BytesIO(raw), "r") as f:
        where = _attrs(f["where"])
        ds = None
        for dname in sorted(k for k in f.keys() if k.startswith("dataset")):
            for kname in sorted(k for k in f[dname].keys() if k.startswith("data")):
                what = {}
                for node in (f[dname].get("what"), f[dname][kname].get("what")):
                    if node is not None:
                        what.update(_attrs(node))
                if str(what.get("quantity", "DBZH")).upper().startswith("DBZ"):
                    ds = (f[dname][kname]["data"][...], what)
                    break
            if ds:
                break
        if ds is None:
            raise ValueError("no DBZH data in file")
    data, what = ds
    gain, offset = float(what.get("gain", 1.0)), float(what.get("offset", 0.0))
    nodata, undetect = what.get("nodata"), what.get("undetect")
    v = data.astype(np.float32) * gain + offset
    if nodata is not None:
        v[data == nodata] = np.nan
    if undetect is not None:
        v[data == undetect] = -np.inf
    return v, where


_map_cache = {}


def _index(where, shape, state_dir):
    """For every pixel of our grid: the row/column in the composite (or -1)."""
    key = (where.get("projdef"), shape, round(float(where["UL_lon"]), 4), round(float(where["UL_lat"]), 4))
    if key in _map_cache:
        return _map_cache[key]
    fn = os.path.join(state_dir, "opera_index.npz")
    if os.path.exists(fn):
        z = np.load(fn, allow_pickle=False)
        if str(z["key"]) == repr(key):
            _map_cache[key] = (z["r"], z["c"])
            return _map_cache[key]
    proj = CRS.from_proj4(where["projdef"])
    tr = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    ulx, uly = tr.transform(float(where["UL_lon"]), float(where["UL_lat"]))
    xs, ys = float(where["xscale"]), float(where["yscale"])
    lon, lat = grid.pixel_lonlat()
    px, py = tr.transform(lon, lat)
    c = np.floor((px - ulx) / xs).astype(np.int32)
    r = np.floor((uly - py) / ys).astype(np.int32)
    bad = (c < 0) | (r < 0) | (c >= shape[1]) | (r >= shape[0])
    c[bad] = -1
    r[bad] = -1
    np.savez_compressed(fn, key=repr(key), r=r, c=c)
    _map_cache[key] = (r, c)
    return r, c


def to_grid(v, where, state_dir):
    """Resample a composite onto our grid, keeping small showers (max of each 3x3 source block)."""
    finite = np.where(np.isfinite(v), v, -32.0)
    peak = ndimage.maximum_filter(finite, size=3)
    peak[np.isnan(v)] = np.nan
    r, c = _index(where, v.shape, state_dir)
    out = np.full(r.shape, np.nan, dtype=np.float32)
    ok = r >= 0
    out[ok] = peak[r[ok], c[ok]]
    return out


def frames(times, state_dir, log):
    """{time: dBZ grid} for each wanted time that has a composite."""
    out = {}
    if not times:
        return out
    files = available(max(times))
    log["opera_files"] = len(files)
    if files:
        log["opera_newest"] = max(files).isoformat()
    for t in times:
        hit = nearest(files, t)
        if not hit:
            continue
        ft, key = hit
        try:
            raw = net.get(BUCKET + key, timeout=90).content
            v, where = read(raw)
            out[t] = to_grid(v, where, state_dir)
            log.setdefault("opera_used", []).append(ft.isoformat())
            if "opera_where" not in log:
                log["opera_where"] = {k: (str(v)[:120]) for k, v in where.items()}
                log["opera_shape"] = list(v.shape)
        except Exception as e:  # one bad file must not stop the rest
            log.setdefault("opera_errors", []).append(f"{key}: {e}"[:200])
    return out


def rain_rate(dbz):
    """mm/h from dBZ with the Marshall-Palmer relation Z = 200 R^1.6."""
    z = np.power(10.0, np.asarray(dbz, dtype=np.float32) / 10.0)
    return np.power(z / 200.0, 1 / 1.6)
