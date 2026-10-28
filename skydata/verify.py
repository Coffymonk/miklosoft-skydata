"""How often were we right? Every run keeps its forecasts (coarsely); later runs compare them
with what the satellite actually saw, over land, and keep a 30-day tally per lead time.

The tally also says how reliable each percentage is (when we said 70 %, was it cloudy 7 times
in 10?), and that is used to correct the percentages: the forecast corrects itself.
"""
import json
import os
import re
import datetime as dt

import numpy as np

from . import grid

F = 8                                    # verification grid: every 8th pixel (208 x 208)
BUCKETS = [("15m", 15), ("30m", 30), ("1h", 60), ("2h", 120), ("3h", 180), ("6h", 360), ("12h", 720)]
NBINS = 11
DAYS = 30
K = 3000.0                               # how much data before the tally overrides the raw number


def bucket_of(lead_min):
    for name, m in BUCKETS:
        if lead_min == m:
            return name
    return None


def _coarse(p):
    return grid.downsample(p.astype(np.float32), F)


def save_forecasts(state_dir, issued, frames):
    """frames: {lead minutes: chance of cloud 0..1 (full grid)} for the verified leads."""
    d = os.path.join(state_dir, "verify")
    os.makedirs(d, exist_ok=True)
    arrs = {f"m{lead}": np.round(_coarse(p) * 250).astype(np.uint8) for lead, p in frames.items() if bucket_of(lead)}
    np.savez_compressed(os.path.join(d, f"fc_{issued:%Y%m%d%H%M}.npz"), **arrs)
    # forget forecasts older than the longest lead
    for fn in os.listdir(d):
        m = re.match(r"fc_(\d{12})\.npz$", fn)
        if m and dt.datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc) < issued - dt.timedelta(hours=13):
            os.remove(os.path.join(d, fn))


def load_stats(state_dir):
    fn = os.path.join(state_dir, "stats.json")
    if os.path.exists(fn):
        with open(fn) as f:
            return json.load(f)
    return {"days": {}}


def score(state_dir, now, observed, land, stats, log):
    """Compare the forecasts made for `now` with the cloud seen now (0 clear, 1 cloud, 255 none)."""
    d = os.path.join(state_dir, "verify")
    if not os.path.isdir(d) or observed is None:
        return
    obs = observed.astype(np.float32)
    obs[observed == 255] = np.nan
    oc = _coarse(obs)
    lm = grid.downsample(land.astype(np.float32), F) > 0.5
    ok = np.isfinite(oc) & lm
    if ok.sum() < 500:
        return
    cloudy = oc[ok] >= 0.5
    day = now.strftime("%Y-%m-%d")
    D = stats["days"].setdefault(day, {})
    for name, lead in BUCKETS:
        issued = now - dt.timedelta(minutes=lead)
        fn = os.path.join(d, f"fc_{issued:%Y%m%d%H%M}.npz")
        if not os.path.exists(fn):
            continue
        z = np.load(fn)
        if f"m{lead}" not in z:
            continue
        p = z[f"m{lead}"].astype(np.float32)[ok] / 250.0
        b = D.setdefault(name, {"n": 0, "hit": 0, "brier": 0.0, "rel": [[0, 0] for _ in range(NBINS)]})
        b["n"] += int(p.size)
        b["hit"] += int(((p >= 0.5) == cloudy).sum())
        b["brier"] += float(((p - cloudy) ** 2).sum())
        idx = np.clip(np.round(p * (NBINS - 1)).astype(int), 0, NBINS - 1)
        for i in range(NBINS):
            sel = idx == i
            b["rel"][i][0] += int(sel.sum())
            b["rel"][i][1] += int(cloudy[sel].sum())
        log.setdefault("verified", []).append(name)
    # keep 30 days
    cut = (now - dt.timedelta(days=DAYS)).strftime("%Y-%m-%d")
    for k in [k for k in stats["days"] if k < cut]:
        del stats["days"][k]


def summary(stats):
    """Per lead: share of hits, number of comparisons, the reliability table and days covered."""
    out = {}
    for name, _ in BUCKETS:
        n = hit = 0
        rel = [[0, 0] for _ in range(NBINS)]
        days = 0
        for D in stats["days"].values():
            b = D.get(name)
            if not b:
                continue
            days += 1
            n += b["n"]
            hit += b["hit"]
            for i in range(NBINS):
                rel[i][0] += b["rel"][i][0]
                rel[i][1] += b["rel"][i][1]
        if n:
            out[name] = {"right": round(100.0 * hit / n, 1), "n": n, "days": days, "rel": rel}
    return out


def calibrate(p, lead_min, summ):
    """Correct a chance of cloud with what the tally says that percentage really meant."""
    keys = [(name, m) for name, m in BUCKETS if name in summ]
    if not keys:
        return p
    name = min(keys, key=lambda x: abs(x[1] - lead_min))[0]
    rel = summ[name]["rel"]
    xs = np.linspace(0, 1, NBINS)
    ys = np.array([(r[1] + K * x) / (r[0] + K) for r, x in zip(rel, xs)], dtype=np.float32)
    ys = np.maximum.accumulate(ys)      # keep it rising: more forecast cloud never means less
    return np.interp(p, xs, ys).astype(np.float32)
