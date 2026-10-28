"""One run of the pipeline: fetch, combine, forecast, check, pack.

    python -m skydata.run --state state --out out
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback

import numpy as np

from . import grid, net, nowcast, pack, verify
from .sources import ens, icon, opera, satellite

UTC = dt.timezone.utc
PAST_MIN = 120          # observed frames back in time
NOW_MIN = 120           # nowcast frames ahead (15-minute steps)
AHEAD_H = 18            # model frames ahead (hourly)

CREDITS = [
    {"name": "EUMETSAT", "what": "Meteosat Cloud Mask and Meteosat Third Generation infrared images",
     "text": "Contains modified EUMETSAT Meteosat data {year}", "url": "https://www.eumetsat.int", "licence": "EUMETSAT data policy (Core data)"},
    {"name": "EUMETNET OPERA", "what": "European weather radar composite",
     "text": "Radar data: EUMETNET OPERA, CC BY 4.0, modified", "url": "https://www.eumetnet.eu/activities/observations-programme/current-activities/opera/", "licence": "CC BY 4.0"},
    {"name": "Deutscher Wetterdienst (DWD)", "what": "ICON-EU forecast model",
     "text": "Source: Deutscher Wetterdienst (DWD), ICON-EU, CC BY 4.0, modified", "url": "https://opendata.dwd.de", "licence": "CC BY 4.0"},
    {"name": "ECMWF", "what": "ECMWF ensemble forecast (open data)",
     "text": "Copyright European Centre for Medium-Range Weather Forecasts (ECMWF), CC BY 4.0, modified", "url": "https://www.ecmwf.int", "licence": "CC BY 4.0"},
]


def iso(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def floor_min(t, m):
    t = t.replace(second=0, microsecond=0)
    return t - dt.timedelta(minutes=t.minute % m)


def model_probability(t, T, icon_f, ens_f):
    """Chance of cloud (>= 50 % cover) from ICON-EU (spread over the neighbourhood) and the ensemble."""
    parts = []
    lead_h = max(0.0, (t - T).total_seconds() / 3600.0)
    if t in icon_f:
        c = icon_f[t][0]
        nanmask = np.isnan(c)
        p = nowcast.neighbourhood(np.where(nanmask, 0.0, (c >= 0.5).astype(np.float32)), 10.0 + 2.0 * lead_h)
        p[nanmask] = np.nan
        parts.append(p)
    e = ens.at(ens_f, t)
    if e is not None:
        parts.append(e)
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    a, b = parts
    return np.where(np.isnan(a), b, 0.5 * a + 0.5 * b).astype(np.float32)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="state")
    ap.add_argument("--out", default="out")
    ap.add_argument("--now", default=None, help="pretend time (ISO) for tests")
    a = ap.parse_args(argv)
    os.makedirs(a.state, exist_ok=True)
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()
    log = {"started": iso(dt.datetime.now(UTC))}
    now = dt.datetime.fromisoformat(a.now) if a.now else dt.datetime.now(UTC)

    # 1. the newest satellite time sets "now" for the observations
    caps = satellite.capabilities(a.state, log)
    if satellite.CLM not in caps:
        raise SystemExit("Cloud Mask times not found")
    T = min(floor_min(caps[satellite.CLM][0], 15), floor_min(now, 15))
    ir_last = caps.get(satellite.IR, (T, 10))[0]
    obs_t = [T - dt.timedelta(minutes=m) for m in range(PAST_MIN, -1, -15)]
    log["T"] = iso(T)
    satellite.prune(a.state, T - dt.timedelta(minutes=PAST_MIN + 60))

    # 2. models
    icon_run, icon_files = icon.fetch(T, AHEAD_H + 1, a.state, log)
    fc_t = [T + dt.timedelta(minutes=m) for m in range(15, NOW_MIN + 1, 15)]
    first_h = (T + dt.timedelta(minutes=NOW_MIN + 59)).replace(minute=0)
    mod_t = [first_h + dt.timedelta(hours=h) for h in range(0, AHEAD_H) if first_h + dt.timedelta(hours=h) <= T + dt.timedelta(hours=AHEAD_H)]
    verify_extra = [T + dt.timedelta(minutes=m) for _, m in verify.BUCKETS if m > NOW_MIN]
    icon_f = icon.fields(icon_run, icon_files, sorted(set(fc_t + mod_t + verify_extra)), log) if icon_run else {}
    tg = icon.surface_temperature(icon_run, icon_files, T) if icon_run else None
    if tg is not None:
        tg = grid.fill_nearest(tg)
    try:
        ens_f = ens.fetch(T, AHEAD_H + 1, a.state, log)
    except Exception as e:
        log["ens_error"] = str(e)[:200]
        ens_f = {}

    # 3. observations: clouds (Cloud Mask, else our infrared test) and rain (radar)
    radar = opera.frames(obs_t, a.state, log)
    land = satellite.land_mask(a.state)
    obs_cloud, ir_frames = {}, {}
    # fetch the missing pictures a few at a time (each is cached for later runs)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as ex:
        jobs = [ex.submit(satellite.ir, min(floor_min(t, 10), ir_last), a.state, log) for t in obs_t]
        jobs += [ex.submit(satellite.clm, t, a.state, log) for t in obs_t]
        for j in jobs:
            j.result()
    for t in obs_t:
        ti = min(floor_min(t, 10), ir_last)
        bt = satellite.ir(ti, a.state, log)
        if bt is not None:
            ir_frames[t] = bt
        c = satellite.clm(t, a.state, log)
        if c is None and bt is None:
            continue
        if c is None:
            c = satellite.ir_cloud(bt, tg)
        elif bt is not None:
            gap = c == 255
            if gap.any():
                c = c.copy()
                c[gap] = satellite.ir_cloud(bt, tg)[gap]
        obs_cloud[t] = c
    log["obs_cloud"] = len(obs_cloud)
    log["obs_radar"] = len(radar)

    def cloud_p(c):
        p = c.astype(np.float32)
        p[c == 255] = np.nan
        return p

    def rain_mmh(dbz):
        r = opera.rain_rate(np.where(np.isneginf(dbz), -32.0, dbz))
        r[np.isnan(dbz)] = np.nan
        return r

    # 4. motion and nowcast
    t_old = T - dt.timedelta(minutes=30)
    if T in ir_frames and t_old in ir_frames:
        u, v = nowcast.motion(ir_frames[t_old], ir_frames[T], 30)
        log["motion"] = "infrared"
    else:
        u = v = np.zeros((grid.H, grid.W), np.float32)
        log["motion"] = "none"
    steps = len(fc_t)
    adv_cloud = adv_rain = adv_cov = None
    if T in obs_cloud:
        p0 = np.nan_to_num(cloud_p(obs_cloud[T]), nan=0.5)
        adv_cloud = nowcast.advect(p0, u, v, steps, fill=0.5)
    if T in radar:
        d0 = radar[T]
        adv_rain = nowcast.advect(np.where(np.isfinite(d0), d0, -32.0), u, v, steps, fill=-32.0)
        adv_cov = nowcast.advect(np.isfinite(d0) | np.isneginf(d0), u, v, steps, fill=0.0)

    summ = verify.summary(verify.load_stats(a.state))

    frames, meta = [], []
    for t in obs_t:
        c = cloud_p(obs_cloud[t]) if t in obs_cloud else np.full((grid.H, grid.W), np.nan, np.float32)
        r = rain_mmh(radar[t]) if t in radar else np.full((grid.H, grid.W), np.nan, np.float32)
        frames.append(pack.encode(c, r))
        meta.append({"t": iso(t), "k": "obs", "c": t in obs_cloud, "r": t in radar})

    to_verify = {}
    for i, t in enumerate(fc_t):
        lead = (i + 1) * 15
        pm = model_probability(t, T, icon_f, ens_f)
        if adv_cloud is not None:
            pn = nowcast.neighbourhood(adv_cloud[i], 3.0 + 12.0 * lead / 60.0)
            w = max(0.0, 1.0 - lead / 180.0)
            p = pn if pm is None else np.where(np.isnan(pm), pn, w * pn + (1 - w) * pm)
        else:
            p = pm if pm is not None else np.full((grid.H, grid.W), np.nan, np.float32)
        p = p.astype(np.float32)
        if verify.bucket_of(lead):
            to_verify[lead] = p
        p = verify.calibrate(p, lead, summ)
        if adv_rain is not None:
            r = opera.rain_rate(adv_rain[i])
            r[adv_rain[i] <= -31.0] = 0.0
            r[adv_cov[i] < 0.5] = np.nan       # outside the radars: not known, as in the measured frames
        elif t in icon_f:
            r = icon_f[t][1]
        else:
            r = np.full((grid.H, grid.W), np.nan, np.float32)
        frames.append(pack.encode(p, r))
        meta.append({"t": iso(t), "k": "now"})

    for t in sorted(set(mod_t + verify_extra)):
        lead = int((t - T).total_seconds() // 60)
        pm = model_probability(t, T, icon_f, ens_f)
        if pm is None:
            continue
        if verify.bucket_of(lead):
            to_verify[lead] = pm
        if t not in mod_t:
            continue
        p = verify.calibrate(pm, lead, summ)
        r = icon_f[t][1] if t in icon_f else np.full((grid.H, grid.W), np.nan, np.float32)
        frames.append(pack.encode(p, r))
        meta.append({"t": iso(t), "k": "fc"})

    # 5. check earlier forecasts against what is seen now, then keep ours for later runs
    stats = verify.load_stats(a.state)
    if T in obs_cloud:
        verify.score(a.state, T, obs_cloud[T], land, stats, log)
    with open(os.path.join(a.state, "stats.json"), "w") as f:
        json.dump(stats, f)
    verify.save_forecasts(a.state, T, to_verify)
    summ = verify.summary(stats)

    # 6. pack
    gen = dt.datetime.now(UTC)
    tag = gen.strftime("%Y%m%dT%H%M")
    names, size = pack.write_tiles(a.out, frames, tag)
    mu, mv = nowcast.motion_kmh(u, v, 64)
    year = T.year
    manifest = {
        "v": 1,
        "generated": iso(gen),
        "set": tag,
        "t0": iso(T),
        "grid": {"zoom": grid.ZOOM, "tile": grid.TILE, "tx": [grid.TX0, grid.TX1], "ty": [grid.TY0, grid.TY1]},
        "frames": meta,
        "tiles": names,
        "rain_mmh": pack.RAIN_EDGES,
        "motion": {"step": 64, "nx": int(mu.shape[1]), "ny": int(mu.shape[0]),
                   "u": [int(x) for x in np.round(np.nan_to_num(mu)).ravel()],
                   "v": [int(x) for x in np.round(np.nan_to_num(mv)).ravel()]},
        "runs": {"icon_eu": log.get("icon_run"), "ecmwf_ens": log.get("ens_run"),
                 "radar_newest": log.get("opera_newest"), "satellite": log.get("sat_latest")},
        "accuracy": {k: {"right": v["right"], "n": v["n"], "days": v["days"]} for k, v in summ.items()},
        "credits": [dict(c, text=c["text"].format(year=year)) for c in CREDITS],
    }
    pack.write_json(os.path.join(a.out, "manifest.json"), manifest)
    pack.write_json(os.path.join(a.out, "accuracy.json"), summ)
    log["tiles"] = len(names)
    log["tiles_bytes"] = size
    log["seconds"] = round(time.time() - t_start, 1)
    log["http"] = net.LOG[-60:]
    pack.write_json(os.path.join(a.out, "log.json"), log)
    print(json.dumps({k: v for k, v in log.items() if k != "http"}, indent=1, default=str))


if __name__ == "__main__":
    code = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)   # skip interpreter teardown: native libraries can crash while unloading
