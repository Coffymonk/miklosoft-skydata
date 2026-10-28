"""Polite HTTP: one identified session, short timeouts, a few retries, and a log of every call."""
import time
import requests

UA = "miklosoft-skydata/1.0 (+https://github.com/Coffymonk/miklosoft-skydata; stargazer@miklosoft.se)"

_s = requests.Session()
_s.headers["User-Agent"] = UA
LOG = []


class FetchError(Exception):
    pass


def get(url, params=None, timeout=60, tries=3, stream=False, headers=None, ok=(200,)):
    last = None
    for i in range(tries):
        t0 = time.time()
        try:
            r = _s.get(url, params=params, timeout=timeout, stream=stream, headers=headers)
            LOG.append(dict(url=r.url[:300], status=r.status_code, s=round(time.time() - t0, 2), bytes=len(r.content) if not stream else None))
            if r.status_code in ok:
                return r
            if r.status_code in (404, 204, 403, 400):
                raise FetchError(f"{r.status_code} {r.url[:200]}")
            last = FetchError(f"{r.status_code} {r.url[:200]}")
        except requests.RequestException as e:
            LOG.append(dict(url=url[:300], error=str(e)[:200]))
            last = FetchError(str(e))
        time.sleep(2 * (i + 1))
    raise last


def head_ok(url, timeout=20):
    try:
        r = _s.head(url, timeout=timeout, allow_redirects=True)
        LOG.append(dict(url=url[:300], status=r.status_code, head=True))
        return r.status_code == 200
    except requests.RequestException as e:
        LOG.append(dict(url=url[:300], error=str(e)[:200], head=True))
        return False
