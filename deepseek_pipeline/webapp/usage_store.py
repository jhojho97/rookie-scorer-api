# -*- coding: utf-8 -*-
"""
usage_store.py
--------------
Where per-user spend is counted.

The original counter lived in a module-level dict, which made the "monthly cap"
a claim the server could not keep: this runs on a free instance that spins down
when idle, and every spin-up starts a fresh process with an empty dict. A user
who waited out a sleep got their full budget back, repeatedly. A local file does
not help either -- the container filesystem is rebuilt from the image on each
start, so it resets at exactly the same moments.

Durability therefore needs a store outside the process. This module provides:

  RedisRestStore  - any Upstash-compatible Redis REST endpoint. Chosen over a
                    client library because it is plain HTTPS through `requests`
                    (already a dependency) and adds no memory to a container
                    that idles at ~408MB of 512MB. Increments are server-side
                    and atomic, which also makes them correct across the batch
                    worker threads.
  MemoryStore     - the previous behaviour, used only when nothing is
                    configured. It reports durable=False so callers can say so
                    instead of implying a guarantee.

Configure with:
    ROOKIE_USAGE_REDIS_URL    https://<name>.upstash.io
    ROOKIE_USAGE_REDIS_TOKEN  <rest token>
"""

import os
import time
import threading
from datetime import datetime, timezone


def month_key(ts=None) -> str:
    d = datetime.fromtimestamp(ts or time.time(), timezone.utc)
    return d.strftime("%Y-%m")


def hour_key(ts=None) -> str:
    d = datetime.fromtimestamp(ts or time.time(), timezone.utc)
    return d.strftime("%Y-%m-%dT%H")


class MemoryStore:
    """Process-local counters. Accurate while the process lives, and gone the
    moment it restarts -- which on a free instance is every idle timeout."""

    durable = False
    name = "memory"

    def __init__(self):
        self._usd = {}       # (uid, month) -> float
        self._n = {}         # (uid, hour)  -> int
        self._lock = threading.Lock()

    def add(self, uid: str, usd: float) -> None:
        with self._lock:
            self._usd[(uid, month_key())] = self._usd.get((uid, month_key()), 0.0) + float(usd or 0)
            self._n[(uid, hour_key())] = self._n.get((uid, hour_key()), 0) + 1
            # keep only the current windows so the dicts cannot grow unbounded
            m, h = month_key(), hour_key()
            for k in [k for k in self._usd if k[1] != m]:
                self._usd.pop(k, None)
            for k in [k for k in self._n if k[1] != h]:
                self._n.pop(k, None)

    def month_usd(self, uid: str) -> float:
        with self._lock:
            return round(self._usd.get((uid, month_key()), 0.0), 6)

    def hour_count(self, uid: str) -> int:
        with self._lock:
            return int(self._n.get((uid, hour_key()), 0))


class RedisRestStore:
    """Upstash-compatible Redis over HTTPS.

    Counters are keyed by calendar month and calendar hour rather than a sliding
    window: a fixed window is one atomic INCR instead of a stored timestamp list,
    and the imprecision it buys (a user may spend up to two hours' allowance
    across an hour boundary) does not matter for a spend guard.
    """

    durable = True
    name = "redis"

    def __init__(self, url: str, token: str, timeout: float = 4.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._session = None

    def _http(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({"Authorization": f"Bearer {self.token}"})
        return self._session

    def _pipeline(self, commands):
        r = self._http().post(f"{self.url}/pipeline", json=commands, timeout=self.timeout)
        r.raise_for_status()
        return [item.get("result") for item in r.json()]

    def _usd_key(self, uid):
        return f"rookie:usd:{uid}:{month_key()}"

    def _n_key(self, uid):
        return f"rookie:n:{uid}:{hour_key()}"

    def add(self, uid: str, usd: float) -> None:
        uk, nk = self._usd_key(uid), self._n_key(uid)
        self._pipeline([
            ["INCRBYFLOAT", uk, str(float(usd or 0))],
            ["EXPIRE", uk, str(60 * 60 * 24 * 40)],   # outlive the month
            ["INCR", nk],
            ["EXPIRE", nk, str(60 * 60 * 3)],
        ])

    def month_usd(self, uid: str) -> float:
        v = self._pipeline([["GET", self._usd_key(uid)]])[0]
        return round(float(v), 6) if v else 0.0

    def hour_count(self, uid: str) -> int:
        v = self._pipeline([["GET", self._n_key(uid)]])[0]
        return int(v) if v else 0


class SafeStore:
    """Wraps a remote store so a network blip cannot take scoring down.

    Reads fall back to a local mirror; writes are mirrored locally too, so a
    temporary outage degrades to the in-process behaviour rather than either
    crashing the request or silently granting an unlimited budget.
    """

    def __init__(self, inner):
        self.inner = inner
        self.mirror = MemoryStore()
        self.name = inner.name
        self.last_error = None

    @property
    def durable(self):
        return self.inner.durable and self.last_error is None

    def add(self, uid, usd):
        self.mirror.add(uid, usd)
        try:
            self.inner.add(uid, usd)
            self.last_error = None
        except Exception as e:                       # noqa: BLE001
            self.last_error = str(e)

    def month_usd(self, uid):
        try:
            v = self.inner.month_usd(uid)
            self.last_error = None
            # The remote is authoritative, but never report LESS than this
            # process has already seen -- otherwise an outage hands back budget.
            return max(v, self.mirror.month_usd(uid))
        except Exception as e:                       # noqa: BLE001
            self.last_error = str(e)
            return self.mirror.month_usd(uid)

    def hour_count(self, uid):
        try:
            v = self.inner.hour_count(uid)
            self.last_error = None
            return max(v, self.mirror.hour_count(uid))
        except Exception as e:                       # noqa: BLE001
            self.last_error = str(e)
            return self.mirror.hour_count(uid)


def make_store():
    url = os.environ.get("ROOKIE_USAGE_REDIS_URL", "").strip()
    token = os.environ.get("ROOKIE_USAGE_REDIS_TOKEN", "").strip()
    if url and token:
        print(f"[usage] durable spend store: {url}", flush=True)
        return SafeStore(RedisRestStore(url, token))
    print("[usage] WARNING: no ROOKIE_USAGE_REDIS_URL/TOKEN set -- spend counters "
          "are in-process and RESET ON RESTART. The monthly cap is not durable.",
          flush=True)
    return MemoryStore()
