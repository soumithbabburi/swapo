from __future__ import annotations
from urllib.parse import urlparse

def _normalize_store_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == "s3":
        key = parsed.path.lstrip("/")
        return f"https://{parsed.netloc}.s3.amazonaws.com/{key}" if key else f"https://{parsed.netloc}.s3.amazonaws.com/"
    if parsed.scheme == "gs":
        key = parsed.path.lstrip("/")
        return f"https://storage.googleapis.com/{parsed.netloc}/{key}" if key else f"https://storage.googleapis.com/{parsed.netloc}/"
    return url

"""What a query costs on object storage, measured rather than assumed.

Bytes read is the wrong objective on S3 or GCS. Every GET pays a fixed
time-to-first-byte of tens of milliseconds before any data moves, so a reader
that issues twenty small requests can be slower than one that issues two large
ones, even though it reads fewer bytes. A layout chosen on bytes alone will
happily recommend thousands of 15 kB files, which is the small-files problem
every lakehouse operator already knows to avoid.

So the cost of a query is::

    seconds  =  requests * latency  +  bytes / bandwidth

This module supplies the two halves of that honestly:

* :func:`measure_remote` times real range GETs against a real bucket and fits
  ``latency`` and ``bandwidth`` from them;
* :class:`RangeServer` serves a directory over HTTP with Range support and logs
  every request, so a real engine (DuckDB ``httpfs``) can be pointed at a
  written table and the requests it *actually* issues can be counted. It can
  also delay each response by a measured latency, so wall-clock time is
  observed rather than modelled.
"""


import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

__all__ = ["StoreCost", "measure_remote", "RangeServer", "Request"]


@dataclass(frozen=True)
class StoreCost:
    """Fixed cost per request and transfer rate of one storage endpoint."""

    latency_s: float
    bandwidth_bps: float
    source: str = "assumed"

    def seconds(self, requests: float, n_bytes: float) -> float:
        return requests * self.latency_s + n_bytes / self.bandwidth_bps

    def summary(self) -> dict:
        return {"latency_ms": round(self.latency_s * 1e3, 1),
                "bandwidth_MBps": round(self.bandwidth_bps / 1e6, 1),
                "source": self.source}


def _orig_measure_remote(url: str, *, sizes: tuple[int, ...] = (1_024, 65_536, 1_048_576,
                                                           4_194_304, 16_777_216),
                   repeats: int = 7, seed: int = 3) -> tuple[StoreCost, list]:
    """Time range GETs of several sizes against ``url`` and fit the cost line.

    Uses one keep-alive connection per size, as a real reader does, and random
    offsets so nothing is served from a cache in front of the bucket. The first
    request of each run pays connection setup and is dropped.
    """
    import http.client
    from urllib.parse import urlsplit

    u = urlsplit(url)
    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head, timeout=30) as r:   # noqa: S310 - caller's URL
        total = int(r.headers["Content-Length"])
    rng = np.random.default_rng(seed)
    samples: list[tuple[int, float]] = []
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    for size in sizes:
        conn = conn_cls(u.netloc, timeout=60)
        for i in range(repeats + 1):
            off = int(rng.integers(0, max(1, total - size)))
            t0 = time.perf_counter()
            conn.request("GET", u.path, headers={"Range": f"bytes={off}-{off + size - 1}"})
            resp = conn.getresponse()
            got = len(resp.read())
            dt = time.perf_counter() - t0
            if i:                       # drop the connection-setup request
                samples.append((got, dt))
        conn.close()
    x = np.array([b for b, _ in samples], dtype=float)
    y = np.array([t for _, t in samples], dtype=float)
    # Robust to the odd slow request: fit the per-size MEDIANS, not every point.
    xs = np.array(sorted(set(x)))
    ys = np.array([np.median(y[x == v]) for v in xs])
    slope, intercept = np.polyfit(xs, ys, 1)
    cost = StoreCost(latency_s=float(max(intercept, 1e-4)),
                     bandwidth_bps=float(1.0 / max(slope, 1e-12)),
                     source=f"measured: {u.netloc}")
    return cost, samples


@dataclass(frozen=True)
class Request:
    path: str
    start: int
    length: int


class _Handler(BaseHTTPRequestHandler):
    server: "_Server"

    def log_message(self, *_a) -> None:   # silence stderr
        pass

    def _resolve(self) -> Path | None:
        rel = self.path.split("?", 1)[0].lstrip("/")
        p = (self.server.root / rel).resolve()
        if self.server.root not in p.parents and p != self.server.root:
            return None
        return p if p.is_file() else None

    def do_HEAD(self) -> None:
        p = self._resolve()
        if p is None:
            self.send_error(404)
            return
        # A HEAD is a round trip like any other, so it is logged and delayed.
        srv = self.server
        with srv.lock:
            srv.log.append(Request(str(p.relative_to(srv.root)), -1, 0))
        if srv.latency_s:
            time.sleep(srv.latency_s)
        self.send_response(200)
        self.send_header("Content-Length", str(p.stat().st_size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self) -> None:
        p = self._resolve()
        if p is None:
            self.send_error(404)
            return
        size = p.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            if a == "":                         # suffix range: last N bytes
                start = max(0, size - int(b))
            else:
                start = int(a)
                end = min(size - 1, int(b)) if b else size - 1
        length = max(0, end - start + 1)
        with p.open("rb") as fh:
            fh.seek(start)
            body = fh.read(length)
        srv = self.server
        with srv.lock:
            srv.log.append(Request(str(p.relative_to(srv.root)), start, length))
        if srv.latency_s or srv.bandwidth_bps:
            time.sleep(srv.latency_s + (length / srv.bandwidth_bps if srv.bandwidth_bps else 0))
        self.send_response(206 if rng else 200)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    root: Path
    log: list[Request]
    lock: threading.Lock
    latency_s: float
    bandwidth_bps: float


@dataclass
class RangeServer:
    """Serve ``root`` over HTTP with Range support and record every request.

    ``latency_s`` and ``bandwidth_bps`` delay each response, so an engine run
    against it sees the request costs of a real bucket. Use as a context
    manager; ``url`` is the base address while it runs.
    """

    root: str | os.PathLike
    latency_s: float = 0.0
    bandwidth_bps: float = 0.0
    _srv: _Server | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __enter__(self) -> "RangeServer":
        srv = _Server(("127.0.0.1", 0), _Handler)
        srv.root = Path(self.root).resolve()
        srv.log, srv.lock = [], threading.Lock()
        srv.latency_s, srv.bandwidth_bps = self.latency_s, self.bandwidth_bps
        self._srv = srv
        self._thread = threading.Thread(target=srv.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()

    @property
    def url(self) -> str:
        host, port = self._srv.server_address[:2]
        return f"http://{host}:{port}"

    def take_log(self) -> list[Request]:
        """Requests since the last call, and clear them."""
        with self._srv.lock:
            out, self._srv.log = self._srv.log, []
        return out

def measure_remote(url: str, *args, **kwargs):
    import time
    norm = _normalize_store_url(url)
    for i in range(3):
        try:
            return _orig_measure_remote(norm, *args, **kwargs)
        except Exception:
            if i == 2: raise
            time.sleep(0.5)
