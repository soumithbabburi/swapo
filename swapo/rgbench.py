"""Check the row-group model against a real engine's real requests.

For each candidate size the real table is written (partitioned by contig, files
of ``file_rows``, row groups of the candidate size), files are planned from
their per-file statistics exactly as a table-format manifest would, and DuckDB
reads them over HTTP from :class:`~swapo.objstore.RangeServer`. Every request
it issues is logged, so requests and bytes are counted, not estimated. Every
query's row count is checked against an unpruned scan of the source table.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .objstore import RangeServer, StoreCost
from .rowgroups import (CANDIDATES, Reader, _contigs, fit_row_group_cost, predict,
                        sort_genomic)

__all__ = ["write_files", "measure", "validate"]


@dataclass(frozen=True)
class FileStat:
    path: str
    contig: str
    start_min: int
    end_max: int
    n_rows: int
    n_bytes: int


def write_files(ordered: pa.Table, root: Path, *, file_rows: int, rg_rows: int,
                compression: str = "zstd") -> list[FileStat]:
    """Contig-partitioned files with row groups of ``rg_rows``; return per-file stats."""
    root.mkdir(parents=True, exist_ok=True)
    out: list[FileStat] = []
    names = ordered.column("contig").to_numpy(zero_copy_only=False)
    change = np.flatnonzero(names[1:] != names[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(names)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        contig = str(names[a])
        part = root / f"contig={contig}"
        part.mkdir(exist_ok=True)
        for k, off in enumerate(range(a, b, file_rows)):
            chunk = ordered.slice(off, min(file_rows, b - off))
            p = part / f"part-{k:05d}.parquet"
            pq.write_table(chunk, p, row_group_size=rg_rows, compression=compression)
            out.append(FileStat(str(p.relative_to(root)).replace("\\", "/"), contig,
                                int(pc.min(chunk.column("start")).as_py()),
                                int(pc.max(chunk.column("end")).as_py()),
                                chunk.num_rows, p.stat().st_size))
    return out


def plan(files: list[FileStat], contig: str, start: int, end: int) -> list[FileStat]:
    """The manifest test: a file is read only if its bounds can overlap."""
    return [f for f in files
            if f.contig == contig and f.start_min < end and start < f.end_max]


@dataclass
class Measured:
    rg_rows: int
    requests: float
    bytes: float
    wall_ms: float
    rows_ok: bool
    per_query: list[tuple[int, int]]


def _truth(by_contig, queries) -> list[int]:
    out = []
    for c, s, e in queries:
        if c not in by_contig:
            out.append(0)
            continue
        st, en = by_contig[c]
        out.append(int(((st < e) & (s < en)).sum()))
    return out


def measure(root: Path, files: list[FileStat], queries: list[tuple[str, int, int]],
            *, select: str = "*", latency_s: float = 0.0, bandwidth_bps: float = 0.0,
            truth: list[int] | None = None, rg_rows: int = 0) -> Measured:
    """Run every query through DuckDB over HTTP and count what it fetched."""
    con = duckdb.connect()
    con.execute("LOAD httpfs")
    con.execute("SET enable_http_metadata_cache=false")
    con.execute("SET enable_object_cache=false")
    con.execute("SET enable_progress_bar=false")
    reqs, byts, per, ok = [], [], [], True
    t_all = 0.0
    with RangeServer(root, latency_s=latency_s, bandwidth_bps=bandwidth_bps) as srv:
        for i, (c, s, e) in enumerate(queries):
            hit = plan(files, c, s, e)
            srv.take_log()
            n = 0
            if hit:
                lst = ", ".join(f"'{srv.url}/{f.path}'" for f in hit)
                sql = (f"SELECT {select} FROM read_parquet([{lst}]) "
                       f"WHERE contig = '{c}' AND start < {e} AND \"end\" > {s}")
                # A transient local socket failure (Windows runs out of
                # ephemeral ports under tens of thousands of short requests)
                # must not count partial requests: drop that attempt's log and
                # rerun the query on a fresh connection.
                for attempt in range(4):
                    try:
                        t0 = time.perf_counter()
                        rows = con.execute(sql).fetchall()
                        t_all += time.perf_counter() - t0
                        break
                    except duckdb.Error:
                        if attempt == 3:
                            raise
                        time.sleep(2.0 * (attempt + 1))
                        srv.take_log()
                        con.close()
                        con = duckdb.connect()
                        for st in ("LOAD httpfs", "SET enable_http_metadata_cache=false",
                                   "SET enable_object_cache=false",
                                   "SET enable_progress_bar=false"):
                            con.execute(st)
                n = len(rows)
            log = srv.take_log()
            r, b = len(log), sum(x.length for x in log)
            reqs.append(r)
            byts.append(b)
            per.append((r, b))
            if truth is not None and n != truth[i]:
                ok = False
    con.close()
    q = max(len(queries), 1)
    return Measured(rg_rows, float(np.mean(reqs)), float(np.mean(byts)),
                    t_all * 1e3 / q, ok, per)


def validate(tbl: pa.Table, query_sets: dict[str, list[tuple[str, int, int]]],
             store: StoreCost | None = None, *, candidates: tuple[int, ...] = CANDIDATES,
             file_rows: int = 4_194_304, current_rows: int = 1_048_576,
             reader: Reader = Reader(), select: str = "*") -> dict:
    """Predicted vs measured requests, bytes and seconds, per query set and size.

    Each candidate layout is written once and every query set is run against
    it. A ``mixed`` set (all sets pooled) is scored from the same measurements.
    """
    ordered = sort_genomic(tbl)
    sizes = tuple(sorted({int(c) for c in candidates} | {int(current_rows)}, reverse=True))
    cost = fit_row_group_cost(ordered, sizes)
    by_contig = _contigs(ordered)
    sets = {k: v for k, v in query_sets.items() if v}
    sets["mixed"] = [q for v in sets.values() for q in v]
    truth = {k: _truth(by_contig, v) for k, v in sets.items() if k != "mixed"}
    rows: dict[str, list[dict]] = {k: [] for k in sets}
    base = Path(tempfile.mkdtemp())
    try:
        for r in sizes:
            d = base / f"rg{r}"
            files = write_files(ordered, d, file_rows=file_rows, rg_rows=r)
            meas = {k: measure(d, files, v, truth=truth[k], rg_rows=r, select=select)
                    for k, v in sets.items() if k != "mixed"}
            per = [x for k in meas for x in meas[k].per_query]
            meas["mixed"] = Measured(r, float(np.mean([a for a, _ in per])),
                                     float(np.mean([b for _, b in per])), 0.0,
                                     all(m.rows_ok for m in meas.values()), per)
            for k, v in sets.items():
                pred = predict(by_contig, v, r, cost, store or StoreCost(0.03, 5e7), file_rows=file_rows,
                               reader=reader)
                m = meas[k]
                rows[k].append({
                    "rg_rows": r,
                    "pred_requests": float(pred.requests), "meas_requests": m.requests,
                    "pred_mb": float(pred.bytes) / 1e6, "meas_mb": m.bytes / 1e6,
                    "rows_ok": m.rows_ok, "files_written": len(files)})
            shutil.rmtree(d, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return {"n_rows": tbl.num_rows, "current_rows": current_rows,
            "queries": {k: len(v) for k, v in sets.items()}, "sets": rows}


def score(rows: list[dict], store: StoreCost, current_rows: int = 1_048_576) -> dict:
    """Choose on predicted seconds, judge on measured seconds, for one store.

    Requests and bytes do not depend on the store, so one measured run can be
    scored exactly at any latency and bandwidth.
    """
    rows = [dict(x, pred_ms=store.seconds(x["pred_requests"], x["pred_mb"] * 1e6) * 1e3,
                 meas_ms=store.seconds(x["meas_requests"], x["meas_mb"] * 1e6) * 1e3)
            for x in rows]
    chosen = min(rows, key=lambda x: x["pred_ms"])
    oracle = min(rows, key=lambda x: x["meas_ms"])
    cur = next(x for x in rows if x["rg_rows"] == current_rows)
    errs = [abs(x["pred_ms"] / x["meas_ms"] - 1) for x in rows if x["meas_ms"]]
    return {
        "store": store.summary(),
        "rows": rows,
        "chosen": chosen["rg_rows"],
        "measured_best": oracle["rg_rows"],
        "loss_vs_best": chosen["meas_ms"] / oracle["meas_ms"] - 1,
        "saving_vs_default": 1 - chosen["meas_ms"] / cur["meas_ms"],
        "mean_abs_err": float(np.mean(errs)) if errs else 0.0,
        "worst_err": float(np.max(errs)) if errs else 0.0,
        "all_rows_correct": all(x["rows_ok"] for x in rows),
    }
