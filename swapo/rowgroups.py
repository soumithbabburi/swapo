"""Choosing the Parquet row-group size by what a region query costs on object storage.

This replaces the file-per-block model in :mod:`swapo.blocksize`, which had two
faults. It wrote every block as its own file, so its recommended 250-row blocks
meant 15 kB files, four thousand per million rows. And it counted bytes only,
when on S3 or GCS a request costs tens of milliseconds before any byte moves.

What a reader actually does, read off the request log of DuckDB 1.x over HTTP
(:class:`swapo.objstore.RangeServer`):

* per file it opens: one HEAD for the size, one 16 kB read of the tail, and a
  second read of the footer if the footer does not fit in that tail;
* per row group whose statistics overlap the query: the needed column chunks,
  coalesced into one range, fetched in pieces of at most 2 MiB.

So row-group size trades three things. Large row groups make a query read
megabytes to return kilobytes. Small row groups grow the footer, which every
query reads whole, by a few hundred bytes per row group. And past 2 MiB a row
group costs extra requests. The minimum of

    seconds = requests * latency + bytes / bandwidth

is what :func:`choose_row_group_size` finds, from the sorted coordinates and a
few sample writes, without rewriting the table. Files are left at whatever size
the table format targets: file size is governed by manifests and compaction,
and moves a query's cost by one footer per file touched.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .objstore import StoreCost

__all__ = ["Reader", "RowGroupCost", "RowGroupChoice", "WRITER_DEFAULTS",
           "fit_row_group_cost", "predict", "choose_row_group_size", "sort_genomic"]

#: Row-group sizes that real writers use when nobody sets one.
WRITER_DEFAULTS = {
    1_048_576: "pyarrow, pyiceberg and delta-rs default (1,048,576 rows)",
    122_880: "DuckDB COPY default (122,880 rows)",
}


@dataclass(frozen=True)
class Reader:
    """How a Parquet reader turns a plan into HTTP requests.

    Defaults are DuckDB 1.x, read off its request log. They are checked against
    the log in :func:`swapo.rgbench.validate`, so a reader that behaves
    differently shows up as a prediction error rather than a silent bias.
    """

    tail_bytes: int = 16_384
    max_request: int = 2 * 1024 * 1024
    head_per_file: bool = True


@dataclass
class RowGroupCost:
    """Measured on-disk cost of one row group, per candidate size."""

    #: Bytes a reader fetches per ROW of a row group of this size: the span
    #: covering the selected column chunks, which is what a coalescing reader
    #: requests, divided by the rows in it. Per row, because a row group is
    #: often smaller than the setting: a contig, or a file, can run out first.
    span_per_row: dict[int, float] = field(default_factory=dict)
    #: Footer bytes added per row group, and the footer's fixed part.
    footer_per_rg: dict[int, float] = field(default_factory=dict)
    footer_base: float = 0.0


def sort_genomic(tbl: pa.Table) -> pa.Table:
    return tbl.take(pc.sort_indices(tbl, sort_keys=[("contig", "ascending"),
                                                     ("start", "ascending")]))


def _span(md: pq.FileMetaData, rg: int, columns: set[str] | None) -> int:
    lo, hi = None, 0
    g = md.row_group(rg)
    for c in range(g.num_columns):
        col = g.column(c)
        if columns is not None and col.path_in_schema not in columns:
            continue
        start = col.dictionary_page_offset if col.has_dictionary_page else col.data_page_offset
        end = start + col.total_compressed_size
        lo = start if lo is None else min(lo, start)
        hi = max(hi, end)
    return 0 if lo is None else hi - lo


def fit_row_group_cost(ordered: pa.Table, sizes: tuple[int, ...], *,
                       columns: set[str] | None = None, anchors: int = 3,
                       groups_per_sample: int = 4, compression: str = "zstd",
                       seed: int = 31) -> RowGroupCost:
    """Write a few sample files per size and read their metadata.

    Every size is sampled at the SAME anchor offsets (paired sampling): the
    search compares sizes, and independent samples per size would make those
    differences carry the sum of two sampling errors.
    """
    rng = np.random.default_rng(seed)
    n = ordered.num_rows
    sizes = tuple(sorted({int(s) for s in sizes if s > 0}))
    biggest = min(n, max(sizes) * groups_per_sample)
    offs = [int(rng.integers(0, max(1, n - biggest))) for _ in range(anchors)]
    tmp = Path(tempfile.mkdtemp())
    cost = RowGroupCost()
    bases: list[float] = []
    try:
        for size in sizes:
            spans, per = [], []
            for i, off in enumerate(offs):
                rows = min(size * groups_per_sample, n - off)
                chunk = ordered.slice(off, rows)
                p = tmp / f"s{size}_{i}.parquet"
                pq.write_table(chunk, p, row_group_size=size, compression=compression)
                md = pq.ParquetFile(p).metadata
                full = [r for r in range(md.num_row_groups)
                        if md.row_group(r).num_rows == size] or [0]
                spans.extend(_span(md, r, columns) / md.row_group(r).num_rows for r in full)
                # One-row-group file at the same anchor gives the fixed part.
                p1 = tmp / f"b{size}_{i}.parquet"
                pq.write_table(ordered.slice(off, min(size, rows)), p1,
                               row_group_size=size, compression=compression)
                base1 = pq.ParquetFile(p1).metadata.serialized_size
                k = md.num_row_groups
                if k > 1:
                    per.append((md.serialized_size - base1) / (k - 1))
                bases.append(base1)
            cost.span_per_row[size] = float(np.mean(spans))
            cost.footer_per_rg[size] = float(np.mean(per)) if per else 0.0
        if bases:
            g = np.median([cost.footer_per_rg[s] for s in sizes]) if sizes else 0.0
            cost.footer_base = float(max(np.median(bases) - g, 0.0))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return cost


def _contigs(ordered: pa.Table) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    contig = ordered.column("contig")
    starts = np.asarray(ordered.column("start")).astype(np.int64)
    ends = np.asarray(ordered.column("end")).astype(np.int64)
    names = contig.to_numpy(zero_copy_only=False)
    change = np.flatnonzero(names[1:] != names[:-1]) + 1
    bounds = np.concatenate([[0], change, [len(names)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            out[str(names[a])] = (starts[a:b], ends[a:b])
    return out


def _groups(starts: np.ndarray, ends: np.ndarray, file_rows: int, rg_rows: int):
    """File and row-group bounds for one contig, row groups restarting per file."""
    n = starts.size
    f_lo = np.arange(0, n, file_rows)
    rg_lo = np.concatenate([np.arange(a, min(a + file_rows, n), rg_rows) for a in f_lo])
    rg_file = np.searchsorted(f_lo, rg_lo, side="right") - 1
    rg_smin = starts[rg_lo]
    rg_emax = np.maximum.reduceat(ends, rg_lo)
    f_smin = starts[f_lo]
    f_emax = np.maximum.reduceat(ends, f_lo)
    f_ngroups = np.bincount(rg_file, minlength=f_lo.size)
    f_hi = np.minimum(f_lo + file_rows, n)
    rg_n = np.minimum(rg_lo + rg_rows, f_hi[rg_file]) - rg_lo
    return f_smin, f_emax, f_ngroups, rg_smin, rg_emax, rg_n


@dataclass
class Prediction:
    rg_rows: int
    requests: float
    bytes: float
    seconds: float
    files: float
    row_groups: float


def predict(by_contig: dict, queries: list[tuple[str, int, int]], rg_rows: int,
            cost: RowGroupCost, store: StoreCost, *, file_rows: int,
            reader: Reader = Reader()) -> Prediction:
    """Mean requests, bytes and seconds per query for row groups of ``rg_rows``."""
    per_row = cost.span_per_row[rg_rows]
    tot_req = tot_bytes = tot_files = tot_rgs = 0.0
    by_q: dict[str, list[tuple[int, int]]] = {}
    for c, s, e in queries:
        by_q.setdefault(c, []).append((s, e))
    for c, qs in by_q.items():
        if c not in by_contig:
            continue
        starts, ends = by_contig[c]
        f_smin, f_emax, f_ng, rg_smin, rg_emax, rg_n = _groups(starts, ends, file_rows, rg_rows)
        rg_bytes = rg_n * per_row
        rg_req = np.maximum(1, np.ceil(rg_bytes / reader.max_request))
        footer = cost.footer_base + cost.footer_per_rg[rg_rows] * f_ng
        big = footer + 8 > reader.tail_bytes
        f_req = 1 + big + (1 if reader.head_per_file else 0)
        f_bytes = reader.tail_bytes + np.where(big, footer, 0.0)
        q = np.array(qs, dtype=np.int64)
        fh = (f_smin[None, :] < q[:, 1:2]) & (q[:, 0:1] < f_emax[None, :])
        gh = (rg_smin[None, :] < q[:, 1:2]) & (q[:, 0:1] < rg_emax[None, :])
        tot_files += fh.sum()
        tot_rgs += gh.sum()
        tot_req += (fh * f_req[None, :]).sum() + (gh * rg_req[None, :]).sum()
        tot_bytes += (fh * f_bytes[None, :]).sum() + (gh * rg_bytes[None, :]).sum()
    n = max(len(queries), 1)
    req, by = tot_req / n, tot_bytes / n
    return Prediction(rg_rows, req, by, store.seconds(req, by), tot_files / n, tot_rgs / n)


@dataclass
class RowGroupChoice:
    rg_rows: int
    current_rows: int
    store: StoreCost
    curve: list[Prediction]
    file_rows: int

    def _at(self, r: int) -> Prediction:
        return next(p for p in self.curve if p.rg_rows == r)

    @property
    def best(self) -> Prediction:
        return self._at(self.rg_rows)

    @property
    def current(self) -> Prediction:
        return self._at(self.current_rows)

    @property
    def improvement(self) -> float:
        cur = self.current.seconds
        return 1.0 - self.best.seconds / cur if cur else 0.0

    def explain(self) -> str:
        b, c = self.best, self.current
        return (
            "row-group size on this table\n"
            f"  current        : {c.rg_rows:,} rows per row group, "
            f"{c.requests:.1f} requests and {c.bytes / 1e6:.2f} MB per query, "
            f"{c.seconds * 1e3:.0f} ms\n"
            f"  recommended    : {b.rg_rows:,} rows per row group, "
            f"{b.requests:.1f} requests and {b.bytes / 1e6:.2f} MB per query, "
            f"{b.seconds * 1e3:.0f} ms\n"
            f"  expected gain  : {self.improvement * 100:.0f}% less time per query "
            f"({c.seconds / max(b.seconds, 1e-9):.1f}x)\n"
            f"  storage cost   : {self.store.latency_s * 1e3:.0f} ms per request, "
            f"{self.store.bandwidth_bps / 1e6:.0f} MB/s ({self.store.source})\n"
            f"  files          : unchanged, up to {self.file_rows:,} rows each\n"
            f"  freshness      : none, the same rows in different row groups\n"
            f"  rollback       : rewrite at {c.rg_rows:,} rows per row group"
        )

    def summary(self) -> dict:
        return {
            "recommend_rows_per_row_group": self.rg_rows,
            "current_rows_per_row_group": self.current_rows,
            "expected_time_reduction": round(self.improvement, 3),
            "store": self.store.summary(),
            "curve": [{"rows": p.rg_rows, "requests": round(p.requests, 2),
                       "megabytes": round(p.bytes / 1e6, 4),
                       "ms": round(p.seconds * 1e3, 2)} for p in self.curve],
        }


CANDIDATES = (1_048_576, 524_288, 262_144, 122_880, 65_536, 32_768, 16_384,
              8_192, 4_096, 2_048)


def choose_row_group_size(
    tbl: pa.Table,
    queries: list[tuple[str, int, int]],
    store: StoreCost,
    *,
    current_rows: int = 1_048_576,
    file_rows: int = 4_194_304,
    columns: set[str] | None = None,
    candidates: tuple[int, ...] = CANDIDATES,
    reader: Reader = Reader(),
    cost: RowGroupCost | None = None,
) -> RowGroupChoice:
    """Row-group size that minimises mean seconds per query on ``store``.

    ``queries`` are ``(contig, start, end)`` from the real workload; positions
    and widths both matter, so a single typical width is not accepted.
    ``columns`` is what the queries select; ``None`` means every column.
    """
    if not queries:
        raise ValueError("need the workload's real query intervals")
    ordered = sort_genomic(tbl)
    sizes = tuple(sorted({int(c) for c in candidates} | {int(current_rows)}, reverse=True))
    cost = cost or fit_row_group_cost(ordered, sizes, columns=columns)
    by_contig = _contigs(ordered)
    curve = [predict(by_contig, queries, r, cost, store, file_rows=file_rows, reader=reader)
             for r in sizes]
    best = min(curve, key=lambda p: p.seconds)
    return RowGroupChoice(best.rg_rows, int(current_rows), store, curve, file_rows)
