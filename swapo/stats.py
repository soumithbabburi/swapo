"""Table statistics: the distribution facts the optimizer reasons about.

These are cheap, single-pass summaries. The one that matters most is
``long_fraction``: the share of records whose span exceeds the threshold at
which they start ruining ``max(end)`` for everything stored beside them.
"""

from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc

from .data import SV_THRESHOLD
from .model import TableStats

__all__ = ["summarise", "length_histogram"]


def summarise(tbl: pa.Table, *, n_bytes: int = 0, n_blocks: int = 0,
              long_threshold: int = SV_THRESHOLD) -> TableStats:
    """One pass over the interval columns."""
    span = pc.subtract(tbl.column("end"), tbl.column("start"))
    n = tbl.num_rows
    long_n = pc.sum(pc.cast(pc.greater(span, long_threshold), pa.int64())).as_py() or 0

    rows_per_contig: dict[str, int] = {}
    if "contig" in tbl.column_names:
        grouped = (pa.table({"contig": tbl.column("contig")})
                   .group_by("contig").aggregate([([], "count_all")]))
        for contig, count in zip(grouped.column("contig").to_pylist(),
                                 grouped.column("count_all").to_pylist()):
            rows_per_contig[str(contig)] = int(count)

    return TableStats(
        n_rows=n,
        n_bytes=n_bytes,
        n_blocks=n_blocks,
        long_fraction=(long_n / n) if n else 0.0,
        max_length=int(pc.max(span).as_py() or 0),
        rows_per_contig=rows_per_contig,
    )


def length_histogram(tbl: pa.Table, edges: list[int] | None = None) -> dict[str, int]:
    """Counts by interval length, for a quick look at the tail."""
    edges = edges or [1, 2, 10, 50, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    span = pc.subtract(tbl.column("end"), tbl.column("start"))
    out: dict[str, int] = {}
    for lo, hi in zip(edges, edges[1:] + [None]):
        if hi is None:
            mask = pc.greater_equal(span, lo)
            label = f">={lo:,}"
        else:
            mask = pc.and_(pc.greater_equal(span, lo), pc.less(span, hi))
            label = f"{lo:,}-{hi - 1:,}"
        out[label] = int(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0)
    return out
