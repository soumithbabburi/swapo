"""Physical layouts: how rows are assigned to blocks, and the statistics that result.

Four layouts are implemented so the benchmark can separate the contribution of
each idea rather than claiming credit for all of them at once:

``RANDOM``
    Rows in arrival order. Every block spans the whole genome, so no coordinate
    statistic can skip anything. This is what an unsorted append-only table looks
    like.

``SORTED``
    Globally sorted by ``(contig, start)``. This is the obvious thing to do and
    what ``ZORDER BY`` or a single-key ``CLUSTER BY`` approximates.

``PARTITIONED``
    Hive-style partitioning by contig, sorted by start inside each partition.
    Equivalent to what a well-tuned Delta table with a contig partition and
    clustering on start gives you.

``SWAPO``
    ``PARTITIONED`` plus the one change this project is really arguing for:
    records longer than a threshold are written to blocks of their own, so they
    cannot inflate ``max(end)`` for the dense majority. Everything else is
    identical, which is what makes the comparison honest: any difference between
    ``PARTITIONED`` and ``SWAPO`` is attributable to that single decision.

Every layout writes the same number of rows into blocks of the same target
size, and every layout gets an :class:`~swapo.zonemap.IntervalZoneMap` built the
same way. The baselines are not handicapped: they are pruned with the same
two-sided overlap test, using their own statistics.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .data import SV_THRESHOLD
from .model import Block
from .zonemap import IntervalZoneMap

__all__ = ["Layout", "LayoutResult", "write_layout", "DEFAULT_ROWS_PER_BLOCK"]

DEFAULT_ROWS_PER_BLOCK = 50_000


class Layout(str, Enum):
    RANDOM = "random"
    SORTED = "sorted"
    PARTITIONED = "partitioned"
    SWAPO = "swapo"


@dataclass
class LayoutResult:
    """A written layout and everything needed to query or cost it."""

    layout: Layout
    root: Path
    zonemap: IntervalZoneMap
    write_seconds: float
    n_bytes: int
    n_rows: int
    n_blocks: int
    #: Rows held in long-interval blocks (SWAPO only).
    long_rows: int = 0
    extras: dict = field(default_factory=dict)

    @property
    def paths(self) -> list[str]:
        return [b.path for b in self.zonemap.blocks]

    def summary(self) -> dict:
        return {
            "layout": self.layout.value,
            "blocks": self.n_blocks,
            "rows": self.n_rows,
            "megabytes": round(self.n_bytes / 1e6, 1),
            "write_seconds": round(self.write_seconds, 2),
            "zonemap_tightness": round(self.zonemap.tightness(), 1),
            "long_rows": self.long_rows,
        }


def _block_stats(path: Path, tbl: pa.Table, long_tail: bool) -> Block:
    """Derive the statistics a reader would get from the file's own metadata."""
    contigs = pc.unique(tbl.column("contig")).to_pylist()
    starts = tbl.column("start")
    ends = tbl.column("end")
    return Block(
        path=str(path),
        # A block spanning several contigs cannot be pruned by contig, which is
        # exactly the weakness of the RANDOM layout. Mark it with "*".
        contig=contigs[0] if len(contigs) == 1 else "*",
        start_min=int(pc.min(starts).as_py()),
        start_max=int(pc.max(starts).as_py()),
        end_max=int(pc.max(ends).as_py()),
        n_rows=tbl.num_rows,
        n_bytes=path.stat().st_size,
        long_tail=long_tail,
    )


def _write_blocks(tbl: pa.Table, root: Path, prefix: str, rows_per_block: int,
                  long_tail: bool, zm: IntervalZoneMap, counter: list[int],
                  compression: str, consolidate: bool = False) -> int:
    """Write ``tbl`` as consecutive blocks (or consolidated multi-row-group file); return bytes written."""
    if tbl.num_rows == 0:
        return 0
    if consolidate:
        path = root / f"{prefix}.parquet"
        counter[0] += 1
        pq.write_table(tbl, path, compression=compression,
                       row_group_size=max(rows_per_block, 1), use_dictionary=True)
        fsize = path.stat().st_size
        n_chunks = max(1, (tbl.num_rows + rows_per_block - 1) // rows_per_block)
        for offset in range(0, tbl.num_rows, rows_per_block):
            chunk = tbl.slice(offset, rows_per_block)
            if chunk.num_rows == 0:
                continue
            blk = _block_stats(path, chunk, long_tail)
            blk.n_bytes = max(1, fsize // n_chunks)
            zm.add(blk)
        return fsize
    total = 0
    for offset in range(0, tbl.num_rows, rows_per_block):
        chunk = tbl.slice(offset, rows_per_block)
        if chunk.num_rows == 0:
            continue
        path = root / f"{prefix}-{counter[0]:05d}.parquet"
        counter[0] += 1
        # One row group per file keeps the block boundary and the file boundary
        # the same thing, so a pruned file is genuinely never opened.
        pq.write_table(chunk, path, compression=compression,
                       row_group_size=max(chunk.num_rows, 1), use_dictionary=True)
        zm.add(_block_stats(path, chunk, long_tail))
        total += path.stat().st_size
    return total


def _sort(tbl: pa.Table, keys: list[tuple[str, str]]) -> pa.Table:
    return tbl.take(pc.sort_indices(tbl, sort_keys=keys))


def write_layout(
    tbl: pa.Table,
    root: str | Path,
    layout: Layout,
    *,
    rows_per_block: int = DEFAULT_ROWS_PER_BLOCK,
    long_threshold: int = SV_THRESHOLD,
    compression: str = "zstd",
    seed: int = 11,
    consolidate: bool = False,
) -> LayoutResult:
    """Write ``tbl`` in ``layout`` under ``root`` and return its zone map.

    ``long_threshold`` is the only knob that distinguishes ``SWAPO`` from
    ``PARTITIONED``. Records whose span exceeds it go to their own blocks. A
    threshold of zero or less means no split is worth making, and the result is
    identical to ``PARTITIONED``; :func:`swapo.threshold.choose_threshold`
    returns zero exactly when that is the case.
    """
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    zm = IntervalZoneMap()
    counter = [0]
    t0 = time.perf_counter()
    n_bytes = 0
    long_rows = 0

    if layout is Layout.RANDOM:
        rng = np.random.default_rng(seed)
        order = rng.permutation(tbl.num_rows)
        shuffled = tbl.take(pa.array(order))
        n_bytes = _write_blocks(shuffled, root, "blk", rows_per_block, False,
                                zm, counter, compression, consolidate)

    elif layout is Layout.SORTED:
        ordered = _sort(tbl, [("contig", "ascending"), ("start", "ascending")])
        n_bytes = _write_blocks(ordered, root, "blk", rows_per_block, False,
                                zm, counter, compression, consolidate)

    elif layout is Layout.PARTITIONED:
        for contig in pc.unique(tbl.column("contig")).to_pylist():
            part = tbl.filter(pc.equal(tbl.column("contig"), contig))
            part = _sort(part, [("start", "ascending")])
            safe = str(contig).replace("*", "_")
            n_bytes += _write_blocks(part, root, f"{safe}", rows_per_block, False,
                                     zm, counter, compression, consolidate)

    elif layout is Layout.SWAPO:
        if long_threshold <= 0:
            # A threshold of zero means the search found no split worth making.
            # That must mean "leave it alone", not "treat every row as long":
            # every span exceeds zero, so the naive reading puts the whole table
            # in the long partition and silently undoes the layout.
            return write_layout(tbl, root, Layout.PARTITIONED,
                                rows_per_block=rows_per_block,
                                compression=compression, seed=seed, consolidate=consolidate)
        span = pc.subtract(tbl.column("end"), tbl.column("start"))
        is_long = pc.greater(span, long_threshold)
        for contig in pc.unique(tbl.column("contig")).to_pylist():
            on_contig = pc.equal(tbl.column("contig"), contig)
            safe = str(contig).replace("*", "_")
            dense = tbl.filter(pc.and_(on_contig, pc.invert(is_long)))
            if dense.num_rows:
                dense = _sort(dense, [("start", "ascending")])
                n_bytes += _write_blocks(dense, root, f"{safe}", rows_per_block,
                                         False, zm, counter, compression, consolidate)
            longs = tbl.filter(pc.and_(on_contig, is_long))
            if longs.num_rows:
                long_rows += longs.num_rows
                # Sorting the long records by start as well keeps their own
                # blocks as tight as their spans allow.
                longs = _sort(longs, [("start", "ascending")])
                n_bytes += _write_blocks(longs, root, f"{safe}-long", rows_per_block,
                                         True, zm, counter, compression, consolidate)
    else:
        raise ValueError(f"unknown layout {layout}")

    zm.finalise()
    elapsed = time.perf_counter() - t0

    result = LayoutResult(
        layout=layout, root=root, zonemap=zm, write_seconds=elapsed,
        n_bytes=n_bytes, n_rows=tbl.num_rows, n_blocks=len(zm), long_rows=long_rows,
    )
    (root / "_swapo_zonemap.json").write_text(
        json.dumps({"layout": layout.value, "blocks": zm.to_json()}), encoding="utf-8"
    )
    return result


def load_layout(root: str | Path) -> IntervalZoneMap:
    """Rebuild a zone map written next to the data by :func:`write_layout`."""
    root = Path(root)
    meta = json.loads((root / "_swapo_zonemap.json").read_text(encoding="utf-8"))
    return IntervalZoneMap.from_json(meta["blocks"])
