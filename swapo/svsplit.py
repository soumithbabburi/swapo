"""Does separating long records pay on a standalone SV / CNV callset? Measured.

The interval-split idea was refuted on annotation and SNV tables because their
length distribution is continuous. The one composition left open was a
standalone structural-variant callset, where long records are a large share of
the table. This writes the real table twice per threshold, unsplit and split
(long records in their own files), at the same row-group size, and runs the
same real queries through DuckDB over HTTP. No model is involved in the verdict.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .objstore import StoreCost
from .rgbench import _truth, measure, write_files
from .rowgroups import _contigs, sort_genomic
from .threshold import separability

__all__ = ["split_test"]


def split_test(tbl: pa.Table, queries: list, store: StoreCost, *, rg_rows: int,
               file_rows: int = 4_194_304,
               percentiles: tuple[float, ...] = (90, 99, 99.9)) -> dict:
    ordered = sort_genomic(tbl)
    span = np.asarray(pc.subtract(ordered.column("end"), ordered.column("start"))).astype(np.int64)
    truth = _truth(_contigs(ordered), queries)
    base = Path(tempfile.mkdtemp())
    out = {"rows": tbl.num_rows, "rg_rows": rg_rows, "separability": separability(tbl),
           "store": store.summary(), "thresholds": []}
    try:
        files = write_files(ordered, base / "unsplit", file_rows=file_rows, rg_rows=rg_rows)
        m0 = measure(base / "unsplit", files, queries, truth=truth)
        out["unsplit"] = {"requests": m0.requests, "megabytes": m0.bytes / 1e6,
                          "ms": store.seconds(m0.requests, m0.bytes) * 1e3, "rows_ok": m0.rows_ok}
        for p in percentiles:
            thr = int(np.percentile(span, p))
            long_mask = pa.array(span > thr)
            root = base / f"split{p}"
            dense = write_files(ordered.filter(pc.invert(long_mask)), root / "dense",
                                file_rows=file_rows, rg_rows=rg_rows)
            longs = write_files(ordered.filter(long_mask), root / "long",
                                file_rows=file_rows, rg_rows=rg_rows)
            both = ([replace(f, path="dense/" + f.path) for f in dense]
                    + [replace(f, path="long/" + f.path) for f in longs])
            m = measure(root, both, queries, truth=truth)
            ms = store.seconds(m.requests, m.bytes) * 1e3
            out["thresholds"].append({
                "percentile": p, "threshold_bp": thr,
                "long_fraction": float((span > thr).mean()),
                "requests": m.requests, "megabytes": m.bytes / 1e6, "ms": ms,
                "saving": 1 - ms / out["unsplit"]["ms"], "rows_ok": m.rows_ok})
            shutil.rmtree(root, ignore_errors=True)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    return out
