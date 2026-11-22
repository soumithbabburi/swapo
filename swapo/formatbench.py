"""End to end on Delta and Iceberg: as written by default, advised, rewritten, measured.

1. write the real table with the format's own defaults, partitioned by contig;
2. read file statistics from the format's metadata, and the row-group size
   from the files' footers, as an operator's tool would find them;
3. check that planning from those statistics selects exactly the files the
   format's own planner selects, for every query;
4. recommend a row-group size and apply it with the format's own writer;
5. run every query through DuckDB over HTTP before and after, counting requests
   and bytes, and check every row count against an unpruned scan;
6. confirm the pre-rewrite version is still readable, which is the rollback.
"""

from __future__ import annotations

import shutil
import statistics
import tempfile
from pathlib import Path

import pyarrow as pa

from .objstore import StoreCost
from .rgbench import _truth, measure, plan
from .rowgroups import _contigs, choose_row_group_size, sort_genomic
from .tableformat import (delta_files, delta_plan, delta_rewrite, iceberg_files,
                          iceberg_plan, iceberg_rewrite, row_group_rows)

__all__ = ["run_format"]


def _flat(sets: dict[str, list]) -> list:
    return [q for v in sets.values() for q in v]


def run_format(fmt: str, tbl: pa.Table, sets: dict[str, list], store: StoreCost,
               *, workdir: str | None = None) -> dict:
    base = Path(workdir or tempfile.mkdtemp())
    queries = _flat(sets)
    truth = _truth(_contigs(sort_genomic(tbl)), queries)
    try:
        if fmt == "delta":
            from deltalake import DeltaTable, write_deltalake

            root = base / "delta"
            write_deltalake(str(root), sort_genomic(tbl), partition_by=["contig"])
            stats = lambda: delta_files(str(root))                       # noqa: E731
            native = lambda c, s, e: delta_plan(str(root), c, s, e)       # noqa: E731
            rewrite = lambda g: delta_rewrite(str(root), tbl, g)          # noqa: E731
            to_abs = lambda f: str(root / f.path)                         # noqa: E731
            rel = lambda f: f.path                                        # noqa: E731

            def old_version_rows() -> int:
                return DeltaTable(str(root), version=0).to_pyarrow_dataset().count_rows()
        elif fmt == "iceberg":
            from pyiceberg.catalog.sql import SqlCatalog

            wh = base / "wh"
            wh.mkdir(parents=True)
            cat = SqlCatalog("swapo", uri=f"sqlite:///{base / 'catalog.db'}", warehouse=str(wh))
            cat.create_namespace("g")
            from pyiceberg.partitioning import PartitionField, PartitionSpec
            from pyiceberg.transforms import IdentityTransform

            t0 = cat.create_table("g.t", schema=tbl.schema)
            with t0.update_spec() as u:
                u.add_identity("contig")
            t0.append(sort_genomic(tbl))
            table = cat.load_table("g.t")
            first_snapshot = table.current_snapshot().snapshot_id
            whn = str(wh).replace("\\", "/")
            stats = lambda: iceberg_files(cat.load_table("g.t"))          # noqa: E731
            native = lambda c, s, e: iceberg_plan(cat.load_table("g.t"), c, s, e)  # noqa: E731
            rewrite = lambda g: iceberg_rewrite(cat.load_table("g.t"), tbl, g)     # noqa: E731
            to_abs = lambda f: f.path                                     # noqa: E731
            rel = lambda f: f.path.removeprefix(whn).lstrip("/")          # noqa: E731
            root = wh

            def old_version_rows() -> int:
                t = cat.load_table("g.t")
                return t.scan(snapshot_id=first_snapshot).to_arrow().num_rows
        else:
            raise ValueError(fmt)

        def snapshot(label: str) -> dict:
            files = stats()
            agree = 0
            for c, s, e in queries:
                ours = {rel(f) for f in plan(files, c, s, e)}
                theirs = {x.removeprefix(str(root).replace("\\", "/")).lstrip("/")
                          for x in native(c, s, e)}
                agree += ours == theirs
            rg = row_group_rows([to_abs(f) for f in files])
            rel_files = [type(f)(rel(f), f.contig, f.start_min, f.end_max, f.n_rows, f.n_bytes)
                         for f in files]
            m = measure(Path(root), rel_files, queries, truth=truth, rg_rows=rg)
            return {"label": label, "files": len(files), "row_group_rows": rg,
                    "file_rows_median": int(statistics.median(f.n_rows for f in files)),
                    "planner_agreement": f"{agree}/{len(queries)}",
                    "requests": m.requests, "megabytes": m.bytes / 1e6,
                    "ms": store.seconds(m.requests, m.bytes) * 1e3,
                    "rows_correct": m.rows_ok}

        before = snapshot("as written by default")
        choice = choose_row_group_size(tbl, queries, store,
                                       current_rows=before["row_group_rows"],
                                       file_rows=before["file_rows_median"])
        rewrite(choice.rg_rows)
        after = snapshot("after the recommendation")
        return {
            "format": fmt, "store": store.summary(),
            "before": before, "after": after,
            "recommended": choice.rg_rows,
            "predicted_ms_before": choice.current.seconds * 1e3,
            "predicted_ms_after": choice.best.seconds * 1e3,
            "measured_saving": 1 - after["ms"] / before["ms"],
            "predicted_saving": choice.improvement,
            "rollback_rows": old_version_rows(), "rows": tbl.num_rows,
        }
    finally:
        if workdir is None:
            shutil.rmtree(base, ignore_errors=True)
