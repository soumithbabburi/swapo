"""Delta Lake and Apache Iceberg adapters.

The zone map was always shaped like a manifest statistic. Here it is read from
the real thing: Delta's per-file ``min``/``max`` in the transaction log, and
Iceberg's ``lower_bounds``/``upper_bounds`` in the manifests. Nothing is
inferred from file names or recomputed from data.

Applying a recommendation uses each format's own writer with its own
row-group setting (``WriterProperties(max_row_group_size=...)`` for delta-rs,
``write.parquet.row-group-limit`` for Iceberg), as a normal overwrite commit,
so it is versioned and the previous snapshot stays readable for rollback.

File pruning is checked against each format's own planner rather than ours:
``DeltaTable.to_pyarrow_dataset().get_fragments(filter)`` and Iceberg's
``scan(row_filter).plan_files()``. If they ever disagree, the adapter is wrong.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .rgbench import FileStat
from .rowgroups import sort_genomic

__all__ = ["delta_files", "delta_plan", "delta_rewrite", "iceberg_files",
           "iceberg_plan", "iceberg_rewrite", "row_group_rows"]


def _norm(p: str) -> str:
    p = p.replace("\\", "/")
    for pre in ("file:///", "file://", "file:"):
        if p.startswith(pre):
            p = p[len(pre):]
    return p


# -- Delta -------------------------------------------------------------------


def delta_files(uri: str) -> list[FileStat]:
    """Per-file stats straight from the Delta log (add actions)."""
    from deltalake import DeltaTable

    dt = DeltaTable(uri)
    aa = dt.get_add_actions(flatten=True)
    aa = aa if isinstance(aa, pa.Table) else pa.table(aa)
    cols = aa.column_names
    contig = (aa.column("partition.contig") if "partition.contig" in cols
              else aa.column("min.contig")).to_pylist()
    return [FileStat(p.replace("\\", "/"), c, int(s), int(e), int(n), int(b))
            for p, c, s, e, n, b in zip(aa.column("path").to_pylist(), contig,
                                        aa.column("min.start").to_pylist(),
                                        aa.column("max.end").to_pylist(),
                                        aa.column("num_records").to_pylist(),
                                        aa.column("size_bytes").to_pylist())]


def delta_plan(uri: str, contig: str, start: int, end: int) -> set[str]:
    """Files delta-rs itself would read for an overlap query."""
    from deltalake import DeltaTable

    root = _norm(str(Path(uri).resolve()))
    flt = ((pc.field("contig") == contig) & (pc.field("start") < end)
           & (pc.field("end") > start))
    frags = DeltaTable(uri).to_pyarrow_dataset().get_fragments(filter=flt)
    return {_norm(f.path).removeprefix(root).lstrip("/") for f in frags}


def delta_rewrite(uri: str, tbl: pa.Table, rg_rows: int, *,
                  target_file_size: int | None = None) -> None:
    """Overwrite with sorted rows at ``rg_rows`` rows per row group (one commit)."""
    from deltalake import WriterProperties, write_deltalake

    write_deltalake(uri, sort_genomic(tbl), mode="overwrite", partition_by=["contig"],
                    target_file_size=target_file_size,
                    writer_properties=WriterProperties(max_row_group_size=rg_rows,
                                                       compression="ZSTD"))


# -- Iceberg -----------------------------------------------------------------


def iceberg_files(table) -> list[FileStat]:
    """Per-file stats from the Iceberg manifests (lower/upper bounds)."""
    fs = table.inspect.files()
    rm = fs.column("readable_metrics").to_pylist()
    part = fs.column("partition").to_pylist()
    out = []
    for path, n, b, m, pt in zip(fs.column("file_path").to_pylist(),
                                 fs.column("record_count").to_pylist(),
                                 fs.column("file_size_in_bytes").to_pylist(), rm, part):
        contig = (pt or {}).get("contig") or m["contig"]["lower_bound"]
        out.append(FileStat(_norm(path), contig, int(m["start"]["lower_bound"]),
                            int(m["end"]["upper_bound"]), int(n), int(b)))
    return out


def iceberg_plan(table, contig: str, start: int, end: int) -> set[str]:
    """Files Iceberg's own planner would read for an overlap query."""
    from pyiceberg.expressions import And, EqualTo, GreaterThan, LessThan

    flt = And(EqualTo("contig", contig), And(LessThan("start", end), GreaterThan("end", start)))
    return {_norm(t.file.file_path) for t in table.scan(row_filter=flt).plan_files()}


def iceberg_rewrite(table, tbl: pa.Table, rg_rows: int) -> None:
    """Set the row-group limit and overwrite with sorted rows (one snapshot)."""
    with table.transaction() as tx:
        tx.set_properties({"write.parquet.row-group-limit": str(rg_rows)})
    table.overwrite(sort_genomic(tbl))


# -- reading the current setting ----------------------------------------------


def row_group_rows(paths: list[str], sample: int = 8) -> int:
    """The row-group size a table was written with, from its files' footers.

    Taken as the largest row group seen, since the last group of every file is
    a remainder.
    """
    best = 0
    for p in paths[:sample]:
        md = pq.ParquetFile(_norm(p)).metadata
        best = max([best] + [md.row_group(i).num_rows for i in range(md.num_row_groups)])
    return best
