"""Execution: prune with the zone map, then hand the surviving files to DuckDB.

Keeping pruning outside the engine is deliberate. DuckDB is perfectly capable of
using Parquet row-group statistics itself, but then the amount of data skipped is
buried inside the engine and hard to attribute. Here SWaPO decides which files to
open and DuckDB opens exactly those, so "bytes read" is a number we can point at
rather than infer, and the comparison between layouts measures the layouts.

The SQL predicate is still applied in full, so results are always correct even if
pruning were to let through a block it should not have.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import duckdb

from .model import Intent, Query, Region
from .zonemap import IntervalZoneMap, PruneResult

__all__ = ["Engine", "QueryResult"]


@dataclass
class QueryResult:
    """What a query returned and what it cost."""

    rows: int
    seconds: float
    blocks_read: int
    blocks_total: int
    bytes_read: int
    prune: PruneResult | None = None
    value: Any = None
    sql: str = ""

    @property
    def blocks_skipped(self) -> int:
        return self.blocks_total - self.blocks_read

    @property
    def skip_fraction(self) -> float:
        return self.blocks_skipped / self.blocks_total if self.blocks_total else 0.0

    def summary(self) -> dict:
        return {
            "rows": self.rows, "ms": round(self.seconds * 1000, 2),
            "blocks_read": self.blocks_read, "blocks_total": self.blocks_total,
            "megabytes_read": round(self.bytes_read / 1e6, 2),
            "skipped_pct": round(self.skip_fraction * 100, 1),
        }


class Engine:
    """Runs genomic queries against a layout, counting what it had to read."""

    def __init__(self, threads: int = 4, gene_table: Path | str | None = None) -> None:
        self.con = duckdb.connect(":memory:")
        self.con.execute(f"PRAGMA threads={threads}")
        self._genes: dict[str, Region] = {}
        if gene_table is not None:
            self.load_genes(gene_table)

    # -- gene model ---------------------------------------------------------

    def load_genes(self, path: Path | str) -> None:
        """Load a gene model so ``Intent.GENE`` can resolve a symbol to coordinates.

        This is the smallest piece of genomic semantics the optimizer needs and
        a generic optimizer does not have: it turns "the queries ask for BRCA2"
        into "the queries hit chr13:32,315,086-32,400,268", which is what any
        layout decision has to reason about.
        """
        rows = self.con.execute(
            "SELECT symbol, gene_id, contig, start, \"end\" FROM read_parquet(?)",
            [str(path)],
        ).fetchall()
        for symbol, gene_id, contig, start, end in rows:
            region = Region(contig, int(start), int(end))
            self._genes[symbol] = region
            self._genes[gene_id] = region

    def resolve(self, gene: str) -> Region | None:
        return self._genes.get(gene)

    # -- execution ----------------------------------------------------------

    @staticmethod
    def _files_clause(paths: Sequence[str]) -> str:
        quoted = ", ".join("'" + p.replace("'", "''") + "'" for p in paths)
        return f"read_parquet([{quoted}])"

    def run(self, query: Query, zonemap: IntervalZoneMap) -> QueryResult:
        """Execute ``query``, reading only the blocks the zone map keeps."""
        region = query.region
        if query.intent is Intent.GENE and query.gene:
            region = self.resolve(query.gene) or region

        total = len(zonemap)
        t0 = time.perf_counter()

        if region is not None:
            pr = zonemap.prune(region.contig, region.start, region.end)
            paths = [b.path for b in pr.blocks]
        else:
            pr = None
            paths = [b.path for b in zonemap.blocks]

        if not paths:
            # Nothing survived pruning: the honest cost of this query is zero
            # bytes, and the answer is empty.
            return QueryResult(0, time.perf_counter() - t0, 0, total, 0, pr, None, "")

        src = self._files_clause(paths)
        params: list[Any] = []
        where: list[str] = []

        if region is not None:
            # The overlap test, written out. Half-open on both sides.
            where.append('contig = ? AND start < ? AND ? < "end"')
            params += [region.contig, region.end, region.start]
        if query.sample_ids:
            marks = ", ".join("?" * len(query.sample_ids))
            where.append(f"sample_id IN ({marks})")
            params += list(query.sample_ids)
        if query.cohort:
            where.append("cohort = ?")
            params.append(query.cohort)

        clause = " WHERE " + " AND ".join(where) if where else ""

        if query.intent is Intent.COHORT_FREQ:
            sql = (f"SELECT variant_id, sum(gt) / (2.0 * count(*)) AS af "
                   f"FROM {src}{clause} GROUP BY variant_id")
        elif query.intent is Intent.PHENOTYPE_JOIN:
            sql = (f"SELECT cohort, avg(gt) AS mean_dosage "
                   f"FROM {src}{clause} GROUP BY cohort")
        elif query.intent is Intent.SCAN:
            sql = f"SELECT count(*) AS n FROM {src}{clause}"
        else:
            sql = f"SELECT * FROM {src}{clause}"

        rel = self.con.execute(sql, params)
        out = rel.fetchall()
        elapsed = time.perf_counter() - t0

        rows = out[0][0] if query.intent is Intent.SCAN and out else len(out)
        return QueryResult(
            rows=int(rows), seconds=elapsed, blocks_read=len(paths),
            blocks_total=total, bytes_read=sum(b.n_bytes for b in (pr.blocks if pr else zonemap.blocks)),
            prune=pr, value=out[:5], sql=sql,
        )

    def count_matching(self, query: Query, zonemap: IntervalZoneMap) -> int:
        """Ground truth for correctness checks: scan every block, no pruning."""
        region = query.region
        if query.intent is Intent.GENE and query.gene:
            region = self.resolve(query.gene) or region
        paths = [b.path for b in zonemap.blocks]
        src = self._files_clause(paths)
        if region is None:
            return int(self.con.execute(f"SELECT count(*) FROM {src}").fetchone()[0])
        return int(self.con.execute(
            f'SELECT count(*) FROM {src} WHERE contig = ? AND start < ? AND ? < "end"',
            [region.contig, region.end, region.start],
        ).fetchone()[0])

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
