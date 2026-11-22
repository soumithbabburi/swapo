"""The cost model, and the what-if estimate that lets the optimizer look before it leaps.

Rewriting a table to find out whether rewriting it was worthwhile is not a
strategy. AutoAdmin solved the equivalent problem for indexes in 1998 with a
"what-if" interface: ask the cost model about a design that does not exist yet.
This module is the genomic version of that, and it is deliberately simple
enough to read in one sitting.

The model
---------
Blocks on a contig are sorted by ``start``. Block *i* accepts rows whose start
falls in a window of width ``span``, and it claims coordinates out to its
``max(end)``, a distance ``reach`` from where it begins. A query of width *w*
must read every block whose claim overlaps it, and the starts of those blocks
occupy a window of width ``w + reach``. So::

    blocks_read  =  clamp(1,  (w + reach) / span,  blocks_on_contig)

Two things follow, and they are the whole argument of this project:

* ``reach`` is measured from the existing zone map, so the cost of the current
  layout is not a guess;
* ``reach`` for a *proposed* layout can be derived from the interval-length
  distribution without writing a byte, which is what makes the recommendation
  cheap enough to make continuously.

Prices default to figures in the right order of magnitude for cloud object
storage and are a constructor argument, because the ranking of actions depends
on their ratio rather than their absolute value.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import INTERVAL_INTENTS, Intent
from .workload import WorkloadProfile
from .zonemap import IntervalZoneMap

__all__ = ["CostModel", "LayoutShape", "expected_blocks"]


def expected_blocks(width: int, reach: float, span: float, n_blocks: int) -> float:
    """Blocks an overlap query of ``width`` must read. See the module docstring."""
    if span <= 0 or n_blocks <= 0:
        return float(max(n_blocks, 0))
    est = (width + reach) / span
    return min(float(n_blocks), max(1.0, est))


@dataclass
class LayoutShape:
    """The few numbers the cost model needs to describe a layout.

    Derived from a real zone map by :meth:`from_zonemap`, or constructed by
    hand to describe a layout that does not exist yet.
    """

    n_blocks: int
    block_bytes: float
    #: Mean width of the start range a block covers, in bp.
    span: float
    #: Mean distance from a block's first start to its furthest end, in bp.
    #: Equal to ``span`` plus the longest record in the block.
    reach: float
    #: Blocks on a typical contig (pruning is per contig when possible).
    blocks_per_contig: int
    #: True when a block can hold rows from more than one contig, which stops
    #: contig pruning working at all.
    mixed_contigs: bool = False

    @classmethod
    def from_zonemap(cls, zm: IntervalZoneMap) -> "LayoutShape":
        blocks = [b for b in zm.blocks if not b.long_tail]
        if not blocks:
            blocks = zm.blocks
        if not blocks:
            return cls(0, 0.0, 1.0, 1.0, 0)
        spans = [max(b.start_max - b.start_min, 1) for b in blocks]
        reaches = [max(b.end_max - b.start_min, 1) for b in blocks]
        contigs = {b.contig for b in zm.blocks}
        mixed = "*" in contigs
        per_contig = max(1, len(zm.blocks) // max(1, len(contigs - {"*"}) or 1))
        return cls(
            n_blocks=len(zm.blocks),
            block_bytes=sum(b.n_bytes for b in zm.blocks) / len(zm.blocks),
            span=sum(spans) / len(spans),
            reach=sum(reaches) / len(reaches),
            blocks_per_contig=len(zm.blocks) if mixed else per_contig,
            mixed_contigs=mixed,
        )


@dataclass
class CostModel:
    """Prices and horizons. Change these to match your storage bill."""

    #: Cost of reading one gigabyte, in arbitrary currency units.
    read_per_gb: float = 1.0
    #: Writing is dearer than reading: a rewrite pays compute as well as I/O.
    write_per_gb: float = 3.0
    #: Cost of keeping a gigabyte for the whole planning horizon.
    storage_per_gb: float = 0.5
    #: How far ahead the optimizer is asked to think, in query windows.
    horizon_windows: float = 30.0
    #: What it costs, in the same units, to serve one answer in the WRONG
    #: frequency class (absent/rare/low/BS1/BA1). The probability of that is
    #: measured (swapo.staleness); this price is the operator's judgement.
    staleness_penalty: float = 400.0

    # -- scanning -----------------------------------------------------------

    def query_blocks(self, profile: WorkloadProfile, shape: LayoutShape) -> float:
        """Expected blocks read per query under ``profile``."""
        if profile.n_queries == 0:
            return 0.0
        total = 0.0
        for intent, count in profile.counts.items():
            share = count / profile.n_queries
            if intent is Intent.SCAN:
                total += share * shape.n_blocks
                continue
            if intent not in INTERVAL_INTENTS and intent not in (
                    Intent.SAMPLE, Intent.COHORT_FREQ, Intent.PHENOTYPE_JOIN):
                total += share * shape.n_blocks
                continue
            width = profile.median_width or 1
            if intent is Intent.LARGE_REGION:
                width = max(width, profile.p95_width or width)
            n = expected_blocks(width, shape.reach, shape.span,
                                shape.blocks_per_contig)
            # Without single-contig blocks, nothing is skipped by contig, so the
            # query touches that fraction of every contig's worth of blocks.
            if shape.mixed_contigs:
                n = max(n, shape.n_blocks * 0.9)
            total += share * n
        return total

    def scan_cost(self, profile: WorkloadProfile, shape: LayoutShape,
                  volume: float | None = None) -> float:
        """Cost of serving the workload over the horizon.

        ``profile`` supplies the *shape* of the workload. ``volume`` supplies how
        many queries of that shape are expected per window. They are separate on
        purpose: a log of fifty statements can describe a service answering
        fifty thousand a day, and charging a rewrite against fifty queries would
        reject every change worth making.
        """
        n = volume if volume is not None else max(profile.n_queries, 1)
        gb = self.query_blocks(profile, shape) * shape.block_bytes / 1e9
        return gb * self.read_per_gb * self.horizon_windows * n

    # -- changing the layout ------------------------------------------------

    def rewrite_cost(self, n_bytes: float) -> float:
        return n_bytes / 1e9 * self.write_per_gb

    def storage_cost(self, n_bytes: float) -> float:
        return n_bytes / 1e9 * self.storage_per_gb

    def refresh_cost(self, n_bytes: float, refreshes_per_horizon: float) -> float:
        """What it costs to keep a derived dataset current over the horizon."""
        return self.rewrite_cost(n_bytes) * refreshes_per_horizon

    def staleness_cost(self, queries_per_horizon: float, stale_fraction: float) -> float:
        """What it costs to serve answers from a dataset that is behind its inputs."""
        return queries_per_horizon * stale_fraction * self.staleness_penalty
