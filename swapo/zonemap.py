"""Augmented interval zone maps: block pruning that survives long intervals.

Why this module exists
----------------------
Columnar formats keep per-block minimum and maximum values for each column, and
engines skip a block when those bounds cannot satisfy the predicate. That works
for equality and for one-sided ranges. A genomic **overlap** test is two-sided::

    feature.start < query.end  AND  query.start < feature.end

With rows sorted by ``start``, the first clause prunes everything after the
query. The second clause is the one that prunes everything *before* it, and it
can only use ``max(end)`` for the block. One megabase-long structural variant,
gene body or repeat region raises ``max(end)`` for its whole block, and every
later query on that contig must read it. Scatter a few thousand such rows
through a table and ``max(end)`` is useless in every block: the reader falls
back to scanning from the start of the contig to the query.

The fix has been known in the interval-search literature since nested
containment lists, and is used by the Augmented Interval List (Feng, Ratan and
Sheffield, *Bioinformatics* 2019): sort by start, keep a running maximum of
end, and lift the small number of *containing* intervals into a component of
their own so they cannot distort the dense majority.

This module applies that idea one level up, to storage blocks rather than to
individual intervals, so it composes with ordinary columnar pruning:

* blocks are sorted by ``start_min`` within a contig, so the upper bound is a
  binary search;
* a prefix maximum over ``end_max`` gives a binary-searchable lower bound, so
  blocks that finish before the query are skipped as a range rather than one by
  one;
* long intervals live in separate blocks, which keeps ``end_max`` tight for the
  dense blocks and confines the unavoidable scan to a handful of small ones.

Correctness is the point of :meth:`IntervalZoneMap.prune`: it may return blocks
that turn out to hold no matching row, but it must never omit one that does.
``tests/test_zonemap.py`` checks that against brute force on random data.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from .model import Block

__all__ = ["IntervalZoneMap", "ContigIndex", "PruneResult", "MIXED"]

#: Contig label for a block whose rows span more than one contig. Such a block
#: cannot be skipped on contig, so it is examined by every region query.
MIXED = "*"


@dataclass(slots=True)
class PruneResult:
    """Which blocks a query must read, and how much that saved."""

    blocks: list[Block]
    considered: int
    #: Blocks eliminated by the contig test.
    skipped_contig: int = 0
    #: Blocks eliminated because they begin after the query ends.
    skipped_after: int = 0
    #: Blocks eliminated because every row in them ends before the query starts.
    #: This is the number the long-interval split is trying to protect.
    skipped_before: int = 0

    @property
    def selected(self) -> int:
        return len(self.blocks)

    @property
    def bytes_read(self) -> int:
        return sum(b.n_bytes for b in self.blocks)

    @property
    def rows_read(self) -> int:
        return sum(b.n_rows for b in self.blocks)


@dataclass(slots=True)
class ContigIndex:
    """Blocks for one contig, ordered by ``start_min`` with a prefix max of ``end_max``."""

    dense: list[Block] = field(default_factory=list)
    #: Blocks holding intervals above the split threshold. Few, and always read
    #: when they overlap, because their bounds are genuinely wide.
    long_tail: list[Block] = field(default_factory=list)
    _starts: list[int] = field(default_factory=list, repr=False)
    _prefix_max_end: list[int] = field(default_factory=list, repr=False)

    def finalise(self) -> None:
        """Sort and build the search structures. Call once after adding blocks."""
        self.dense.sort(key=lambda b: (b.start_min, b.end_max))
        self.long_tail.sort(key=lambda b: (b.start_min, b.end_max))
        self._starts = [b.start_min for b in self.dense]
        running = 0
        self._prefix_max_end = []
        for b in self.dense:
            # Monotonic by construction, which is what makes it binary-searchable.
            running = b.end_max if b.end_max > running else running
            self._prefix_max_end.append(running)

    def prune(self, start: int, end: int, out: PruneResult) -> None:
        """Append the blocks that could hold a row overlapping ``[start, end)``."""
        n = len(self.dense)
        # Upper bound: the first block whose start_min is at or past the query
        # end cannot contain an overlapping row, nor can any block after it.
        hi = bisect.bisect_left(self._starts, end)
        out.skipped_after += n - hi
        # Lower bound: every block before `lo` has end_max <= start, because the
        # prefix maximum up to lo-1 is <= start. Skip the whole range at once.
        lo = bisect.bisect_right(self._prefix_max_end, start, 0, hi)
        out.skipped_before += lo
        for i in range(lo, hi):
            b = self.dense[i]
            if b.overlaps(start, end):
                out.blocks.append(b)
            else:
                out.skipped_before += 1
        for b in self.long_tail:
            if b.overlaps(start, end):
                out.blocks.append(b)
            else:
                out.skipped_before += 1


class IntervalZoneMap:
    """Block statistics for one table, queryable by genomic region.

    Build it once after writing a layout, then ask it which files a region
    query needs. It is small: a few hundred bytes per block, so a table of ten
    thousand blocks costs a couple of megabytes and is cheap to keep in memory
    or beside the data as JSON.
    """

    def __init__(self, blocks: list[Block] | None = None) -> None:
        self._by_contig: dict[str, ContigIndex] = {}
        self._blocks: list[Block] = []
        if blocks:
            for b in blocks:
                self.add(b)
            self.finalise()

    # -- construction -------------------------------------------------------

    def add(self, block: Block) -> None:
        idx = self._by_contig.setdefault(block.contig, ContigIndex())
        (idx.long_tail if block.long_tail else idx.dense).append(block)
        self._blocks.append(block)

    def finalise(self) -> "IntervalZoneMap":
        for idx in self._by_contig.values():
            idx.finalise()
        return self

    # -- querying -----------------------------------------------------------

    def prune(self, contig: str | None, start: int, end: int) -> PruneResult:
        """Blocks that must be read for an overlap query.

        ``contig=None`` means the query spans the whole genome, which only the
        two-sided coordinate test can narrow.

        A block holding rows from more than one contig is recorded under
        :data:`MIXED` and can never be excluded by contig, only by coordinate.
        Leaving those out would be a correctness bug rather than an
        optimization, so they are always examined.
        """
        res = PruneResult(blocks=[], considered=len(self._blocks))
        if contig is None:
            for idx in self._by_contig.values():
                idx.prune(start, end, res)
            return res

        searched = 0
        for name in (contig, MIXED):
            idx = self._by_contig.get(name)
            if idx is None:
                continue
            searched += len(idx.dense) + len(idx.long_tail)
            idx.prune(start, end, res)
        res.skipped_contig = len(self._blocks) - searched
        return res

    def prune_brute_force(self, contig: str | None, start: int, end: int) -> list[Block]:
        """Reference implementation. Used by the tests to check :meth:`prune`."""
        return [
            b for b in self._blocks
            if (contig is None or b.contig == contig or b.contig == MIXED)
            and b.overlaps(start, end)
        ]

    # -- reporting ----------------------------------------------------------

    @property
    def blocks(self) -> list[Block]:
        return list(self._blocks)

    @property
    def n_bytes(self) -> int:
        return sum(b.n_bytes for b in self._blocks)

    @property
    def n_rows(self) -> int:
        return sum(b.n_rows for b in self._blocks)

    def tightness(self) -> float:
        """Mean ratio of block span to the coordinate range it really covers.

        1.0 is perfect. Large values mean ``end_max`` overshoots ``start_max``,
        which is the signature of long intervals mixed into dense blocks and the
        signal the optimizer uses to propose a split.
        """
        ratios = []
        for b in self._blocks:
            if b.long_tail:
                continue
            span = max(b.start_max - b.start_min, 1)
            reach = max(b.end_max - b.start_min, 1)
            ratios.append(reach / span)
        return sum(ratios) / len(ratios) if ratios else 1.0

    def to_json(self) -> list[dict]:
        return [b.to_dict() for b in self._blocks]

    @classmethod
    def from_json(cls, rows: list[dict]) -> "IntervalZoneMap":
        return cls([Block(**r) for r in rows])

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        return (f"IntervalZoneMap(blocks={len(self._blocks)}, "
                f"contigs={len(self._by_contig)}, tightness={self.tightness():.1f})")
