"""Choosing where "long" begins, from the data rather than from a constant.

The first version of this project used a fixed 50 bp threshold, borrowed from
the conventional boundary between an indel and a structural variant. Real data
broke that immediately: in GENCODE v44, 93% of records are longer than 50 bp,
because an exon is a few hundred bases. Splitting at 50 bp would put almost the
whole table in the "long" partition and achieve nothing.

The threshold is not a property of biology, it is a property of the join between
the length distribution and the block size. A record is worth separating only
when it reaches far enough past its block to force other queries to read that
block. If blocks span 300 kb of coordinates, a 2 kb exon is irrelevant and a
500 kb gene is not.

So it is chosen by search. For each candidate threshold the expected bytes read
is estimated for the workload, and the cheapest is taken. The estimate is exact
in the sense that it uses the real block boundaries the layout would produce; it
simply does not write them, which makes the whole search cost one sort.

See :func:`choose_threshold`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

__all__ = ["choose_threshold", "ThresholdChoice", "simulate_blocks", "separability"]


@dataclass
class ThresholdChoice:
    """The chosen threshold and the evidence for it."""

    threshold: int
    #: Expected blocks read per query at this threshold.
    expected_blocks: float
    #: Expected blocks read with no split at all, for comparison.
    baseline_blocks: float
    long_fraction: float
    #: Every candidate considered: ``(threshold, expected_blocks, long_fraction)``.
    candidates: list[tuple[int, float, float]]
    query_width: int

    @property
    def improvement(self) -> float:
        if not self.baseline_blocks:
            return 0.0
        return 1.0 - self.expected_blocks / self.baseline_blocks

    @property
    def worth_doing(self) -> bool:
        """Below a few percent the rewrite will not repay itself.

        ``threshold == 0`` means no candidate beat leaving the table alone.
        """
        return self.threshold > 0 and self.improvement > 0.05

    def summary(self) -> dict:
        return {
            "threshold_bp": self.threshold or None,   # None == do not split
            "long_fraction": round(self.long_fraction, 5),
            "expected_blocks_read": round(self.expected_blocks, 2),
            "baseline_blocks_read": round(self.baseline_blocks, 2),
            "improvement": round(self.improvement, 3),
            "worth_doing": self.worth_doing,
            "query_width_bp": self.query_width,
        }


def simulate_blocks(starts: np.ndarray, ends: np.ndarray,
                    rows_per_block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Block bounds for rows already sorted by start, without writing anything.

    Returns ``(start_min, start_max, end_max)`` per block. This is the whole
    trick that makes the search affordable: the layout is a deterministic
    function of the sort order, so its statistics can be computed directly.
    """
    n = starts.size
    if n == 0:
        return (np.empty(0, np.int64),) * 3
    edges = np.arange(0, n, rows_per_block)
    start_min = starts[edges]
    last = np.append(edges[1:], n) - 1
    start_max = starts[last]
    # max(end) within each block.
    end_max = np.maximum.reduceat(ends, edges)
    return start_min, start_max, end_max


def _expected_blocks(start_min: np.ndarray, end_max: np.ndarray,
                     width: int, probe: np.ndarray) -> float:
    """Mean blocks whose claimed range overlaps a query of ``width`` at each probe.

    The same two-sided test the engine uses: a block is read when
    ``start_min < query_end`` and ``query_start < end_max``.
    """
    if start_min.size == 0:
        return 0.0
    qs = probe[:, None]
    qe = qs + width
    hit = (start_min[None, :] < qe) & (qs < end_max[None, :])
    return float(hit.sum(axis=1).mean())


def choose_threshold(
    tbl: pa.Table,
    *,
    rows_per_block: int = 8_000,
    query_width: int = 50_000,
    candidates: tuple[int, ...] | None = None,
    n_probes: int = 160,
    seed: int = 19,
) -> ThresholdChoice:
    """Pick the split threshold that minimises expected blocks read.

    ``query_width`` should be the typical region width in the observed
    workload; :class:`~swapo.workload.WorkloadProfile` reports it as
    ``median_width``. The choice genuinely depends on it, which is the point:
    the right physical design for gene-sized lookups is not the right one for
    whole-arm sweeps.
    """
    rng = np.random.default_rng(seed)
    contigs = pc.unique(tbl.column("contig")).to_pylist()
    # Work on the contig holding the most rows: it dominates the cost and keeps
    # the search to one sort.
    counts = (pa.table({"contig": tbl.column("contig")})
              .group_by("contig").aggregate([([], "count_all")]))
    names = counts.column("contig").to_pylist()
    sizes = counts.column("count_all").to_pylist()
    contig = names[int(np.argmax(sizes))]

    sub = tbl.filter(pc.equal(tbl.column("contig"), contig))
    order = pc.sort_indices(sub, sort_keys=[("start", "ascending")])
    sub = sub.take(order)
    starts = np.asarray(sub.column("start")).astype(np.int64)
    ends = np.asarray(sub.column("end")).astype(np.int64)
    spans = ends - starts
    if starts.size == 0:
        return ThresholdChoice(0, 0.0, 0.0, 0.0, [], query_width)

    lo, hi = int(starts.min()), int(starts.max())
    probe = rng.integers(lo, max(hi, lo + 1), size=n_probes)

    # Baseline: no split at all.
    b_min, _, b_end = simulate_blocks(starts, ends, rows_per_block)
    baseline = _expected_blocks(b_min, b_end, query_width, probe)

    if candidates is None:
        # Span quantiles, so the candidates follow the data rather than a guess.
        qs = np.unique(np.quantile(spans, [.5, .75, .9, .95, .98, .99, .995, .999]))
        candidates = tuple(int(x) for x in qs if x > 0)

    evaluated: list[tuple[int, float, float]] = []
    # Threshold 0 means "do not split". It is the incumbent and a candidate has
    # to beat it outright, which is how the method is able to return nothing.
    best = (0, baseline, 0.0)
    best_cost = baseline

    for t in candidates:
        is_long = spans > t
        frac = float(is_long.mean())
        if frac >= 0.9 or frac == 0.0:
            # Splitting almost everything, or nothing, is not a split.
            evaluated.append((int(t), baseline, frac))
            continue
        d_start, d_end = starts[~is_long], ends[~is_long]
        l_start, l_end = starts[is_long], ends[is_long]
        dm, _, de = simulate_blocks(d_start, d_end, rows_per_block)
        lm, _, le = simulate_blocks(l_start, l_end, rows_per_block)
        cost = (_expected_blocks(dm, de, query_width, probe)
                + _expected_blocks(lm, le, query_width, probe))
        evaluated.append((int(t), cost, frac))
        if cost < best_cost:
            best_cost, best = cost, (int(t), cost, frac)

    return ThresholdChoice(
        threshold=best[0], expected_blocks=best_cost, baseline_blocks=baseline,
        long_fraction=best[2], candidates=evaluated, query_width=query_width,
    )


def separability(tbl: "pa.Table") -> dict:
    """How cleanly a length threshold could cut this table in two.

    Separating long records only pays when the distribution is **bimodal**: a
    dense short majority plus a sparse minority that is longer by orders of
    magnitude. Then a threshold exists that moves very few rows and removes
    almost all of the reach.

    When the distribution is continuous, every threshold is a bad compromise.
    Lifting out enough rows to tighten the dense blocks creates a long partition
    with enough blocks of its own that the query has to read those instead, and
    the total gets worse rather than better.

    The ratio of the 99.99th to the 99th percentile span measures this directly,
    and it costs one pass. In our measurements a ratio in the tens means no
    threshold helps; a ratio in the tens of thousands means one helps a great
    deal. It is reported alongside the recommendation so the reason for a
    refusal is visible rather than mysterious.
    """
    import numpy as _np
    import pyarrow.compute as _pc

    span = _pc.subtract(tbl.column("end"), tbl.column("start"))
    arr = _np.asarray(span).astype(_np.int64)
    if arr.size == 0:
        return {"bimodality": 0.0}
    p50, p90, p99, p999, p9999 = (
        float(_np.percentile(arr, q)) for q in (50, 90, 99, 99.9, 99.99))
    return {
        "median_bp": int(p50), "p90_bp": int(p90), "p99_bp": int(p99),
        "p99_9_bp": int(p999), "p99_99_bp": int(p9999), "max_bp": int(arr.max()),
        # Guarded so a table of single-base records does not divide by zero.
        "bimodality": round(p9999 / max(p99, 1.0), 1),
        "separable": p9999 / max(p99, 1.0) > 1000.0,
    }
