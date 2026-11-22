"""Choosing the block size, which is where the money actually is.

Our measurements kept showing the same thing. Separating long records was worth
nothing on real data, but moving a real chr1 table from 8,000 rows per block to
250 cut bytes read by 87%. Nobody computes that number today: block size is set
by convention (128 MB row groups, or whatever the writer defaults to) and then
left alone, even though the right value depends on the query width and the data
density, both of which are measurable.

It is a genuine optimum rather than "smaller is better", because two costs pull
against each other:

* **fewer rows per block** means narrower blocks, so a region query overlaps
  fewer of them — bytes read falls roughly as ``1/rows_per_block``;
* **fewer rows per block** also means more blocks, and every block carries fixed
  overhead: a Parquet footer, column chunk headers, dictionary pages. Past some
  point the overhead dominates and bytes read climbs again.

So the cost curve is U-shaped, and the minimum is what this module finds::

    bytes(r)  =  expected_blocks(width, reach(r), span(r))  ×  (overhead + r × bytes_per_row)

``reach`` and ``span`` come from simulating block boundaries on the real sorted
data without writing it (:func:`swapo.threshold.simulate_blocks`). ``overhead``
and ``bytes_per_row`` are fitted from two small sample writes, because they
depend on the schema, the compression and the writer.

Unlike the long-record split, this always applies: every table has a block size,
and it is almost never the one that minimises reads.
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

from .threshold import _expected_blocks, simulate_blocks


def _blocks_for_queries(s_min: np.ndarray, e_max: np.ndarray,
                        qs: np.ndarray, qe: np.ndarray) -> float:
    """Mean blocks touched by a set of real queries of differing widths.

    Evaluating at a single median width is not the same as averaging over the
    real distribution: blocks read grows with width, so the wide tail of a
    workload pulls the mean above the median's value. With the real intervals
    in hand there is no reason to approximate.
    """
    if s_min.size == 0 or qs.size == 0:
        return 0.0
    hit = (s_min[None, :] < qe[:, None]) & (qs[:, None] < e_max[None, :])
    return float(hit.sum(axis=1).mean())

__all__ = ["BlockSizeChoice", "fit_block_cost", "choose_block_size"]


@dataclass
class BlockCost:
    """How many bytes a block of ``r`` rows occupies on disk.

    Backed by direct measurement at the sizes we care about, with a linear fit
    kept only for extrapolating outside the measured range. A two-parameter fit
    alone was systematically 3.5-6.7% low across the whole curve: real block
    cost is not quite linear in row count, because dictionaries and page
    headers amortise unevenly. Since the candidate sizes are known in advance
    and sample blocks are cheap to write, there is no reason to model what can
    be measured.
    """

    overhead_bytes: float
    bytes_per_row: float
    samples: list[tuple[int, int]] = field(default_factory=list)
    #: Mean measured bytes per block, keyed by rows per block.
    measured: dict[int, float] = field(default_factory=dict)

    def block_bytes(self, rows_per_block: int) -> float:
        if rows_per_block in self.measured:
            return self.measured[rows_per_block]
        if len(self.measured) >= 2:
            sizes = sorted(self.measured)
            lo = [x for x in sizes if x < rows_per_block]
            hi = [x for x in sizes if x > rows_per_block]
            if lo and hi:
                # Log-log interpolation: bytes grow close to a power law in
                # rows, so this is far tighter than interpolating linearly.
                a, b = lo[-1], hi[0]
                fa, fb = self.measured[a], self.measured[b]
                w = (np.log(rows_per_block) - np.log(a)) / (np.log(b) - np.log(a))
                return float(np.exp(np.log(fa) + w * (np.log(fb) - np.log(fa))))
        return self.overhead_bytes + self.bytes_per_row * rows_per_block


@dataclass
class BlockSizeChoice:
    """The recommended block size and the curve it was chosen from."""

    rows_per_block: int
    expected_bytes: float
    current_rows_per_block: int
    current_expected_bytes: float
    query_width: int
    cost: BlockCost
    #: ``(rows_per_block, expected_blocks_read, expected_bytes)`` for every candidate.
    curve: list[tuple[int, float, float]] = field(default_factory=list)
    #: Candidates whose forecast is within the measurement noise of the best.
    #: Near the optimum the curve is flat and sample variance is a few percent,
    #: so naming a single winner would be spurious precision.
    tied: list[int] = field(default_factory=list)
    #: Relative standard error of the block-byte samples.
    noise: float = 0.0

    @property
    def improvement(self) -> float:
        if not self.current_expected_bytes:
            return 0.0
        return 1.0 - self.expected_bytes / self.current_expected_bytes

    @property
    def speedup(self) -> float:
        if not self.expected_bytes:
            return 1.0
        return self.current_expected_bytes / self.expected_bytes

    def summary(self) -> dict:
        return {
            "recommend_rows_per_block": self.rows_per_block,
            "from_rows_per_block": self.current_rows_per_block,
            "expected_bytes_per_query": round(self.expected_bytes),
            "current_bytes_per_query": round(self.current_expected_bytes),
            "expected_reduction": round(self.improvement, 3),
            "expected_speedup": round(self.speedup, 2),
            "query_width_bp": self.query_width,
            "fitted_overhead_bytes": round(self.cost.overhead_bytes),
            "fitted_bytes_per_row": round(self.cost.bytes_per_row, 2),
            "tied_within_noise": self.tied,
            "sample_noise": round(self.noise, 4),
        }

    @property
    def tie_note(self) -> str:
        if len(self.tied) <= 1:
            return ""
        others = ", ".join(f"{x:,}" for x in self.tied if x != self.rows_per_block)
        return (f"{others} forecast within measurement noise "
                f"({self.noise * 100:.1f}%) of this; a larger size there would "
                f"mean fewer files for about the same reads")

    def explain(self) -> str:
        tie = ("%s  also fine     : %s" % (chr(10), self.tie_note)) if self.tie_note else ""
        return (
            f"block size on this table\n"
            f"  current        : {self.current_rows_per_block:,} rows/block, "
            f"about {self.current_expected_bytes / 1e6:.3f} MB read per query\n"
            f"  recommended    : {self.rows_per_block:,} rows/block, "
            f"about {self.expected_bytes / 1e6:.3f} MB read per query\n"
            f"  expected gain  : {self.improvement * 100:.0f}% fewer bytes "
            f"({self.speedup:.1f}x)\n"
            f"  why            : at {self.rows_per_block:,} rows a block covers "
            f"roughly the width of a typical {self.query_width:,} bp query, so a "
            f"query reads few blocks while per-block overhead "
            f"({self.cost.overhead_bytes / 1e3:.0f} kB each) stays small\n"
            f"  freshness      : none, this rewrites rows into different blocks\n"
            f"  rollback       : rewrite at {self.current_rows_per_block:,} rows/block"
            + tie
        )


def fit_block_cost(
    tbl: pa.Table,
    *,
    sample_sizes: tuple[int, ...] = (125, 250, 500, 1_000, 2_000, 4_000,
                                     8_000, 16_000, 32_000, 64_000),
    windows_per_size: int = 3,
    compression: str = "zstd",
    seed: int = 29,
) -> BlockCost:
    """Fit per-block overhead and per-row bytes by writing sample blocks.

    Both depend on the schema, the compression codec and the writer, so they are
    measured rather than assumed. Two details matter for accuracy, and getting
    either wrong produced a two-fold error in our first attempt:

    **Sample contiguous windows of the sorted table, not the head.** A real
    block holds rows that are adjacent after sorting by position, so its columns
    are locally correlated and dictionary-compress accordingly. Taking the first
    N rows of a concatenated table instead samples whichever source was
    concatenated first — for us, annotation, which compresses far better than
    the variant rows that dominate the table.

    **Fit across several sizes, not two.** Bytes per row drifts slightly with
    block size as dictionaries amortise, so a least-squares fit over a range is
    more faithful than a two-point slope.
    """
    rng = np.random.default_rng(seed)
    ordered = tbl.take(pc.sort_indices(
        tbl, sort_keys=[("contig", "ascending"), ("start", "ascending")]))
    n = ordered.num_rows
    sizes = sorted({int(x) for x in sample_sizes if 0 < int(x) <= n})
    if not sizes:
        return BlockCost(overhead_bytes=0.0, bytes_per_row=1e-6)

    # Paired sampling. Every candidate size is measured at the SAME anchor
    # offsets, so each size sees the same stretches of genome. Giving each size
    # its own random windows makes the per-size means independent, and the
    # *differences* between adjacent sizes — which is all the search actually
    # compares — then carry the sum of both errors. Near a flat optimum that
    # noise is larger than the real difference, and the ranking flips. Sharing
    # anchors cancels most of it, because a region that compresses unusually
    # well does so at every size.
    largest = sizes[-1]
    anchors = [int(rng.integers(0, max(1, n - largest))) for _ in range(windows_per_size)]

    tmp = Path(tempfile.mkdtemp())
    points: list[tuple[int, int]] = []
    per_size: dict[int, list[int]] = {z: [] for z in sizes}
    try:
        for ai, offset in enumerate(anchors):
            for size in sizes:
                chunk = ordered.slice(offset, size)
                if chunk.num_rows == 0:
                    continue
                path = tmp / f"a{ai}_s{size}.parquet"
                pq.write_table(chunk, path, compression=compression,
                               row_group_size=max(chunk.num_rows, 1),
                               use_dictionary=True)
                b = path.stat().st_size
                points.append((chunk.num_rows, b))
                per_size[size].append(b)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    measured = {z: float(np.mean(v)) for z, v in per_size.items() if v}

    if len(points) < 2:
        total = points[0][1] if points else 0
        return BlockCost(overhead_bytes=float(total), bytes_per_row=0.0,
                         samples=points,
                         measured={points[0][0]: float(total)} if points else {})

    rows = np.array([r for r, _ in points], dtype=float)
    size_b = np.array([b for _, b in points], dtype=float)
    # Fit on RELATIVE error, not absolute. An ordinary least-squares fit is
    # dominated by the largest samples, which leaves the small-block end — the
    # end where the fixed overhead actually matters, and where the optimum
    # lives — badly wrong. Dividing each equation by its own byte count turns
    # "minimise relative error" into a linear system:
    #
    #     overhead * (1/b) + per_row * (r/b) = 1
    design = np.stack([1.0 / size_b, rows / size_b], axis=1)
    (overhead, per_row), *_ = np.linalg.lstsq(design, np.ones_like(size_b), rcond=None)
    return BlockCost(overhead_bytes=float(max(overhead, 0.0)),
                     bytes_per_row=float(max(per_row, 1e-6)),
                     samples=points, measured=measured)


def choose_block_size(
    tbl: pa.Table,
    *,
    query_width: int,
    current_rows_per_block: int = 8_000,
    probes: "np.ndarray | list[int] | None" = None,
    queries: "list[tuple[int, int]] | None" = None,
    candidates: tuple[int, ...] = (64_000, 32_000, 16_000, 8_000, 4_000, 2_000,
                                   1_000, 500, 250, 125),
    n_probes: int = 160,
    compression: str = "zstd",
    seed: int = 23,
) -> BlockSizeChoice:
    """Find the block size that minimises expected bytes read per region query.

    Pass ``queries`` as ``(start, end)`` pairs from a real workload for a true
    forecast: the estimate then averages over the real positions *and* the real
    widths. Failing that, ``probes`` sets the positions and ``query_width`` a
    single width. With neither, probes are drawn from the table's own row
    starts, which matches the density profile genomic workloads tend to have.

    Nothing is written except a few small samples for the cost fit, so the whole
    search costs one sort of the busiest contig.
    """
    rng = np.random.default_rng(seed)
    wanted = tuple(sorted({int(c) for c in candidates} | {int(current_rows_per_block)}))
    cost = fit_block_cost(tbl, sample_sizes=wanted, compression=compression)

    counts = (pa.table({"contig": tbl.column("contig")})
              .group_by("contig").aggregate([([], "count_all")]))
    names = counts.column("contig").to_pylist()
    sizes = counts.column("count_all").to_pylist()
    contig = names[int(np.argmax(sizes))]

    sub = tbl.filter(pc.equal(tbl.column("contig"), contig))
    sub = sub.take(pc.sort_indices(sub, sort_keys=[("start", "ascending")]))
    starts = np.asarray(sub.column("start")).astype(np.int64)
    ends = np.asarray(sub.column("end")).astype(np.int64)
    if starts.size == 0:
        raise ValueError("table has no rows on its busiest contig")

    q_start = q_end = None
    if queries:
        # The most faithful forecast: the real intervals, widths and all.
        q_start = np.array([a for a, _ in queries], dtype=np.int64)
        q_end = np.array([b for _, b in queries], dtype=np.int64)
    if probes is not None and len(probes):
        probe = np.asarray(probes, dtype=np.int64)
    else:
        # Probe where the queries actually go, which is where the rows are.
        # Sampling uniformly across the contig is wrong in a way that matters:
        # genomic queries cluster on genes, genes sit in record-dense regions,
        # and dense regions have narrower blocks, so a uniform probe set
        # systematically understates how many blocks a real query touches.
        # Drawing probes from the row starts themselves weights by density for
        # free; pass `probes` to use a real workload instead.
        probe = rng.choice(starts, size=min(n_probes, starts.size), replace=False)

    def bytes_for(rpb: int) -> tuple[float, float]:
        s_min, _s_max, e_max = simulate_blocks(starts, ends, rpb)
        if q_start is not None:
            blocks = _blocks_for_queries(s_min, e_max, q_start, q_end)
        else:
            blocks = _expected_blocks(s_min, e_max, query_width, probe)
        return blocks, blocks * cost.block_bytes(rpb)

    curve: list[tuple[int, float, float]] = []
    best = (current_rows_per_block, float("inf"))
    for rpb in sorted(set(candidates) | {current_rows_per_block}, reverse=True):
        if rpb < 1 or rpb > sub.num_rows:
            continue
        blocks, total = bytes_for(rpb)
        curve.append((rpb, round(blocks, 3), total))
        if total < best[1]:
            best = (rpb, total)

    _, current_bytes = bytes_for(current_rows_per_block)
    curve.sort(key=lambda c: -c[0])

    # Relative spread of the block-byte samples, used as the noise floor.
    # With paired anchors the meaningful uncertainty is in the ratio between
    # adjacent sizes, not in their absolute levels, so measure it that way.
    noise = 0.0
    sizes_m = sorted(cost.measured)
    ratios = []
    for a, b in zip(sizes_m, sizes_m[1:]):
        va = [y for r, y in cost.samples if r == a]
        vb = [y for r, y in cost.samples if r == b]
        if len(va) == len(vb) and len(va) > 1:
            per = np.array(vb, dtype=float) / np.maximum(np.array(va, dtype=float), 1.0)
            if per.mean() > 0:
                ratios.append(float(np.std(per, ddof=1)) / per.mean() / np.sqrt(len(per)))
    if ratios:
        noise = float(np.mean(ratios))

    # Candidates inside the noise band are reported as ties so the operator can
    # see the curve is flat there. The recommendation itself stays the minimum:
    # preferring the larger of a "tied" pair to save files cost 16.6% of the
    # available read saving on one of our real tables, which is not a trade
    # worth making silently.
    band = best[1] * (1.0 + max(noise, 0.02))
    tied = sorted({r for r, _b, by in curve if by <= band})
    chosen, chosen_bytes = best[0], best[1]

    return BlockSizeChoice(
        rows_per_block=chosen, expected_bytes=chosen_bytes,
        tied=tied, noise=noise,
        current_rows_per_block=current_rows_per_block,
        current_expected_bytes=current_bytes,
        query_width=query_width, cost=cost, curve=curve,
    )
