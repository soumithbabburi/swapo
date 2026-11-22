"""SWaPO: the decision layer.

It looks at four things together, which is the only claim this project makes
that the shipping alternatives do not already cover:

1. **what the queries mean** — gene lookups and whole-arm sweeps are different
   problems even though they filter the same columns (:mod:`swapo.workload`);
2. **how the data is shaped** — in particular the interval-length tail, which is
   what decides whether coordinate statistics can skip anything at all
   (:mod:`swapo.zonemap`);
3. **how often the data changes** — a layout that must be rebuilt after every
   cohort delivery can cost more than it saves;
4. **what depends on it** — a materialized projection carries a refresh
   obligation and a staleness risk (:mod:`swapo.provenance`).

It runs in recommendation mode. Every recommendation states the expected
benefit, the cost of getting it, what it would touch, how it affects freshness,
and how to undo it. Nothing is applied without an explicit call, and after a
change is applied the predicted and measured savings are compared so the next
estimate is better calibrated.

The estimates are what-if estimates (see :mod:`swapo.cost`). They are checked
against measurement in ``swapo.bench``; treat the predicted numbers as a ranking
signal, not as a result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .objstore import StoreCost
from .rowgroups import choose_row_group_size
from .staleness import CALIBRATED_1KG_CHR21, flip_probability
from .cost import CostModel, LayoutShape, expected_blocks
from .data import SV_THRESHOLD
from .model import Intent, TableStats
from .provenance import DatasetKind, Lineage
from .workload import WorkloadProfile
from .zonemap import IntervalZoneMap

__all__ = ["Action", "Recommendation", "SWaPO"]


class Action(str, Enum):
    NOTHING = "do_nothing"
    SPLIT_LONG_INTERVALS = "split_long_intervals"
    PARTITION_BY_CONTIG = "partition_by_contig"
    SORT_BY_POSITION = "sort_by_position"
    MATERIALIZE_SAMPLE_PROJECTION = "materialize_sample_projection"
    MATERIALIZE_COHORT_FREQ = "materialize_cohort_frequency"
    COMPACT = "compact_small_blocks"
    RESIZE_ROW_GROUPS = "resize_row_groups"


@dataclass
class Recommendation:
    """One proposed change, with everything an operator needs to judge it."""

    action: Action
    table: str
    rationale: str
    #: Expected reduction in bytes read per query, as a fraction of current.
    expected_read_reduction: float
    benefit: float
    rewrite_cost: float
    storage_cost: float
    freshness_cost: float = 0.0
    #: benefit - every cost. Only positive values are worth doing.
    net: float = 0.0
    confidence: float = 0.5
    affected: list[str] = field(default_factory=list)
    freshness_impact: str = "none: this rearranges existing rows only"
    rollback: str = ""
    params: dict = field(default_factory=dict)

    def explain(self) -> str:
        lines = [
            f"{self.action.value} on {self.table}",
            f"  why            : {self.rationale}",
            f"  expected gain  : {self.expected_read_reduction * 100:.0f}% less read cost per query",
            f"  benefit        : {self.benefit:,.1f}",
            f"  rewrite cost   : {self.rewrite_cost:,.1f}",
            f"  storage cost   : {self.storage_cost:,.1f}",
        ]
        if self.freshness_cost:
            lines.append(f"  freshness cost : {self.freshness_cost:,.1f}")
        lines += [
            f"  net            : {self.net:,.1f}",
            f"  confidence     : {self.confidence:.2f}",
            f"  affects        : {', '.join(self.affected) or 'nothing downstream'}",
            f"  freshness      : {self.freshness_impact}",
            f"  rollback       : {self.rollback or 'rewrite in the previous layout'}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["action"] = self.action.value
        return d


@dataclass
class Observation:
    """Everything the optimizer was told, kept so a decision can be re-examined."""

    profile: WorkloadProfile
    stats: TableStats
    shape: LayoutShape
    zonemap: IntervalZoneMap
    updates_per_horizon: float = 1.0
    #: Queries of this shape expected per horizon window.
    volume: float = 1.0
    #: The observed queries themselves, when available.
    queries: list = field(default_factory=list)
    #: Fractional growth of the cohort per data update (0.1 = 10% more samples).
    cohort_growth: float = 0.1


class SWaPO:
    """Semantic Workload and Provenance Optimizer."""

    def __init__(self, cost: CostModel | None = None,
                 lineage: Lineage | None = None,
                 long_threshold: int = SV_THRESHOLD,
                 store: StoreCost | None = None) -> None:
        self.cost = cost or CostModel()
        #: Request latency and bandwidth of the storage the table lives on.
        #: Measure yours with swapo.objstore.measure_remote; this default is
        #: what we measured against GCS from a laptop.
        self.store = store or StoreCost(0.170, 5.2e6, "measured: GCS from a laptop")
        self.lineage = lineage or Lineage()
        self.long_threshold = long_threshold
        #: Multiplied into every benefit estimate. Starts at 1.0 and is moved
        #: towards reality by :meth:`record_outcome`.
        self.calibration: float = 1.0
        self._outcomes: list[tuple[float, float]] = []

    # -- the entry point ----------------------------------------------------

    def recommend(
        self,
        table: str,
        profile: WorkloadProfile,
        stats: TableStats,
        zonemap: IntervalZoneMap,
        *,
        updates_per_horizon: float = 1.0,
        query_volume: float | None = None,
        tbl=None,
        observed_queries: list | None = None,
        current_rg_rows: int = 1_048_576,
        file_rows: int = 4_194_304,
        cohort_growth_per_update: float = 0.1,
    ) -> list[Recommendation]:
        """Rank the actions worth taking on ``table``, best first.

        An empty list means do nothing, which is a real answer and the right one
        far more often than an optimizer that must always act would suggest.
        """
        shape = LayoutShape.from_zonemap(zonemap)
        volume = query_volume if query_volume is not None else max(profile.n_queries, 1)
        obs = Observation(profile, stats, shape, zonemap, updates_per_horizon, volume,
                          list(observed_queries or []), cohort_growth_per_update)
        current = self.cost.scan_cost(profile, shape, volume)

        candidates = [
            self._consider_row_groups(table, obs, current, tbl, current_rg_rows, file_rows),
            self._consider_split_long(table, obs, current),
            self._consider_partition(table, obs, current),
            self._consider_sample_projection(table, obs, current),
            self._consider_cohort_freq(table, obs, current),
            self._consider_compact(table, obs, current),
        ]
        recs = [r for r in candidates if r is not None and r.net > 0]
        recs.sort(key=lambda r: r.net * r.confidence, reverse=True)
        return recs

    # -- individual actions -------------------------------------------------

    def _consider_row_groups(self, table: str, obs: Observation, current: float,
                             tbl, current_rg_rows: int,
                             file_rows: int) -> Recommendation | None:
        """Resize Parquet row groups to what the real queries cost on this storage.

        The lever that pays on real genomic tables. Row groups are left at a
        writer default (1,048,576 rows for pyarrow, pyiceberg and delta-rs),
        so a single-gene query fetches megabytes to return kilobytes. Files are
        not touched. Needs the table and the real query intervals.
        """
        queries = [(q.region.contig, q.region.start, q.region.end)
                   for q in obs.queries if q.region is not None]
        if tbl is None or not queries:
            return None
        try:
            choice = choose_row_group_size(tbl, queries, self.store,
                                           current_rows=current_rg_rows,
                                           file_rows=file_rows)
        except (ValueError, KeyError):
            return None
        if choice.improvement <= 0.05:
            return None
        benefit = current * choice.improvement * self.calibration
        rewrite = self.cost.rewrite_cost(obs.stats.n_bytes)
        b, c = choice.best, choice.current
        return Recommendation(
            action=Action.RESIZE_ROW_GROUPS, table=table,
            rationale=(
                f"Row groups hold {current_rg_rows:,} rows, so the real queries "
                f"fetch {c.bytes / 1e6:.2f} MB in {c.requests:.1f} requests each. At "
                f"{choice.rg_rows:,} rows they fetch {b.bytes / 1e6:.2f} MB in "
                f"{b.requests:.1f} requests: {c.seconds * 1e3:.0f} ms to "
                f"{b.seconds * 1e3:.0f} ms on {self.store.source}."),
            expected_read_reduction=choice.improvement, benefit=benefit,
            rewrite_cost=rewrite, storage_cost=0.0,
            net=benefit - rewrite, confidence=0.85,
            freshness_impact="none: the same rows and files, different row groups",
            rollback=(f"rewrite at {current_rg_rows:,} rows per row group, or read "
                      "the previous table version"),
            params={"rows_per_row_group": choice.rg_rows,
                    "ms_before": round(c.seconds * 1e3, 1),
                    "ms_after": round(b.seconds * 1e3, 1)},
        )

    def _consider_split_long(self, table: str, obs: Observation,
                             current: float) -> Recommendation | None:
        """Separate long records so they stop inflating ``max(end)``.

        This is the action the whole project is built around, so it is also the
        one that must decline most carefully: when the table has no long tail,
        or when nobody asks interval questions, it is pure cost.
        """
        shape, stats, profile = obs.shape, obs.stats, obs.profile
        if profile.interval_share < 0.15:
            return None
        if stats.long_fraction <= 0 or stats.max_length <= self.long_threshold:
            return None
        # Already tight: the long records are not mixed into the dense blocks.
        if shape.reach <= shape.span * 1.5:
            return None

        # What-if: with long records removed, a dense block reaches only as far
        # as the longest short record beyond its last start.
        after = LayoutShape(
            n_blocks=shape.n_blocks, block_bytes=shape.block_bytes,
            span=shape.span, reach=shape.span + self.long_threshold,
            blocks_per_contig=shape.blocks_per_contig,
            mixed_contigs=shape.mixed_contigs,
        )
        projected = self.cost.scan_cost(profile, after, obs.volume)
        # The long records still have to be read when they genuinely overlap.
        projected += current * stats.long_fraction
        benefit = (current - projected) * self.calibration
        if benefit <= 0:
            return None

        rewrite = self.cost.rewrite_cost(stats.n_bytes)
        reduction = 1.0 - (projected / current) if current else 0.0
        # Confidence is highest when the tail is clearly present but small: that
        # is the regime the model was built for.
        conf = 0.85 if 1e-4 < stats.long_fraction < 0.05 else 0.6

        return Recommendation(
            action=Action.SPLIT_LONG_INTERVALS, table=table,
            rationale=(
                f"{stats.long_fraction * 100:.2f}% of rows are longer than "
                f"{self.long_threshold} bp, the longest is {stats.max_length:,} bp, and "
                f"blocks reach {shape.reach / max(shape.span, 1):.0f}x further than the "
                f"range of starts they hold. Every interval query therefore reads blocks "
                f"whose rows all end before it. {profile.interval_share * 100:.0f}% of "
                f"queries are interval queries."
            ),
            expected_read_reduction=reduction, benefit=benefit,
            rewrite_cost=rewrite, storage_cost=0.0,
            net=benefit - rewrite, confidence=conf,
            freshness_impact="none: the same rows, written to different blocks",
            rollback="rewrite without the long/short split; no derived data is affected",
            params={"long_threshold": self.long_threshold},
        )

    def _consider_partition(self, table: str, obs: Observation,
                            current: float) -> Recommendation | None:
        """Give each contig its own blocks so contig pruning can work."""
        shape, profile, stats = obs.shape, obs.profile, obs.stats
        if not shape.mixed_contigs or profile.interval_share < 0.15:
            return None
        n_contigs = max(len(profile.contigs), 1)
        after = LayoutShape(
            n_blocks=shape.n_blocks, block_bytes=shape.block_bytes,
            span=shape.span, reach=shape.reach,
            blocks_per_contig=max(1, shape.n_blocks // n_contigs),
            mixed_contigs=False,
        )
        projected = self.cost.scan_cost(profile, after, obs.volume)
        benefit = (current - projected) * self.calibration
        rewrite = self.cost.rewrite_cost(stats.n_bytes)
        if benefit <= 0:
            return None
        return Recommendation(
            action=Action.PARTITION_BY_CONTIG, table=table,
            rationale=(
                f"Blocks span more than one contig, so no block can be skipped on "
                f"contig alone. The workload names a contig in "
                f"{profile.interval_share * 100:.0f}% of queries."
            ),
            expected_read_reduction=1.0 - (projected / current) if current else 0.0,
            benefit=benefit, rewrite_cost=rewrite, storage_cost=0.0,
            net=benefit - rewrite, confidence=0.8,
            freshness_impact="none: the same rows, written to different blocks",
            rollback="rewrite as a single sorted sequence",
        )

    def _consider_sample_projection(self, table: str, obs: Observation,
                                    current: float) -> Recommendation | None:
        """A sample-ordered copy, for workloads dominated by per-sample extraction.

        This is a derived dataset, so unlike the layout actions it carries a
        refresh obligation and a staleness risk, and it is charged for both.
        """
        profile, stats = obs.profile, obs.stats
        share = profile.share(Intent.SAMPLE)
        if share < 0.15:
            return None
        # A sample-ordered copy turns a full scan per sample into a small read.
        saved_fraction = share * 0.9
        benefit = current * saved_fraction * self.calibration
        rewrite = self.cost.rewrite_cost(stats.n_bytes)
        storage = self.cost.storage_cost(stats.n_bytes)
        refresh = self.cost.refresh_cost(stats.n_bytes, obs.updates_per_horizon)
        # Between a cohort landing and the refresh finishing, answers from the
        # projection are behind the base table.
        stale_fraction = min(0.5, obs.updates_per_horizon / self.cost.horizon_windows)
        stale = self.cost.staleness_cost(obs.volume * share, stale_fraction)
        freshness = refresh + stale
        net = benefit - rewrite - storage - freshness
        downstream = self.lineage.descendants(table) if table in self.lineage else []
        return Recommendation(
            action=Action.MATERIALIZE_SAMPLE_PROJECTION, table=table,
            rationale=(
                f"{share * 100:.0f}% of queries extract one sample, which in a "
                f"coordinate-ordered table means touching every block. A sample-ordered "
                f"copy answers them from a few. It is a second copy of the data and it "
                f"goes stale on every cohort delivery, so both are charged here."
            ),
            expected_read_reduction=saved_fraction, benefit=benefit,
            rewrite_cost=rewrite, storage_cost=storage, freshness_cost=freshness,
            net=net, confidence=0.55,
            affected=downstream,
            freshness_impact=(
                f"creates a derived dataset that must be refreshed after each of the "
                f"~{obs.updates_per_horizon:.0f} expected updates; queries served from it "
                f"between delivery and refresh are behind the base table"
            ),
            rollback="drop the projection and route queries back to the base table",
        )

    def _consider_cohort_freq(self, table: str, obs: Observation,
                              current: float) -> Recommendation | None:
        """Precomputed allele frequencies, the classic stale-science hazard."""
        profile, stats = obs.profile, obs.stats
        share = profile.share(Intent.COHORT_FREQ)
        if share < 0.1:
            return None
        benefit = current * share * 0.95 * self.calibration
        # The aggregate is far smaller than the genotypes it came from.
        agg_bytes = stats.n_bytes * 0.02
        rewrite = self.cost.rewrite_cost(stats.n_bytes)   # it reads everything to build
        storage = self.cost.storage_cost(agg_bytes)
        refresh = self.cost.refresh_cost(stats.n_bytes, obs.updates_per_horizon)
        # A stale allele frequency is a scientific error, not a slow query. The
        # chance that a stale answer is in the WRONG decision class is measured,
        # not assumed: 1000 Genomes phase 1 -> phase 3 (swapo.staleness). The
        # cost of one wrong class is the operator's staleness_penalty.
        stale_fraction = min(0.6, obs.updates_per_horizon / self.cost.horizon_windows)
        p_wrong = flip_probability(obs.cohort_growth, CALIBRATED_1KG_CHR21)
        stale = self.cost.staleness_cost(obs.volume * share, stale_fraction * p_wrong)
        freshness = refresh + stale
        net = benefit - rewrite - storage - freshness
        return Recommendation(
            action=Action.MATERIALIZE_COHORT_FREQ, table=table,
            rationale=(
                f"{share * 100:.0f}% of queries recompute allele frequency over the "
                f"cohort. Materializing it is cheap to store but it is a scientific "
                f"claim with a date on it. With the cohort growing "
                f"{obs.cohort_growth * 100:.0f}% per update, a stale answer is in the "
                f"wrong frequency class {p_wrong * 100:.1f}% of the time (measured on "
                f"1000 Genomes chr21, phase 1 to phase 3)."
            ),
            expected_read_reduction=share * 0.95, benefit=benefit,
            rewrite_cost=rewrite, storage_cost=storage, freshness_cost=freshness,
            net=net, confidence=0.5,
            affected=self.lineage.descendants(table) if table in self.lineage else [],
            freshness_impact=(
                "creates a derived dataset whose values change with every cohort "
                "delivery; it must carry the cohort version it was computed from"
            ),
            rollback="drop the table and compute frequencies on demand",
        )

    def _consider_compact(self, table: str, obs: Observation,
                          current: float) -> Recommendation | None:
        """Merge blocks that have become too small to be worth opening."""
        shape, stats = obs.shape, obs.stats
        if shape.n_blocks < 8 or shape.block_bytes > 4e6:
            return None
        target = 16e6
        factor = min(8.0, target / max(shape.block_bytes, 1))
        if factor < 2:
            return None
        benefit = current * (1 - 1 / factor) * 0.25 * self.calibration
        rewrite = self.cost.rewrite_cost(stats.n_bytes)
        return Recommendation(
            action=Action.COMPACT, table=table,
            rationale=(
                f"Mean block is {shape.block_bytes / 1e6:.1f} MB across "
                f"{shape.n_blocks} blocks. Per-file overhead dominates; merging to "
                f"about {target / 1e6:.0f} MB cuts the number of opens."
            ),
            expected_read_reduction=0.0, benefit=benefit,
            rewrite_cost=rewrite, storage_cost=0.0,
            net=benefit - rewrite, confidence=0.45,
            freshness_impact="none: the same rows, fewer blocks",
            rollback="rewrite at the previous block size",
        )

    # -- closing the loop ---------------------------------------------------

    def record_outcome(self, rec: Recommendation, measured_reduction: float) -> float:
        """Compare a prediction with what happened and move the calibration.

        Without this the optimizer would keep making the same mistake
        confidently. The adjustment is deliberately slow, so one unusual table
        cannot swing the model.
        """
        predicted = rec.expected_read_reduction
        if predicted <= 0:
            return self.calibration
        ratio = max(0.05, min(4.0, measured_reduction / predicted))
        self._outcomes.append((predicted, measured_reduction))
        self.calibration = math.exp(
            0.75 * math.log(self.calibration) + 0.25 * math.log(ratio * self.calibration)
        )
        self.calibration = max(0.1, min(5.0, self.calibration))
        return self.calibration

    @property
    def outcomes(self) -> list[tuple[float, float]]:
        return list(self._outcomes)
