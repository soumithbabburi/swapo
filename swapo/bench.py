"""The benchmark: build every layout, run the same workload, measure, compare.

Rules this harness follows, because a benchmark that flatters its author is
worth nothing:

* every layout holds exactly the same rows and the same target block size;
* every layout is pruned with the same two-sided overlap test, using its own
  statistics. The baselines are not handicapped by being denied ``max(end)``;
* every query is checked against an unpruned scan, so a layout cannot look fast
  by returning fewer rows;
* timings are the median of repeated runs, and bytes read are counted from the
  files actually opened rather than estimated.

The headline comparison is ``PARTITIONED`` against ``SWAPO``. They differ in one
decision only: whether long records are written to blocks of their own. Any gap
between them is attributable to that and nothing else.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from .cost import CostModel, LayoutShape
from .data import DatasetSpec, simulate_features, simulate_genes, simulate_samples
from .engine import Engine, QueryResult
from .layout import Layout, LayoutResult, write_layout
from .model import INTERVAL_INTENTS, Intent, Query, Region
from .optimizer import SWaPO
from .provenance import Dataset, DatasetKind, Lineage
from .stats import length_histogram, summarise
from .workload import DEFAULT_MIX, DRIFTED_MIX, WorkloadLog, generate_workload

__all__ = ["BenchResult", "run_benchmark", "run_sweep"]


@dataclass
class LayoutScore:
    layout: str
    median_ms: float
    mean_ms: float
    total_bytes: int
    total_blocks: int
    blocks_available: int
    write_seconds: float
    store_bytes: int
    by_intent: dict[str, dict] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "layout": self.layout,
            "median_query_ms": round(self.median_ms, 2),
            "mean_query_ms": round(self.mean_ms, 2),
            "gigabytes_read_total": round(self.total_bytes / 1e9, 3),
            "mean_blocks_read": round(self.total_blocks / max(len(self.by_intent), 1), 1),
            "store_megabytes": round(self.store_bytes / 1e6, 1),
            "write_seconds": round(self.write_seconds, 2),
        }


@dataclass
class BenchResult:
    scores: dict[str, LayoutScore]
    dataset: dict
    workload: dict
    recommendations: list[dict]
    verification: dict
    drift: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "dataset": self.dataset,
            "workload": self.workload,
            "layouts": {k: v.summary() for k, v in self.scores.items()},
            "per_intent": {k: v.by_intent for k, v in self.scores.items()},
            "recommendations": self.recommendations,
            "verification": self.verification,
            "drift": self.drift,
            "calibration": self.calibration,
        }, indent=2), encoding="utf-8")


def _run_workload(engine: Engine, result: LayoutResult, queries: list[Query],
                  repeats: int) -> LayoutScore:
    times: list[float] = []
    total_bytes = 0
    total_blocks = 0
    by_intent: dict[str, dict] = {}

    for q in queries:
        runs = []
        res: QueryResult | None = None
        for _ in range(repeats):
            res = engine.run(q, result.zonemap)
            runs.append(res.seconds)
        assert res is not None
        t = statistics.median(runs)
        times.append(t)
        total_bytes += res.bytes_read
        total_blocks += res.blocks_read
        slot = by_intent.setdefault(q.intent.value, {
            "n": 0, "ms": 0.0, "bytes": 0, "blocks": 0})
        slot["n"] += 1
        slot["ms"] += t * 1000
        slot["bytes"] += res.bytes_read
        slot["blocks"] += res.blocks_read

    for slot in by_intent.values():
        n = slot["n"]
        slot["mean_ms"] = round(slot["ms"] / n, 2)
        slot["mean_megabytes"] = round(slot["bytes"] / n / 1e6, 2)
        slot["mean_blocks"] = round(slot["blocks"] / n, 1)
        del slot["ms"]

    return LayoutScore(
        layout=result.layout.value,
        median_ms=statistics.median(times) * 1000,
        mean_ms=statistics.fmean(times) * 1000,
        total_bytes=total_bytes,
        total_blocks=total_blocks,
        blocks_available=len(result.zonemap),
        write_seconds=result.write_seconds,
        store_bytes=result.n_bytes,
        by_intent=by_intent,
    )


def run_sweep(
    root: str | Path = "bench_out/sweep",
    *,
    n_features: int = 3_000_000,
    long_fractions: tuple[float, ...] = (0.0, 0.0005, 0.004, 0.02),
    block_sizes: tuple[int, ...] = (6_000, 25_000),
    n_queries: int = 45,
    query_width: int = 50_000,
    verbose: bool = True,
) -> list[dict]:
    """Measure where the long-interval split helps, and where it does not.

    Two knobs are swept, because the benefit is a function of both and quoting a
    single number would be misleading:

    ``long_fraction``
        With no long records there is nothing to separate and SWaPO should match
        the baseline exactly. That row of the table is the control.

    ``rows_per_block``
        Smaller blocks prune better in principle. The point of the sweep is that
        they only do so once ``max(end)`` is tight: otherwise shrinking blocks
        buys far less than it should.
    """
    import numpy as np

    from .model import GRCH38

    root = Path(root)
    say = print if verbose else (lambda *a, **k: None)
    rng = np.random.default_rng(17)
    rows: list[dict] = []
    engine = Engine()

    for lf in long_fractions:
        spec = DatasetSpec(n_features=n_features, long_fraction=lf)
        features = simulate_features(spec)
        queries: list[Query] = []
        for c in spec.contigs:
            L = GRCH38[c]
            queries += [
                Query(Intent.INTERVAL, region=Region(c, int(p), int(p) + query_width))
                for p in rng.integers(1_000_000, L - 3_000_000, n_queries // len(spec.contigs))
            ]
        for rpb in block_sizes:
            measured: dict[str, dict] = {}
            for lay in (Layout.PARTITIONED, Layout.SWAPO):
                lr = write_layout(features, root / f"lf{lf}-rpb{rpb}-{lay.value}",
                                  lay, rows_per_block=rpb)
                tb = tk = 0
                for q in queries:
                    r = engine.run(q, lr.zonemap)
                    tb += r.bytes_read
                    tk += r.blocks_read
                measured[lay.value] = {
                    "blocks_per_contig": lr.n_blocks // len(spec.contigs),
                    "tightness": round(lr.zonemap.tightness(), 2),
                    "mean_blocks_read": round(tk / len(queries), 2),
                    "mean_megabytes_read": round(tb / len(queries) / 1e6, 3),
                    "store_megabytes": round(lr.n_bytes / 1e6, 1),
                }
            base = measured["partitioned"]["mean_megabytes_read"]
            new = measured["swapo"]["mean_megabytes_read"]
            row = {
                "long_fraction": lf,
                "rows_per_block": rpb,
                "partitioned": measured["partitioned"],
                "swapo": measured["swapo"],
                "read_reduction": round(1 - new / base, 3) if base else 0.0,
                "speedup_x": round(base / new, 2) if new else float("inf"),
            }
            rows.append(row)
            say(f"  long_fraction={lf:<7} rows/block={rpb:<6} "
                f"partitioned {base:6.3f} MB -> swapo {new:6.3f} MB  "
                f"({row['read_reduction'] * 100:5.1f}% less, {row['speedup_x']}x)")
    engine.close()
    return rows


def run_benchmark(
    root: str | Path = "bench_out",
    *,
    spec: DatasetSpec | None = None,
    n_queries: int = 120,
    repeats: int = 3,
    rows_per_block: int = 8_000,
    layouts: list[Layout] | None = None,
    verbose: bool = True,
) -> BenchResult:
    """Build the data, write every layout, run the workload, and compare."""
    spec = spec or DatasetSpec()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    layouts = layouts or [Layout.RANDOM, Layout.SORTED, Layout.PARTITIONED, Layout.SWAPO]
    say = print if verbose else (lambda *a, **k: None)

    # -- data ---------------------------------------------------------------
    say(f"Simulating {spec.n_features:,} features "
        f"(long tail {spec.long_fraction * 100:.2f}%) ...")
    features = simulate_features(spec)
    genes = simulate_genes(spec)
    samples = simulate_samples(spec)
    gene_path = root / "genes.parquet"
    pq.write_table(genes, gene_path)

    stats = summarise(features)
    say(f"  {stats.n_rows:,} rows, longest interval {stats.max_length:,} bp, "
        f"{stats.long_fraction * 100:.3f}% long, contig skew {stats.contig_skew:.2f}")

    # -- provenance ---------------------------------------------------------
    lineage = Lineage()
    lineage.register(Dataset("features", DatasetKind.BASE, n_rows=features.num_rows,
                             activity="simulate", assembly=spec.assembly.value))

    # -- workload -----------------------------------------------------------
    gene_pairs = [(s, Region(c, int(a), int(b))) for s, c, a, b in zip(
        genes.column("symbol").to_pylist(), genes.column("contig").to_pylist(),
        genes.column("start").to_pylist(), genes.column("end").to_pylist())][:4000]
    sample_ids = samples.column("sample_id").to_pylist()
    queries = generate_workload(n_queries, genes=gene_pairs, samples=sample_ids,
                                mix=DEFAULT_MIX, contigs=spec.contigs, seed=5)
    # Only the feature-table intents are timed here; the genotype intents are
    # reported separately because they run against a different table.
    feature_queries = [q for q in queries if q.intent in INTERVAL_INTENTS
                       or q.intent is Intent.SCAN]
    log = WorkloadLog()
    log.extend(feature_queries)
    profile = log.profile()
    say(f"  workload: {len(feature_queries)} feature queries, "
        f"median width {profile.median_width:,} bp")

    # -- layouts ------------------------------------------------------------
    results: dict[str, LayoutResult] = {}
    scores: dict[str, LayoutScore] = {}
    engine = Engine(gene_table=gene_path)

    for lay in layouts:
        say(f"Writing layout {lay.value} ...")
        lr = write_layout(features, root / lay.value, lay,
                          rows_per_block=rows_per_block)
        results[lay.value] = lr
        say(f"  {lr.n_blocks} blocks, {lr.n_bytes / 1e6:.0f} MB, "
            f"tightness {lr.zonemap.tightness():.1f}x, "
            f"written in {lr.write_seconds:.1f}s")
        lineage.register(Dataset(f"features@{lay.value}", DatasetKind.LAYOUT,
                                 inputs=["features"], n_rows=lr.n_rows,
                                 n_bytes=lr.n_bytes, activity="write_layout",
                                 assembly=spec.assembly.value,
                                 params={"layout": lay.value}))
        say(f"  running {len(feature_queries)} queries x{repeats} ...")
        scores[lay.value] = _run_workload(engine, lr, feature_queries, repeats)

    # -- correctness --------------------------------------------------------
    # Pruning that drops a matching row is a bug, not an optimization. Check
    # every layout returns exactly what an unpruned scan would.
    say("Verifying pruning never drops a matching row ...")
    checked = 0
    mismatches: list[dict] = []
    base = results[layouts[0].value]
    for q in feature_queries:
        if q.region is None and q.intent is not Intent.GENE:
            continue
        truth = engine.count_matching(q, base.zonemap)
        for name, lr in results.items():
            got = engine.run(q, lr.zonemap).rows
            checked += 1
            if got != truth:
                mismatches.append({"layout": name, "expected": truth, "got": got,
                                   "region": str(q.region), "intent": q.intent.value})
    say(f"  {checked} layout/query checks, {len(mismatches)} mismatches")

    # -- what the optimizer would have said ---------------------------------
    opt = SWaPO(CostModel(), lineage)
    recs: list[dict] = []
    for name in ("random", "sorted", "partitioned", "swapo"):
        if name not in results:
            continue
        lr = results[name]
        st = summarise(features, n_bytes=lr.n_bytes, n_blocks=lr.n_blocks)
        for r in opt.recommend(f"features@{name}", profile, st, lr.zonemap,
                               updates_per_horizon=2.0):
            recs.append({"from_layout": name, **r.to_dict()})

    # -- did the prediction hold? -------------------------------------------
    calibration: dict = {}
    if "partitioned" in scores and "swapo" in scores:
        a, b = scores["partitioned"], scores["swapo"]
        measured = 1 - (b.total_bytes / a.total_bytes) if a.total_bytes else 0.0
        predicted = next((r["expected_read_reduction"] for r in recs
                          if r["from_layout"] == "partitioned"
                          and r["action"] == "split_long_intervals"), None)
        if predicted is not None:
            before = opt.calibration
            lr_rec = next(r for r in opt.recommend(
                "features@partitioned",
                profile,
                summarise(features, n_bytes=results["partitioned"].n_bytes,
                          n_blocks=results["partitioned"].n_blocks),
                results["partitioned"].zonemap, updates_per_horizon=2.0)
                if r.action.value == "split_long_intervals")
            after = opt.record_outcome(lr_rec, measured)
            calibration = {
                "predicted_read_reduction": round(predicted, 3),
                "measured_read_reduction": round(measured, 3),
                "calibration_before": round(before, 3),
                "calibration_after": round(after, 3),
            }

    # -- workload drift -----------------------------------------------------
    drift: dict = {}
    if "swapo" in results:
        drifted = generate_workload(n_queries, genes=gene_pairs, samples=sample_ids,
                                    mix=DRIFTED_MIX, contigs=spec.contigs, seed=9)
        dlog = WorkloadLog()
        dlog.extend([q for q in drifted if q.intent in INTERVAL_INTENTS
                     or q.intent is Intent.SCAN])
        dprofile = dlog.profile()
        lr = results["swapo"]
        st = summarise(features, n_bytes=lr.n_bytes, n_blocks=lr.n_blocks)
        drift = {
            "drifted_profile": dprofile.summary(),
            "recommendations_after_drift": [
                r.to_dict() for r in opt.recommend("features@swapo", dprofile, st,
                                                   lr.zonemap, updates_per_horizon=6.0)
            ],
        }

    engine.close()
    lineage.save(root / "lineage.prov.json")

    return BenchResult(
        scores=scores,
        dataset={
            "rows": stats.n_rows,
            "long_fraction": stats.long_fraction,
            "max_interval_bp": stats.max_length,
            "contig_skew": round(stats.contig_skew, 2),
            "length_histogram": length_histogram(features),
            "assembly": spec.assembly.value,
            "seed": spec.seed,
        },
        workload=profile.summary(),
        recommendations=recs,
        verification={"checks": checked, "mismatches": mismatches,
                      "passed": not mismatches},
        drift=drift,
        calibration=calibration,
    )
