"""The benchmark on real public files, with real query regions.

:mod:`swapo.bench` runs on simulated data and can be made to say whatever the
simulator was told to say. This module runs the same comparison on files that
real pipelines hold, with three differences that matter:

* the interval-length distribution is whatever GENCODE, ClinVar and the 1000
  Genomes Project actually contain, not a draw from a chosen distribution;
* the query regions are **real gene coordinates** taken from the annotation,
  so the workload has the real clustering of genes along the genome rather than
  a uniform scatter;
* the split threshold is chosen by search over the real length distribution
  (:mod:`swapo.threshold`) rather than fixed, because the fixed value that
  seemed obvious for variants is wrong for annotation.

Every run records the SHA-256 of each input file, so a number in the paper can
be traced to the bytes it came from.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .data import read_vcf
from .engine import Engine
from .layout import Layout, write_layout
from .model import Intent, Query, Region
from .real import combine, describe_file, harmonise_contigs, read_gtf
from .stats import length_histogram, summarise
from .threshold import choose_threshold
from .workload import WorkloadLog

__all__ = ["RealDataset", "load_real", "run_real_benchmark"]


@dataclass
class RealDataset:
    table: pa.Table
    name: str
    sources: list[dict]
    contigs: tuple[str, ...]
    #: Real gene regions, for drawing a realistic workload.
    genes: list[tuple[str, Region]] = field(default_factory=list)


def load_real(
    data_dir: str | Path = "swapo_data",
    *,
    contigs: tuple[str, ...] = ("chr1",),
    use_gencode: bool = True,
    use_clinvar: bool = True,
    extra_vcf: str | Path | None = None,
    name: str = "gencode+clinvar",
) -> RealDataset:
    """Build a platform-shaped table from whichever real files are present."""
    data_dir = Path(data_dir)
    parts: list[pa.Table] = []
    sources: list[dict] = []
    genes: list[tuple[str, Region]] = []

    gtf = data_dir / "gencode.v44.basic.annotation.gtf.gz"
    if use_gencode and gtf.exists():
        g = read_gtf(gtf, contigs=contigs)
        parts.append(g)
        sources.append({**describe_file(gtf), "records": g.num_rows,
                        "role": "annotation (genes, transcripts, exons, CDS, UTR)"})
        # Real gene spans drive the query workload.
        only_genes = g.filter(pc.equal(g.column("kind"), "gene"))
        for c, s, e, gid in zip(only_genes.column("contig").to_pylist(),
                                only_genes.column("start").to_pylist(),
                                only_genes.column("end").to_pylist(),
                                only_genes.column("gene_id").to_pylist()):
            genes.append((gid, Region(c, int(s), int(e))))

    cv = data_dir / "clinvar.vcf.gz"
    if use_clinvar and cv.exists():
        v = harmonise_contigs(read_vcf(cv))
        v = v.filter(pc.is_in(v.column("contig"), value_set=pa.array(list(contigs))))
        parts.append(v)
        sources.append({**describe_file(cv), "records": v.num_rows,
                        "role": "clinically reported variants"})

    if extra_vcf:
        extra_vcf = Path(extra_vcf)
        x = harmonise_contigs(read_vcf(extra_vcf))
        x = x.filter(pc.is_in(x.column("contig"), value_set=pa.array(list(contigs))))
        parts.append(x)
        sources.append({**describe_file(extra_vcf), "records": x.num_rows,
                        "role": "population callset"})

    if not parts:
        raise FileNotFoundError(
            f"no real inputs found in {data_dir}. Run: python -m swapo fetch")

    table = combine(*parts, contigs=contigs)
    return RealDataset(table=table, name=name, sources=sources,
                       contigs=contigs, genes=genes)


def _real_workload(ds: RealDataset, n: int, seed: int) -> list[Query]:
    """Queries a variant-interpretation service would actually issue.

    Two thirds are gene lookups drawn from the real annotation, which means the
    query positions inherit the real clustering of genes; the rest are windows
    and single positions around those genes, which is how a clinical review
    actually moves around the genome.
    """
    rng = np.random.default_rng(seed)
    if not ds.genes:
        raise ValueError("no gene model loaded; cannot build a realistic workload")
    out: list[Query] = []
    for _ in range(n):
        gid, region = ds.genes[int(rng.integers(0, len(ds.genes)))]
        roll = rng.random()
        if roll < 0.45:
            out.append(Query(Intent.GENE, region=region, gene=gid))
        elif roll < 0.75:
            mid = (region.start + region.end) // 2
            half = int(10 ** rng.uniform(3.3, 4.8))
            out.append(Query(Intent.INTERVAL,
                             region=Region(region.contig, max(0, mid - half), mid + half)))
        else:
            p = int(rng.integers(region.start, max(region.start + 1, region.end)))
            out.append(Query(Intent.POINT, region=Region(region.contig, p, p + 1)))
    return out


def run_real_benchmark(
    data_dir: str | Path = "swapo_data",
    out: str | Path = "real_out",
    *,
    contigs: tuple[str, ...] = ("chr1",),
    rows_per_block: int = 4_000,
    n_queries: int = 150,
    repeats: int = 3,
    extra_vcf: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Load real files, choose a threshold, write both layouts, measure, verify."""
    say = print if verbose else (lambda *a, **k: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    ds = load_real(data_dir, contigs=contigs, extra_vcf=extra_vcf)
    stats = summarise(ds.table)
    say(f"Real table: {ds.table.num_rows:,} rows on {', '.join(contigs)}")
    for s in ds.sources:
        say(f"  {s['file']}  {s['records']:,} records  sha256 {s['sha256'][:16]}...")
    say(f"  longest interval {stats.max_length:,} bp; "
        f"{stats.long_fraction * 100:.1f}% longer than 50 bp")

    queries = _real_workload(ds, n_queries, seed=31)
    log = WorkloadLog()
    log.extend(queries)
    profile = log.profile()
    say(f"  workload: {len(queries)} real gene-derived queries, "
        f"median width {profile.median_width:,} bp")

    # The threshold is chosen from this table, at this block size, for this
    # workload. It is not a constant and it is not assumed.
    choice = choose_threshold(ds.table, rows_per_block=rows_per_block,
                              query_width=max(profile.median_width, 1))
    say(f"  chosen split threshold: {choice.summary()}")

    layouts = {}
    for lay in (Layout.PARTITIONED, Layout.SWAPO):
        kw = {"long_threshold": choice.threshold} if lay is Layout.SWAPO else {}
        lr = write_layout(ds.table, out / lay.value, lay,
                          rows_per_block=rows_per_block, **kw)
        layouts[lay.value] = lr
        say(f"  {lay.value:12} {lr.n_blocks:4} blocks  {lr.n_bytes / 1e6:6.1f} MB  "
            f"tightness {lr.zonemap.tightness():5.1f}x  long rows {lr.long_rows:,}")

    engine = Engine()
    results: dict[str, dict] = {}
    for name, lr in layouts.items():
        times, tb, tk = [], 0, 0
        per_intent: dict[str, dict] = {}
        for q in queries:
            runs = [engine.run(q, lr.zonemap) for _ in range(repeats)]
            t = statistics.median(r.seconds for r in runs)
            r = runs[-1]
            times.append(t)
            tb += r.bytes_read
            tk += r.blocks_read
            slot = per_intent.setdefault(q.intent.value,
                                         {"n": 0, "bytes": 0, "blocks": 0, "ms": 0.0})
            slot["n"] += 1
            slot["bytes"] += r.bytes_read
            slot["blocks"] += r.blocks_read
            slot["ms"] += t * 1000
        for slot in per_intent.values():
            n = slot["n"]
            slot["mean_megabytes"] = round(slot["bytes"] / n / 1e6, 3)
            slot["mean_blocks"] = round(slot["blocks"] / n, 2)
            slot["mean_ms"] = round(slot["ms"] / n, 3)
        results[name] = {
            "blocks": lr.n_blocks,
            "store_megabytes": round(lr.n_bytes / 1e6, 1),
            "tightness": round(lr.zonemap.tightness(), 2),
            "total_gigabytes_read": round(tb / 1e9, 4),
            "mean_blocks_read": round(tk / len(queries), 2),
            "median_ms": round(statistics.median(times) * 1000, 3),
            "mean_ms": round(statistics.fmean(times) * 1000, 3),
            "per_intent": per_intent,
        }

    # Correctness: every layout must return what an unpruned scan returns.
    say("  verifying ...")
    mismatches = []
    base = layouts["partitioned"]
    for q in queries:
        truth = engine.count_matching(q, base.zonemap)
        for name, lr in layouts.items():
            got = engine.run(q, lr.zonemap).rows
            if got != truth:
                mismatches.append({"layout": name, "region": str(q.region),
                                   "expected": truth, "got": got})
    engine.close()

    a = results["partitioned"]["total_gigabytes_read"]
    b = results["swapo"]["total_gigabytes_read"]
    report = {
        "dataset": {
            "name": ds.name, "contigs": list(contigs),
            "rows": ds.table.num_rows,
            "longest_interval_bp": stats.max_length,
            "fraction_over_50bp": round(stats.long_fraction, 5),
            "length_histogram": length_histogram(ds.table),
            "sources": ds.sources,
        },
        "workload": profile.summary(),
        "threshold_choice": choice.summary(),
        "threshold_candidates": [
            {"threshold_bp": t, "expected_blocks": round(c, 3), "long_fraction": round(f, 5)}
            for t, c, f in choice.candidates],
        "rows_per_block": rows_per_block,
        "layouts": results,
        "read_reduction": round(1 - b / a, 4) if a else 0.0,
        "verification": {"queries": len(queries) * len(layouts),
                         "mismatches": mismatches, "passed": not mismatches},
    }
    (out / "real_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    say(f"  partitioned {a:.4f} GB -> swapo {b:.4f} GB "
        f"({report['read_reduction'] * 100:.1f}% less), "
        f"{len(mismatches)} mismatches")
    return report
