"""The experiments reported in ``docs/paper.md``, as runnable code.

Each function returns plain dictionaries and writes JSON, so a number in the
paper can be regenerated and diffed. Nothing here is tuned: the real tables are
whatever the public files contain, and the simulated tables exist only to reach
regimes the public files do not cover.

The central experiment is :func:`crossover`, which puts real and simulated
tables on one axis — block tightness — and finds where interval-aware layout
starts to pay.
"""

from __future__ import annotations

import json
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from .data import DatasetSpec, read_vcf, simulate_features
from .engine import Engine
from .layout import Layout, write_layout
from .model import GRCH38, Intent, Query, Region
from .real import combine, describe_file, harmonise_contigs, read_gtf, read_rmsk
from .stats import length_histogram, summarise
from .threshold import choose_threshold, separability

__all__ = ["load_platform_table", "length_profiles", "crossover", "Point"]


@dataclass
class Point:
    """One measured (tightness, saving) observation."""

    table: str
    source: str            # "real" or "simulated"
    rows: int
    rows_per_block: int
    blocks: int
    block_span_kb: float
    tightness: float
    threshold_bp: int
    partitioned_mb: float
    swapo_mb: float
    saving: float

    @property
    def decision_correct(self) -> bool:
        """Did the optimizer act exactly when acting helped?

        A split was proposed (``threshold > 0``) if and only if splitting
        measurably reduced bytes read. This is the property that matters for a
        recommendation system: being right about when to do nothing.
        """
        proposed = self.threshold_bp > 0
        helped = self.saving > 0.01
        return proposed == helped

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["decision_correct"] = self.decision_correct
        return d


# ---------------------------------------------------------------------------
# Real data
# ---------------------------------------------------------------------------


def load_platform_table(data_dir: str | Path, contig: str = "chr1") -> tuple[pa.Table, list[dict]]:
    """Everything a genomics platform would hold for one contig, from real files.

    Annotation, clinically reported variants, repeats and structural variants.
    They are kept in one table because that is how a lakehouse stores them, and
    because the mixture is what produces the interval-length distribution the
    layout question turns on.
    """
    data_dir = Path(data_dir)
    sources: list[dict] = []
    parts: list[pa.Table] = []

    def add(path: Path, tbl: pa.Table, role: str, assembly: str = "GRCh38") -> None:
        if tbl.num_rows:
            parts.append(tbl)
            sources.append({**describe_file(path), "records": tbl.num_rows,
                            "role": role, "assembly": assembly})

    def on_contig(t: pa.Table) -> pa.Table:
        t = harmonise_contigs(t)
        return t.filter(pc.equal(t.column("contig"), contig))

    gtf = data_dir / "gencode.v44.basic.annotation.gtf.gz"
    if gtf.exists():
        add(gtf, read_gtf(gtf, contigs=(contig,)), "annotation")
    cv = data_dir / "clinvar.vcf.gz"
    if cv.exists():
        add(cv, on_contig(read_vcf(cv)), "clinical variants")
    rm = data_dir / "rmsk.txt.gz"
    if rm.exists():
        add(rm, read_rmsk(rm, contigs=(contig,)), "repeats")
    sv = data_dir / "1kg_sv.vcf.gz"
    if sv.exists():
        # 1000 Genomes phase 3 is on GRCh37, unlike the other sources.
        add(sv, on_contig(read_vcf(sv)), "structural variants", "GRCh37")
    kg = data_dir / f"1kg.{contig}.vcf.gz"
    if kg.exists():
        add(kg, on_contig(read_vcf(kg)), "population callset", "GRCh37")

    if not parts:
        raise FileNotFoundError(f"no real inputs in {data_dir}")
    assemblies = {s["assembly"] for s in sources}
    if len(assemblies) > 1:
        # Disclosed rather than silently combined. Coordinates are not
        # comparable across assemblies, so this table is valid for measuring
        # interval-length distribution, density and block pruning, and is NOT
        # valid for drawing any biological conclusion about which features
        # overlap which variants. See the paper's threats to validity.
        for s in sources:
            s["assembly_warning"] = (
                "table mixes " + ", ".join(sorted(assemblies))
                + "; coordinates are not comparable across assemblies")
    return combine(*parts, contigs=(contig,)), sources


def length_profiles(data_dir: str | Path, contig: str = "chr1") -> dict:
    """Interval-length distribution of each real source, and of the simulator.

    Reported because the gap between them is the reason the simulated result in
    the first version of this work did not survive contact with real files.
    """
    data_dir = Path(data_dir)
    out: dict[str, dict] = {}

    def profile(name: str, tbl: pa.Table, path: Path | None = None) -> None:
        if not tbl.num_rows:
            return
        span = (np.asarray(tbl.column("end")).astype(np.int64)
                - np.asarray(tbl.column("start")).astype(np.int64))
        out[name] = {
            "records": int(tbl.num_rows),
            "median_bp": int(np.median(span)),
            "p99_bp": int(np.percentile(span, 99)),
            "p99_99_bp": int(np.percentile(span, 99.99)),
            "max_bp": int(span.max()),
            "fraction_over_50bp": round(float((span > 50).mean()), 5),
            "histogram": length_histogram(tbl),
            **({"sha256": describe_file(path)["sha256"]} if path else {}),
        }

    gtf = data_dir / "gencode.v44.basic.annotation.gtf.gz"
    if gtf.exists():
        profile("GENCODE v44 (annotation)", read_gtf(gtf, contigs=(contig,)), gtf)
    cv = data_dir / "clinvar.vcf.gz"
    if cv.exists():
        t = harmonise_contigs(read_vcf(cv))
        profile("ClinVar (clinical variants)",
                t.filter(pc.equal(t.column("contig"), contig)), cv)
    rm = data_dir / "rmsk.txt.gz"
    if rm.exists():
        profile("RepeatMasker (repeats)", read_rmsk(rm, contigs=(contig,)), rm)
    sv = data_dir / "1kg_sv.vcf.gz"
    if sv.exists():
        profile("1000G structural variants", read_vcf(sv), sv)
    kg = data_dir / f"1kg.{contig}.vcf.gz"
    if kg.exists():
        t = harmonise_contigs(read_vcf(kg))
        profile("1000G population callset",
                t.filter(pc.equal(t.column("contig"), contig)), kg)

    sim = simulate_features(DatasetSpec(n_features=300_000, long_fraction=0.004))
    profile("simulator (long_fraction=0.004)", sim)
    return out


# ---------------------------------------------------------------------------
# The crossover experiment
# ---------------------------------------------------------------------------


def _queries_around(contig: str, centres: list[int], n: int, seed: int
                    ) -> tuple[list[Query], int]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = centres[int(rng.integers(0, len(centres)))]
        half = int(10 ** rng.uniform(3.3, 4.6))
        out.append(Query(Intent.INTERVAL,
                         region=Region(contig, max(0, c - half), c + half)))
    return out, int(statistics.median(q.region.length for q in out))


def _measure(tbl: pa.Table, contig: str, label: str, source: str, rows_per_block: int,
             queries: list[Query], width: int, engine: Engine, tmp: Path) -> Point:
    choice = choose_threshold(tbl, rows_per_block=rows_per_block, query_width=width)
    got: dict[str, tuple[float, object]] = {}
    for lay in (Layout.PARTITIONED, Layout.SWAPO):
        kw = {"long_threshold": choice.threshold} if lay is Layout.SWAPO else {}
        lr = write_layout(tbl, tmp / f"{label}-{rows_per_block}-{lay.value}", lay,
                          rows_per_block=rows_per_block, **kw)
        mb = sum(engine.run(q, lr.zonemap).bytes_read for q in queries) / len(queries) / 1e6
        got[lay.value] = (mb, lr)
    p, s = got["partitioned"][0], got["swapo"][0]
    lr = got["partitioned"][1]
    return Point(
        table=label, source=source, rows=tbl.num_rows, rows_per_block=rows_per_block,
        blocks=lr.n_blocks,
        block_span_kb=round(GRCH38.get(contig, 1) / max(lr.n_blocks, 1) / 1000, 1),
        tightness=round(lr.zonemap.tightness(), 2),
        threshold_bp=int(choice.threshold),
        partitioned_mb=round(p, 4), swapo_mb=round(s, 4),
        saving=round(1 - s / p, 4) if p else 0.0,
    )


def crossover(
    data_dir: str | Path = "swapo_data",
    out: str | Path = "paper_out",
    *,
    contig: str = "chr1",
    real_block_sizes: tuple[int, ...] = (8_000, 2_000, 1_000, 500, 250),
    sim_long_fractions: tuple[float, ...] = (0.0, 0.0005, 0.004, 0.02),
    sim_block_sizes: tuple[int, ...] = (8_000, 2_000),
    n_queries: int = 100,
    verbose: bool = True,
) -> dict:
    """Put real and simulated tables on one axis and find where splitting pays.

    The axis is **tightness**: the mean ratio of how far a block's rows reach to
    the range of starts it holds. It costs nothing to compute, because it comes
    from block metadata that already exists.
    """
    say = print if verbose else (lambda *a, **k: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp())
    engine = Engine()
    points: list[Point] = []

    real, sources = load_platform_table(data_dir, contig)
    sep = separability(real)
    say(f"Real platform table, {contig}: {real.num_rows:,} rows from "
        f"{len(sources)} public files")
    say(f"  separability: p99 {sep['p99_bp']:,} bp, p99.99 {sep['p99_99_bp']:,} bp, "
        f"ratio {sep['bimodality']}x, separable={sep['separable']}")

    gtf = Path(data_dir) / "gencode.v44.basic.annotation.gtf.gz"
    genes = read_gtf(gtf, feature_types=("gene",), contigs=(contig,))
    centres = [(int(s) + int(e)) // 2 for s, e in
               zip(genes.column("start").to_pylist(), genes.column("end").to_pylist())]
    queries, width = _queries_around(contig, centres, n_queries, seed=31)
    say(f"  {len(queries)} queries around real gene coordinates, "
        f"median width {width:,} bp")

    for rpb in real_block_sizes:
        pt = _measure(real, contig, f"real/{contig}", "real", rpb, queries, width,
                      engine, tmp)
        points.append(pt)
        say(f"  real  rpb={rpb:<6} tightness {pt.tightness:>5.2f}  "
            f"saving {pt.saving * 100:>5.1f}%  threshold {pt.threshold_bp or '-'}")

    # Simulated tables reach tightness values the real files do not.
    sim_contig = "chr21"
    sim_centres = [int(x) for x in
                   np.random.default_rng(7).integers(2_000_000, 44_000_000, 400)]
    sim_queries, sim_width = _queries_around(sim_contig, sim_centres, n_queries, seed=31)
    for lf in sim_long_fractions:
        sim = simulate_features(DatasetSpec(n_features=1_000_000, long_fraction=lf))
        sim = sim.filter(pc.equal(sim.column("contig"), sim_contig))
        for rpb in sim_block_sizes:
            pt = _measure(sim, sim_contig, f"sim/long={lf}", "simulated", rpb,
                          sim_queries, sim_width, engine, tmp)
            points.append(pt)
            say(f"  sim   lf={lf:<7} rpb={rpb:<6} tightness {pt.tightness:>5.2f}  "
                f"saving {pt.saving * 100:>5.1f}%  threshold {pt.threshold_bp or '-'}")
    engine.close()

    ordered = sorted(points, key=lambda p: p.tightness)
    zero = [p for p in ordered if p.saving <= 0.01]
    nonzero = [p for p in ordered if p.saving > 0.01]
    crossing = (max(p.tightness for p in zero) if zero else None,
                min(p.tightness for p in nonzero) if nonzero else None)
    correct = sum(p.decision_correct for p in points)

    report = {
        "contig": contig,
        "real_separability": sep,
        "real_sources": sources,
        "real_rows": real.num_rows,
        "query_width_bp": width,
        "points": [p.to_dict() for p in ordered],
        "crossover": {"highest_tightness_with_no_gain": crossing[0],
                      "lowest_tightness_with_gain": crossing[1]},
        "decisions_correct": f"{correct}/{len(points)}",
        "real_tightness_range": [min(p.tightness for p in points if p.source == "real"),
                                 max(p.tightness for p in points if p.source == "real")],
    }
    (out / "crossover.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    say(f"\n  crossover between tightness {crossing[0]} and {crossing[1]}; "
        f"optimizer decisions correct {correct}/{len(points)}")
    return report
