"""Datasets: a realistic simulator, and readers for real VCF and BED files.

The simulator exists so the benchmark can be run by anyone in a minute without
downloading a cohort, and so the interval-length distribution can be dialled up
and down to show exactly where the optimizer helps and where it does not. It is
seeded, so a run is reproducible.

What it tries to get right, because these are the properties the layout
decisions turn on:

* variants fall across contigs roughly in proportion to contig length;
* density is uneven, with hotspots, because real callsets are;
* most records are a single base, a minority are short indels, and a small
  tail are long: structural variants, CNVs and gene bodies. That tail is the
  whole problem (see :mod:`swapo.zonemap`), so its size is a parameter.

Real data is read by :func:`read_vcf` and :func:`read_bed`, which produce the
same schema, so every other module is indifferent to where the rows came from.
"""

from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from .model import GRCH38, Assembly

__all__ = [
    "FEATURE_SCHEMA", "SAMPLE_SCHEMA", "GENOTYPE_SCHEMA",
    "simulate_features", "simulate_samples", "simulate_genotypes",
    "read_vcf", "read_bed", "DatasetSpec",
]

# A single base change is length 1; a long interval is anything above this.
# 50 bp is the conventional boundary between an indel and a structural variant.
SV_THRESHOLD = 50

FEATURE_SCHEMA = pa.schema([
    pa.field("contig", pa.string()),
    pa.field("start", pa.int32()),          # 0-based, half-open
    pa.field("end", pa.int32()),
    pa.field("variant_id", pa.string()),
    pa.field("ref", pa.string()),
    pa.field("alt", pa.string()),
    pa.field("kind", pa.string()),          # snv | indel | sv
    pa.field("gene_id", pa.string()),
    pa.field("consequence", pa.string()),
    pa.field("af", pa.float32()),
])

SAMPLE_SCHEMA = pa.schema([
    pa.field("sample_id", pa.string()),
    pa.field("cohort", pa.string()),
    pa.field("sex", pa.string()),
    pa.field("phenotype", pa.int8()),       # 0 control, 1 case
])

GENOTYPE_SCHEMA = pa.schema([
    pa.field("variant_id", pa.string()),
    pa.field("contig", pa.string()),
    pa.field("start", pa.int32()),
    pa.field("sample_id", pa.string()),
    pa.field("cohort", pa.string()),
    pa.field("gt", pa.int8()),              # 0, 1 or 2 alternate alleles
])

CONSEQUENCES = np.array([
    "intergenic", "intron", "synonymous", "missense", "stop_gained",
    "splice_acceptor", "frameshift", "upstream", "downstream", "utr3",
])
CONSEQUENCE_P = np.array([.33, .34, .10, .11, .01, .01, .02, .04, .03, .01])


@dataclass(frozen=True)
class DatasetSpec:
    """Knobs for :func:`simulate_features`."""

    n_features: int = 2_000_000
    n_samples: int = 64
    #: Fraction of records longer than :data:`SV_THRESHOLD`. Real short-variant
    #: callsets sit near 0.001; a callset merged with an SV or annotation track
    #: is far higher. Set to 0.0 to see the optimizer correctly decline to act.
    long_fraction: float = 0.004
    #: Longest record, bp. 2.4 Mb is about the span of DMD, the longest human gene.
    max_long_length: int = 2_400_000
    n_genes: int = 20_000
    seed: int = 20261116
    assembly: Assembly = Assembly.GRCH38
    #: Which contigs to simulate. The default is three small autosomes rather
    #: than the whole genome, so that a few million rows still give a realistic
    #: *density* — on the order of a million records per contig, which is what a
    #: population callset looks like. Spreading the same rows over all 24
    #: contigs would make each block span tens of megabases, which is not what
    #: a real table does and would quietly flatter every layout equally.
    contigs: tuple[str, ...] = ("chr20", "chr21", "chr22")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def simulate_features(spec: DatasetSpec = DatasetSpec()) -> pa.Table:
    """Build a variant/feature table with a realistic interval-length tail."""
    rng = _rng(spec.seed)
    n = spec.n_features
    contigs = np.array(list(spec.contigs))
    lengths = np.array([GRCH38[c] for c in spec.contigs], dtype=np.int64)

    # Contig assignment in proportion to length, so chr1 really does dominate.
    p = lengths / lengths.sum()
    contig_idx = rng.choice(len(contigs), size=n, p=p)

    # Positions: a uniform background plus hotspots, because real density is
    # clumpy and even clumping changes how well fixed-size blocks bound a range.
    pos = np.empty(n, dtype=np.int64)
    uni = rng.random(n)
    contig_len = lengths[contig_idx]
    hotspot_mask = rng.random(n) < 0.25
    n_hot = 400
    hot_centres = rng.random(n_hot)
    centre_pick = rng.integers(0, n_hot, size=n)
    hot_pos = (hot_centres[centre_pick] + rng.normal(0, 0.004, size=n)) % 1.0
    frac = np.where(hotspot_mask, hot_pos, uni)
    pos = np.clip((frac * np.maximum(contig_len - spec.max_long_length - 2, 1)).astype(np.int64), 0, None)

    # Lengths: 1 bp for most, short indels next, a heavy tail of long records.
    u = rng.random(n)
    is_long = u < spec.long_fraction
    is_indel = (~is_long) & (u < spec.long_fraction + 0.08)
    length = np.ones(n, dtype=np.int64)
    n_indel = int(is_indel.sum())
    if n_indel:
        length[is_indel] = rng.integers(2, SV_THRESHOLD, size=n_indel)
    n_long = int(is_long.sum())
    if n_long:
        # Log-uniform between the SV threshold and the cap: plenty of kilobase
        # records, a few megabase ones. That shape is what defeats max(end).
        lo, hi = np.log10(SV_THRESHOLD + 1), np.log10(spec.max_long_length)
        length[is_long] = np.power(10.0, rng.uniform(lo, hi, size=n_long)).astype(np.int64)

    start = pos
    end = start + length
    end = np.minimum(end, contig_len - 1)
    end = np.maximum(end, start + 1)

    kind = np.where(is_long, "sv", np.where(is_indel, "indel", "snv"))
    bases = np.array(["A", "C", "G", "T"])
    ref = bases[rng.integers(0, 4, size=n)]
    alt = bases[rng.integers(0, 4, size=n)]
    alt = np.where(alt == ref, bases[(np.searchsorted(bases, ref) + 1) % 4], alt)

    gene = rng.integers(0, spec.n_genes, size=n)
    gene_id = np.char.add("ENSG", np.char.zfill(gene.astype(str), 11))
    cons = CONSEQUENCES[rng.choice(len(CONSEQUENCES), size=n, p=CONSEQUENCE_P)]
    # Allele frequency: most variants are rare. A beta draw gives that shape.
    af = rng.beta(0.3, 12.0, size=n).astype(np.float32)

    contig_col = contigs[contig_idx]
    variant_id = np.char.add(
        np.char.add(contig_col, "-"),
        np.char.add(np.char.zfill(start.astype(str), 9), np.char.add("-", alt)),
    )

    return pa.table({
        "contig": pa.array(contig_col, pa.string()),
        "start": pa.array(start, pa.int32()),
        "end": pa.array(end, pa.int32()),
        "variant_id": pa.array(variant_id, pa.string()),
        "ref": pa.array(ref, pa.string()),
        "alt": pa.array(alt, pa.string()),
        "kind": pa.array(kind, pa.string()),
        "gene_id": pa.array(gene_id, pa.string()),
        "consequence": pa.array(cons, pa.string()),
        "af": pa.array(af, pa.float32()),
    }, schema=FEATURE_SCHEMA)


def simulate_samples(spec: DatasetSpec = DatasetSpec()) -> pa.Table:
    rng = _rng(spec.seed + 1)
    n = spec.n_samples
    sample_id = np.array([f"NA{100000 + i}" for i in range(n)])
    cohort = rng.choice(np.array(["discovery", "replication", "control"]),
                        size=n, p=[0.45, 0.3, 0.25])
    sex = rng.choice(np.array(["male", "female"]), size=n)
    phen = rng.integers(0, 2, size=n).astype(np.int8)
    return pa.table({
        "sample_id": pa.array(sample_id, pa.string()),
        "cohort": pa.array(cohort, pa.string()),
        "sex": pa.array(sex, pa.string()),
        "phenotype": pa.array(phen, pa.int8()),
    }, schema=SAMPLE_SCHEMA)


def simulate_genotypes(features: pa.Table, samples: pa.Table,
                       n_variants: int = 120_000, seed: int = 7) -> pa.Table:
    """Long-format genotypes for a subset of variants, every sample.

    Long format is what makes sample and cohort queries interesting: the table
    is ``n_variants x n_samples`` rows, so the layout question is whether to
    order by coordinate (good for region queries) or by sample (good for
    per-sample extraction). The optimizer has to choose, and that choice is the
    kind of trade-off the paper is about.
    """
    rng = _rng(seed)
    n_variants = min(n_variants, features.num_rows)
    pick = np.sort(rng.choice(features.num_rows, size=n_variants, replace=False))
    sub = features.take(pa.array(pick))
    vid = np.asarray(sub.column("variant_id"))
    contig = np.asarray(sub.column("contig"))
    start = np.asarray(sub.column("start"))
    af = np.asarray(sub.column("af"))

    sid = np.asarray(samples.column("sample_id"))
    coh = np.asarray(samples.column("cohort"))
    ns = len(sid)

    # Genotypes drawn under Hardy-Weinberg from each variant's allele frequency.
    p = np.repeat(af.astype(np.float64), ns)
    draw = rng.random(n_variants * ns)
    gt = np.where(draw < p * p, 2, np.where(draw < p * p + 2 * p * (1 - p), 1, 0)).astype(np.int8)

    return pa.table({
        "variant_id": pa.array(np.repeat(vid, ns), pa.string()),
        "contig": pa.array(np.repeat(contig, ns), pa.string()),
        "start": pa.array(np.repeat(start, ns).astype(np.int32), pa.int32()),
        "sample_id": pa.array(np.tile(sid, n_variants), pa.string()),
        "cohort": pa.array(np.tile(coh, n_variants), pa.string()),
        "gt": pa.array(gt, pa.int8()),
    }, schema=GENOTYPE_SCHEMA)


def simulate_genes(spec: DatasetSpec = DatasetSpec()) -> pa.Table:
    """A gene model, so ``Intent.GENE`` queries resolve to real coordinates."""
    rng = _rng(spec.seed + 2)
    n = spec.n_genes
    contigs = np.array(list(spec.contigs))
    lengths = np.array([GRCH38[c] for c in spec.contigs], dtype=np.int64)
    idx = rng.choice(len(contigs), size=n, p=lengths / lengths.sum())
    clen = lengths[idx]
    start = (rng.random(n) * np.maximum(clen - 2_500_000, 1)).astype(np.int64).clip(0)
    # Gene spans are log-normal: most tens of kilobases, a few over a megabase.
    span = np.power(10.0, rng.normal(4.4, 0.55, size=n)).astype(np.int64).clip(200, 2_400_000)
    gene_id = np.char.add("ENSG", np.char.zfill(np.arange(n).astype(str), 11))
    return pa.table({
        "gene_id": pa.array(gene_id, pa.string()),
        "symbol": pa.array(np.char.add("GENE", np.arange(n).astype(str)), pa.string()),
        "contig": pa.array(contigs[idx], pa.string()),
        "start": pa.array(start, pa.int32()),
        "end": pa.array(start + span, pa.int32()),
    })


# ---------------------------------------------------------------------------
# Real files
# ---------------------------------------------------------------------------


def _open_maybe_gzip(path: str | os.PathLike) -> io.TextIOBase:
    path = str(path)
    if path.endswith((".gz", ".bgz")):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def read_vcf(path: str | os.PathLike, max_records: int | None = None) -> pa.Table:
    """Read a VCF or VCF.gz into :data:`FEATURE_SCHEMA`.

    VCF is 1-based and inclusive. The end of a record is taken from the ``END``
    INFO key when present (which is how symbolic structural variants carry
    their span) and otherwise from the length of ``REF``. Both are converted to
    0-based half-open here, once, so nothing downstream has to think about it.
    """
    contig: list[str] = []
    start: list[int] = []
    end: list[int] = []
    vid: list[str] = []
    ref_l: list[str] = []
    alt_l: list[str] = []
    kind_l: list[str] = []
    gene_l: list[str] = []
    cons_l: list[str] = []
    af_l: list[float] = []

    with _open_maybe_gzip(path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            chrom, pos_s, id_s, ref, alt = f[0], f[1], f[2], f[3], f[4].split(",")[0]
            info = f[7]
            pos = int(pos_s)
            s = pos - 1                       # 0-based
            info_d: dict[str, str] = {}
            for kv in info.split(";"):
                k, _, v = kv.partition("=")
                info_d[k] = v
            if "END" in info_d and info_d["END"].isdigit():
                e = int(info_d["END"])        # END is inclusive, 1-based -> half-open
            else:
                e = s + max(len(ref), 1)
            e = max(e, s + 1)
            span = e - s
            if alt.startswith("<") or span > SV_THRESHOLD:
                k = "sv"
            elif len(ref) != len(alt):
                k = "indel"
            else:
                k = "snv"
            af_s = info_d.get("AF", "").split(",")[0]
            try:
                af = float(af_s)
            except ValueError:
                af = 0.0
            contig.append(chrom); start.append(s); end.append(e)
            vid.append(id_s if id_s not in (".", "") else f"{chrom}-{s:09d}-{alt}")
            ref_l.append(ref); alt_l.append(alt); kind_l.append(k)
            gene_l.append(info_d.get("GENE", info_d.get("Gene", "")))
            cons_l.append(info_d.get("CSQ", info_d.get("Consequence", "")).split("|")[0])
            af_l.append(af)
            if max_records and len(start) >= max_records:
                break

    return pa.table({
        "contig": pa.array(contig, pa.string()),
        "start": pa.array(start, pa.int32()),
        "end": pa.array(end, pa.int32()),
        "variant_id": pa.array(vid, pa.string()),
        "ref": pa.array(ref_l, pa.string()),
        "alt": pa.array(alt_l, pa.string()),
        "kind": pa.array(kind_l, pa.string()),
        "gene_id": pa.array(gene_l, pa.string()),
        "consequence": pa.array(cons_l, pa.string()),
        "af": pa.array(af_l, pa.float32()),
    }, schema=FEATURE_SCHEMA)


def read_bed(path: str | os.PathLike) -> pa.Table:
    """Read a BED file. BED is already 0-based half-open, so coordinates pass through."""
    contig: list[str] = []
    start: list[int] = []
    end: list[int] = []
    name: list[str] = []
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            contig.append(f[0]); start.append(int(f[1])); end.append(int(f[2]))
            name.append(f[3] if len(f) > 3 else "")
    n = len(start)
    span = [e - s for s, e in zip(start, end)]
    return pa.table({
        "contig": pa.array(contig, pa.string()),
        "start": pa.array(start, pa.int32()),
        "end": pa.array(end, pa.int32()),
        "variant_id": pa.array(name or [""] * n, pa.string()),
        "ref": pa.array([""] * n, pa.string()),
        "alt": pa.array([""] * n, pa.string()),
        "kind": pa.array(["sv" if x > SV_THRESHOLD else "snv" for x in span], pa.string()),
        "gene_id": pa.array([""] * n, pa.string()),
        "consequence": pa.array([""] * n, pa.string()),
        "af": pa.array([0.0] * n, pa.float32()),
    }, schema=FEATURE_SCHEMA)
