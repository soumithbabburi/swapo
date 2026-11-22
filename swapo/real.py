"""Readers for real public genomic files, and a small registry of them.

The simulator in :mod:`swapo.data` exists so the benchmark can run anywhere in a
minute. It is not evidence. These readers exist so the same benchmark can run on
files that real pipelines actually hold, which is the only way the interval
pathology this project is about can be shown to be real rather than assumed.

Three formats, all plain text and all read as data only:

``GTF``   gene annotation (GENCODE, Ensembl). Genes, transcripts, exons and CDS
          in one file, which is where the long tail genuinely comes from: an
          exon is a few hundred bases, the gene containing it can be megabases.
``VCF``   variants (ClinVar, gnomAD, 1000 Genomes). Handled in :mod:`swapo.data`.
``rmsk``  UCSC RepeatMasker, a large interval track with its own length profile.

Coordinate conventions differ between all three and are normalised here, once:
GTF is 1-based inclusive, VCF is 1-based inclusive, BED and UCSC table dumps are
0-based half-open. Everything downstream sees 0-based half-open.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa

from .data import FEATURE_SCHEMA, SV_THRESHOLD, _open_maybe_gzip

__all__ = ["read_gtf", "read_rmsk", "SOURCES", "fetch", "describe_file",
           "harmonise_contigs", "combine", "fetch_gnomad_window",
           "GNOMAD_CHR21_URL"]


@dataclass(frozen=True)
class Source:
    """A public file, recorded so a result can be traced back to its input."""

    name: str
    url: str
    filename: str
    reader: str
    assembly: str
    description: str
    approx_mb: int


#: Files the benchmark knows how to fetch. Nothing is downloaded unless asked.
SOURCES: dict[str, Source] = {
    "gencode": Source(
        name="gencode",
        url=("https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/"
             "release_44/gencode.v44.basic.annotation.gtf.gz"),
        filename="gencode.v44.basic.annotation.gtf.gz",
        reader="gtf", assembly="GRCh38",
        description="GENCODE v44 basic annotation: genes, transcripts, exons, CDS",
        approx_mb=29,
    ),
    "clinvar": Source(
        name="clinvar",
        url="https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
        filename="clinvar.vcf.gz",
        reader="vcf", assembly="GRCh38",
        description="ClinVar variant summary, GRCh38",
        approx_mb=192,
    ),
    "1kg_chr21": Source(
        name="1kg_chr21",
        url=("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
             "ALL.chr21.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"),
        filename="1kg.chr21.vcf.gz",
        reader="vcf", assembly="GRCh37",
        description="1000 Genomes phase 3 chr21 population callset (1.1M variants)",
        approx_mb=200,
    ),
    "1kg_sv": Source(
        name="1kg_sv",
        url=("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/integrated_sv_map/"
             "ALL.wgs.mergedSV.v8.20130502.svs.genotypes.vcf.gz"),
        filename="1kg_sv.vcf.gz",
        reader="vcf", assembly="GRCh37",
        description="1000 Genomes integrated structural variant map v8",
        approx_mb=18,
    ),
    "rmsk": Source(
        name="rmsk",
        url="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/rmsk.txt.gz",
        filename="rmsk.txt.gz",
        reader="rmsk", assembly="GRCh38",
        description="UCSC RepeatMasker track for hg38",
        approx_mb=148,
    ),
}


def fetch(name: str, root: str | os.PathLike = "swapo_data") -> Path:
    """Download ``name`` into ``root`` if it is not already there, and return its path."""
    src = SOURCES[name]
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / src.filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(src.url, tmp)   # noqa: S310 - fixed, known URLs
    tmp.replace(dest)
    return dest


def describe_file(path: str | os.PathLike) -> dict:
    """Size and SHA-256, so a run can state exactly which bytes it used."""
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"file": path.name, "bytes": path.stat().st_size,
            "sha256": h.hexdigest()}


def _gtf_attr(attrs: str, key: str) -> str:
    """Pull one value out of a GTF attribute string without a full parse."""
    i = attrs.find(key + ' "')
    if i < 0:
        return ""
    j = attrs.find('"', i + len(key) + 2)
    return attrs[i + len(key) + 2:j] if j > 0 else ""


def read_gtf(
    path: str | os.PathLike,
    *,
    feature_types: tuple[str, ...] = ("gene", "transcript", "exon", "CDS", "UTR"),
    contigs: tuple[str, ...] | None = None,
    max_records: int | None = None,
) -> pa.Table:
    """Read a GENCODE/Ensembl GTF into :data:`~swapo.data.FEATURE_SCHEMA`.

    GTF columns 4 and 5 are 1-based and inclusive, so ``start`` loses one and
    ``end`` is kept as-is to give a half-open interval.

    Keeping several feature types in one table is deliberate and is the point of
    using this file: exons are short, genes are long, and a platform that stores
    both in one table has exactly the length distribution that ruins ``max(end)``.
    """
    wanted = set(feature_types)
    keep_contigs = set(contigs) if contigs else None

    contig_l: list[str] = []
    start_l: list[int] = []
    end_l: list[int] = []
    id_l: list[str] = []
    kind_l: list[str] = []
    gene_l: list[str] = []
    cons_l: list[str] = []

    with _open_maybe_gzip(path) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            ftype = f[2]
            if ftype not in wanted:
                continue
            chrom = f[0]
            if keep_contigs is not None and chrom not in keep_contigs:
                continue
            s = int(f[3]) - 1          # 1-based inclusive -> 0-based
            e = int(f[4])              # inclusive end == half-open end
            if e <= s:
                e = s + 1
            attrs = f[8]
            contig_l.append(chrom)
            start_l.append(s)
            end_l.append(e)
            id_l.append(_gtf_attr(attrs, "transcript_id")
                        or _gtf_attr(attrs, "gene_id") or f"{chrom}:{s}")
            kind_l.append(ftype)
            gene_l.append(_gtf_attr(attrs, "gene_id"))
            cons_l.append(_gtf_attr(attrs, "gene_type"))
            if max_records and len(start_l) >= max_records:
                break

    n = len(start_l)
    span = np.array(end_l, dtype=np.int64) - np.array(start_l, dtype=np.int64)
    return pa.table({
        "contig": pa.array(contig_l, pa.string()),
        "start": pa.array(start_l, pa.int32()),
        "end": pa.array(end_l, pa.int32()),
        "variant_id": pa.array(id_l, pa.string()),
        "ref": pa.array([""] * n, pa.string()),
        "alt": pa.array([""] * n, pa.string()),
        # "kind" keeps the GTF feature type, so the length profile can be
        # attributed to the biology rather than guessed at.
        "kind": pa.array(kind_l, pa.string()),
        "gene_id": pa.array(gene_l, pa.string()),
        "consequence": pa.array(cons_l, pa.string()),
        "af": pa.array(np.zeros(n, dtype=np.float32), pa.float32()),
    }, schema=FEATURE_SCHEMA)


def read_rmsk(path: str | os.PathLike, *, contigs: tuple[str, ...] | None = None,
              max_records: int | None = None) -> pa.Table:
    """Read a UCSC RepeatMasker table dump. Its coordinates are 0-based half-open."""
    keep = set(contigs) if contigs else None
    contig_l: list[str] = []
    start_l: list[int] = []
    end_l: list[int] = []
    id_l: list[str] = []
    fam_l: list[str] = []
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 17:
                continue
            chrom, s, e = f[5], int(f[6]), int(f[7])
            if keep is not None and chrom not in keep:
                continue
            contig_l.append(chrom)
            start_l.append(s)
            end_l.append(max(e, s + 1))
            id_l.append(f[10])
            fam_l.append(f[12])
            if max_records and len(start_l) >= max_records:
                break
    n = len(start_l)
    span = np.array(end_l, dtype=np.int64) - np.array(start_l, dtype=np.int64)
    return pa.table({
        "contig": pa.array(contig_l, pa.string()),
        "start": pa.array(start_l, pa.int32()),
        "end": pa.array(end_l, pa.int32()),
        "variant_id": pa.array(id_l, pa.string()),
        "ref": pa.array([""] * n, pa.string()),
        "alt": pa.array([""] * n, pa.string()),
        "kind": pa.array(np.where(span > SV_THRESHOLD, "repeat_long", "repeat"), pa.string()),
        "gene_id": pa.array([""] * n, pa.string()),
        "consequence": pa.array(fam_l, pa.string()),
        "af": pa.array(np.zeros(n, dtype=np.float32), pa.float32()),
    }, schema=FEATURE_SCHEMA)


# ---------------------------------------------------------------------------
# Harmonising sources
# ---------------------------------------------------------------------------


def harmonise_contigs(tbl: pa.Table, style: str = "chr") -> pa.Table:
    """Put contig names into one style so sources can be combined.

    ClinVar writes ``1``; GENCODE and UCSC write ``chr1``. Joining the two
    without reconciling this produces an empty result rather than an error,
    which is the kind of failure that survives review. The mapping also folds
    the mitochondrial aliases ``MT`` and ``chrMT`` onto ``chrM``.
    """
    names = tbl.column("contig").to_pylist()
    out: list[str] = []
    for c in names:
        c = str(c)
        bare = c[3:] if c.startswith("chr") else c
        if bare in ("MT", "M"):
            bare = "M"
        out.append(("chr" + bare) if style == "chr" else bare)
    return tbl.set_column(tbl.schema.get_field_index("contig"),
                          pa.field("contig", pa.string()),
                          pa.array(out, pa.string()))


def combine(*tables: pa.Table, contigs: tuple[str, ...] | None = None) -> pa.Table:
    """Concatenate feature tables after harmonising their contig naming.

    This is what a genomic platform table actually is: variants and annotation
    in one place. It is also where the interval-length distribution becomes
    bimodal, with single-base variants beside megabase gene bodies.
    """
    parts = [harmonise_contigs(t) for t in tables if t.num_rows]
    merged = pa.concat_tables(parts, promote_options="default")
    if contigs:
        import pyarrow.compute as _pc
        keep = _pc.is_in(merged.column("contig"), value_set=pa.array(list(contigs)))
        merged = merged.filter(keep)
    return merged


# ---------------------------------------------------------------------------
# gnomAD, read as a slice rather than downloaded whole
# ---------------------------------------------------------------------------

#: The gnomAD chr21 genomes sites VCF. 7.2 GB, so it is never fetched whole;
#: :func:`fetch_gnomad_window` reads a contiguous slice with BGZF range reads.
GNOMAD_CHR21_URL = ("https://gnomad-public-us-east-1.s3.amazonaws.com/release/4.1/"
                    "vcf/genomes/gnomad.genomes.v4.1.sites.chr21.vcf.bgz")


def fetch_gnomad_window(root: str | os.PathLike = "swapo_data", *,
                        start_byte: int = 2_000_000_000,
                        n_bytes: int = 420_000_000,
                        url: str = GNOMAD_CHR21_URL) -> Path:
    """Read a contiguous window of a remote gnomAD VCF into a Parquet file.

    Returns the path. Re-reading is skipped if the file already exists. The
    window is whatever genomic interval those bytes happen to cover; the result
    carries its true density, which is the point, but it is not a uniform
    sample of the chromosome.
    """
    import re as _re

    import pyarrow.parquet as _pq

    from .bgzf import stream_lines

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "gnomad.chr21.window.parquet"
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    contig: list[str] = []
    start: list[int] = []
    end: list[int] = []
    vid: list[str] = []
    ref_l: list[str] = []
    alt_l: list[str] = []
    kind_l: list[str] = []
    end_re = _re.compile(r"(?:^|;)END=(\d+)")

    for line, _info in stream_lines(url, start_byte, n_bytes):
        if not line or line[0] == "#":
            continue
        f = line.split("\t", 8)
        if len(f) < 8:
            continue
        pos, ref, alt = int(f[1]), f[3], f[4].split(",")[0]
        s = pos - 1
        e = s + max(len(ref), 1)
        if alt.startswith("<"):
            m = end_re.search(f[7])
            if m:
                e = int(m.group(1))
        e = max(e, s + 1)
        span = e - s
        contig.append(f[0]); start.append(s); end.append(e)
        vid.append(f[2] if f[2] != "." else f"{f[0]}-{s}-{alt}")
        ref_l.append(ref[:20]); alt_l.append(alt[:20])
        kind_l.append("sv" if span > SV_THRESHOLD
                      else ("indel" if len(ref) != len(alt) else "snv"))

    n = len(start)
    tbl = pa.table({
        "contig": pa.array(contig, pa.string()),
        "start": pa.array(start, pa.int32()),
        "end": pa.array(end, pa.int32()),
        "variant_id": pa.array(vid, pa.string()),
        "ref": pa.array(ref_l, pa.string()),
        "alt": pa.array(alt_l, pa.string()),
        "kind": pa.array(kind_l, pa.string()),
        "gene_id": pa.array([""] * n, pa.string()),
        "consequence": pa.array([""] * n, pa.string()),
        "af": pa.array(np.zeros(n, dtype=np.float32), pa.float32()),
    }, schema=FEATURE_SCHEMA)
    _pq.write_table(tbl, dest, compression="zstd")
    return dest


# ---------------------------------------------------------------------------
# Standalone structural-variant callsets
# ---------------------------------------------------------------------------


def _feature_table(contig, start, end, vid, kind, af=None) -> pa.Table:
    n = len(start)
    return pa.table({
        "contig": pa.array(contig, pa.string()),
        "start": pa.array(start, pa.int32()),
        "end": pa.array(end, pa.int32()),
        "variant_id": pa.array(vid, pa.string()),
        "ref": pa.array([""] * n, pa.string()),
        "alt": pa.array([""] * n, pa.string()),
        "kind": pa.array(kind, pa.string()),
        "gene_id": pa.array([""] * n, pa.string()),
        "consequence": pa.array([""] * n, pa.string()),
        "af": pa.array(af if af is not None else np.zeros(n, np.float32), pa.float32()),
    }, schema=FEATURE_SCHEMA)


def _info(info: str, key: str) -> str:
    i = info.find(key + "=")
    while i > 0 and info[i - 1] != ";":
        i = info.find(key + "=", i + 1)
    if i < 0:
        return ""
    j = info.find(";", i)
    return info[i + len(key) + 1: j if j >= 0 else None]


def read_sv_vcf(path: str | os.PathLike, *, contigs: tuple[str, ...] | None = None) -> pa.Table:
    """A sites-only SV VCF (gnomAD-SV, 1000G SV), reading only what a layout needs.

    The span is ``END`` when the record stays on one contig. Breakends and
    translocations (``CHR2`` differs) are stored at their own position with a
    one-base span, because ``END`` then names a coordinate on another contig.
    """
    keep = set(contigs) if contigs else None
    c_l, s_l, e_l, id_l, k_l, af_l = [], [], [], [], [], []
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 8)
            chrom = f[0] if f[0].startswith("chr") else "chr" + f[0]
            if keep is not None and chrom not in keep:
                continue
            info = f[7]
            s = int(f[1]) - 1
            chr2 = _info(info, "CHR2")
            end = _info(info, "END")
            same = not chr2 or chr2.removeprefix("chr") == chrom.removeprefix("chr")
            e = int(end) if (end.isdigit() and same) else s + max(len(f[3]), 1)
            e = max(e, s + 1)
            af = _info(info, "AF").split(",")[0]
            c_l.append(chrom); s_l.append(s); e_l.append(e)
            id_l.append(f[2])
            k_l.append(_info(info, "SVTYPE") or "sv")
            try:
                af_l.append(float(af))
            except ValueError:
                af_l.append(0.0)
    return _feature_table(c_l, s_l, e_l, id_l, k_l, np.array(af_l, np.float32))


def read_dgv(path: str | os.PathLike, *, contigs: tuple[str, ...] | None = None) -> pa.Table:
    """UCSC ``dgvMerged`` dump of the Database of Genomic Variants (0-based)."""
    keep = set(contigs) if contigs else None
    c_l, s_l, e_l, id_l, k_l = [], [], [], [], []
    with _open_maybe_gzip(path) as fh:
        for line in fh:
            f = line.split("\t", 11)
            if len(f) < 11 or "_" in f[1]:
                continue
            if keep is not None and f[1] not in keep:
                continue
            s, e = int(f[2]), int(f[3])
            c_l.append(f[1]); s_l.append(s); e_l.append(max(e, s + 1))
            id_l.append(f[4]); k_l.append(f[10])
    return _feature_table(c_l, s_l, e_l, id_l, k_l)
