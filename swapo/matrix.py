"""The row-group experiment over every real table and every real query shape.

Five tables, chosen to differ on the things that could change the answer:

==================  ======================================================  ========
table               contents                                                assembly
==================  ======================================================  ========
grch38_platform     GENCODE + ClinVar + RepeatMasker, all primary contigs    GRCh38
gnomad_window       gnomAD v4.1 genomes, 2.33 Mb of chr21                    GRCh38
grch37_1kg_chr21    1000 Genomes phase 3 chr21 + GENCODE v44lift37 chr21     GRCh37
gnomad_sv           gnomAD-SV v4.1, every site, standalone SV callset        GRCh38
dgv                 Database of Genomic Variants (merged), standalone CNVs   GRCh38
==================  ======================================================  ========

Each table is read once from the published files and cached as Parquet, with
the SHA-256 of every input recorded beside it.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import real_queries as rq
from .data import read_vcf
from .real import (combine, describe_file, harmonise_contigs, read_dgv, read_gtf,
                   read_rmsk, read_sv_vcf)

PRIMARY = tuple(f"chr{c}" for c in [*range(1, 23), "X", "Y"])
GNOMAD_WINDOW = ("chr21", 16_918_424, 19_246_885)


def _primary(t: pa.Table) -> pa.Table:
    t = harmonise_contigs(t)
    return t.filter(pc.is_in(t.column("contig"), value_set=pa.array(PRIMARY)))


def _build(name: str, d: Path) -> tuple[pa.Table, list[dict], str]:
    if name == "grch38_platform":
        files = ["gencode.v44.basic.annotation.gtf.gz", "clinvar.vcf.gz", "rmsk.txt.gz"]
        t = combine(_primary(read_gtf(d / files[0])), _primary(read_vcf(d / files[1])),
                    _primary(read_rmsk(d / files[2])))
        return t, files, "GRCh38"
    if name == "gnomad_window":
        return pq.read_table(d / "gnomad.chr21.window.parquet"), \
            ["gnomad.chr21.window.parquet"], "GRCh38"
    if name == "grch37_1kg_chr21":
        files = ["1kg.chr21.vcf.gz", "gencode.v44lift37.basic.annotation.gtf.gz"]
        t = combine(_primary(read_vcf(d / files[0])),
                    read_gtf(d / files[1], contigs=("chr21",)), contigs=("chr21",))
        return t, files, "GRCh37"
    if name == "gnomad_sv":
        f = "gnomad.v4.1.sv.sites.vcf.gz"
        return _primary(read_sv_vcf(d / f)), [f], "GRCh38"
    if name == "dgv":
        f = "hg38.dgvMerged.txt.gz"
        return _primary(read_dgv(d / f)), [f], "GRCh38"
    raise KeyError(name)


TABLES = ("grch38_platform", "gnomad_window", "grch37_1kg_chr21", "gnomad_sv", "dgv")


def load_table(name: str, data: str | os.PathLike) -> tuple[pa.Table, dict]:
    d = Path(data)
    cache = d / "tables" / f"{name}.parquet"
    meta_p = cache.with_suffix(".json")
    if cache.exists() and meta_p.exists():
        return pq.read_table(cache), json.loads(meta_p.read_text())
    t, files, assembly = _build(name, d)
    cache.parent.mkdir(exist_ok=True)
    pq.write_table(t, cache)
    span = np.asarray(pc.subtract(t.column("end"), t.column("start"))).astype(np.int64)
    meta = {"table": name, "assembly": assembly, "rows": t.num_rows,
            "contigs": sorted(set(t.column("contig").to_pylist())),
            "span_pct": {str(q): int(np.percentile(span, q))
                         for q in (50, 90, 99, 99.9, 99.99, 100)},
            "inputs": [describe_file(d / f) for f in files]}
    meta_p.write_text(json.dumps(meta, indent=2))
    return t, meta


def query_sets(name: str, meta: dict, data: str | os.PathLike, n: int = 100,
               seed: int = 7) -> dict[str, list]:
    d = Path(data)
    assembly = meta["assembly"]
    cache = d / f"queries_{assembly.lower()}.pkl"
    if cache.exists():
        sets = pickle.loads(cache.read_bytes())
    else:
        sets = rq.all_sets(d, assembly=assembly)
        cache.write_bytes(pickle.dumps(sets))
    window = GNOMAD_WINDOW if name == "gnomad_window" else None
    contigs = set(meta["contigs"])
    out = {k: rq.sample(v, n, seed=seed, contigs=contigs, window=window)
           for k, v in sets.items()}
    return {k: v for k, v in out.items() if v}
