"""Query sets built from real public resources.

No genomic lakehouse publishes its query log, so the workloads here are the
coordinates people actually ask about, taken from the resources that define
them. Each is a different shape:

=============  ======================================  ===================
name           source                                  typical width
=============  ======================================  ===================
``gene``       GENCODE protein-coding gene bodies       10 kb to 2 Mb
``variant``    ClinVar pathogenic / likely pathogenic   1 bp
``gwas``       GWAS Catalog lead SNP, +/- 250 kb        500 kb
``panel``      exons of the ACMG SF v3.2 genes          100 bp
``band``       UCSC cytogenetic bands                   1 to 20 Mb
=============  ======================================  ===================

All coordinates are 0-based half-open, on the assembly the source file is on.
"""

from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np

__all__ = ["ACMG_SF_V32", "gene_queries", "panel_queries", "variant_queries",
           "gwas_queries", "band_queries", "sample", "all_sets"]

Query = tuple[str, int, int]

#: ACMG secondary findings list v3.2 (Miller et al., Genet Med 2023).
ACMG_SF_V32 = frozenset("""
ACTA2 ACTC1 ACVRL1 APC APOB ATP7B BAG3 BMPR1A BRCA1 BRCA2 BTD CACNA1S CALM1 CALM2
CALM3 CASQ2 COL3A1 DES DSC2 DSG2 DSP ENG FBN1 FLNC GAA GLA HFE HNF1A KCNH2 KCNQ1
LDLR LMNA MAX MEN1 MLH1 MSH2 MSH6 MUTYH MYBPC3 MYH11 MYH7 MYL2 MYL3 NF2 OTC PALB2
PCSK9 PKP2 PMS2 PRKAG2 PTEN RB1 RBM20 RET RPE65 RYR1 RYR2 SCN5A SDHAF2 SDHB SDHC
SDHD SMAD3 SMAD4 STK11 TGFBR1 TGFBR2 TMEM127 TMEM43 TNNC1 TNNI3 TNNT2 TP53 TPM1
TRDN TSC1 TSC2 TTN TTR VHL WT1
""".split())


def _open(path: str | os.PathLike):
    p = str(path)
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.endswith(".gz") \
        else open(p, encoding="utf-8", errors="replace")


def _attr(attrs: str, key: str) -> str:
    i = attrs.find(key + ' "')
    if i < 0:
        return ""
    j = attrs.find('"', i + len(key) + 2)
    return attrs[i + len(key) + 2:j]


def _chr(c: str) -> str:
    c = c if c.startswith("chr") else "chr" + c
    return "chrM" if c in ("chrMT",) else c


def gene_queries(gtf: str | os.PathLike) -> list[Query]:
    """Protein-coding gene bodies. GTF is 1-based inclusive."""
    out = []
    with _open(gtf) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 9)
            if len(f) < 9 or f[2] != "gene" or 'gene_type "protein_coding"' not in f[8]:
                continue
            out.append((_chr(f[0]), int(f[3]) - 1, int(f[4])))
    return out


def panel_queries(gtf: str | os.PathLike, genes=ACMG_SF_V32) -> list[Query]:
    """Distinct exons of the panel genes on their Ensembl canonical transcript."""
    seen, out = set(), []
    with _open(gtf) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 9)
            if len(f) < 9 or f[2] != "exon" or "Ensembl_canonical" not in f[8]:
                continue
            if _attr(f[8], "gene_name") not in genes:
                continue
            q = (_chr(f[0]), int(f[3]) - 1, int(f[4]))
            if q not in seen:
                seen.add(q)
                out.append(q)
    return out


def variant_queries(clinvar_vcf: str | os.PathLike) -> list[Query]:
    """ClinVar pathogenic and likely pathogenic sites, as the REF span."""
    out = []
    with _open(clinvar_vcf) as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.split("\t", 8)
            info = f[7]
            i = info.find("CLNSIG=")
            if i < 0:
                continue
            sig = info[i + 7:info.find(";", i) if info.find(";", i) > 0 else None]
            if not sig.startswith(("Pathogenic", "Likely_pathogenic")):
                continue
            s = int(f[1]) - 1
            out.append((_chr(f[0]), s, s + max(len(f[3]), 1)))
    return out


def gwas_queries(ucsc_gwas: str | os.PathLike, flank: int = 250_000) -> list[Query]:
    """GWAS Catalog lead SNPs (UCSC dump, 0-based) widened to a locus window."""
    seen, out = set(), []
    with _open(ucsc_gwas) as fh:
        for line in fh:
            f = line.split("\t", 5)
            c, s = f[1], int(f[2])
            if "_" in c or (c, s) in seen:
                continue
            seen.add((c, s))
            out.append((c, max(0, s - flank), s + flank))
    return out


def band_queries(ucsc_cytoband: str | os.PathLike) -> list[Query]:
    out = []
    with _open(ucsc_cytoband) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) >= 3 and "_" not in f[0]:
                out.append((f[0], int(f[1]), int(f[2])))
    return out


def sample(queries: list[Query], n: int, *, seed: int = 0,
           contigs: set[str] | None = None,
           window: tuple[str, int, int] | None = None) -> list[Query]:
    """``n`` queries, restricted to contigs or a window the table actually covers."""
    pool = queries
    if contigs is not None:
        pool = [q for q in pool if q[0] in contigs]
    if window is not None:
        c, lo, hi = window
        pool = [q for q in pool if q[0] == c and q[1] < hi and lo < q[2]]
    if len(pool) <= n:
        return list(pool)
    idx = np.random.default_rng(seed).choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(idx)]


def all_sets(data: str | os.PathLike, *, assembly: str = "GRCh38") -> dict[str, list[Query]]:
    """Every query set available for ``assembly`` under ``data``."""
    d = Path(data)
    if assembly == "GRCh38":
        gtf = d / "gencode.v44.basic.annotation.gtf.gz"
        sets = {"gene": gene_queries(gtf), "panel": panel_queries(gtf),
                "variant": variant_queries(d / "clinvar.vcf.gz"),
                "gwas": gwas_queries(d / "hg38.gwasCatalog.txt.gz"),
                "band": band_queries(d / "hg38.cytoBand.txt.gz")}
    else:
        gtf = d / "gencode.v44lift37.basic.annotation.gtf.gz"
        sets = {"gene": gene_queries(gtf), "panel": panel_queries(gtf)}
        if (d / "hg19.cytoBand.txt.gz").exists():
            sets["band"] = band_queries(d / "hg19.cytoBand.txt.gz")
    return sets
