"""Workload observation: turning a query log into genomic intent.

A generic optimizer records which columns were filtered and how selective the
filters were. That is enough to choose a clustering key, and Databricks'
automatic liquid clustering does exactly that. What it cannot see is that
``contig = 'chr13' AND start < 32400268 AND 32315086 < end`` is *one gene*, that
the gene is 85 kb long, and that the next hundred queries will be other genes of
similar size rather than arbitrary windows.

That difference matters because the right layout for "many narrow windows
scattered across the genome" is not the right layout for "a few whole-arm
sweeps", even though both filter the same three columns.

This module is deliberately pragmatic: it pattern-matches a query log rather
than parsing SQL properly. A production version would read the engine's parsed
plan. The classification it produces is the input to :mod:`swapo.optimizer`.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median

import numpy as np

from .model import GRCH38, Intent, Query, Region

__all__ = ["classify", "WorkloadLog", "WorkloadProfile", "generate_workload"]

# Windows at or below this are "a gene or smaller"; above it, a region sweep.
GENE_SCALE = 3_000_000

_RE_CONTIG = re.compile(r"""contig\s*=\s*'([^']+)'""", re.I)
_RE_START_LT = re.compile(r"""start\s*<\s*(\d+)""", re.I)
# Both spellings of the lower bound: `S < "end"` and `"end" > S`.
_RE_END_GT = re.compile(r"""(\d+)\s*<\s*"?end"?|"?\bend"?\s*>\s*(\d+)""", re.I)
_RE_POS_EQ = re.compile(r"""\b(?:start|pos)\s*=\s*(\d+)""", re.I)
_RE_SAMPLE = re.compile(r"""sample_id\s*(?:=|in)\s*""", re.I)
_RE_COHORT = re.compile(r"""cohort\s*=\s*'([^']+)'""", re.I)
_RE_GENE = re.compile(r"""(?:gene_id|symbol)\s*=\s*'([^']+)'""", re.I)
_RE_GROUP_VARIANT = re.compile(r"""group\s+by\s+variant_id""", re.I)
_RE_PHENO = re.compile(r"""\bphenotype\b|\bgroup\s+by\s+cohort\b""", re.I)


def classify(sql: str, gene_lookup: dict[str, Region] | None = None) -> Query:
    """Infer the genomic intent of one observed SQL statement."""
    contig_m = _RE_CONTIG.search(sql)
    gene_m = _RE_GENE.search(sql)
    sample = bool(_RE_SAMPLE.search(sql))
    cohort_m = _RE_COHORT.search(sql)

    region: Region | None = None
    if contig_m:
        contig = contig_m.group(1)
        lt, gt = _RE_START_LT.search(sql), _RE_END_GT.search(sql)
        eq = _RE_POS_EQ.search(sql)
        if lt and gt:
            region = Region(contig, int(gt.group(1) or gt.group(2)), int(lt.group(1)))
        elif eq:
            p = int(eq.group(1))
            region = Region(contig, p, p + 1)

    gene = gene_m.group(1) if gene_m else None
    if gene and gene_lookup and gene in gene_lookup and region is None:
        region = gene_lookup[gene]

    if _RE_GROUP_VARIANT.search(sql):
        intent = Intent.COHORT_FREQ
    elif _RE_PHENO.search(sql):
        intent = Intent.PHENOTYPE_JOIN
    elif gene:
        intent = Intent.GENE
    elif sample:
        intent = Intent.SAMPLE
    elif region is None:
        intent = Intent.SCAN
    elif region.length <= 2:
        intent = Intent.POINT
    elif region.length > GENE_SCALE:
        intent = Intent.LARGE_REGION
    else:
        intent = Intent.INTERVAL

    return Query(
        intent=intent, region=region, gene=gene,
        cohort=cohort_m.group(1) if cohort_m else None,
        sample_ids=("*",) if sample else (),
    )


@dataclass
class WorkloadProfile:
    """What the observed workload looks like, as the optimizer sees it."""

    n_queries: int
    counts: dict[Intent, int]
    #: Fraction of queries whose cost is dominated by interval overlap.
    interval_share: float
    median_width: int
    p95_width: int
    #: Fraction of interval queries narrower than a typical gene.
    narrow_share: float
    contigs: dict[str, int] = field(default_factory=dict)
    samples_share: float = 0.0
    cohort_share: float = 0.0

    def share(self, intent: Intent) -> float:
        return self.counts.get(intent, 0) / self.n_queries if self.n_queries else 0.0

    def summary(self) -> dict:
        return {
            "queries": self.n_queries,
            "interval_share": round(self.interval_share, 3),
            "median_width_bp": self.median_width,
            "p95_width_bp": self.p95_width,
            "narrow_share": round(self.narrow_share, 3),
            "sample_share": round(self.samples_share, 3),
            "cohort_share": round(self.cohort_share, 3),
            "by_intent": {k.value: v for k, v in sorted(
                self.counts.items(), key=lambda kv: -kv[1])},
        }


class WorkloadLog:
    """Accumulates observed queries and summarises them."""

    def __init__(self) -> None:
        self._queries: list[Query] = []

    def record(self, q: Query) -> None:
        self._queries.append(q)

    def record_sql(self, sql: str, gene_lookup: dict[str, Region] | None = None) -> Query:
        q = classify(sql, gene_lookup)
        self.record(q)
        return q

    def extend(self, queries: list[Query]) -> None:
        self._queries.extend(queries)

    @property
    def queries(self) -> list[Query]:
        return list(self._queries)

    def profile(self) -> WorkloadProfile:
        from .model import INTERVAL_INTENTS

        n = len(self._queries)
        counts = Counter(q.intent for q in self._queries)
        widths = [q.region.length for q in self._queries if q.region is not None]
        contigs: Counter[str] = Counter(
            q.region.contig for q in self._queries if q.region is not None
        )
        interval_n = sum(counts[i] for i in INTERVAL_INTENTS)
        narrow = sum(1 for w in widths if w <= GENE_SCALE)
        return WorkloadProfile(
            n_queries=n,
            counts=dict(counts),
            interval_share=interval_n / n if n else 0.0,
            median_width=int(median(widths)) if widths else 0,
            p95_width=int(np.percentile(widths, 95)) if widths else 0,
            narrow_share=narrow / len(widths) if widths else 0.0,
            contigs=dict(contigs),
            samples_share=sum(1 for q in self._queries if q.sample_ids) / n if n else 0.0,
            cohort_share=sum(1 for q in self._queries if q.cohort) / n if n else 0.0,
        )

    def __len__(self) -> int:
        return len(self._queries)


# ---------------------------------------------------------------------------
# A realistic query mix for the benchmark
# ---------------------------------------------------------------------------

#: Roughly what a variant-interpretation service sees in a day: mostly narrow
#: lookups around genes and reported positions, a minority of wider sweeps, and
#: a steady trickle of per-sample and cohort work.
DEFAULT_MIX: dict[Intent, float] = {
    Intent.POINT: 0.30,
    Intent.GENE: 0.26,
    Intent.INTERVAL: 0.22,
    Intent.LARGE_REGION: 0.06,
    Intent.SAMPLE: 0.08,
    Intent.COHORT_FREQ: 0.05,
    Intent.PHENOTYPE_JOIN: 0.02,
    Intent.SCAN: 0.01,
}

#: A workload that has drifted towards population-scale analysis: fewer single
#: lookups, far more cohort work. Used to test whether a layout chosen for the
#: first mix is still the right one later.
DRIFTED_MIX: dict[Intent, float] = {
    Intent.POINT: 0.08,
    Intent.GENE: 0.12,
    Intent.INTERVAL: 0.15,
    Intent.LARGE_REGION: 0.15,
    Intent.SAMPLE: 0.20,
    Intent.COHORT_FREQ: 0.25,
    Intent.PHENOTYPE_JOIN: 0.04,
    Intent.SCAN: 0.01,
}


def generate_workload(
    n: int = 200,
    *,
    genes: list[tuple[str, Region]] | None = None,
    samples: list[str] | None = None,
    mix: dict[Intent, float] | None = None,
    contigs: tuple[str, ...] | None = None,
    seed: int = 42,
) -> list[Query]:
    """Draw ``n`` queries from a mix of genomic intents.

    ``contigs`` must be the set the data actually covers. Drawing queries from
    contigs that hold no rows would make every layout look excellent, because
    the honest answer to such a query is to read nothing at all.
    """
    rng = np.random.default_rng(seed)
    mix = mix or DEFAULT_MIX
    intents = list(mix)
    probs = np.array([mix[i] for i in intents], dtype=float)
    probs = probs / probs.sum()

    names = list(contigs) if contigs else list(GRCH38)
    contig_arr = np.array(names)
    clen = np.array([GRCH38[c] for c in names], dtype=np.int64)
    cp = clen / clen.sum()

    out: list[Query] = []
    for _ in range(n):
        intent = intents[int(rng.choice(len(intents), p=probs))]
        ci = int(rng.choice(len(contig_arr), p=cp))
        contig, length = str(contig_arr[ci]), int(clen[ci])

        if intent is Intent.GENE and genes:
            symbol, region = genes[int(rng.integers(0, len(genes)))]
            out.append(Query(intent, region=region, gene=symbol))
            continue

        if intent in (Intent.POINT, Intent.INTERVAL, Intent.LARGE_REGION,
                      Intent.ANNOTATION_JOIN):
            pos = int(rng.random() * max(length - 5_000_000, 1))
            width = {
                Intent.POINT: 1,
                Intent.INTERVAL: int(10 ** rng.uniform(3, 5.5)),
                Intent.LARGE_REGION: min(int(10 ** rng.uniform(6.3, 7.0)), length // 4),
                Intent.ANNOTATION_JOIN: int(10 ** rng.uniform(4, 6)),
            }[intent]
            out.append(Query(intent, region=Region(contig, pos, pos + width)))
            continue

        if intent is Intent.SAMPLE and samples:
            sid = samples[int(rng.integers(0, len(samples)))]
            pos = int(rng.random() * max(length - 5_000_000, 1))
            out.append(Query(intent, region=Region(contig, pos, pos + 2_000_000),
                             sample_ids=(sid,), table="genotypes"))
            continue

        if intent in (Intent.COHORT_FREQ, Intent.PHENOTYPE_JOIN):
            pos = int(rng.random() * max(length - 5_000_000, 1))
            out.append(Query(intent, region=Region(contig, pos, pos + 1_000_000),
                             cohort="discovery", table="genotypes"))
            continue

        out.append(Query(Intent.SCAN, table="features"))
    return out
