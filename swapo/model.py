"""Core types: genomic intervals, storage blocks, query intents.

Coordinates are half-open and 0-based throughout (``[start, end)``), the BED
convention. VCF is 1-based and closed, so :func:`swapo.data.read_vcf` converts on
the way in. Mixing the two is the most common source of off-by-one errors in
genomics, so the convention is enforced in one place and asserted in the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Reference assemblies
# ---------------------------------------------------------------------------

#: Contig lengths for GRCh38, primary assembly (bp). Source: GRCh38.p14 report.
GRCH38: dict[str, int] = {
    "chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559,
    "chr4": 190_214_555, "chr5": 181_538_259, "chr6": 170_805_979,
    "chr7": 159_345_973, "chr8": 145_138_636, "chr9": 138_394_717,
    "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189,
    "chr16": 90_338_345, "chr17": 83_257_441, "chr18": 80_373_285,
    "chr19": 58_617_616, "chr20": 64_444_167, "chr21": 46_709_983,
    "chr22": 50_818_468, "chrX": 156_040_895, "chrY": 57_227_415,
}

CONTIG_ORDER: list[str] = list(GRCH38)
CONTIG_INDEX: dict[str, int] = {c: i for i, c in enumerate(CONTIG_ORDER)}


class Assembly(str, Enum):
    """A layout is only valid for the assembly it was built from."""

    GRCH38 = "GRCh38"
    GRCH37 = "GRCh37"


# ---------------------------------------------------------------------------
# Query intents
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    """The genomic access patterns SWaPO distinguishes.

    A generic optimizer sees predicates on columns. SWaPO sees the biological
    question, because the right physical layout differs between them even when
    the predicate columns are identical. ``POINT`` and ``INTERVAL`` both filter
    ``contig``/``start``/``end``, but a point lookup wants tight blocks near one
    coordinate while a cohort aggregate wants every sample for a few variants.
    """

    POINT = "point"                  # one coordinate: chr7:140753336
    INTERVAL = "interval"            # a window: chr7:140,700,000-140,800,000
    GENE = "gene"                    # by gene symbol, resolved to an interval
    LARGE_REGION = "large_region"    # a cytoband or whole arm
    SAMPLE = "sample"                # all variants for one sample
    COHORT_FREQ = "cohort_freq"      # allele frequency across a cohort
    ANNOTATION_JOIN = "annotation"   # variants joined to a feature track
    PHENOTYPE_JOIN = "phenotype"     # genotypes joined to sample phenotypes
    SCAN = "scan"                    # unselective, whole-table


#: Intents whose cost is dominated by interval overlap tests.
INTERVAL_INTENTS = frozenset(
    {Intent.POINT, Intent.INTERVAL, Intent.GENE, Intent.LARGE_REGION,
     Intent.ANNOTATION_JOIN}
)


@dataclass(frozen=True, slots=True)
class Region:
    """A half-open genomic interval on one contig."""

    contig: str
    start: int
    end: int

    def __post_init__(self) -> None:
        # Coordinates often arrive as numpy integers from a dataframe. Database
        # drivers reject those, so normalise once, here.
        object.__setattr__(self, "contig", str(self.contig))
        object.__setattr__(self, "start", int(self.start))
        object.__setattr__(self, "end", int(self.end))
        if self.end < self.start:
            raise ValueError(f"end {self.end} precedes start {self.start}")

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, start: int, end: int) -> bool:
        """Half-open overlap: ``[a, b)`` meets ``[c, d)`` when ``a < d and c < b``."""
        return self.start < end and start < self.end

    @classmethod
    def parse(cls, text: str) -> "Region":
        """Parse ``chr7:140,700,000-140,800,000`` or ``chr7:140753336``.

        The input is 1-based inclusive (what a genome browser shows) and the
        result is 0-based half-open.
        """
        contig, _, rest = text.partition(":")
        if not rest:
            raise ValueError(f"no coordinates in {text!r}")
        rest = rest.replace(",", "").replace("_", "")
        lo, _, hi = rest.partition("-")
        start = int(lo) - 1
        end = int(hi) if hi else int(lo)
        return cls(contig, start, end)

    def __str__(self) -> str:
        return f"{self.contig}:{self.start + 1:,}-{self.end:,}"


@dataclass(frozen=True, slots=True)
class Query:
    """One observed query: what was asked, and what it cost."""

    intent: Intent
    region: Region | None = None
    sample_ids: tuple[str, ...] = ()
    cohort: str | None = None
    gene: str | None = None
    table: str = "features"
    #: How many times this shape was seen in the observation window.
    frequency: int = 1

    def key(self) -> tuple[Any, ...]:
        """Shape identity, ignoring the particular coordinates."""
        return (self.table, self.intent, bool(self.sample_ids), self.cohort is not None)


# ---------------------------------------------------------------------------
# Storage blocks
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Block:
    """One physical unit of storage, with the statistics used to skip it.

    In this implementation a block is one Parquet file, which keeps pruning
    decisions explicit and measurable: the engine is handed a file list and
    reads nothing else. In a Delta or Iceberg deployment a block maps to a row
    group and the same statistics live in the manifest.
    """

    path: str
    contig: str
    start_min: int
    start_max: int
    #: ``max(end)`` over the rows in this block. This is the statistic that
    #: decides whether a two-sided overlap test can skip the block, and the one
    #: a single long interval destroys.
    end_max: int
    n_rows: int
    n_bytes: int
    #: True when the block holds intervals longer than the split threshold.
    #: These are kept apart so they cannot inflate ``end_max`` for dense blocks.
    long_tail: bool = False

    def overlaps(self, start: int, end: int) -> bool:
        """Could this block hold a row overlapping ``[start, end)``?

        Both sides matter. ``start_min < end`` rejects blocks that begin after
        the query; ``start < end_max`` rejects blocks whose rows all finish
        before it. Neither test alone is enough, and the second is the one that
        needs a layout that keeps ``end_max`` tight.
        """
        return self.start_min < end and start < self.end_max

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TableStats:
    """Distribution facts the cost model needs."""

    n_rows: int = 0
    n_bytes: int = 0
    n_blocks: int = 0
    #: Fraction of rows longer than the split threshold.
    long_fraction: float = 0.0
    #: Longest interval seen, in bp.
    max_length: int = 0
    #: Rows per contig, for skew.
    rows_per_contig: dict[str, int] = field(default_factory=dict)

    @property
    def contig_skew(self) -> float:
        """Max/mean rows per contig. 1.0 is even; higher means one contig dominates."""
        if not self.rows_per_contig:
            return 1.0
        counts = list(self.rows_per_contig.values())
        mean = sum(counts) / len(counts)
        return max(counts) / mean if mean else 1.0
