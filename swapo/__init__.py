"""SWaPO: a semantic, workload- and provenance-aware optimizer for genomic tables.

The short version of what this package argues:

Generic lakehouse optimizers already choose clustering keys from observed query
predicates and weigh the saving against the cost of the rewrite. What they do
not do is understand that a genomic overlap predicate is two-sided, that a small
tail of very long records destroys the statistic the second side depends on, or
that a materialized genomic aggregate is a dated scientific claim rather than a
cache. SWaPO adds those three things and nothing else.

Typical use::

    from swapo import DatasetSpec, simulate_features, write_layout, Layout
    from swapo import Engine, SWaPO, summarise, WorkloadLog

    features = simulate_features(DatasetSpec(n_features=500_000))
    written = write_layout(features, "data/variants", Layout.PARTITIONED)

    log = WorkloadLog()
    log.record_sql("SELECT * FROM v WHERE contig = 'chr7' "
                   "AND start < 140800000 AND 140700000 < end")

    opt = SWaPO()
    for rec in opt.recommend("variants", log.profile(),
                             summarise(features, n_bytes=written.n_bytes),
                             written.zonemap):
        print(rec.explain())
"""

from .blocksize import BlockSizeChoice, choose_block_size, fit_block_cost
from .rowgroups import RowGroupChoice, choose_row_group_size
from .objstore import StoreCost, measure_remote
from .cost import CostModel, LayoutShape, expected_blocks
from .data import (DatasetSpec, read_bed, read_vcf, simulate_features,
                   simulate_genes, simulate_genotypes, simulate_samples)
from .engine import Engine, QueryResult
from .layout import Layout, LayoutResult, load_layout, write_layout
from .model import (Assembly, Block, GRCH38, Intent, Query, Region, TableStats)
from .optimizer import Action, Recommendation, SWaPO
from .provenance import Dataset, DatasetKind, Freshness, Lineage
from .real import SOURCES, combine, fetch, harmonise_contigs, read_gtf, read_rmsk
from .stats import length_histogram, summarise
from .threshold import ThresholdChoice, choose_threshold, separability
from .workload import (WorkloadLog, WorkloadProfile, classify, generate_workload)
from .zonemap import IntervalZoneMap, PruneResult

__version__ = "0.1.0"

__all__ = [
    "Action", "Assembly", "Block", "CostModel", "Dataset", "DatasetKind",
    "DatasetSpec", "Engine", "Freshness", "GRCH38", "Intent", "IntervalZoneMap",
    "Layout", "LayoutResult", "LayoutShape", "Lineage", "PruneResult", "Query",
    "QueryResult", "Recommendation", "Region", "SWaPO", "TableStats",
    "WorkloadLog", "WorkloadProfile", "__version__", "classify",
    "expected_blocks", "generate_workload", "length_histogram", "load_layout",
    "BlockSizeChoice", "choose_block_size", "fit_block_cost",
    "RowGroupChoice", "choose_row_group_size", "StoreCost", "measure_remote",
    "SOURCES", "ThresholdChoice", "choose_threshold", "combine", "fetch",
    "harmonise_contigs", "read_bed", "read_gtf", "read_rmsk", "read_vcf",
    "separability", "simulate_features", "simulate_genes",
    "simulate_genotypes", "simulate_samples", "summarise", "write_layout",
]
