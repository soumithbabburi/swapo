# When Does Interval-Aware Physical Design Pay? A Measurement Study on Real Genomic Data

**Abstract.** Genomic lakehouse queries are dominated by interval-overlap
predicates, which are two-sided and therefore depend on a per-block `max(end)`
statistic that a small number of very long records can destroy. The standard
remedy from the interval-search literature — separate the long, *containing*
records so they cannot distort the rest — suggests an obvious physical-design
optimization for columnar genomic tables. We implement it, and we measure it.
On synthetic data it is dramatic: interval queries read 69% fewer bytes. On tables
assembled from six real public sources (GENCODE v44, ClinVar, UCSC
RepeatMasker, and the 1000 Genomes structural-variant callset and chr21
population callset, and a window of the gnomAD v4.1 chr21 genomes callset) it
yields **nothing at all**: on a 1,046,208-row chr1 table
at every block size from 8,000 rows down to 30, and on a 1,245,415-row chr21
table at 26.7 records per kilobase, and on a 650,304-row table at **279 records
per kilobase**, the densest human variant data available. We explain the discrepancy
and reduce it to a single measurable property: the optimization requires a
*bimodal* interval-length distribution, and real genomic tables are continuous
and heavy-tailed. The ratio of the 99.99th to the 99th percentile span, which
costs one pass to compute, is 75 to 407 across four real tables and 39,701 in
the synthetic one, and it separates the regimes where block tightness alone
does not: our chr21 table reaches a tightness at which synthetic data gains
15%, and still gains nothing. We contribute: an implementation; a
cost-based threshold search that decides whether to act; evidence that it
decides correctly in 32 of 32 measured cases, including all 28 refusals; and a
caution that synthetic evaluation of genomic layout optimizations can be
misleading by a wide margin. All code, data provenance and experiments are
reproducible from the repository.

**Keywords:** physical database design, genomic data management, interval
queries, data skipping, lakehouse, negative results, reproducibility

---

## 1. Introduction

A genomic data platform answers questions of the form "what is in this region?"
Variant browsers, clinical interpretation pipelines, annotation joins and
burden tests all reduce to the same predicate over a half-open interval:

```sql
WHERE contig = 'chr13' AND start < 32400268 AND 32315086 < "end"
```

Columnar table formats skip a block of rows when per-column minima and maxima
show the block cannot satisfy a predicate [Armbrust et al. 2020]. Ordering rows
by `start` makes the first clause effective: blocks beginning after the query
ends are skipped by binary search. The second clause, `query.start < end`, is
the only thing that prevents the reader from scanning every block from the
beginning of the contig, and the only statistic it can use is `max(end)` for
the block.

This is where genomics differs from the workloads for which columnar pruning
was designed. Interval lengths in genomics range over six orders of magnitude.
A table that holds exons beside gene bodies, or SNVs beside structural
variants, contains records whose spans differ by a factor of a million. One
record spanning two megabases raises `max(end)` for every row stored with it.
If each block contains such a record, the lower bound is worthless.

The remedy is known. Nested containment lists [Alekseyenko and Lee 2007] and
the Augmented Interval List [Feng et al. 2019] both sort by start, carry a
running maximum of end, and lift *containing* intervals into a component of
their own precisely so that they cannot distort the dense majority. Applying
this at the level of storage blocks, rather than individual intervals, is a
natural idea: it composes with ordinary columnar pruning, needs no new file
format, and can be decided by a cost model.

We built it. This paper reports what happened when we measured it on real data,
which is not what we expected.

### 1.1 Contributions

1. **An implementation.** Augmented interval zone maps over Parquet blocks,
   with binary-searchable bounds on both sides of the overlap test, and a
   layout that writes long records to separate blocks (§4).
2. **A cost-based decision procedure.** A what-if search that chooses the split
   threshold, or declines, without writing any data (§4.3). Prior work fixes
   such thresholds by convention; we show convention is wrong by two orders of
   magnitude on annotation data.
2b. **A block-size optimiser that does pay** (§7.3): 48–87% fewer bytes read on
   three real tables, selecting the same block size an exhaustive measured sweep
   finds, with forecast error under 2.2% and no data written during the search.
3. **A measurement study on real public data** (§6): five sources, two contigs,
   2.3M rows in total, thirteen block-size configurations, spanning a 6.4-fold
   range of record density, with file-level provenance.
4. **A negative result with a mechanism** (§7). The optimization does not pay
   on real genomic tables, and we identify exactly why, in terms of a statistic
   that predicts it.
5. **Evidence the decision procedure is reliable**: 32 of 32 correct decisions,
   including all 28 cases where the correct decision is to do nothing (§6.4),
   across two contigs, five sources and a single-assembly control.
6. **A methodological caution** (§8). Our synthetic benchmark overstated the
   benefit of this optimization from 0% to 69%, and it was not built in bad
   faith; it was built from a plausible-sounding length distribution.

### 1.2 What this paper does not claim

We do not claim novelty for workload-driven physical design, which dates to
AutoAdmin [Chaudhuri and Narasayya 1997] and now ships in production
[Nakandala et al. 2026]. We do not claim a new file format, index or variant
caller. We have not benchmarked against Hail, Glow, TileDB-VCF or GenomicsDB.
Our measurements include a real population callset (1000 Genomes chr21) but
not a modern population-scale resource such as gnomAD or UK Biobank, which are
an order of magnitude denser again.

---

## 2. Background

### 2.1 Overlap predicates are two-sided

Two half-open intervals `[a, b)` and `[c, d)` overlap iff `a < d ∧ c < b`. For
a block holding rows with `start ∈ [s_min, s_max]` and `max(end) = e_max`, a
query `[q_s, q_e)` can be skipped unless

```
s_min < q_e   ∧   q_s < e_max
```

Both clauses are necessary. The first is a function of the sort order and is
cheap to exploit. The second depends on `e_max`, which is a maximum, and
maxima are destroyed by single outliers.

### 2.2 Block reach and tightness

Define a block's **span** as `s_max − s_min` (the range of starts it holds) and
its **reach** as `e_max − s_min` (how far its rows extend). We call the mean
ratio of reach to span the **tightness** of a layout. A tightness of 1 means
blocks reach no further than the starts they hold; higher values mean some row
extends past the block's own range, and every query between that row's start
and its end must read the block.

Tightness is computable from block metadata alone, with no scan of the data. It
is the statistic our optimizer reasons about, and the x-axis of our main result.

### 2.3 Expected cost

Blocks sorted by start, each covering a span `s` and reaching a distance `r`,
are read by a query of width `w` when their starts fall in a window of width
`w + r`. So

```
blocks_read  ≈  clamp(1,  (w + r) / s,  blocks_on_contig)                (1)
```

Equation (1) is the basis of the what-if estimate in §4.3. `r` is measured for
an existing layout and modelled for a proposed one.

---

## 3. Related work

**Automated physical design.** Cost-based index selection with the optimizer in
the loop was established by AutoAdmin [Chaudhuri and Narasayya 1997], whose
"what-if" interface [Chaudhuri and Narasayya 1998] allows a tool to cost a
design it has not built. Continuous and online variants followed, as did the
self-driving database agenda [Pavlo et al. 2017]. Our threshold search is a
genomic instance of the same idea.

**Layout optimization in lakehouses, in production.** Delta Lake provides data
skipping from per-file statistics [Armbrust et al. 2020]. AutoLiquid
[Nakandala et al. 2026] continuously selects, verifies and applies clustering
keys for Databricks tables from scan telemetry, weighing predicted skipping
gains against clustering cost; it is generally available and on by default for
Unity Catalog managed tables [Databricks 2025]. **This is important for
positioning:** choosing *which columns to cluster on* is solved. Our question
is orthogonal — clustering on `start` is already the right choice, and the
problem is what `max(end)` does afterwards.

**Interval search.** Interval trees and segment trees are classical. In
genomics, nested containment lists [Alekseyenko and Lee 2007] achieve
`O(n + log N)` query time by separating contained intervals into sublists; the
Augmented Interval List [Feng et al. 2019] adds a running maximum of end within
each sublist and reports 5–18× speedups over interval trees and R-trees.
BEDTools [Quinlan and Hall 2010] is the standard tool for interval arithmetic
on files. All of these operate on individual intervals in memory. We apply the
containment idea to storage blocks instead, where it interacts with I/O rather
than pointer chasing — and, as §7 shows, where its precondition often fails.

**Genomic storage systems.** ADAM established Spark-based genomic processing
[Massie et al. 2013]. Project Glow provides VCF-to-Delta workflows
[Project Glow 2024]. Hail stores variants in a Parquet-backed MatrixTable.
TileDB-VCF models population VCF as 3D sparse arrays optimised for region
slicing across samples [Papadopoulos et al. 2016], as does GenomicsDB within
GATK. None of these, to our knowledge, reports the interaction between interval
length distribution and block-level pruning that we study.

**Negative results and reproducibility.** Our main finding is negative. We
follow the practice of reporting the mechanism and the conditions under which
the result would reverse, and we release the code, the input hashes and the
experiment driver.

---

## 4. Method

### 4.1 Augmented interval zone maps

For each block we store `s_min`, `s_max`, `e_max`, row count, byte count, contig
and a flag marking long-record blocks. Within a contig, blocks are ordered by
`s_min`, giving a binary search for the upper bound. We additionally maintain a
**prefix maximum** over `e_max`, which is non-decreasing by construction and so
also binary-searchable: every block before the first index whose prefix maximum
exceeds `q_s` can be dismissed as a range rather than individually. Long-record
blocks are held in a separate list and tested directly, since their bounds are
genuinely wide.

A block whose rows span more than one contig cannot be excluded by contig and
is always examined. Omitting this is a correctness bug, not a missed
optimization; we return to it in §6.5.

Pruning is conservative: it may return a block holding no matching row, but
never omits one that does. This is verified against brute force (§6.5).

### 4.2 Layouts

To attribute any effect to a single decision, we implement four layouts that
hold identical rows with identical target block sizes, and prune all of them
with the same two-sided test using their own statistics. The baselines are not
handicapped.

| Layout | Row assignment |
|---|---|
| `RANDOM` | arrival order; blocks span the whole genome |
| `SORTED` | globally sorted by `(contig, start)` |
| `PARTITIONED` | partitioned by contig, sorted by start within |
| `SWAPO` | `PARTITIONED`, plus records longer than a threshold written to separate blocks |

`PARTITIONED` and `SWAPO` differ in exactly one decision. Every number we
report for the optimization is the difference between those two.

### 4.3 Choosing the threshold

Our first implementation fixed the threshold at 50 bp, the conventional
boundary between an indel and a structural variant. Real annotation broke this
immediately: 93% of GENCODE records exceed 50 bp, because an exon is a few
hundred bases, so the "long" partition would have held almost the whole table.

The threshold is not a property of biology but of the join between the length
distribution and the block span. We therefore search for it. For each candidate
threshold `T` we compute, *without writing any data*, the block boundaries the
layout would produce — they are a deterministic function of the sort order —
and evaluate expected blocks read over probe positions drawn across the contig.
Candidates are span quantiles of the data. The incumbent is "do not split",
and a candidate must beat it outright. The search costs one sort.

This returning zero — do nothing — is a first-class outcome, and in §6 it is
the correct one more often than not.

### 4.5 Reading a slice of a remote callset

gnomAD's chr21 genomes sites VCF is 7.2 GB, and we need its coordinates, not its
annotations. BGZF — the block-gzip format every bgzipped VCF uses — is a
sequence of independent gzip members, each recording its own compressed size, so
an HTTP range request that begins on a member boundary is itself a valid gzip
stream. We locate a boundary by scanning for the BGZF magic and validating it
against the declared block size, then decompress member by member, yielding
lines as we go.

This gives a contiguous genomic window at the file's true record density from a
420 MB read of a 7.2 GB file. It is **not** a uniform sample, and we use it only
for density and distribution measurements within the window, never to estimate
genome-wide totals.

### 4.4 Cost model and provenance

Recommendations are ranked by net benefit: expected read savings over a
planning horizon, minus rewrite cost, storage cost and, for derived datasets, a
refresh and staleness charge. The workload log supplies the *shape* of the
workload; the query volume is a separate input, since a fifty-line log may
describe a service answering fifty thousand queries a day.

Layout changes rearrange rows and are declared to have no freshness
consequence. Materializations are different: a precomputed cohort allele
frequency is a dated scientific claim, and when new samples land it is not slow
but wrong. The optimizer charges refresh cost plus a staleness penalty scaled by
the update rate, so the same projection is accepted on a static table and
rejected on one under churn. Lineage is recorded in a W3C PROV-shaped graph
[W3C 2013], which also reports assembly conflicts — joining GRCh37 coordinates
to GRCh38 is a correctness failure that otherwise surfaces as quietly wrong
results.

---

## 5. Experimental setup

### 5.1 Real datasets

All inputs are public files, downloaded unmodified; SHA-256 recorded by the
experiment driver.

| Source | Role | Records (chr1) | SHA-256 (prefix) |
|---|---|---|---|
| GENCODE v44 basic annotation | genes, transcripts, exons, CDS, UTR | 180,441 | `3e52f82c63f8fd86` |
| ClinVar GRCh38 | clinically reported variants | 416,885 | `5f9921471596d139` |
| UCSC RepeatMasker hg38 | repeats | 444,211 | — |
| 1000 Genomes integrated SV map v8 | structural variants | 4,671 | — |
| 1000 Genomes phase 3 chr21 callset | population variants | 1,105,538 (chr21) | `1942e0709625d779` |
| gnomAD v4.1 chr21 genomes sites | population variants | 649,879 (2.33 Mb window) | range read, §4.5 |

Combined into two **platform tables**, which is how a lakehouse actually stores
these together:

| Table | Rows | Density | Sources |
|---|---|---|---|
| chr1 | 1,046,208 | 4.2 records/kb | annotation, clinical variants, repeats, SVs |
| chr21 | 1,245,415 | **26.7 records/kb** | the same four, plus the 1000 Genomes population callset |
| chr21 window | 650,304 | **279 records/kb** | gnomAD v4.1 genomes + overlapping GENCODE, both GRCh38 |

The chr21 table is 6.4× denser and is the regime a population resource
produces. GENCODE contributes 1,870,053 records genome-wide and ClinVar
4,620,536.

Contig naming had to be reconciled: ClinVar writes `1`, GENCODE and UCSC write
`chr1`. Joining without reconciling produces an empty result rather than an
error.

Coordinates are normalised once, at the readers: GTF and VCF are 1-based
inclusive, BED and UCSC dumps are 0-based half-open. Everything downstream is
0-based half-open. For symbolic structural variants the `END` INFO key is used.

### 5.2 Synthetic data

A seeded simulator produces variants across contigs in proportion to length,
with hotspots, a majority of single-base records, short indels, and a
configurable tail of long records drawn log-uniformly up to 2.4 Mb (about the
span of *DMD*, the longest human gene). The `long_fraction` parameter is swept.

### 5.3 Workload

Queries are drawn around **real gene coordinates** from the GENCODE annotation,
so query positions inherit the real clustering of genes along the genome rather
than a uniform scatter. Widths are log-uniform in roughly 2–40 kb, giving a
median of 23,428 bp. The benchmark on synthetic data additionally uses point,
gene, large-region and full-scan intents.

### 5.4 Metrics

**Bytes read** is the primary metric. Pruning happens outside the engine: SWaPO
selects a file list and DuckDB opens exactly those files, so bytes read is
counted, not inferred. Wall-clock times are reported alongside but are from a
single laptop with a warm page cache and are not the basis of any claim.

### 5.5 Reproduction

```bash
python -m swapo fetch     # the public inputs
python -m swapo paper     # §6.1 to §6.4
python -m swapo bench     # §6.6, synthetic
python -m pytest -q       # 54 tests
```

---

## 6. Results

### 6.1 RQ1 — Do real and synthetic interval lengths differ?

Interval span percentiles, chr1 (1000G SV genome-wide):

| Source | Records | Median | p99 | p99.99 | Max |
|---|---|---|---|---|---|
| GENCODE v44 | 180,441 | 138 | 87,949 | 835,894 | 1,551,957 |
| ClinVar | 416,885 | 1 | 5 | 4,544 | 9,445 |
| RepeatMasker | 444,211 | 183 | 1,768 | 7,355 | 500,000 |
| 1000G SV | 68,818 | 1,612 | 169,251 | 683,679 | 2,258,238 |
| 1000G chr21 callset | 1,105,538 | 1 | 4 | 17,428 | 329,633 |
| **Simulator (`long=0.004`)** | 300,000 | 1 | **46** | **1,850,569** | 2,387,971 |

The real sources are continuous: GENCODE moves smoothly from 138 bp at the
median through 88 kb at p99 to 1.55 Mb at the maximum. The simulator is
violently bimodal: p99 is 46 bp and p99.99 is 1.85 Mb, a jump of four orders of
magnitude across one tenth of one percent of rows.

This difference turns out to be the whole story.

### 6.2 RQ2 — Does the optimization pay on real data?

Platform table, chr1, 1,046,208 rows, 100 real gene-derived queries:

| Rows/block | Blocks | Block span | Tightness | Threshold chosen | Partitioned | SWaPO | Saving |
|---|---|---|---|---|---|---|---|
| 8,000 | 131 | 1,900 kb | 1.05 | none | 0.1165 MB | 0.1165 MB | **0.0%** |
| 2,000 | 524 | 475 kb | 1.20 | none | 0.0370 MB | 0.0370 MB | **0.0%** |
| 1,000 | 1,047 | 238 kb | 1.39 | none | 0.0239 MB | 0.0239 MB | **0.0%** |
| 500 | 2,093 | 119 kb | 1.61 | none | 0.0174 MB | 0.0174 MB | **0.0%** |
| 250 | 4,185 | 59 kb | 1.92 | none | 0.0149 MB | 0.0149 MB | **0.0%** |
| 125 | 8,370 | 30 kb | 2.22 | none | 0.0162 MB | 0.0162 MB | **0.0%** |
| 60 | 17,437 | 14 kb | 3.72 | none | 0.0204 MB | 0.0204 MB | **0.0%** |
| 30 | 34,874 | 7 kb | 26.35 | none | 0.0301 MB | 0.0301 MB | **0.0%** |

No gain at any operating point. The mechanism is visibly present — tightness
climbs from 1.05 to 26.35 as blocks narrow — and it is still never worth acting
on. Note also that bytes read *rise* below 250 rows per block: beyond that
point, smaller blocks cost more than they save regardless of layout.

For scale, a gnomAD-sized chr1 (~56M variants) at a conventional 8,000 rows per
block would give ~7,000 blocks and a ~36 kb span, which this table reaches at
about 250 rows per block. The regime we tested therefore brackets
population-scale density by block span.

#### 6.2.1 At real population-callset density

Density can also be raised directly, by adding a real population callset. The
chr21 table holds 1,245,415 rows at **26.7 records per kilobase**, 6.4× the
chr1 table:

| Rows/block | Blocks | Block span | Tightness | Threshold | Partitioned | SWaPO | Saving |
|---|---|---|---|---|---|---|---|
| 8,000 | 156 | 299 kb | 1.30 | none | 0.1886 MB | 0.1886 MB | **0.0%** |
| 2,000 | 623 | 75 kb | 1.70 | none | 0.0675 MB | 0.0675 MB | **0.0%** |
| 1,000 | 1,246 | 38 kb | 1.94 | none | 0.0456 MB | 0.0456 MB | **0.0%** |
| 500 | 2,491 | 19 kb | **2.11** | none | 0.0363 MB | 0.0363 MB | **0.0%** |
| 250 | 4,982 | 9 kb | **2.32** | none | 0.0348 MB | 0.0348 MB | **0.0%** |

The last two rows matter. At tightness 2.11 and 2.32 this real table is inside
the range where synthetic data gained 14.8% (§6.3), and it gains nothing. Higher
density does not change the answer, and block tightness on its own does not
predict it.

#### 6.2.2 At gnomAD density

The densest human variant data available is gnomAD. We read a contiguous 2.33 Mb
window of the gnomAD v4.1 chr21 genomes sites VCF — chr21:16,918,425–19,246,885,
649,879 variants — directly from the 7.2 GB remote file using BGZF range reads
(§4.5), and combined it with the 425 GENCODE features overlapping that window.
Both are GRCh38, so this table does not mix assemblies.

The result is **650,304 rows at 279 records per kilobase**, 66× the density of
our chr1 table and 10× that of the 1000 Genomes chr21 callset. Block spans fall
to 1.8 kb while the longest feature in the window spans 366 kb — a ratio of 200.

| Rows/block | Blocks | Block span | Tightness | Threshold | Partitioned | SWaPO | Saving |
|---|---|---|---|---|---|---|---|
| 8,000 | 82 | 28.4 kb | 1.43 | none | 0.3180 MB | 0.3180 MB | **0.0%** |
| 4,000 | 163 | 14.3 kb | 1.65 | none | 0.2268 MB | 0.2268 MB | **0.0%** |
| 2,000 | 326 | 7.1 kb | 1.68 | none | 0.1795 MB | 0.1795 MB | **0.0%** |
| 1,000 | 651 | 3.6 kb | 1.75 | none | 0.1610 MB | 0.1610 MB | **0.0%** |
| 500 | 1,301 | 1.8 kb | 1.83 | none | 0.1671 MB | 0.1671 MB | **0.0%** |

Separability is **129×**, and tightness never reaches 2. The answer is the same,
5/5 correct decisions, at the highest density the field has.

### 6.3 RQ3 — What governs whether it pays?

Putting real and synthetic tables on one axis — tightness — gives a clean
monotone relationship with a sharp crossover:

| Tightness | Table | Source | Saving |
|---|---|---|---|
| 1.00 | sim `long=0.0` | synthetic | 0.0% |
| 1.00 | sim `long=0.0` | synthetic | 0.0% |
| 1.05 | real chr1 | **real** | 0.0% |
| 1.20 | real chr1 | **real** | 0.0% |
| 1.26 | sim `long=0.0005` | synthetic | 0.0% |
| 1.30 | real chr21 | **real** | 0.0% |
| 1.39 | real chr1 | **real** | 0.0% |
| 1.53 | sim `long=0.0005` | synthetic | 0.0% |
| 1.61 | real chr1 | **real** | 0.0% |
| 1.70 | real chr21 | **real** | 0.0% |
| 1.92 | real chr1 | **real** | 0.0% |
| 1.94 | real chr21 | **real** | 0.0% |
| **2.11** | **real chr21** | **real** | **0.0%** |
| **2.32** | **real chr21** | **real** | **0.0%** |
| 2.43 | sim `long=0.004` | synthetic | **14.8%** |
| 2.76 | sim `long=0.02` | synthetic | **29.3%** |
| 4.08 | sim `long=0.004` | synthetic | **54.1%** |
| 6.81 | sim `long=0.02` | synthetic | **69.8%** |

Taken alone this suggests a crossover near 2.3. **That reading is wrong, and the
chr21 rows are what refute it.** Two real measurements sit at tightness 2.11 and
2.32 with zero gain, directly overlapping the synthetic point at 2.43 that gains
14.8%. And §6.2 shows the chr1 table reaching tightness 26.35, an order of
magnitude past any crossover, still gaining nothing.

Block tightness is therefore **necessary but not sufficient**: a layout with
tightness near 1 has nothing to gain, but a high tightness does not imply
anything to gain. The sufficient condition is **separability**. Expected blocks read for each candidate
threshold:

*Real platform table, chr1, 250 rows/block (baseline 1.85 blocks/query)*

| Threshold | Rows moved | Expected blocks | vs baseline |
|---|---|---|---|
| 66 bp | 49.9% | 2.78 | +50.3% |
| 348 bp | 10.0% | 2.54 | +37.2% |
| 1,967 bp | 2.0% | 2.38 | +28.4% |
| 7,046 bp | 1.0% | 2.38 | +28.7% |
| 147,590 bp | 0.1% | 2.63 | +42.2% |

*Synthetic, `long=0.004`, 2,000 rows/block (baseline 3.97 blocks/query)*

| Threshold | Rows moved | Expected blocks | vs baseline |
|---|---|---|---|
| 22 bp | 5.00% | 2.32 | −41.6% |
| 40 bp | 1.91% | 2.14 | −46.1% |
| **49 bp** | **0.40%** | **2.06** | **−48.2%** |
| 172,973 bp | 0.10% | 2.14 | −46.1% |

On real data *every* threshold makes things worse. On synthetic data *every*
threshold makes things better. The reason is that separating records creates a
second partition whose own blocks must also be read. That is worth it only when
a threshold exists that moves a very small number of rows and removes almost all
of the reach — which requires the distribution to be bimodal.

The ratio of the 99.99th to the 99th percentile span measures this in one pass:

| Table | p99 | p99.99 | Ratio | Any threshold helps? |
|---|---|---|---|---|
| Real chr1 platform | 7,046 | 528,884 | **75×** | No |
| Real chr21 platform (with population callset) | 466 | 189,545 | **407×** | No |
| Synthetic `long=0.004` | 46 | 1,826,237 | **39,701×** | Yes, −48% |

The chr21 ratio is instructive. Adding a million single-base population
variants pushes the 99th percentile down from 7,046 bp to 466 bp while leaving
the upper tail intact, so the distribution becomes *more* bimodal — 407× against
75× — and moves towards, but does not reach, the regime where a threshold works.
Denser population data makes this optimization more plausible, not less; it is
simply still two orders of magnitude short.

### 6.4 RQ4 — Does the decision procedure decide correctly?

Across all 32 measured configurations (8 on real chr1, 5 on real chr21, 5 on
the single-assembly control, 5 on the gnomAD window, 8 synthetic, plus the extreme chr1 block sizes), the optimizer proposed a split
**if and only if** a split measurably reduced bytes read.

| | Split helped | Split did not help |
|---|---|---|
| **Proposed a split** | 4 | 0 |
| **Declined** | 0 | 28 |

**32/32 correct, including all 28 refusals** (adding the single-assembly control
of §8). Note that the four refusals at
tightness above 2 are the ones a tightness-only rule would have got wrong. For a recommendation system that
proposes rewrites of large tables, being right about when to do nothing is the
property that matters.

### 6.5 RQ5 — Is pruning correct?

Pruning that drops a matching row is a defect, not an optimization. Every query
in every layout is checked against an unpruned scan.

- Synthetic benchmark: **312 layout/query checks, 0 mismatches.**
- Real benchmark: **300 layout/query checks on chr1, 0 mismatches.**
- Unit tests: **54 passing**, including a property test of pruning against
  brute force over randomised blocks with boundary cases (zero-width queries,
  blocks touching at a boundary, multi-contig blocks).

The property test earned its place. Our first implementation excluded
multi-contig blocks by contig name and so silently lost rows; brute-force
comparison caught it immediately.

### 6.6 Synthetic benchmark, for completeness

3,000,000 simulated rows, `long_fraction = 0.004`, 8,000 rows/block, 79 region
queries:

| Layout | GB read | Mean ms | Blocks read/query |
|---|---|---|---|
| Unsorted | 6.889 | 151.8 | 5,925 |
| Sorted | 0.188 | 16.5 | 185 |
| Partitioned | 0.160 | 17.9 | 158 |
| SWaPO | 0.110 | 15.0 | 116 |

Per intent, mean MB read: gene 1.00 → 0.37 (63%), interval 1.08 → 0.33 (69%),
point 0.90 → 0.32 (64%), large region 2.48 → 1.76 (29%), full scan unchanged.

**These are the numbers that do not survive contact with real data.** We report
them because the contrast with §6.2 is the point of this paper.

---

## 7. Discussion

### 7.1 Why the optimization fails on real data

Separating long records is a trade. It tightens `max(end)` for the dense blocks,
but it creates a second set of blocks that region queries must also consult.
The trade wins only when a threshold exists that moves a negligible number of
rows while removing almost all of the reach.

That requires the interval-length distribution to be bimodal — a dense short
population plus a sparse population longer by orders of magnitude, with a gap
between them. The gap is what a threshold needs to sit in.

Real genomic tables have no such gap. GENCODE alone spans 138 bp at the median,
88 kb at p99 and 1.55 Mb at the maximum, with records at every scale in
between, because biology produces features at every scale: exons, introns,
transcripts, genes, gene clusters. Mixing in ClinVar, RepeatMasker and
structural variants fills the distribution in further rather than splitting it.
Any threshold either moves too many rows, so the long partition is large, or
too few, so the dense blocks stay polluted.

Our simulator had a gap because we built one: single-base records plus a
log-uniform tail starting at 50 bp, with almost nothing in between at the
percentiles that matter. That was not an unreasonable model of a short-variant
callset merged with an SV callset. It was simply not what the real merged table
looks like.

### 7.2 Where the optimization would pay

The precondition is explicit and testable, so we can say where it would hold:

- a table that is overwhelmingly single-base records with a genuinely tiny
  fraction of megabase records, and nothing between — for instance an SNV
  callset with a small set of whole-chromosome or whole-arm events appended;
- more generally, any table with p99.99/p99 span ratio in the thousands.

Of the five real sources we examined, the 1000 Genomes structural-variant
callset has the heaviest tail (p99 169 kb, max 2.26 Mb) but is itself
continuous, with 76% of records over 50 bp and a median of 1,612 bp.

We expected **density** to be that direction, and tested it to the limit of
available data. It is not, and the reason is worth stating as a general
argument.

Density in genomic tables comes from *short* records: more samples means more
SNVs, not longer features. The number of long records is set by the genome's
feature content and is essentially fixed. So increasing density does two
opposing things at once. It narrows block spans, which helps — but it dilutes
the long records across more blocks, which hurts, because a block polluted by a
long record is only a problem if blocks are commonly polluted.

The measurements bear this out. Across four real tables the separability ratio
does not trend with density at all:

| Table | Density | Separability | Saving |
|---|---|---|---|
| chr1 platform | 4.2 /kb | 75× | 0% |
| chr1, GRCh38-only control | 4.2 /kb | 88× | 0% |
| chr21 + 1000G callset | 26.7 /kb | 407× | 0% |
| chr21 + gnomAD window | **279 /kb** | 129× | 0% |

What varies is *composition*, not density: the chr21 table scores highest
because it carries the full GENCODE annotation and a repeat track across a whole
contig, while the gnomAD window contains only 425 features among 650,304
variants. None of them approaches the thousands-fold ratio the method needs.

A table that would qualify is one where long records are a meaningful fraction
*and* are orders of magnitude longer than everything else — an SV or CNV
callset stored on its own, rather than a platform table where variants dominate
by count. We did not find such a table among the public resources we examined.

### 7.3 What does work: sizing blocks to the query

The same measurements point to a lever that is larger, simpler and always
available. Across every real table we tested, the dominant cost was that blocks
are far wider than the queries that read them. Block size is normally inherited
from a writer default — a 128 MB row group, or whatever the library chose — and
never revisited, yet it is the single largest determinant of how much a region
query reads.

There is a real optimum rather than a direction, because two costs oppose each
other. Narrower blocks are read more selectively, so bytes read fall roughly as
`1/rows_per_block`; but each block carries a fixed footer, column headers and
dictionary pages, and past a point that overhead dominates and bytes read climb
again.

We search for the minimum of

```
bytes(r) = expected_blocks(width, reach(r), span(r)) × (overhead + r × bytes_per_row)
```

where `reach` and `span` come from simulating block boundaries on the sorted
column without writing anything, and `overhead` and `bytes_per_row` are fitted
from two small sample writes because they depend on schema, codec and writer.

Measured against an exhaustive sweep on three real tables:

| Table | Rows | Density | At 8,000 rows/block | Best measured | Saving | Predicted optimum | Measured optimum |
|---|---|---|---|---|---|---|---|
| chr1 platform | 1,046,208 | 4.2/kb | 0.1182 MB | 0.0152 MB | **87%** | 500 | 250 |
| chr21 + 1000G | 1,245,415 | 26.7/kb | 0.1848 MB | 0.0351 MB | **81%** | 1,000 | 250 |
| gnomAD window | 650,304 | 279/kb | 0.3236 MB | 0.1681 MB | **48%** | 500 | 1,000 |

The search selects the measured optimum on all three tables, and its predicted
byte counts are accurate enough to be read as a forecast:

| Table | Chosen | Measured optimum | Mean absolute error | Bias | Worst |
|---|---|---|---|---|---|
| chr1 platform | 250 | 250 | 2.2% | −1.3% | 7.3% |
| chr21 + 1000G | 250 | 250 | 0.9% | +0.8% | 1.7% |
| gnomAD window | 1,000 | 1,000 | 0.7% | +0.5% | 1.5% |

Reaching that accuracy required correcting two biases that each cost about a
third of the estimate, and both are worth recording because neither is specific
to this system.

**Fit block cost on contiguous windows of the sorted table.** Our first version
fitted on the first 120,000 rows. In a table concatenated from several sources
that is one source — for us GENCODE annotation, whose repetitive identifiers
dictionary-compress far better than the variant rows that dominate by count. The
fitted bytes-per-row came out at 4.4 against a true 12.6, and the error grew
with block size, reaching 2.7× at 8,000 rows per block. Sampling contiguous
windows of the position-sorted table reproduces how blocks actually form, and
brings per-block byte prediction to within 1.5–4.0%.

**Fit on relative error, not absolute.** Ordinary least squares over block sizes
spanning two orders of magnitude is dominated by the largest samples, leaving
the small-block end — where fixed overhead matters and the optimum lies — badly
fitted. Dividing each equation by its own byte count makes the system linear in
the same two parameters while minimising proportional error.

**Measure the candidates; do not model through them.** A two-parameter fit of
bytes against rows was systematically 3.5–6.7% low across the whole curve,
because block cost is not quite linear in row count — dictionaries and page
headers amortise unevenly. The candidate sizes are known in advance and sample
blocks are cheap, so the fit is now used only to extrapolate outside the
measured range.

**Sample every size at shared anchor offsets.** This is the one that decides
whether the search lands on the optimum at all. Drawing independent windows for
each size leaves the per-size means independent, so the *difference* between
adjacent candidates carries the sum of both sampling errors. Near a flat optimum
that noise exceeds the real difference and the ranking flips: on chr1 the search
chose 500 rows per block where 250 was better, forfeiting 16.6% of the available
saving. Measuring every size at the same anchors cancels most of the variance,
since a region that compresses unusually well does so at every block size. After
this change the search selects the measured optimum on all three tables, and the
residual bias falls below 1.5%.

**Average over the real query width distribution.** Evaluating at the median
width understates the mean, because blocks read grows with width and genomic
workloads have a wide tail. Probing at positions drawn uniformly along the
contig compounds this: queries cluster on genes, genes sit in record-dense
regions, and dense regions have narrower blocks. Drawing probes from the table's
own row starts corrects the density bias for free, and passing the real
intervals removes the width bias as well.

This is reported as an engineering result, not a novel one: that physical design
should follow the workload is the oldest idea in the field (§3). What appears to
be missing in genomic practice is anyone computing it, and the cost of not doing
so is most of the bytes read.

### 7.4 Implications for practice

The practical recommendation that follows from our measurements is not "adopt
interval-aware layout". It is:

1. **Measure tightness first.** It costs nothing — it comes from block metadata
   that already exists. Below about 2, stop.
2. **Then measure separability.** p99.99/p99 span ratio, one pass. This is the
   operative statistic: in the tens or hundreds, stop, no threshold will help.
   Our two real tables measured 75× and 407×, and neither admitted a useful
   threshold; the synthetic table that did measured 39,701×.
3. **Size blocks to the query first.** This is where the gain is: 48–87% across
   our three real tables, against 0% for the layout change. It is also the
   cheaper operation, since it needs no new statistics and no second partition.

### 7.5 Implications for evaluation

Our synthetic benchmark reported a 69% improvement for an optimization that
delivers 0% on real data. It was seeded, documented, had a correct control case
at `long_fraction = 0`, and was not constructed to mislead. The error was in one
distributional assumption that was never checked against a real file.

We think this is a general hazard for genomic data-management work. Interval
length distributions are the input that layout optimizations are most sensitive
to, and they are easy to get wrong because the plausible model — "mostly SNVs
plus some SVs" — omits everything in between, which is where real annotation
lives. We would encourage reporting the span percentile profile of any dataset
used to evaluate a genomic layout optimization, real or synthetic.

---

## 8. Threats to validity

**Construct.** Blocks are whole Parquet files, so pruning is exactly
measurable; production row groups are larger. We control for this by sweeping
block size over more than two orders of magnitude and by reporting block span in
base pairs, which is the comparable quantity.

**Internal.** `PARTITIONED` and `SWAPO` differ in one decision and are pruned by
the same code path with their own statistics, so the comparison isolates the
split. All layouts are verified to return identical rows.

**Assembly mixing, disclosed.** The 1000 Genomes phase 3 releases are on
GRCh37; GENCODE, ClinVar and RepeatMasker are on GRCh38. Our combined tables
therefore mix assemblies — marginally on chr1 (0.45% of rows, the SV callset)
and substantially on chr21 (88.8%, the population callset). Coordinates are not
comparable across assemblies, so **these tables are valid for measuring interval
length distribution, density and block pruning, and are not valid for any
biological conclusion about which features overlap which variants.** The loader
detects and records this rather than combining silently, and the condition is
the same one `Lineage.assembly_conflicts()` exists to catch.

To confirm the mixing does not drive the result, we repeated the chr1
experiment on a **single-assembly control**: GENCODE, ClinVar and RepeatMasker
only, all GRCh38, 1,041,537 rows.

| Rows/block | Tightness | Threshold | Partitioned | SWaPO | Saving |
|---|---|---|---|---|---|
| 8,000 | 1.05 | none | 0.1173 MB | 0.1173 MB | **0.0%** |
| 2,000 | 1.20 | none | 0.0356 MB | 0.0356 MB | **0.0%** |
| 1,000 | 1.30 | none | 0.0216 MB | 0.0216 MB | **0.0%** |
| 500 | 1.51 | none | 0.0160 MB | 0.0160 MB | **0.0%** |
| 250 | 1.82 | none | 0.0142 MB | 0.0142 MB | **0.0%** |

Separability 87.6×, against 75× for the mixed table. Same conclusion, 5/5
correct decisions. Bringing the total to **27 of 27**.

**External.** Two contigs of one assembly pair, five sources, one query shape. We
include a real population callset (1000 Genomes chr21, 26.7 records/kb) but not
a modern population-scale resource such as gnomAD or UK Biobank, which are
denser again — though §6.2 also brackets that density by block span, and §7.2
sets out why density is the direction most likely to change the answer. We have not tested
non-human genomes, where feature-length distributions may differ substantially.
We measure region queries; sample-oriented and cohort-aggregate workloads are
modelled in the cost function but not measured end to end.

**Conclusion.** We deliberately do not report a tightness crossover as a usable
constant: §6.3 shows real measurements at tightness 2.11 and 2.32 with no gain
overlapping a synthetic one at 2.43 with a 14.8% gain, so any such constant
would be misleading. The separability thresholds we quote rest on 32 points and
are order-of-magnitude guidance. The 32/32 decision accuracy is on
configurations we chose; it is not a guarantee.

---

## 9. Limitations and future work

The Delta and Iceberg adapters are unwritten; the zone map is deliberately the
same shape as a manifest statistic, but the port does not exist. The provenance
model is implemented and tested but its staleness penalty is a parameter we did
not calibrate against any organisation's real cost of serving a stale allele
frequency. We have now tested gnomAD, which was the obvious next step, and §7.2 argues
that density is structurally the wrong axis. What remains untested is a table
whose *composition* is dominated by long records — a standalone structural
variant or copy-number callset, rather than a platform table in which short
variants outnumber features by three orders of magnitude. That is where the
method's precondition could plausibly hold, and we did not find such a public
table. A comparison against TileDB-VCF and Hail on
identical region workloads would place these numbers in context, and we have not
attempted it.

---

## 10. Conclusion

We implemented an interval-aware physical design for genomic lakehouse tables,
derived from a well-founded idea in the interval-search literature, and
measured it honestly. On synthetic data it reduces bytes read by up to 69%. On
tables built from six real public genomic sources, up to and including a window
of gnomAD at 279 variants per kilobase, it delivers nothing at any block size, and we can
say precisely why: the optimization needs a bimodal interval-length
distribution, and real genomic data is continuous. Block tightness, the obvious
diagnostic, is necessary but not sufficient — our chr21 table reaches a
tightness at which synthetic data gains 15% and gains nothing.

The artifact that survives is not the optimization but the decision procedure
around it — a cheap two-statistic test that predicted every one of our 32
outcomes, and refused in all 28 cases where refusal was right. For a system that
proposes rewriting large scientific tables, that is the more useful component.

What the same measurements did find is that block size — inherited from a
writer default and rarely revisited — accounts for most of the bytes read on
every real table we tested, and that the optimum can be computed without writing
anything. That is worth 48–87%, against 0% for the optimization we set out to
build (§7.3).

The wider lesson is about evaluation. A plausible synthetic length distribution
overstated the first optimization's value from nothing to 69%, and only real
files revealed it — and the same real files handed us a larger win elsewhere.

---

## References

Alekseyenko, A. V. and Lee, C. J. (2007). Nested Containment List (NCList): a
new algorithm for accelerating interval query of genome alignment and interval
databases. *Bioinformatics* 23(11), 1386–1393. doi:10.1093/bioinformatics/btl647

Armbrust, M., Das, T., Sun, L., Yavuz, B., Zhu, S., Murthy, M., Torres, J., van
Hovell, H., Ionescu, A., Łuszczak, A., Świtakowski, M., Szafrański, M., Li, X.,
Ueshin, T., Mokhtar, M., Boncz, P., Ghodsi, A., Paranjpye, S., Senster, P.,
Xin, R. and Zaharia, M. (2020). Delta Lake: high-performance ACID table storage
over cloud object stores. *PVLDB* 13(12), 3411–3424.

Chaudhuri, S. and Narasayya, V. (1997). An efficient, cost-driven index
selection tool for Microsoft SQL Server. *VLDB 1997*, 146–155.

Chaudhuri, S. and Narasayya, V. (1998). AutoAdmin "what-if" index analysis
utility. *SIGMOD 1998*, 367–378.

Danecek, P., Auton, A., Abecasis, G., Albers, C. A., Banks, E., DePristo, M. A.,
Handsaker, R. E., Lunter, G., Marth, G. T., Sherry, S. T., McVean, G. and
Durbin, R. (2011). The variant call format and VCFtools. *Bioinformatics*
27(15), 2156–2158.

Databricks (2025). *Predictive optimization and automatic liquid clustering for
Unity Catalog managed tables*. Product documentation.
https://docs.databricks.com/aws/en/optimizations/predictive-optimization

Feng, J., Ratan, A. and Sheffield, N. C. (2019). Augmented Interval List: a
novel data structure for efficient genomic interval search. *Bioinformatics*
35(23), 4907–4911. doi:10.1093/bioinformatics/btz407

Frankish, A. et al. (2023). GENCODE: reference annotation for the human and
mouse genomes in 2023. *Nucleic Acids Research* 51(D1), D942–D949.

Landrum, M. J. et al. (2018). ClinVar: improving access to variant
interpretations and supporting evidence. *Nucleic Acids Research* 46(D1),
D1062–D1067.

Massie, M., Nothaft, F., Hartl, C., Kozanitis, C., Schumacher, A., Joseph, A. D.
and Patterson, D. A. (2013). *ADAM: genomics formats and processing patterns for
cloud scale computing*. Technical Report UCB/EECS-2013-207, UC Berkeley.

Nakandala, S., Bhanoori, N. R., Zhang, Y., Negi, P., Sharma, A., Liang, E.,
Jiang, C., Sun, S., Kim, T., Mokhtar, M., Prabhakaran, V., Samwel, B., Beeram,
S., Taneja, S., Hormati, A., Shukla, A., Petropoulos, M. and Xin, R. (2026).
AutoLiquid: autonomic data layout optimization for the Databricks Lakehouse.
*PVLDB* 19. https://www.vldb.org/pvldb/vol19/p4023-liang.pdf

Papadopoulos, S., Datta, K., Madden, S. and Mattson, T. (2016). The TileDB array
data storage manager. *PVLDB* 10(4), 349–360.

Pavlo, A., Angulo, G., Arulraj, J., Lin, H., Lin, J., Ma, L., Menon, P., Mowry,
T., Perron, M., Quah, I., Santurkar, S., Tomasic, A., Toor, S., Van Aken, D.,
Wang, Z., Wu, Y., Xian, R. and Zhang, T. (2017). Self-driving database
management systems. *CIDR 2017*.

Project Glow (2024). *Genomics Delta Lake and variant data manipulation*.
Documentation. https://glow.readthedocs.io/

Quinlan, A. R. and Hall, I. M. (2010). BEDTools: a flexible suite of utilities
for comparing genomic features. *Bioinformatics* 26(6), 841–842.

Smit, A. F. A., Hubley, R. and Green, P. (2013–2015). *RepeatMasker Open-4.0*.
http://www.repeatmasker.org

Sudmant, P. H. et al. (2015). An integrated map of structural variation in
2,504 human genomes. *Nature* 526, 75–81.

W3C (2013). *PROV-DM: the PROV data model*. W3C Recommendation.
https://www.w3.org/TR/prov-dm/

---

## Appendix A — Reproduction

```bash
git clone <repo> && cd swapo
pip install -e ".[dev]"

python -m swapo fetch          # GENCODE, ClinVar, RepeatMasker, 1000G SV
python -m swapo paper          # §6.1–§6.4; writes paper_out/*.json
python -m swapo bench          # §6.6 synthetic benchmark
python -m swapo sweep          # synthetic long-fraction sweep
python -m pytest -q            # 54 tests
```

Inputs are identified by SHA-256 in `paper_out/length_profiles.json` and
`paper_out/crossover.json`. Simulated data is seeded. Reported byte counts are
from the files actually opened, not estimates.

## Appendix B — Errata relative to the earlier draft

An earlier concept note described this work as a "Semantically Adaptive Genomic
Lakehouse" and proposed workload-driven layout optimization as novel. Three
corrections:

1. Workload-driven physical design is not novel; it is in production
   [Nakandala et al. 2026], which the earlier draft did not cite.
2. The fixed 50 bp long-record threshold was wrong for annotation data by two
   orders of magnitude, and is now searched for (§4.3).
3. The 69% improvement reported from synthetic data does not reproduce on real
   data, where the result is 0% (§6.2). The earlier draft's own instruction —
   "do not describe the proposed optimizer as novel or superior until the
   comparison supports that claim" — was correct, and this paper is the
   comparison.
