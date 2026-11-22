# SWaPO — physical design for genomic tables, chosen by measurement

**Cuts bytes read by 48–87% on real genomic tables**, by computing the block
size that fits your queries instead of inheriting a writer default.

```
$ swapo advise data/variants queries.sql

block size on this table
  current        : 8,000 rows/block, about 0.044 MB read per query
  recommended    : 500 rows/block, about 0.014 MB read per query
  expected gain  : 69% fewer bytes (3.2x)
  why            : at 500 rows a block covers roughly the width of a typical
                   26,095 bp query, so a query reads few blocks while per-block
                   overhead (6 kB each) stays small
  freshness      : none, this rewrites rows into different blocks
  rollback       : rewrite at 8,000 rows/block
```

Measured on real public data (GENCODE, ClinVar, RepeatMasker, 1000 Genomes, gnomAD):

| Table | Rows | 8,000 rows/block | Best | Saving |
|---|---|---|---|---|
| chr1 platform | 1,046,208 | 0.1182 MB | 0.0152 MB | **87%** |
| chr21 + 1000G | 1,245,415 | 0.1848 MB | 0.0351 MB | **81%** |
| gnomAD window | 650,304 | 0.3236 MB | 0.1681 MB | **48%** |

The optimum is real, not "smaller is always better": narrower blocks are read
more selectively, but each carries a fixed footer and header, so past a point
bytes read climb again. The search finds the turn **without writing any data** —
it simulates block boundaries on the sorted column and fits per-block cost from
a few sample writes.

**The predicted megabytes are a forecast, not just a ranking.** Against an
exhaustive measured sweep:

| Table | Chosen | Measured optimum | Mean abs. error | Bias |
|---|---|---|---|---|
| chr1 platform | 250 | **250** | 2.2% | −1.3% |
| chr21 + 1000G | 250 | **250** | 0.9% | +0.8% |
| gnomAD window | 1,000 | **1,000** | 0.7% | +0.5% |

It picks the measured optimum on all three, so acting on the recommendation
gives the full 87%, 81% and 48% rather than an approximation of it.

Four things were needed, each worth roughly a third of the error and each wrong
in the first version:

- **fit block cost on contiguous windows of the sorted table**, not its head —
  a concatenated table's head is one source, and GENCODE's repetitive IDs
  compress unlike the variant rows that dominate by count (2.7× error at 8,000
  rows/block);
- **fit on relative error**, since least squares across two orders of magnitude
  is dominated by the largest samples and misfits the small-block end where the
  optimum lives;
- **measure the candidate sizes directly** instead of extrapolating a
  two-parameter model through them, which was systematically 3.5–6.7% low;
- **sample every size at the same anchor offsets.** Independent windows per
  size make the *differences* between adjacent candidates carry the sum of both
  errors, which near a flat optimum is larger than the real difference, so the
  ranking flips. Sharing anchors cancels it: a region that compresses unusually
  well does so at every size.

## What we tried first, and why it failed

The rest of this README is the study that got us here. It is kept because the
negative result is the reason to trust the positive one.

## The interval-length idea, and the result

Genomic queries are interval-overlap tests, which are two-sided:

```sql
WHERE contig = 'chr13' AND start < 32400268 AND 32315086 < "end"
```

Sorting by `start` bounds such a query from above. Only `max(end)` per block
bounds it from below — and one gene body or structural variant spanning a
megabase raises `max(end)` for everything stored beside it. The standard fix
from the interval-search literature is to separate those long records.

| | Bytes read, interval queries |
|---|---|
| On **synthetic** data | **69% less** |
| Real chr1 table (GENCODE + ClinVar + RepeatMasker + 1000G SV), 1,046,208 rows | **0% less**, every block size |
| Real chr21 table, **+ 1000 Genomes population callset**, 1,245,415 rows, 26.7 records/kb | **0% less**, every block size |
| Real **gnomAD v4.1** window + GENCODE, 650,304 rows, **279 records/kb** | **0% less**, every block size |

The optimization needs a **bimodal** interval-length distribution. Real genomic
data is continuous.

| | p99 span | p99.99 span | Ratio | Any threshold helps? |
|---|---|---|---|---|
| Real chr1 platform table | 7,046 bp | 528,884 bp | **75×** | No |
| Real chr21 + population callset | 466 bp | 189,545 bp | **407×** | No |
| Real gnomAD window + GENCODE | 11 bp | 1,419 bp | **129×** | No |
| Our simulator | 46 bp | 1,826,237 bp | **39,701×** | Yes, −48% |

Biology produces features at every scale — exons, introns, transcripts, genes,
gene clusters — so there is no gap for a threshold to sit in. Our simulator had
a gap because we built one.

## What is actually useful here

The decision procedure. It computes two cheap statistics and tells you whether
to act, and **it was right in 32 of 32 measured cases, including all 28 times
the right answer was to do nothing.**

```
                      proposed a split    declined
split helped                 4                0
split did not help           0               28
```

Four of those refusals are ones a naive rule would have got wrong — see below.

Run it on your own table:

```bash
python -m swapo advise data/variants queries.sql --volume 50000
```

## Reproduce the paper

```bash
pip install -e ".[dev]"

python -m swapo fetch             # GENCODE, ClinVar, RepeatMasker, 1000G SV
python -m swapo paper             # real-data experiments on chr1
python -m swapo paper --contig chr21   # with the population callset
python -m swapo bench     # the synthetic benchmark, for contrast
python -m pytest -q       # 54 tests
```

Input files are identified by SHA-256 in the output JSON. Simulated data is
seeded. Byte counts come from the files actually opened — pruning happens
outside DuckDB, which is handed an explicit file list — so they are counted,
not estimated.

[`docs/paper.md`](docs/paper.md) is the full write-up: method, related work,
five research questions, threats to validity, references.

## Measurements

**Real chr21 table with the population callset, 1,245,415 rows at 26.7 records/kb:**

| Rows/block | Block span | Tightness | Partitioned | SWaPO | Saving |
|---|---|---|---|---|---|
| 8,000 | 299 kb | 1.30 | 0.1886 MB | 0.1886 MB | 0.0% |
| 1,000 | 38 kb | 1.94 | 0.0456 MB | 0.0456 MB | 0.0% |
| 500 | 19 kb | **2.11** | 0.0363 MB | 0.0363 MB | 0.0% |
| 250 | 9 kb | **2.32** | 0.0348 MB | 0.0348 MB | 0.0% |

Adding a million population variants raised separability from 75× to 407× — it
moves *towards* the regime where the method works, without reaching it. Whether
a gnomAD-scale resource crosses is the open question.

**Real platform table, chr1, 1,046,208 rows, 100 queries around real gene coordinates:**

| Rows/block | Block span | Tightness | Partitioned | SWaPO | Saving |
|---|---|---|---|---|---|
| 8,000 | 1,900 kb | 1.05 | 0.1165 MB | 0.1165 MB | 0.0% |
| 1,000 | 238 kb | 1.39 | 0.0239 MB | 0.0239 MB | 0.0% |
| 250 | 59 kb | 1.92 | 0.0149 MB | 0.0149 MB | 0.0% |
| 60 | 14 kb | 3.72 | 0.0204 MB | 0.0204 MB | 0.0% |
| 30 | 7 kb | 26.35 | 0.0301 MB | 0.0301 MB | 0.0% |

Two things worth noting. The mechanism is real — tightness climbs to 26 — and
it is still never worth acting on. And **block-size tuning beat every layout
change**: 8,000 → 250 rows per block cut bytes read 87%, then reversed.

**Block tightness looks like the answer, and isn't.** Real chr21 reaches
tightness 2.11 and 2.32 — overlapping the synthetic point at 2.43 that gains
14.8% — and gains nothing. Real chr1 reaches 26.35 and gains nothing.

```
tightness   1.0 ──────── 2.11  2.32 │ 2.43 ──────── 6.81        26.35
saving       0%            0%    0% │ 14.8%        69.8%           0%
            └──── real ─────────────┘ └── synthetic ──┘      └─ real ─┘
```

Tightness is necessary but not sufficient. **Separability is the operative
statistic**, and the table above is what the optimizer actually keys on.

**Correctness.** Pruning that drops a matching row is a defect, not an
optimization. Every query is checked against an unpruned scan: **312 synthetic +
300 real layout/query checks, 0 mismatches.** The property test against brute
force caught a real bug, where blocks spanning two contigs were excluded by
contig name and silently lost rows.

## Install and use

```python
from swapo import Engine, Layout, SWaPO, WorkloadLog, summarise, write_layout
from swapo.real import read_gtf
from swapo.threshold import choose_threshold, separability

features = read_gtf("gencode.v44.basic.annotation.gtf.gz", contigs=("chr1",))

print(separability(features))        # is a split even possible?
choice = choose_threshold(features, rows_per_block=8000, query_width=25_000)
print(choice.summary())              # threshold_bp: None -> do not split

written = write_layout(features, "data/ann", Layout.SWAPO,
                       long_threshold=choice.threshold)
```

Reads real VCF, GTF, BED and UCSC table dumps. Coordinates are normalised once,
at the reader: GTF and VCF are 1-based inclusive, BED is 0-based half-open,
everything downstream is 0-based half-open. Contig naming is reconciled too —
ClinVar writes `1`, GENCODE writes `chr1`, and joining them without reconciling
returns an empty result rather than an error.

## Modules

| Module | What it does |
|---|---|
| `zonemap.py` | Augmented interval zone maps; binary search on both bounds of the overlap test |
| `threshold.py` | Chooses the split threshold by what-if search, or declines. `separability()` is the predictor |
| `layout.py` | Four layouts differing by one decision, so the comparison isolates it |
| `cost.py` | `blocks_read ≈ (width + reach) / span`; measured for a real layout, modelled for a proposed one |
| `optimizer.py` | Ranks actions by net benefit; recalibrates from measured outcomes |
| `provenance.py` | Lineage, freshness, assembly conflicts; W3C PROV-shaped export |
| `real.py` | Readers for GENCODE, ClinVar, RepeatMasker, 1000G; SHA-256 provenance |
| `bgzf.py` | Range-reads a slice of a remote bgzipped VCF, so gnomAD costs 420 MB not 7.2 GB |
| `experiments.py` | The paper's experiments, as runnable code |
| `engine.py` | Prunes, then hands DuckDB an explicit file list so bytes read are counted |

## Provenance

Rearranging rows is not a scientific event. A materialized allele frequency is:
when new samples land it is not slow, it is **wrong**. Layout changes declare no
freshness impact; materializations are charged refresh cost plus a staleness
penalty scaled by the update rate, so the same projection is accepted on a
static table and rejected on one under churn.

```python
lin.bump("variants")                  # a cohort lands
lin.freshness("af_by_cohort").stale   # True
lin.assembly_conflicts()              # GRCh37 coordinates joined to GRCh38
```

## We tested gnomAD. It does not cross.

Density was the obvious place the method might start to pay, so we read a
contiguous 2.33 Mb window of the **gnomAD v4.1 chr21 genomes** sites VCF —
649,879 variants at **279 per kilobase**, 66× our chr1 table — straight out of
the 7.2 GB remote file with BGZF range reads, no full download.

| Rows/block | Block span | Tightness | Partitioned | SWaPO | Saving |
|---|---|---|---|---|---|
| 8,000 | 28.4 kb | 1.43 | 0.3180 MB | 0.3180 MB | 0.0% |
| 2,000 | 7.1 kb | 1.68 | 0.1795 MB | 0.1795 MB | 0.0% |
| 500 | 1.8 kb | 1.83 | 0.1671 MB | 0.1671 MB | 0.0% |

Block spans fall to 1.8 kb against a longest feature of 366 kb — a 200× ratio —
and tightness still never reaches 2.

**Density is structurally the wrong axis**, and that is the general result.
Density in genomic tables comes from *short* records: more samples means more
SNVs, never longer genes. The count of long records is fixed by the genome. So
adding variants narrows blocks (which helps) while diluting long records across
more of them (which hurts), and the two cancel:

| Table | Density | Separability | Saving |
|---|---|---|---|
| chr1 platform | 4.2 /kb | 75× | 0% |
| chr21 + 1000G | 26.7 /kb | 407× | 0% |
| chr21 + gnomAD | **279 /kb** | 129× | 0% |

Separability tracks *composition*, not density, and none of it comes near the
thousands-fold ratio the method needs.

```bash
python -m swapo fetch gnomad_chr21_window   # 420 MB range read, not 7.2 GB
```

## One caveat we disclose

The 1000 Genomes releases are **GRCh37**; GENCODE, ClinVar and RepeatMasker are
**GRCh38**. Our combined tables mix them — 0.45% of rows on chr1, 88.8% on
chr21. That is fine for measuring length distribution, density and block
pruning, and **not** fine for concluding which feature overlaps which variant.
The loader records it rather than combining silently.

To show it does not drive the result, we repeated chr1 on a GRCh38-only control
(GENCODE + ClinVar + RepeatMasker, 1,041,537 rows): separability 87.6×,
tightness 1.05 → 1.82, saving **0.0%** at every block size, 5/5 correct
decisions. Same answer.

## What this is not

Not a new file format, index or variant caller. Not benchmarked against Hail,
Glow, TileDB-VCF or GenomicsDB. Tested on real population callsets including a gnomAD
v4.1 window, but on two contigs of one assembly pair, and not on a standalone
structural-variant callset — the one composition where the method's
precondition might hold. The
Delta and Iceberg adapters are unwritten.

Workload-driven layout selection is not novel and ships in production; see
AutoLiquid (PVLDB 19) and AutoAdmin (VLDB 1997), both discussed in the paper.

## Licence

MIT.
