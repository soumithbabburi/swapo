"""End-to-end checks: coordinates, layouts, the optimizer's judgement, provenance.

The optimizer tests are mostly about what it *declines* to do. An optimizer that
always finds something to recommend is not an optimizer, it is a rewrite loop
with a cost centre attached.
"""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from swapo import (CostModel, DatasetSpec, Engine, Intent, Layout, Lineage,
                   Query, Region, SWaPO, WorkloadLog, classify,
                   generate_workload, read_bed, read_vcf, simulate_features,
                   summarise, write_layout)
from swapo.provenance import Dataset, DatasetKind

SMALL = DatasetSpec(n_features=60_000, long_fraction=0.01, contigs=("chr21", "chr22"))

#: Density close to a real callset: enough rows per contig that a block spans a
#: megabase or two, which is the regime where a 2 Mb record actually hurts. With
#: only a handful of very wide blocks per contig the split is correctly declined,
#: and two of the tests below check exactly that.
DENSE = DatasetSpec(n_features=300_000, long_fraction=0.004, contigs=("chr21", "chr22"))
DENSE_BLOCK = 2_000


# ---------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------


def test_region_parsing_is_one_based_in_zero_based_out() -> None:
    r = Region.parse("chr7:140,753,336")
    assert (r.contig, r.start, r.end) == ("chr7", 140_753_335, 140_753_336)
    assert r.length == 1
    r2 = Region.parse("chr7:140700000-140800000")
    assert r2.start == 140_699_999 and r2.end == 140_800_000


def test_region_overlap_is_half_open() -> None:
    r = Region("chr1", 100, 200)
    assert r.overlaps(199, 250)
    assert not r.overlaps(200, 250)
    assert not r.overlaps(50, 100)


def test_region_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValueError):
        Region("chr1", 500, 100)


def test_vcf_is_converted_to_half_open_and_end_is_honoured(tmp_path: Path) -> None:
    vcf = tmp_path / "t.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t100\trs1\tA\tG\t.\t.\tAF=0.25\n"            # SNV at 1-based 100
        "chr1\t200\t.\tACGT\tA\t.\t.\tAF=0.01\n"           # deletion, REF length 4
        "chr1\t500\tsv1\tN\t<DEL>\t.\t.\tEND=900000;AF=0.001\n",  # symbolic SV
        encoding="utf-8")
    t = read_vcf(vcf)
    starts = t.column("start").to_pylist()
    ends = t.column("end").to_pylist()
    kinds = t.column("kind").to_pylist()
    assert starts == [99, 199, 499]                 # 1-based -> 0-based
    assert ends == [100, 203, 900_000]              # half-open; END used for the SV
    assert kinds == ["snv", "indel", "sv"]
    assert t.column("af").to_pylist()[0] == pytest.approx(0.25)


def test_bed_coordinates_pass_through(tmp_path: Path) -> None:
    bed = tmp_path / "t.bed"
    bed.write_text("chr1\t0\t100\tfeatureA\nchr1\t500\t600\tfeatureB\n", encoding="utf-8")
    t = read_bed(bed)
    assert t.column("start").to_pylist() == [0, 500]
    assert t.column("end").to_pylist() == [100, 600]


# ---------------------------------------------------------------------------
# Layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", list(Layout))
def test_every_layout_keeps_every_row(tmp_path: Path, layout: Layout) -> None:
    features = simulate_features(SMALL)
    r = write_layout(features, tmp_path / layout.value, layout, rows_per_block=5_000)
    assert r.n_rows == features.num_rows
    assert sum(b.n_rows for b in r.zonemap.blocks) == features.num_rows


def test_swapo_layout_separates_long_records_and_tightens_bounds(tmp_path: Path) -> None:
    features = simulate_features(SMALL)
    part = write_layout(features, tmp_path / "p", Layout.PARTITIONED, rows_per_block=5_000)
    swap = write_layout(features, tmp_path / "s", Layout.SWAPO, rows_per_block=5_000)
    assert swap.long_rows > 0
    assert part.long_rows == 0
    # The dense blocks of the SWaPO layout must be tighter, which is the
    # mechanism behind every performance claim here.
    assert swap.zonemap.tightness() < part.zonemap.tightness()


def test_layouts_return_identical_answers(tmp_path: Path) -> None:
    features = simulate_features(SMALL)
    written = {lay: write_layout(features, tmp_path / lay.value, lay, rows_per_block=5_000)
               for lay in Layout}
    eng = Engine()
    queries = generate_workload(25, contigs=SMALL.contigs, seed=3)
    for q in queries:
        # Only the feature-table intents belong here; the genotype intents need
        # columns this table does not have.
        if q.region is None or q.table != "features":
            continue
        counts = {lay.value: eng.run(q, w.zonemap).rows for lay, w in written.items()}
        assert len(set(counts.values())) == 1, (q.region, counts)
    eng.close()


# ---------------------------------------------------------------------------
# Workload classification
# ---------------------------------------------------------------------------


def test_sql_is_classified_by_genomic_intent() -> None:
    cases = {
        "SELECT * FROM v WHERE contig = 'chr7' AND start < 140753337 "
        "AND 140753335 < end": Intent.POINT,
        "SELECT * FROM v WHERE contig = 'chr7' AND start < 140800000 "
        "AND 140700000 < end": Intent.INTERVAL,
        "SELECT * FROM v WHERE contig = 'chr1' AND start < 90000000 "
        "AND 1000000 < end": Intent.LARGE_REGION,
        "SELECT * FROM v WHERE gene_id = 'ENSG00000012048'": Intent.GENE,
        "SELECT * FROM g WHERE sample_id = 'NA100001'": Intent.SAMPLE,
        "SELECT variant_id, sum(gt) FROM g GROUP BY variant_id": Intent.COHORT_FREQ,
        "SELECT count(*) FROM v": Intent.SCAN,
    }
    for sql, expected in cases.items():
        assert classify(sql).intent is expected, sql


def test_profile_reports_the_interval_share() -> None:
    log = WorkloadLog()
    log.extend(generate_workload(80, contigs=("chr21",), seed=2))
    p = log.profile()
    assert p.n_queries == 80
    assert 0.0 <= p.interval_share <= 1.0
    assert p.median_width > 0


# ---------------------------------------------------------------------------
# The optimizer's judgement
# ---------------------------------------------------------------------------


def _profile(n: int = 100):
    log = WorkloadLog()
    log.extend(generate_workload(n, contigs=("chr21", "chr22"), seed=4))
    return log.profile()


def test_optimizer_declines_when_there_is_no_long_tail(tmp_path: Path) -> None:
    """The control case. No long records means nothing to split, and the right
    answer is to leave the table alone."""
    flat = simulate_features(DatasetSpec(n_features=40_000, long_fraction=0.0,
                                         contigs=("chr21", "chr22")))
    w = write_layout(flat, tmp_path / "flat", Layout.PARTITIONED, rows_per_block=4_000)
    stats = summarise(flat, n_bytes=w.n_bytes, n_blocks=w.n_blocks)
    recs = SWaPO().recommend("t", _profile(), stats, w.zonemap)
    assert not [r for r in recs if r.action.value == "split_long_intervals"]


def test_optimizer_proposes_the_split_when_the_tail_is_there(tmp_path: Path) -> None:
    features = simulate_features(DENSE)
    w = write_layout(features, tmp_path / "p", Layout.PARTITIONED,
                     rows_per_block=DENSE_BLOCK)
    stats = summarise(features, n_bytes=w.n_bytes, n_blocks=w.n_blocks)
    recs = SWaPO().recommend("t", _profile(), stats, w.zonemap)
    split = [r for r in recs if r.action.value == "split_long_intervals"]
    assert split, [r.action.value for r in recs]
    assert split[0].net > 0
    assert "rows are longer than" in split[0].rationale
    assert split[0].rollback


def test_optimizer_declines_to_split_a_table_nobody_queries_by_region(tmp_path: Path) -> None:
    features = simulate_features(SMALL)
    w = write_layout(features, tmp_path / "p", Layout.PARTITIONED, rows_per_block=4_000)
    stats = summarise(features, n_bytes=w.n_bytes, n_blocks=w.n_blocks)
    log = WorkloadLog()
    log.extend([Query(Intent.SCAN) for _ in range(50)])
    recs = SWaPO().recommend("t", log.profile(), stats, w.zonemap)
    assert not [r for r in recs if r.action.value == "split_long_intervals"]


def test_derived_data_is_charged_for_going_stale(tmp_path: Path) -> None:
    """A materialization under heavy update should be rejected where the same
    one under a static table is accepted."""
    features = simulate_features(SMALL)
    w = write_layout(features, tmp_path / "p", Layout.PARTITIONED, rows_per_block=4_000)
    stats = summarise(features, n_bytes=w.n_bytes, n_blocks=w.n_blocks)
    log = WorkloadLog()
    log.extend([Query(Intent.SAMPLE, region=Region("chr21", 0, 10_000_000),
                      sample_ids=("NA1",)) for _ in range(60)])
    profile = log.profile()
    opt = SWaPO(CostModel())
    calm = opt.recommend("t", profile, stats, w.zonemap, updates_per_horizon=0.1)
    busy = opt.recommend("t", profile, stats, w.zonemap, updates_per_horizon=40.0)

    def net_of(recs):
        m = [r for r in recs if r.action.value == "materialize_sample_projection"]
        return m[0].net if m else None

    calm_net, busy_net = net_of(calm), net_of(busy)
    if calm_net is not None and busy_net is not None:
        assert busy_net < calm_net
    else:
        # Rejected outright under churn, which is the same conclusion.
        assert busy_net is None


def test_calibration_moves_towards_measurement(tmp_path: Path) -> None:
    features = simulate_features(DENSE)
    w = write_layout(features, tmp_path / "p", Layout.PARTITIONED,
                     rows_per_block=DENSE_BLOCK)
    stats = summarise(features, n_bytes=w.n_bytes, n_blocks=w.n_blocks)
    opt = SWaPO()
    rec = [r for r in opt.recommend("t", _profile(), stats, w.zonemap)
           if r.action.value == "split_long_intervals"][0]
    start = opt.calibration
    # Pretend the change delivered only a third of what was predicted.
    opt.record_outcome(rec, rec.expected_read_reduction / 3)
    assert opt.calibration < start


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_derived_dataset_goes_stale_when_its_input_moves() -> None:
    lin = Lineage()
    lin.register(Dataset("variants", DatasetKind.BASE, n_rows=10))
    lin.register(Dataset("af_by_cohort", DatasetKind.DERIVED, inputs=["variants"]))
    assert not lin.freshness("af_by_cohort").stale
    lin.bump("variants", n_rows=20)
    f = lin.freshness("af_by_cohort")
    assert f.stale and f.behind == ["variants"]
    assert lin.descendants("variants") == ["af_by_cohort"]


def test_a_layout_change_does_not_make_anything_stale() -> None:
    """Rearranging rows is not a scientific event. Only derived data can go stale."""
    lin = Lineage()
    lin.register(Dataset("variants", DatasetKind.BASE))
    lin.register(Dataset("variants@swapo", DatasetKind.LAYOUT, inputs=["variants"]))
    lin.bump("variants")
    assert not lin.freshness("variants@swapo").stale


def test_assembly_mismatch_is_reported() -> None:
    lin = Lineage()
    lin.register(Dataset("v37", DatasetKind.BASE, assembly="GRCh37"))
    lin.register(Dataset("joined", DatasetKind.DERIVED, inputs=["v37"],
                         assembly="GRCh38"))
    assert lin.assembly_conflicts()


def test_unknown_input_is_rejected() -> None:
    lin = Lineage()
    with pytest.raises(KeyError):
        lin.register(Dataset("x", DatasetKind.DERIVED, inputs=["nope"]))


def test_optimizer_declines_when_blocks_are_already_wider_than_any_record(
        tmp_path: Path) -> None:
    """A 2 Mb record cannot hurt a block that already spans 6 Mb of coordinates.

    This is the model earning its keep: the same table and the same long tail,
    but at a block size where separating them would cost a rewrite and return
    almost nothing.
    """
    features = simulate_features(SMALL)
    w = write_layout(features, tmp_path / "wide", Layout.PARTITIONED,
                     rows_per_block=4_000)
    stats = summarise(features, n_bytes=w.n_bytes, n_blocks=w.n_blocks)
    recs = SWaPO().recommend("t", _profile(), stats, w.zonemap)
    assert not [r for r in recs if r.action.value == "split_long_intervals"]


# ---------------------------------------------------------------------------
# Distribution shape, the statistic that predicts whether a split can pay
# ---------------------------------------------------------------------------


def test_separability_distinguishes_bimodal_from_continuous() -> None:
    """A split pays on a bimodal length distribution and not on a continuous one.

    The simulator is bimodal by construction: almost every record is one base,
    and a tiny tail is megabases. Real annotation is continuous, with records at
    every scale between. This statistic is what tells them apart, and it is why
    the optimizer refuses on real data.
    """
    import numpy as np
    import pyarrow as pa

    from swapo.threshold import separability

    rng = np.random.default_rng(0)

    def table(spans: np.ndarray) -> pa.Table:
        start = np.arange(spans.size, dtype=np.int64) * 1000
        return pa.table({
            "contig": pa.array(["chr1"] * spans.size, pa.string()),
            "start": pa.array(start, pa.int32()),
            "end": pa.array(start + spans, pa.int32()),
        })

    # Bimodal: single bases plus a 0.4% tail of megabase records.
    n = 100_000
    spans = np.ones(n, dtype=np.int64)
    tail = rng.choice(n, size=int(n * 0.004), replace=False)
    spans[tail] = rng.integers(500_000, 2_000_000, size=tail.size)
    bimodal = separability(table(spans))

    # Continuous: log-normal, every scale represented, like real annotation.
    cont = separability(table(
        np.power(10.0, rng.normal(2.2, 1.0, size=n)).astype(np.int64).clip(1)))

    assert bimodal["bimodality"] > 1000
    assert bimodal["separable"]
    assert cont["bimodality"] < 1000
    assert not cont["separable"]
