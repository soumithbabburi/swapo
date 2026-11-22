"""Row-group model, table-format adapters and staleness, on a real slice.

Fixture: GENCODE v44 basic (sha256 3e52f82c63f8fd86...) plus ClinVar GRCh38
(sha256 a0984d79c24a6735...), chr22:20,000,000-24,000,000, 17,933 rows.
"""

from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from swapo.objstore import StoreCost
from swapo.rgbench import _truth, measure, plan, write_files
from swapo.rowgroups import _contigs, choose_row_group_size, fit_row_group_cost, predict
from swapo.staleness import Counts, classify, flip_curve
from swapo.workload import classify as classify_sql

FIX = Path(__file__).parent / "data" / "chr22_20_24mb.parquet"
STORE = StoreCost(0.03, 50e6)


@pytest.fixture(scope="module")
def table():
    return pq.read_table(FIX)


@pytest.fixture(scope="module")
def queries(table):
    genes = table.filter(pc.equal(table.column("kind"), "gene"))
    s, e = genes.column("start").to_pylist(), genes.column("end").to_pylist()
    return [("chr22", a, b) for a, b in zip(s, e)][::12]


def test_model_matches_duckdb_request_log(table, queries, tmp_path):
    """Requests exactly, bytes within 5%, every row returned."""
    sizes = (16_384, 2_048, 512)
    cost = fit_row_group_cost(table, sizes)
    by = _contigs(table)
    truth = _truth(by, queries)
    for r in sizes:
        files = write_files(table, tmp_path / f"r{r}", file_rows=8_192, rg_rows=r)
        m = measure(tmp_path / f"r{r}", files, queries, truth=truth, rg_rows=r)
        p = predict(by, queries, r, cost, STORE, file_rows=8_192)
        assert m.rows_ok
        assert p.requests == pytest.approx(m.requests, abs=0.05)
        assert p.bytes == pytest.approx(m.bytes, rel=0.05)


def test_small_contig_is_not_charged_a_full_row_group(table, queries):
    """The DGV bug: a contig shorter than the row-group size is one smaller group."""
    cost = fit_row_group_cost(table, (1_048_576, 16_384))
    by = _contigs(table)
    big = predict(by, queries, 1_048_576, cost, STORE, file_rows=4_194_304)
    whole = table.num_rows * cost.span_per_row[1_048_576]
    assert big.bytes < whole * 1.2 + 64_000


def test_chooses_smaller_than_writer_default_and_declines_nothing_silly(table, queries):
    c = choose_row_group_size(table, queries, STORE, candidates=(1_048_576, 16_384, 2_048, 256))
    assert c.rg_rows < 1_048_576 and c.improvement > 0
    assert c.rg_rows > 256          # the footer and request costs stop the slide


def test_plan_never_drops_a_matching_file(table, queries, tmp_path):
    files = write_files(table, tmp_path / "p", file_rows=1_000, rg_rows=500)
    for c, s, e in queries:
        chosen = {f.path for f in plan(files, c, s, e)}
        for f in files:
            if f.start_min < e and s < f.end_max:
                assert f.path in chosen


def test_sql_lower_bound_both_spellings():
    a = classify_sql("SELECT * FROM t WHERE contig = 'chr1' AND start < 200 AND 100 < \"end\"")
    b = classify_sql("SELECT * FROM t WHERE contig = 'chr1' AND start < 200 AND \"end\" > 100")
    assert a.region == b.region and b.region.start == 100 and b.region.end == 200


def test_frequency_classes_and_flips():
    assert classify(np.array([0, 0.0005, 0.005, 0.02, 0.2])).tolist() == [0, 1, 2, 3, 4]
    # variant 1: absent in the old cohort, 2/200 = 1% after -> flips into BS1.
    # variant 2: 50% -> 25%, still BA1 -> no flip.
    c = Counts(np.array([0, 50]), 100, np.array([[2], [0]]), np.array([100]), ["NEW"])
    r = flip_curve(c)[-1]
    assert r["absent_to_present"] == 1 and r["flip_rate_all"] == 0.5


def test_delta_stats_plan_equals_delta_rs_planner_and_rewrite_is_reversible(table, queries, tmp_path):
    from deltalake import DeltaTable, write_deltalake

    from swapo.tableformat import delta_files, delta_plan, delta_rewrite, row_group_rows

    uri = str(tmp_path / "d")
    write_deltalake(uri, table, partition_by=["contig"], target_file_size=60_000)
    files = delta_files(uri)
    assert len(files) > 1
    for c, s, e in queries:
        assert {f.path for f in plan(files, c, s, e)} == delta_plan(uri, c, s, e)
    delta_rewrite(uri, table, 1_024)
    after = delta_files(uri)
    assert row_group_rows([str(Path(uri) / f.path) for f in after]) == 1_024
    assert DeltaTable(uri).to_pyarrow_dataset().count_rows() == table.num_rows
    assert DeltaTable(uri, version=0).to_pyarrow_dataset().count_rows() == table.num_rows
