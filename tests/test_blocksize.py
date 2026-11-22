"""Block sizing: the recommendation that actually pays on real genomic tables.

The cost curve is U-shaped — narrower blocks are read more selectively, but
each one carries fixed overhead — so there is a real optimum rather than a
direction. These tests check the shape, the fit and the refusal.
"""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest

from swapo import DatasetSpec, simulate_features
from swapo.blocksize import choose_block_size, fit_block_cost

SPEC = DatasetSpec(n_features=120_000, long_fraction=0.001, contigs=("chr21",))


@pytest.fixture(scope="module")
def table() -> pa.Table:
    return simulate_features(SPEC)


def test_cost_fit_is_positive_and_ordered(table: pa.Table) -> None:
    cost = fit_block_cost(table)
    assert cost.overhead_bytes >= 0
    assert cost.bytes_per_row > 0
    # A bigger block must cost more bytes than a smaller one.
    assert cost.block_bytes(8_000) > cost.block_bytes(1_000)


def test_chosen_size_is_the_minimum_of_the_curve(table: pa.Table) -> None:
    choice = choose_block_size(table, query_width=25_000,
                               current_rows_per_block=32_000)
    best = min(choice.curve, key=lambda c: c[2])
    assert choice.rows_per_block == best[0]
    assert choice.expected_bytes == pytest.approx(best[2])


def test_cost_curve_turns_up_once_overhead_dominates() -> None:
    """On a dense table the optimum is interior: narrower blocks eventually cost
    more, because each one carries a fixed footer and header regardless of how
    few rows it holds. A sparse table has no such turning point, since a query
    reads about one block however small they get — which is why this fixture is
    deliberately denser than the module-level one."""
    dense = simulate_features(DatasetSpec(n_features=600_000, long_fraction=0.001,
                                          contigs=("chr21",)))
    # Swept below the default floor of 125 rows, which is where the turn lies
    # for this density. Real tables are denser still and turn earlier.
    choice = choose_block_size(
        dense, query_width=25_000, current_rows_per_block=32_000,
        candidates=(32_000, 8_000, 2_000, 500, 250, 125, 60, 30))
    ordered = sorted(choice.curve, key=lambda c: c[0])      # ascending rows/block
    costs = [c[2] for c in ordered]
    best = int(np.argmin(costs))
    # The minimum is interior, not at either end of the sweep.
    assert 0 < best < len(costs) - 1, costs
    assert costs[0] > costs[best] and costs[-1] > costs[best]


def test_recommendation_beats_the_starting_point(table: pa.Table) -> None:
    choice = choose_block_size(table, query_width=25_000,
                               current_rows_per_block=32_000)
    assert choice.rows_per_block < 32_000
    assert choice.improvement > 0.2
    assert choice.speedup > 1.2
    assert "rows/block" in choice.explain()


def test_wider_queries_want_bigger_blocks(table: pa.Table) -> None:
    """The optimum tracks the query, which is why a fixed default is wrong.

    A narrow lookup wants blocks sized to it; a megabase sweep reads whole
    blocks anyway, so the per-block overhead should push the answer upward.
    """
    narrow = choose_block_size(table, query_width=2_000, current_rows_per_block=32_000)
    wide = choose_block_size(table, query_width=2_000_000, current_rows_per_block=32_000)
    assert wide.rows_per_block >= narrow.rows_per_block


def test_declines_when_already_well_sized(table: pa.Table) -> None:
    """Started at the optimum, there is nothing to recommend."""
    best = choose_block_size(table, query_width=25_000,
                             current_rows_per_block=32_000).rows_per_block
    again = choose_block_size(table, query_width=25_000,
                              current_rows_per_block=best)
    assert again.improvement <= 0.05


def test_forecast_tracks_a_measured_sweep(tmp_path, table: pa.Table) -> None:
    """The predicted bytes must be a forecast, not just a ranking signal.

    An earlier version was right about the ordering but a third low on the
    absolute number, because it fitted block cost on the head of the table and
    evaluated a single median query width. Both are fixed; this pins it.
    """
    from swapo.engine import Engine
    from swapo.layout import Layout, write_layout
    from swapo.model import Intent, Query, Region

    rng = np.random.default_rng(5)
    lo = int(pa.compute.min(table.column("start")).as_py())
    hi = int(pa.compute.max(table.column("end")).as_py())
    qs = []
    for _ in range(40):
        a = int(rng.integers(lo, hi))
        qs.append(Query(Intent.INTERVAL, region=Region("chr21", a, a + int(10 ** rng.uniform(3.5, 4.6)))))

    choice = choose_block_size(
        table, query_width=25_000, current_rows_per_block=16_000,
        queries=[(q.region.start, q.region.end) for q in qs])

    eng = Engine()
    errors = []
    try:
        for rpb, _blocks, predicted in choice.curve:
            if rpb > table.num_rows or rpb < 500:
                continue
            lr = write_layout(table, tmp_path / f"b{rpb}", Layout.PARTITIONED,
                              rows_per_block=rpb)
            measured = np.mean([eng.run(q, lr.zonemap).bytes_read for q in qs])
            if measured:
                errors.append(abs(predicted / measured - 1))
    finally:
        eng.close()

    assert errors, "no block sizes were evaluated"
    assert np.mean(errors) < 0.20, f"mean forecast error {np.mean(errors):.1%}"
