"""The pruning must never drop a block that holds a matching row.

Everything else in this project is a performance argument. This file is the
correctness argument, and it is the one that matters: a layout that returns the
wrong variants is worse than a slow one. The main test is a property test
against brute force over randomly generated blocks, including the cases that
break naive implementations — zero-length queries, blocks that touch at a
boundary, long intervals that start far to the left, and blocks that span more
than one contig.
"""

from __future__ import annotations

import random

import pytest

from swapo.model import Block
from swapo.zonemap import MIXED, IntervalZoneMap


def _random_blocks(rng: random.Random, n: int) -> list[Block]:
    contigs = ["chr20", "chr21", MIXED]
    out = []
    for i in range(n):
        contig = rng.choice(contigs)
        start_min = rng.randrange(0, 1_000_000)
        span = rng.randrange(1, 50_000)
        start_max = start_min + span
        # Sometimes a block reaches far past its last start: a long interval.
        reach = rng.choice([span + 1, span + 1, span + rng.randrange(0, 400_000)])
        out.append(Block(
            path=f"b{i}", contig=contig, start_min=start_min, start_max=start_max,
            end_max=start_min + reach, n_rows=rng.randrange(1, 1000),
            n_bytes=rng.randrange(1, 10_000),
            long_tail=reach > span + 10_000,
        ))
    return out


@pytest.mark.parametrize("seed", range(25))
def test_prune_matches_brute_force(seed: int) -> None:
    rng = random.Random(seed)
    blocks = _random_blocks(rng, 120)
    zm = IntervalZoneMap(blocks)
    for _ in range(60):
        contig = rng.choice(["chr20", "chr21", "chr22", None])
        start = rng.randrange(0, 1_100_000)
        end = start + rng.choice([0, 1, 1, rng.randrange(1, 200_000)])
        fast = {b.path for b in zm.prune(contig, start, end).blocks}
        slow = {b.path for b in zm.prune_brute_force(contig, start, end)}
        assert fast == slow, (contig, start, end, fast ^ slow)


def test_mixed_contig_blocks_are_never_excluded_by_contig() -> None:
    """A block spanning contigs cannot be skipped on contig. Forgetting this
    silently loses rows, which is how the first version of this code was wrong."""
    zm = IntervalZoneMap([
        Block("mixed", MIXED, 100, 200, 300, 10, 100),
        Block("chr20-only", "chr20", 100, 200, 300, 10, 100),
    ])
    paths = {b.path for b in zm.prune("chr21", 150, 160).blocks}
    assert paths == {"mixed"}


def test_half_open_boundaries() -> None:
    """``[a, b)`` touching ``[b, c)`` is not an overlap."""
    zm = IntervalZoneMap([Block("b", "chr1", 100, 100, 200, 1, 10)])
    assert zm.prune("chr1", 200, 300).blocks == []   # starts where the block ends
    assert zm.prune("chr1", 50, 100).blocks == []    # ends where the block starts
    assert len(zm.prune("chr1", 199, 201).blocks) == 1
    assert len(zm.prune("chr1", 100, 101).blocks) == 1


def test_long_interval_split_tightens_the_zone_map() -> None:
    """The whole point: one long row ruins ``max(end)`` for its block."""
    dense = [Block(f"d{i}", "chr1", i * 1000, i * 1000 + 999, i * 1000 + 1000, 100, 1000)
             for i in range(20)]
    tight = IntervalZoneMap(list(dense))
    # Put a 500 kb record in the first block and watch its reach explode.
    polluted = list(dense)
    polluted[0] = Block("d0", "chr1", 0, 999, 500_000, 101, 1000)
    loose = IntervalZoneMap(polluted)

    assert tight.tightness() < 1.5
    assert loose.tightness() > 20

    # A query late on the contig now has to read the polluted block.
    assert len(tight.prune("chr1", 19_000, 19_500).blocks) == 1
    assert len(loose.prune("chr1", 19_000, 19_500).blocks) == 2


def test_empty_and_unknown_contig() -> None:
    zm = IntervalZoneMap([])
    assert zm.prune("chr1", 0, 10).blocks == []
    zm2 = IntervalZoneMap([Block("a", "chr1", 0, 10, 20, 1, 1)])
    res = zm2.prune("chrZZ", 0, 10)
    assert res.blocks == [] and res.skipped_contig == 1


def test_round_trip_json() -> None:
    blocks = _random_blocks(random.Random(1), 10)
    zm = IntervalZoneMap(blocks)
    again = IntervalZoneMap.from_json(zm.to_json())
    assert len(again) == len(zm)
    assert {b.path for b in again.prune("chr20", 0, 500_000).blocks} == \
           {b.path for b in zm.prune("chr20", 0, 500_000).blocks}
