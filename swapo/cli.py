"""Command line: ``swapo bench``, ``swapo sweep``, ``swapo ingest``, ``swapo advise``.

Run ``python -m swapo --help`` for the options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bench import run_benchmark, run_sweep
from .experiments import crossover, length_profiles
from .real import SOURCES, fetch, fetch_gnomad_window
from .data import DatasetSpec, read_bed, read_vcf
from .layout import Layout, write_layout
from .optimizer import SWaPO
from .stats import length_histogram, summarise
from .workload import WorkloadLog


def _cmd_bench(args: argparse.Namespace) -> int:
    spec = DatasetSpec(n_features=args.rows, long_fraction=args.long_fraction)
    res = run_benchmark(args.out, spec=spec, n_queries=args.queries,
                        repeats=args.repeats, rows_per_block=args.rows_per_block, consolidate=args.consolidate)
    print("\nBytes read over the whole workload (lower is better)")
    print(f"  {'layout':14}{'GB read':>10}{'mean ms':>10}{'blocks read':>14}")
    for name, s in res.scores.items():
        print(f"  {name:14}{s.total_bytes / 1e9:>10.3f}{s.mean_ms:>10.2f}"
              f"{s.total_blocks:>14,}")
    print("\nPer intent, mean megabytes read")
    intents = sorted({i for s in res.scores.values() for i in s.by_intent})
    header = "".join(f"{n:>14}" for n in res.scores)
    print(f"  {'intent':16}{header}")
    for i in intents:
        row = "".join(
            f"{res.scores[n].by_intent.get(i, {}).get('mean_megabytes', 0):>14.3f}"
            for n in res.scores)
        print(f"  {i:16}{row}")
    v = res.verification
    print(f"\nCorrectness: {v['checks']} layout/query checks, "
          f"{len(v['mismatches'])} mismatches -> {'PASS' if v['passed'] else 'FAIL'}")
    if res.calibration:
        c = res.calibration
        print(f"What-if accuracy: predicted {c['predicted_read_reduction']:.0%} "
              f"saving, measured {c['measured_read_reduction']:.0%}")
    out = Path(args.out) / "result.json"
    res.to_json(out)
    print(f"Full results: {out}")
    return 0 if v["passed"] else 1


def _cmd_sweep(args: argparse.Namespace) -> int:
    rows = run_sweep(args.out, n_features=args.rows)
    Path(args.out, "sweep.json").parent.mkdir(parents=True, exist_ok=True)
    Path(args.out, "sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n{'long tail':>10}{'rows/block':>12}{'partitioned MB':>16}"
          f"{'swapo MB':>11}{'saving':>9}")
    for r in rows:
        print(f"{r['long_fraction']:>10}{r['rows_per_block']:>12}"
              f"{r['partitioned']['mean_megabytes_read']:>16.3f}"
              f"{r['swapo']['mean_megabytes_read']:>11.3f}"
              f"{r['read_reduction']:>8.0%}")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    path = Path(args.path)
    reader = read_bed if path.suffix.lower().startswith(".bed") else read_vcf
    tbl = reader(path)
    print(f"Read {tbl.num_rows:,} records from {path.name}")
    print("Interval lengths:", json.dumps(length_histogram(tbl), indent=2))
    res = write_layout(tbl, args.out, Layout(args.layout),
                       rows_per_block=args.rows_per_block, consolidate=args.consolidate)
    print(json.dumps(res.summary(), indent=2))
    return 0


def _load_for_advice(path: str, fmt: str):
    """The table, its per-file stats and the paths of its data files."""
    import pyarrow.parquet as pq

    from .tableformat import _norm, delta_files, iceberg_files

    p = Path(path)
    if fmt == "auto":
        fmt = ("delta" if (p / "_delta_log").exists()
               else "iceberg" if p.suffix == ".json" else "parquet")
    if fmt == "delta":
        from deltalake import DeltaTable

        files = delta_files(path)
        tbl = DeltaTable(path).to_pyarrow_table()
        paths = [str(p / f.path) for f in files]
    elif fmt == "iceberg":
        from pyiceberg.table import StaticTable

        t = StaticTable.from_metadata(path)
        files = iceberg_files(t)
        tbl = t.scan().to_arrow()
        paths = [f.path for f in files]
    else:
        paths = sorted(str(x) for x in p.rglob("*.parquet"))
        if not paths:
            raise SystemExit(f"no Parquet files under {path}")
        tbl = pq.read_table(paths)
        files = None
    return fmt, tbl, files, [_norm(x) for x in paths]


def _cmd_advise(args: argparse.Namespace) -> int:
    """Read a table and a query log, and print what to change."""
    import statistics

    import pyarrow.parquet as pq

    from .objstore import StoreCost, measure_remote
    from .rgbench import FileStat
    from .tableformat import row_group_rows
    from .zonemap import IntervalZoneMap
    from .model import Block

    fmt, tbl, files, paths = _load_for_advice(args.table_path, args.format)
    log = WorkloadLog()
    for line in Path(args.query_log).read_text(encoding="utf-8").splitlines():
        if line.strip():
            log.record_sql(line)
    profile = log.profile()

    if args.store_url:
        store, _ = measure_remote(args.store_url)
    else:
        store = StoreCost(args.latency_ms / 1e3, args.bandwidth_mbps * 1e6,
                          "given on the command line")
    current = row_group_rows(paths)
    if files is None:
        import pyarrow.compute as pc

        files = []
        for x in paths:
            t = pq.read_table(x, columns=["contig", "start", "end"])
            files.append(FileStat(x, str(t.column("contig")[0]),
                                  int(pc.min(t.column("start")).as_py()),
                                  int(pc.max(t.column("end")).as_py()),
                                  t.num_rows, Path(x).stat().st_size))
    zm = IntervalZoneMap([Block(path=f.path, contig=f.contig, start_min=f.start_min,
                                start_max=f.start_min, end_max=f.end_max,
                                n_rows=f.n_rows, n_bytes=f.n_bytes) for f in files])
    stats = summarise(tbl, n_bytes=sum(f.n_bytes for f in files), n_blocks=len(files))
    file_rows = int(statistics.median(f.n_rows for f in files))

    print(f"{fmt} table, {tbl.num_rows:,} rows in {len(files)} files, "
          f"{current:,} rows per row group; {len(log.queries)} queries in the log")
    print(f"storage: {store.latency_s * 1e3:.0f} ms per request, "
          f"{store.bandwidth_bps / 1e6:.1f} MB/s ({store.source})\n")
    recs = SWaPO(store=store).recommend(
        args.table, profile, stats, zm, updates_per_horizon=args.updates,
        query_volume=args.volume, tbl=tbl, observed_queries=log.queries,
        current_rg_rows=current, file_rows=file_rows,
        cohort_growth_per_update=args.cohort_growth)
    if not recs:
        print("No change is worth making. Leave the table as it is.")
        return 0
    for r in recs:
        print(r.explain())
        print()
    rg = next((r for r in recs if r.action.value == "resize_row_groups"), None)
    if rg and args.apply:
        if fmt != "delta":
            print("--apply writes Delta tables only; for Iceberg use "
                  "swapo.tableformat.iceberg_rewrite with your catalog.")
            return 1
        from .tableformat import delta_rewrite

        delta_rewrite(args.table_path, tbl, rg.params["rows_per_row_group"])
        print(f"applied: rewritten at {rg.params['rows_per_row_group']:,} rows per row "
              "group as a new Delta version; the previous version is still readable")
    return 0


def summarise_from_zonemap(zm) -> "object":
    """Approximate table statistics from block metadata alone.

    Used by ``advise`` when the data itself is not being read. ``long_fraction``
    cannot be known exactly from block bounds, so it is inferred from how far
    blocks reach beyond their own range of starts, which is the signal that
    matters anyway.
    """
    from .model import TableStats

    blocks = zm.blocks
    if not blocks:
        return TableStats()
    reach_ratio = zm.tightness()
    rows = sum(b.n_rows for b in blocks)
    per_contig: dict[str, int] = {}
    for b in blocks:
        per_contig[b.contig] = per_contig.get(b.contig, 0) + b.n_rows
    return TableStats(
        n_rows=rows,
        n_bytes=sum(b.n_bytes for b in blocks),
        n_blocks=len(blocks),
        long_fraction=0.002 if reach_ratio > 1.5 else 0.0,
        max_length=max((b.end_max - b.start_min) for b in blocks),
        rows_per_contig=per_contig,
    )


def _cmd_fetch(args: argparse.Namespace) -> int:
    names = args.which or list(SOURCES)
    if "gnomad_chr21_window" in names:
        print("gnomad_chr21_window: a 420 MB contiguous slice of the 7.2 GB "
              "gnomAD v4.1 chr21 genomes VCF, read in place")
        print("  ->", fetch_gnomad_window(args.out))
        names = [n for n in names if n != "gnomad_chr21_window"]
    for name in names:
        src = SOURCES[name]
        print(f"{name}: {src.description} (~{src.approx_mb} MB)")
        p = fetch(name, args.out)
        print(f"  -> {p}")
    return 0


def _cmd_paper(args: argparse.Namespace) -> int:
    """Reproduce the experiments reported in docs/paper.md."""
    print("Interval-length profiles of each real source" + chr(10))
    prof = length_profiles(args.data, args.contig)
    print(f"  {'source':34}{'records':>10}{'median':>9}{'p99':>10}{'p99.99':>11}{'max':>12}")
    for name, v in prof.items():
        print(f"  {name:34}{v['records']:>10,}{v['median_bp']:>9,}"
              f"{v['p99_bp']:>10,}{v['p99_99_bp']:>11,}{v['max_bp']:>12,}")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    Path(args.out, "length_profiles.json").write_text(json.dumps(prof, indent=2),
                                                      encoding="utf-8")
    print(chr(10) + "Crossover experiment" + chr(10))
    rep = crossover(args.data, args.out, contig=args.contig)
    print(chr(10) + f"  {'table':22}{'source':11}{'rows/blk':>9}"
          f"{'tightness':>11}{'saving':>9}{'ok':>5}")
    for p in rep["points"]:
        print(f"  {p['table']:22}{p['source']:11}{p['rows_per_block']:>9}"
              f"{p['tightness']:>11.2f}{p['saving'] * 100:>8.1f}%"
              f"{'  yes' if p['decision_correct'] else '   NO':>5}")
    print(chr(10) + f"  Optimizer decisions correct: {rep['decisions_correct']}")
    print(f"  Real data tightness range: {rep['real_tightness_range']}")
    c = rep["crossover"]
    print(f"  No gain up to tightness {c['highest_tightness_with_no_gain']}; "
          f"gain from {c['lowest_tightness_with_gain']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="swapo",
        description="Semantic, workload- and provenance-aware optimizer for genomic tables.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bench", help="build every layout, run a workload, compare")
    b.add_argument("--out", default="bench_out")
    b.add_argument("--rows", type=int, default=3_000_000)
    b.add_argument("--long-fraction", type=float, default=0.004, dest="long_fraction")
    b.add_argument("--queries", type=int, default=90)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--rows-per-block", type=int, default=8_000, dest="rows_per_block")
    b.set_defaults(func=_cmd_bench)

    s = sub.add_parser("sweep", help="where the long-interval split helps, and where it does not")
    s.add_argument("--out", default="bench_out/sweep")
    s.add_argument("--rows", type=int, default=3_000_000)
    s.set_defaults(func=_cmd_sweep)

    i = sub.add_parser("ingest", help="load a real VCF or BED and write a layout")
    i.add_argument("path")
    i.add_argument("--out", default="data/ingested")
    i.add_argument("--layout", default="swapo", choices=[x.value for x in Layout])
    i.add_argument("--rows-per-block", type=int, default=50_000, dest="rows_per_block")
    i.add_argument("--consolidate", action="store_true", help="pack row groups into consolidated Parquet files per partition for S3/GCS")
    i.set_defaults(func=_cmd_ingest)

    a = sub.add_parser("advise", help="recommend changes for a Parquet, Delta or Iceberg table")
    a.add_argument("table_path", help="Parquet directory, Delta table directory, or "
                                      "Iceberg metadata.json")
    a.add_argument("query_log", help="file of SQL statements, one per line")
    a.add_argument("--format", default="auto", choices=["auto", "parquet", "delta", "iceberg"])
    a.add_argument("--table", default="variants")
    a.add_argument("--store-url", dest="store_url", default=None,
                   help="a large object in your bucket; its latency and bandwidth are measured")
    a.add_argument("--latency-ms", dest="latency_ms", type=float, default=170.0)
    a.add_argument("--bandwidth-mbps", dest="bandwidth_mbps", type=float, default=5.2)
    a.add_argument("--updates", type=float, default=1.0,
                   help="expected data updates per planning horizon")
    a.add_argument("--cohort-growth", dest="cohort_growth", type=float, default=0.1,
                   help="fraction by which each update grows the cohort")
    a.add_argument("--volume", type=float, default=10_000,
                   help="queries of this shape expected per horizon window; the "
                        "log supplies the shape, this supplies the rate")
    a.add_argument("--apply", action="store_true",
                   help="apply the row-group recommendation (Delta tables)")
    a.set_defaults(func=_cmd_advise)

    f = sub.add_parser("fetch", help="download the public datasets used by the paper")
    f.add_argument("which", nargs="*",
                   choices=list(SOURCES) + ["gnomad_chr21_window"],
                   help="names to fetch; default all the downloadable ones")
    f.add_argument("--out", default="swapo_data")
    f.set_defaults(func=_cmd_fetch)

    pp = sub.add_parser("paper", help="reproduce the experiments in docs/paper.md")
    pp.add_argument("--data", default="swapo_data")
    pp.add_argument("--out", default="paper_out")
    pp.add_argument("--contig", default="chr1")
    pp.set_defaults(func=_cmd_paper)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
