import os, time, duckdb, numpy as np, pyarrow as pa, pyarrow.parquet as pq
from swapo.formatbench import run_format
from swapo.objstore import StoreCost

con = duckdb.connect()
chr22 = con.execute('SELECT contig, start, "end", ref, alt FROM read_parquet("chr22_sorted_pq/*.parquet")').to_arrow_table()
GENES = {'clinical': [('22', 19941772, 19970744), ('22', 20080197, 20111641), ('22', 23180509, 23318037)]}
store = StoreCost(0.035, 100e6, 'S3 standard')

print('=== TRACK 2A: APACHE ICEBERG & DELTA LAKE PUSHDOWN & ROLLBACK VERIFICATION ===')
for fmt in ('iceberg', 'delta'):
    try:
        res = run_format(fmt, chr22, GENES, store)
        pa1, pa2 = res['before']['planner_agreement'], res['after']['planner_agreement']
        rok, rbr = res['after']['rows_correct'], res['rollback_rows']
        sav = res['measured_saving'] * 100
        print(f'{fmt.upper():8s} | planner_agreement={pa1} -> {pa2} | rows_ok={rok} | rollback_rows={rbr:,} | saving={sav:.1f}%')
    except Exception as e:
        print(f'{fmt.upper():8s} skipped/err: {e}')

print('\n=== TRACK 2B: PYSPARK CATALYST PARTITION & PARQUET PUSHDOWN PLAN ===')
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.master('local[2]').appName('SWaPO-Pushdown').config('spark.ui.enabled', 'false').getOrCreate()
    spark.sparkContext.setLogLevel('ERROR')
    df = spark.read.parquet('chr22_swapo_pq')
    q = df.filter('contig = "22" AND end >= 19941772 AND start <= 19970744').select('contig', 'start', 'end')
    plan_str = q._jdf.queryExecution().executedPlan().toString()
    for line in plan_str.splitlines():
        if any(k in line for k in ('FileScan', 'PushedFilters', 'PartitionFilters')):
            print(' ', line.strip())
    print('  Spark COMT hit count:', q.count())
    spark.stop()
except Exception as e:
    print('  PySpark not installed/configured:', e)

print('\n=== TRACK 3: MULTI-SAMPLE COHORT GENOTYPE MATRIX BENCHMARK (N SAMPLES x 1.1M CHR22 SITES) ===')
rng = np.random.default_rng(7)
n_vars = chr22.num_rows
starts = chr22.column('start').to_numpy()
ends = chr22.column('end').to_numpy()
contigs = chr22.column('contig')
is_long = (ends - starts) >= 500

for n_samples in [100, 500, 2504]:
    nnz_per_row = np.maximum(1, rng.poisson(max(1, n_samples * 0.015), size=n_vars))
    total_nnz = int(nnz_per_row.sum())
    sample_ids = rng.integers(0, n_samples, size=total_nnz, dtype=np.int32)
    gts = rng.choice(np.array([1, 2], dtype=np.int8), size=total_nnz, p=[0.85, 0.15])
    offsets = np.zeros(n_vars + 1, dtype=np.int32)
    np.cumsum(nnz_per_row, out=offsets[1:])
    sid_arr = pa.ListArray.from_arrays(pa.array(offsets), pa.array(sample_ids))
    gt_arr = pa.ListArray.from_arrays(pa.array(offsets), pa.array(gts))
    cohort = pa.table({'contig': contigs, 'start': chr22.column('start'), 'end': chr22.column('end'), 'is_long': is_long, 'sample_ids': sid_arr, 'gts': gt_arr})

    idx_base = pa.compute.sort_indices(cohort, [('start', 'ascending')])
    t0 = time.perf_counter()
    pq.write_table(cohort.take(idx_base), f'cohort_{n_samples}_sorted.parquet', compression='zstd', row_group_size=50_000)
    t_base_w = time.perf_counter() - t0

    idx_swapo = pa.compute.sort_indices(cohort, [('is_long', 'ascending'), ('start', 'ascending')])
    t0 = time.perf_counter()
    pq.write_table(cohort.take(idx_swapo), f'cohort_{n_samples}_swapo.parquet', compression='zstd', row_group_size=50_000)
    t_sw_w = time.perf_counter() - t0

    c2 = duckdb.connect()
    q_base = f'SELECT count(*) FROM "cohort_{n_samples}_sorted.parquet" WHERE contig=\'22\' AND "end" >= 19941772 AND start <= 19970744 AND list_has_any(sample_ids, [0,1,2,3,4,5,6,7,8,9])'
    q_sw = f'SELECT count(*) FROM "cohort_{n_samples}_swapo.parquet" WHERE contig=\'22\' AND "end" >= 19941772 AND start <= 19970744 AND list_has_any(sample_ids, [0,1,2,3,4,5,6,7,8,9])'
    for _ in range(2):
        c2.execute(q_base).fetchone()
        c2.execute(q_sw).fetchone()
    t0 = time.perf_counter(); h_b = c2.execute(q_base).fetchone()[0]; ms_b = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter(); h_s = c2.execute(q_sw).fetchone()[0]; ms_s = (time.perf_counter() - t0) * 1000
    assert h_b == h_s
    mb_b = os.path.getsize(f'cohort_{n_samples}_sorted.parquet') / 1e6
    mb_s = os.path.getsize(f'cohort_{n_samples}_swapo.parquet') / 1e6
    print(f'N={n_samples:4d} samples ({total_nnz/1e6:5.1f}M GTs) | Disk: {mb_b:5.1f} vs {mb_s:5.1f} MB | Write: {t_base_w:4.2f}s vs {t_sw_w:4.2f}s | Locus+Sample Query: {ms_b:5.1f} ms vs {ms_s:5.1f} ms ({ms_b/ms_s:.2f}x faster, hits={h_s})')
