"""Does separating long records pay on a real bimodal callset?

Table: 1000 Genomes phase 3 chr22 (GRCh37), SNVs + indels + SVs in one VCF,
already loaded as chr22_sorted_pq/. Queries are drawn from the table itself
(density weighted), so no cross-assembly coordinates are involved.

Cost is read from the Parquet footers that a planner would use (row group
min/max), not from wall clock. Seconds use the DuckDB httpfs request pattern
measured in rowgroups.py at 170 ms/request and 5.2 MB/s (GCS from this laptop).
Every layout is checked row for row against a brute-force count.
"""
import gzip, json, os, re, sys
import duckdb, numpy as np, pyarrow as pa, pyarrow.parquet as pq

OUT = os.environ.get('OUT', 'results/split_1kg_chr22.json')
TMP = os.environ.get('SWAPO_TMP', 'bench_out/split_chr22')
LAT, BW = 0.170, 5.2e6
TAIL = 16 * 1024
CHUNK = 2 * 1024 * 1024
SIZES = [int(x) for x in os.environ.get('SIZES', '8192,50000,122880,1048576').split(',')]
TAUS = [int(x) for x in os.environ.get('TAUS', '50,500,5000,50000').split(',')]
N_Q = 1000
rng = np.random.default_rng(20261122)

tbl = duckdb.connect().execute(
    'SELECT contig, start, "end", ref, alt FROM read_parquet(\'chr22_sorted_pq/*.parquet\') ORDER BY start'
).to_arrow_table()
S = tbl.column('start').to_numpy().astype(np.int64)
E = tbl.column('end').to_numpy().astype(np.int64)
L = E - S
n = len(S)

pct = {p: float(np.percentile(L, p)) for p in (50, 99, 99.9, 99.99)}
profile = {'rows': n, 'median_bp': pct[50], 'p99_bp': pct[99], 'p99_9_bp': pct[99.9],
           'p99_99_bp': pct[99.99], 'max_bp': int(L.max()),
           'separability': pct[99.99] / max(pct[99], 1.0),
           'records_ge_500bp': int((L >= 500).sum())}

# gene widths: GENCODE v44 gene lengths (width only; positions come from this table)
widths = []
with gzip.open('swapo_data/gencode.v44.basic.annotation.gtf.gz', 'rt') as fh:
    for line in fh:
        if line.startswith('#'):
            continue
        f = line.split('\t', 4)
        if f[2] == 'gene':
            widths.append(int(f[4].split('\t')[0]) - int(f[3]) + 1)
widths = np.array(widths)


def draw(kind):
    c = S[rng.integers(0, n, N_Q)]
    if kind == 'variant':
        return np.stack([c, c + 1], 1)
    if kind == 'gene':
        w = rng.choice(widths, N_Q)
        return np.stack([c, c + w], 1)
    return np.stack([c - 250_000, c + 250_000], 1)  # gwas: lead variant +/- 250 kb


QUERIES = {k: draw(k) for k in ('variant', 'gene', 'gwas')}


def write(path, parts, rg):
    w = None
    for p in parts:
        if p.num_rows == 0:
            continue
        if w is None:
            w = pq.ParquetWriter(path, p.schema, compression='snappy')
        w.write_table(p, row_group_size=rg)  # each part starts its own row group
    w.close()
    return path


def plan(path):
    pf = pq.ParquetFile(path)
    md = pf.metadata
    names = md.schema.names
    si, ei = names.index('start'), names.index('end')
    rgs = []
    for i in range(md.num_row_groups):
        r = md.row_group(i)
        rgs.append((r.column(si).statistics.min, r.column(ei).statistics.max,
                    sum(r.column(j).total_compressed_size for j in range(r.num_columns))))
    rgs = np.array(rgs, dtype=np.int64)
    footer = md.serialized_size
    return rgs, footer, os.path.getsize(path)


def cost(rgs, footer, qs):
    hit = (rgs[:, 0] < qs[:, 1:2]) & (rgs[:, 1] > qs[:, 0:1])   # half-open overlap
    by = (hit * rgs[:, 2]).sum(1)
    ranges = (hit * np.ceil(rgs[:, 2] / CHUNK)).sum(1)
    req = 1 + 1 + (footer + 8 > TAIL) + ranges                    # HEAD, tail, footer, data
    b = by + TAIL + (footer if footer + 8 > TAIL else 0)
    return {'row_groups': float(hit.sum(1).mean()), 'requests': float(req.mean()),
            'megabytes': float(b.mean() / 1e6), 'ms': float((req * LAT + b / BW).mean() * 1000)}


def check(path, qs):
    con = duckdb.connect()
    for a, b in qs[:: max(1, len(qs) // 100)]:
        got = con.execute(f'SELECT count(*) FROM read_parquet(\'{path}\') WHERE start < {b} AND "end" > {a}').fetchone()[0]
        if got != int(((S < b) & (E > a)).sum()):
            return False
    return True


os.makedirs(TMP, exist_ok=True)
res = {'table': '1000 Genomes phase 3 chr22 (GRCh37)', 'profile': profile,
       'store': {'latency_ms': LAT * 1000, 'bandwidth_MBps': BW / 1e6}, 'n_queries': N_Q, 'runs': []}
for tau in TAUS:
    long = L >= tau
    for rg in SIZES:
        if tau != 500 and rg not in (50_000, 122_880):
            continue
        base = write(f'{TMP}/sorted_{rg}.parquet', [tbl], rg)
        split = write(f'{TMP}/split_{tau}_{rg}.parquet', [tbl.filter(pa.array(~long)), tbl.filter(pa.array(long))], rg)
        row = {'tau_bp': tau, 'rg_rows': rg, 'long_rows': int(long.sum())}
        for name, path in (('sorted', base), ('split', split)):
            rgs, footer, size = plan(path)
            row[name] = {'file_mb': size / 1e6, 'row_groups': len(rgs),
                         'rows_correct': check(path, np.concatenate(list(QUERIES.values()))),
                         **{k: cost(rgs, footer, q) for k, q in QUERIES.items()}}
        row['saving_ms'] = {k: 1 - row['split'][k]['ms'] / row['sorted'][k]['ms'] for k in QUERIES}
        res['runs'].append(row)
        print(tau, rg, {k: f"{v:+.1%}" for k, v in row['saving_ms'].items()},
              row['sorted']['rows_correct'], row['split']['rows_correct'], flush=True)

json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(profile, indent=1))
