import os, shutil, subprocess, sys, time, duckdb

VCF = 'gnomad_real_chr22.vcf'
GENES = [('COMT', '22', 19941772, 19970744), ('DGCR8', '22', 20080197, 20111641), ('BCR', '22', 23180509, 23318037)]

def dir_mb(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) / (1024 * 1024)

for layout in ('sorted', 'swapo'):
    out = f'chr22_{layout}_pq'
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'swapo', 'ingest', VCF, '--out', out, '--layout', layout], check=True)
    dt = time.perf_counter() - t0
    print(f'{layout:10s} | Ingest: {dt:6.2f}s | Disk: {dir_mb(out):6.2f} MB')

q, dq = chr(39), chr(34)
con = duckdb.connect()
for gene, chrom, qs, qe in GENES:
    parts = [f'{gene:6s}']
    for layout in ('sorted', 'swapo'):
        sql = f'SELECT count(*) FROM read_parquet({q}chr22_{layout}_pq/*.parquet{q}) WHERE contig = {q}{chrom}{q} AND {dq}end{dq} >= {qs} AND start <= {qe}'
        hits = con.execute(sql).fetchone()[0]
        t0 = time.perf_counter()
        for _ in range(5):
            con.execute(sql).fetchone()
        ms = ((time.perf_counter() - t0) / 5.0) * 1000
        parts.append(f'{layout}: {ms:5.2f} ms (hits={hits})')
    print(' | '.join(parts))
