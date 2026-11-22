import os, shutil, subprocess, sys, time
import tiledbvcf, duckdb, pyarrow as pa, pyarrow.parquet as pq

VCF_IN = '/data/gnomad_real_chr22.vcf'
VCF_SAMPLE = '/tmp/chr22_sample.vcf'
VCF_SORTED = '/tmp/chr22_sorted.vcf'
VCF_GZ = '/tmp/chr22_sorted.vcf.gz'
GENES = [
    ('COMT', '22', 19941772, 19970744),
    ('DGCR8', '22', 20080197, 20111641),
    ('BCR', '22', 23180509, 23318037),
]

def dir_mb(path):
    if os.path.isfile(path):
        return os.path.getsize(path) / (1024 * 1024)
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total / (1024 * 1024)

print('=== PREPARING SAMPLE-COLUMN VCF (gnomad_real_chr22.vcf) ===')
with open(VCF_IN, 'r') as fin, open(VCF_SAMPLE, 'w') as fout:
    has_format = False
    for line in fin:
        if line.startswith('##FORMAT='):
            has_format = True
            fout.write(line)
        elif line.startswith('#CHROM'):
            if not has_format:
                fout.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
            fout.write('##INFO=<ID=END,Number=1,Type=Integer,Description="End position">\n##INFO=<ID=SVLEN,Number=1,Type=Integer,Description="SV length">\n##INFO=<ID=SVTYPE,Number=1,Type=String,Description="SV type">\n')
            cols = line.rstrip('\n').split('\t')
            fout.write('\t'.join(cols[:8] + ['FORMAT', 'SAMPLE0']) + '\n' if len(cols) <= 8 else line)
        elif line.startswith('#'):
            fout.write(line)
        else:
            cols = line.rstrip('\n').split('\t')
            fout.write('\t'.join(cols[:8] + ['GT', '0/1']) + '\n' if len(cols) <= 8 else line)

# 1. Tabix (requires coordinate sort + bgzip + tbi)
t0_tbx = time.perf_counter()
subprocess.run(f'(grep "#" {VCF_SAMPLE}; grep -v "#" {VCF_SAMPLE} | LC_ALL=C sort -k1,1 -k2,2n) > {VCF_SORTED} && bgzip -f -c {VCF_SORTED} > {VCF_GZ} && tabix -f -p vcf {VCF_GZ}', shell=True, check=True)
dt_tbx = time.perf_counter() - t0_tbx
print(f'Tabix (Sort+BGZF+.tbi)     | Ingest: {dt_tbx:6.2f}s | Disk: {dir_mb(VCF_GZ) + dir_mb(VCF_GZ + ".tbi"):6.2f} MB')

# 2. TileDB-VCF (anchor_gap = 1000 and 10000)
for gap in (1000, 10000):
    uri = f'/tmp/tiledb_gap_{gap}'
    shutil.rmtree(uri, ignore_errors=True)
    t0 = time.perf_counter()
    ds = tiledbvcf.Dataset(uri, mode='w')
    ds.create_dataset(extra_attrs=['info_END'], anchor_gap=gap)
    ds.ingest_samples(sample_uris=[VCF_GZ])
    dt = time.perf_counter() - t0
    print(f'TileDB-VCF (anchor_gap={gap:5d}) | Ingest: {dt:6.2f}s | Disk: {dir_mb(uri):6.2f} MB')

# 3. SWaPO vs Coordinate-Sorted Parquet (via local swapo package if present)
sys.path.insert(0, '/data')
sys.path.insert(0, '/data/src')
try:
    import swapo
    for mode, out_dir in [('Sorted-Parquet', '/tmp/chr22_sorted_pq'), ('SWaPO-Parquet', '/tmp/chr22_swapo_pq')]:
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir, exist_ok=True)
        t0 = time.perf_counter()
        if hasattr(swapo, 'optimize_vcf'):
            swapo.optimize_vcf(VCF_IN, out_dir, isolate_long=(mode == 'SWaPO-Parquet'))
        else:
            subprocess.run([sys.executable, '-m', 'swapo', 'ingest', VCF_IN, out_dir] + ([] if mode == 'SWaPO-Parquet' else ['--no-isolate']), check=True)
        dt = time.perf_counter() - t0
        print(f'{mode:27s} | Ingest: {dt:6.2f}s | Disk: {dir_mb(out_dir):6.2f} MB')
except Exception as e:
    print(f'SWaPO local import note: {e}')

print('\n=== CLINICAL GENE QUERY LATENCY & HIT PARITY (WARM CACHE, 5-RUN MEAN) ===')
q, dq = chr(39), chr(34)
con = duckdb.connect()
for gene, chrom, qs, qe in GENES:
    row_parts = [f'{gene:6s}']
    region = f'{chrom}:{qs}-{qe}'
    out0 = subprocess.check_output(['tabix', VCF_GZ, region], text=True)
    tbx_hits = len([l for l in out0.splitlines() if l])
    t0 = time.perf_counter()
    for _ in range(5):
        subprocess.check_output(['tabix', VCF_GZ, region])
    tbx_ms = ((time.perf_counter() - t0) / 5.0) * 1000
    row_parts.append(f'Tabix: {tbx_ms:5.2f} ms (hits={tbx_hits})')

    for gap in (1000, 10000):
        uri = f'/tmp/tiledb_gap_{gap}'
        ds = tiledbvcf.Dataset(uri, mode='r')
        df0 = ds.read(regions=[region], attrs=['contig', 'pos_start', 'pos_end'])
        hits = len(df0)
        t0 = time.perf_counter()
        for _ in range(5):
            ds.read(regions=[region], attrs=['contig', 'pos_start', 'pos_end'])
        ms = ((time.perf_counter() - t0) / 5.0) * 1000
        row_parts.append(f'TileDB(gap={gap}): {ms:5.2f} ms (hits={hits})')

    for label, dpath in [('Sorted-PQ', '/tmp/chr22_sorted_pq'), ('SWaPO-PQ', '/tmp/chr22_swapo_pq')]:
        if os.path.exists(dpath) and os.listdir(dpath):
            sql = f'SELECT count(*) FROM read_parquet({q}{dpath}/*.parquet{q}) WHERE contig = {q}{chrom}{q} AND {dq}end{dq} >= {qs} AND start <= {qe}'
            hits = con.execute(sql).fetchone()[0]
            t0 = time.perf_counter()
            for _ in range(5):
                con.execute(sql).fetchone()
            ms = ((time.perf_counter() - t0) / 5.0) * 1000
            row_parts.append(f'{label}: {ms:5.2f} ms (hits={hits})')
    print(' | '.join(row_parts))
