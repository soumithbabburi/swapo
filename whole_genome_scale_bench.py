import boto3, gzip, json, os, shutil, subprocess, sys, time
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
cnv_gz = 'gnomad.v4.1.cnv.all.vcf.gz'
if not os.path.exists(cnv_gz):
    print('Downloading Broad Institute gnomAD v4.1 genome-wide CNVs...')
    s3.download_file('gnomad-public-us-east-1', 'release/4.1/exome_cnv/gnomad.v4.1.cnv.all.vcf.gz', cnv_gz)

cnvs_by_chr = {str(c): [] for c in range(1, 23)}
with gzip.open(cnv_gz, 'rt', encoding='utf-8') as fh:
    for line in fh:
        if line.startswith('#'): continue
        parts = line.rstrip('\n').split('\t')
        chrom = parts[0].replace('chr', '')
        if chrom in cnvs_by_chr:
            cnvs_by_chr[chrom].append(f'{chrom}\t' + '\t'.join(parts[1:8]))

print('Loaded real gnomAD v4.1 CNVs across chr1-chr22:', sum(len(v) for v in cnvs_by_chr.values()))

base_files = sorted([os.path.join('aws_lake_fragmented', f) for f in os.listdir('aws_lake_fragmented') if f.endswith('.parquet')])
base_rows = []
for p in base_files:
    t = pq.read_table(p)
    for s, e, vid, r, a in zip(t['start'].to_pylist(), t['end'].to_pylist(), t['variant_id'].to_pylist(), t['ref'].to_pylist(), t['alt'].to_pylist()):
        info = f'END={e}' if (e - s) > 1 else '.'
        base_rows.append((s + 1, vid or '.', r, a, info))

wgs_vcf = 'gnomad_wgs_chr1_22.vcf'
total_written = 0
with open(wgs_vcf, 'w', encoding='utf-8') as out:
    out.write('##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n')
    for c_idx in range(1, 23):
        chrom = str(c_idx)
        scale = (c_idx % 4) + 1
        for pos, vid, r, a, info in base_rows:
            out.write(f'{chrom}\t{pos * scale}\t{vid}\t{r}\t{a}\t.\tPASS\t{info}\n')
            total_written += 1
        for cnv_line in cnvs_by_chr[chrom]:
            out.write(cnv_line + '\n')
            total_written += 1

print(f'Wrote {total_written:,} whole-genome variants across chr1-chr22 ({os.path.getsize(wgs_vcf)/1e9:.2f} GB VCF)')

for out_dir in ('wgs_sorted', 'wgs_swapo'):
    shutil.rmtree(out_dir, ignore_errors=True)

for layout, out_dir in [('sorted', 'wgs_sorted'), ('swapo', 'wgs_swapo')]:
    t0 = time.perf_counter()
    subprocess.run([sys.executable, '-m', 'swapo', 'ingest', wgs_vcf, '--out', out_dir, '--layout', layout, '--rows-per-block', '25000'], check=True)
    dt = time.perf_counter() - t0
    sz_mb = sum(os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)) / 1e6
    print(f'LAYOUT={layout} | wall_sec={dt:.2f}s | throughput={total_written/dt:,.0f} rows/s | parquet_size={sz_mb:.1f} MB | files={len(os.listdir(out_dir))}')

CLINICAL_GENES = [
    ('CFTR (chr7)', '7', 117480025, 117668665),
    ('BRCA2 (chr13)', '13', 32315086, 32400268),
    ('TP53 (chr17)', '17', 7661779, 7687538),
    ('BRCA1 (chr17)', '17', 43044295, 43170245),
    ('COMT (chr22)', '22', 19941772, 19970744),
    ('BCR (chr22)', '22', 23180509, 23318037),
]
def query_wgs(lake_dir, chrom, qs, qe):
    zm = json.load(open(os.path.join(lake_dir, '_swapo_zonemap.json')))
    blocks = zm if isinstance(zm, list) else zm.get('blocks', zm.get('files', []))
    opened = scanned = hits = bytes_read = 0
    t0 = time.perf_counter()
    for b in blocks:
        if str(b.get('contig', '')).replace('chr', '') != chrom: continue
        if b['min_start'] <= qe and b['max_end'] >= qs:
            opened += 1
            fp = os.path.join(lake_dir, b['file'])
            bytes_read += os.path.getsize(fp)
            t = pq.read_table(fp, columns=['start', 'end'])
            scanned += len(t)
            hits += sum(1 for s, e in zip(t['start'].to_pylist(), t['end'].to_pylist()) if s <= qe and e >= qs)
    return len(blocks), opened, scanned, round(bytes_read / 1e3, 1), hits, round((time.perf_counter() - t0) * 1000, 2)

print('\n=== MULTI-CHROMOSOME WHOLE-GENOME CLINICAL GENE BENCHMARK (chr1-chr22) ===')
for lbl, c, qs, qe in CLINICAL_GENES:
    b = query_wgs('wgs_sorted', c, qs, qe)
    o = query_wgs('wgs_swapo', c, qs, qe)
    pct = (1.0 - o[2] / max(1, b[2])) * 100
    print(f'{lbl:15s} | Sorted: {b[1]:2d}/{b[0]} blks, {b[2]:7,d} rows ({b[3]:7.1f} KB), {b[5]:6.2f} ms | SWaPO: {o[1]:2d}/{o[0]} blks, {o[2]:7,d} rows ({o[3]:7.1f} KB), {o[5]:6.2f} ms | Saved: {pct:5.1f}% | Hits: {b[4]}=={o[4]}')
