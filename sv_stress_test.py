import os, shutil, random, subprocess, sys, pyarrow as pa, pyarrow.parquet as pq
QUERIES = [("COMT (chr22:19.94M-19.97M)", "22", 19941371, 19970316), ("DGCR8 (chr22:20.08M-20.11M)", "22", 20080054, 20109536), ("BCR (chr22:23.18M-23.32M)", "22", 23180366, 23318037)]
def scan_dir(folder, contig, qs, qe):
    files = sorted([os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs if f.endswith(".parquet")])
    tot_rg = open_rg = rows_scan = bytes_scan = hits = 0
    for p in files:
        pf = pq.ParquetFile(p)
        cols = pf.schema_arrow.names
        ci, si, ei = cols.index("contig"), cols.index("start"), cols.index("end")
        for rg in range(pf.num_row_groups):
            tot_rg += 1
            m = pf.metadata.row_group(rg)
            cs, ss, es = m.column(ci).statistics, m.column(si).statistics, m.column(ei).statistics
            if cs and ss and es:
                if str(cs.max) != contig and str(cs.min) != contig: continue
                if ss.min > qe or es.max < qs: continue
            open_rg += 1
            rows_scan += m.num_rows
            bytes_scan += m.total_byte_size
            t = pf.read_row_group(rg, columns=["contig", "start", "end"])
            for c, s, e in zip(t["contig"].to_pylist(), t["start"].to_pylist(), t["end"].to_pylist()):
                if str(c) == contig and s <= qe and e >= qs: hits += 1
    return tot_rg, open_rg, rows_scan, round(bytes_scan/1024.0, 1), hits
src, polluted, opt, vcf = "aws_lake_fragmented", "aws_lake_sv_polluted", "aws_lake_sv_swapo", "sv_polluted_merged.vcf"
for d in (polluted, opt): shutil.rmtree(d, ignore_errors=True)
shutil.copytree(src, polluted)
files = sorted([os.path.join(polluted, f) for f in os.listdir(polluted) if f.endswith(".parquet")])
rng = random.Random(42)
with open(vcf, "w", encoding="utf-8") as out:
    out.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    for i, p in enumerate(files):
        tbl = pq.read_table(p)
        starts, ends = tbl["start"].to_pylist(), tbl["end"].to_pylist()
        if i < 60:
            idx = rng.randint(0, len(ends)-1)
            ends[idx] = starts[idx] + rng.randint(2_000_000, 15_000_000)
            cols = {n: tbl[n] for n in tbl.column_names}
            cols["end"] = pa.array(ends, type=tbl.schema.field("end").type)
            pq.write_table(pa.table(cols, schema=tbl.schema), p, row_group_size=10000)
        for c, s, e, vid, r, a in zip(tbl["contig"].to_pylist(), starts, ends, tbl["variant_id"].to_pylist(), tbl["ref"].to_pylist(), tbl["alt"].to_pylist()):
            info = f"END={e}" if (e - s) > 1 else "."
            out.write(f"{c}\t{s+1}\t{vid or "."}\t{r}\t{a}\t.\tPASS\t{info}\n")
subprocess.run([sys.executable, "-m", "swapo", "ingest", vcf, "--out", opt, "--layout", "swapo", "--rows-per-block", "10000"], check=True)
for lbl, contig, qs, qe in QUERIES:
    b = scan_dir(polluted, contig, qs, qe)
    o = scan_dir(opt, contig, qs, qe)
    red = (1.0 - (o[2] / max(1, b[2]))) * 100.0
    print(f"\n=== SV STRESS: {lbl} ===")
    print(f"  SV-Polluted Baseline : opened {b[1]}/{b[0]} blocks - scanned {b[2]:,} rows ({b[3]} KB) - hits={b[4]}")
    print(f"  SWaPO Quarantined    : opened {o[1]}/{o[0]} blocks - scanned {o[2]:,} rows ({o[3]} KB) - hits={o[4]}")
    print(f"  - rows eliminated: {red:.1f}%")