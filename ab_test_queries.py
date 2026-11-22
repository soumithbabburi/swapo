import os, time, pyarrow.parquet as pq
QUERIES = [("COMT (chr22:19.94M-19.97M)", "22", 19941371, 19970316), ("DGCR8 (chr22:20.08M-20.11M)", "22", 20080054, 20109536), ("BCR (chr22:23.18M-23.32M)", "22", 23180366, 23318037)]
def scan_dir(folder, contig, qs, qe):
    files = sorted([os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs if f.endswith(".parquet")])
    t0 = time.perf_counter()
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
    return tot_rg, open_rg, rows_scan, round(bytes_scan/1024.0, 1), hits, round((time.perf_counter()-t0)*1000, 2)
for lbl, contig, qs, qe in QUERIES:
    b = scan_dir("aws_lake_fragmented", contig, qs, qe)
    o = scan_dir("aws_lake_swapo_10k", contig, qs, qe)
    red = (1.0 - (o[2] / max(1, b[2]))) * 100.0
    print(f"\n=== {lbl} ===")
    print(f"  Sorted (Baseline) : opened {b[1]}/{b[0]} blocks - scanned {b[2]:,} rows ({b[3]} KB) - hits={b[4]} - {b[5]} ms")
    print(f"  SWaPO (Optimized) : opened {o[1]}/{o[0]} blocks - scanned {o[2]:,} rows ({o[3]} KB) - hits={o[4]} - {o[5]} ms")
    print(f"  - rows eliminated: {red:.1f}%")