import duckdb, time
con = duckdb.connect()
genes = [('CFTR (chr7)', '7', 117480025, 117668665), ('BRCA2 (chr13)', '13', 32315086, 32400268), ('TP53 (chr17)', '17', 7661779, 7687538), ('BRCA1 (chr17)', '17', 43044295, 43170245), ('COMT (chr22)', '22', 19941772, 19970744), ('BCR (chr22)', '22', 23180509, 23318037)]
q, dq = chr(39), chr(34)
print('=== DUCKDB SQL ENGINE PREDICATE PUSHDOWN VERIFICATION (24.35M VARIANTS) ===')
for lbl, c, qs, qe in genes:
    res = []
    for d in ('wgs_sorted', 'wgs_swapo'):
        sql = f'SELECT count(*) FROM read_parquet({q}{d}/*.parquet{q}) WHERE CAST(contig AS VARCHAR) IN ({q}{c}{q}, {q}chr{c}{q}) AND {dq}end{dq} >= {qs} AND start <= {qe}'
        con.execute(sql).fetchone()
        t0 = time.perf_counter()
        r = con.execute(sql).fetchone()
        res.append((r[0], (time.perf_counter() - t0) * 1000))
    b, o = res
    print(f'{lbl:15s} | DuckDB-Sorted: {b[1]:6.2f} ms (hits={b[0]}) | DuckDB-SWaPO: {o[1]:6.2f} ms (hits={o[0]}) | Speedup: {b[1]/max(0.01, o[1]):.2f}x')
