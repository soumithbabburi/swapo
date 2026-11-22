import duckdb, numpy as np

con = duckdb.connect()
base = con.execute('SELECT contig, start, "end" FROM read_parquet("chr22_sorted_pq/*.parquet")').fetchnumpy()
starts, ends = base['start'], base['end']
n = len(starts)

GENES = [('COMT', 19941772, 19970744), ('DGCR8', 20080197, 20111641), ('BCR', 23180509, 23318037)]
BLOCK_SIZE = 50_000

def eval_layout(s_arr, e_arr, tau):
    lengths = e_arr - s_arr + 1
    short_mask = lengths < tau
    s_short, e_short = s_arr[short_mask], e_arr[short_mask]
    s_long, e_long = s_arr[~short_mask], e_arr[~short_mask]

    idx_s = np.argsort(s_short, kind='mergesort')
    s_short, e_short = s_short[idx_s], e_short[idx_s]
    idx_l = np.argsort(s_long, kind='mergesort')
    s_long, e_long = s_long[idx_l], e_long[idx_l]

    rg_total, rg_scanned, span_sum = 0, 0.0, 0.0
    for part_s, part_e in [(s_short, e_short), (s_long, e_long)]:
        m = len(part_s)
        if m == 0: continue
        for i in range(0, m, BLOCK_SIZE):
            bs, be = part_s[i:i+BLOCK_SIZE], part_e[i:i+BLOCK_SIZE]
            z_min, z_max = int(bs.min()), int(be.max())
            span_sum += (z_max - z_min + 1)
            rg_total += 1
            rg_scanned += sum(1 for _, qs, qe in GENES if z_max >= qs and z_min <= qe) / len(GENES)

    b_total, b_scanned, b_span = 0, 0.0, 0.0
    idx_all = np.argsort(s_arr, kind='mergesort')
    sa, ea = s_arr[idx_all], e_arr[idx_all]
    for i in range(0, len(sa), BLOCK_SIZE):
        bs, be = sa[i:i+BLOCK_SIZE], ea[i:i+BLOCK_SIZE]
        z_min, z_max = int(bs.min()), int(be.max())
        b_span += (z_max - z_min + 1)
        b_total += 1
        b_scanned += sum(1 for _, qs, qe in GENES if z_max >= qs and z_min <= qe) / len(GENES)

    pruned_pct = (1.0 - (rg_scanned / rg_total)) * 100.0
    base_pruned_pct = (1.0 - (b_scanned / b_total)) * 100.0
    delta_frac = max((b_scanned / b_total) - (rg_scanned / rg_total), 1e-6)
    q_star = 0.015 / (0.005 * delta_frac)
    return len(s_long)/n*100, rg_total, rg_scanned, pruned_pct, base_pruned_pct, span_sum/b_span, q_star

print('=== PART A: REAL GNOMAD CHR22 THRESHOLD (TAU) SENSITIVITY ===')
print(f'{ "Tau (bp)":>10s} | { "Long %":>7s} | { "RGs Scanned":>13s} | { "SWaPO Pruned":>12s} | { "Base Pruned":>11s} | { "Zone Span Ratio":>15s} | { "Break-Even Q*":>13s}')
for tau in [50, 500, 5000, 50000]:
    lp, rgt, rgs, pp, bpp, sr, qs = eval_layout(starts, ends, tau)
    print(f'{tau:10,d} | {lp:6.2f}% | {rgs:5.2f} / {rgt:<4d} | {pp:11.1f}% | {bpp:10.1f}% | {sr:14.3f}x | {qs:12.1f}')

print('\n=== PART B: ADVERSARIAL SV SKEW SWEEP (0.2% GERMLINE TO 40% SOMATIC CANCER SV, TAU=500 BP) ===')
rng = np.random.default_rng(42)
for sv_frac in [0.002, 0.05, 0.20, 0.40]:
    s_sim = starts.copy()
    e_sim = s_sim + 1
    k = int(n * sv_frac)
    sv_idx = rng.choice(n, size=k, replace=False)
    sv_lens = rng.integers(1_000, 2_000_000, size=k)
    e_sim[sv_idx] = np.minimum(s_sim[sv_idx] + sv_lens, 50_818_468)
    lp, rgt, rgs, pp, bpp, sr, qs = eval_layout(s_sim, e_sim, 500)
    print(f'SV={sv_frac*100:5.1f}% | Long={lp:5.2f}% | RGs={rgs:5.2f}/{rgt:<3d} | SWaPO Pruned={pp:5.1f}% vs Base={bpp:5.1f}% | Span Ratio={sr:5.3f}x | Q*={qs:6.1f}')
