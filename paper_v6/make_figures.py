"""Figures for the manuscript. Every number is read from results/ or paper_out/."""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = lambda p: json.load(open(os.path.join(ROOT, p)))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
                     'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 600, 'axes.linewidth': 0.6})
C = ['#1b4f72', '#c0392b', '#7f8c8d', '#27ae60']
STORES = {'Local object store (1 ms, 500 MB/s)': (0.001, 500e6),
          'Same region cloud (30 ms, 90 MB/s)': (0.030, 90e6),
          'Remote cloud, measured (170 ms, 5.2 MB/s)': (0.170, 5.2e6)}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), bbox_inches='tight')
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


# Figure 1: mechanism
fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.2), sharex=True, gridspec_kw={'hspace': 0.55})
q = (46, 52)
for ax, split, title in ((axes[0], False, 'A  Sorted by start: one long record raises max(end) of its row group'),
                         (axes[1], True, 'B  Long records stored as their own row group')):
    groups = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
    for i, (a, b) in enumerate(groups):
        reach = b + (38 if (i == 1 and not split) else 0)
        hit = a < q[1] and reach > q[0]
        ax.add_patch(Rectangle((a, 1.1), b - a - 0.8, 0.8, fc='#d6e4f0' if not hit else '#f5b7b1', ec='#1b4f72', lw=0.6))
        ax.text((a + b) / 2, 1.5, f'RG {i + 1}', ha='center', va='center', fontsize=7)
        if reach > b:
            ax.annotate('', xy=(reach, 1.0), xytext=(b, 1.0), arrowprops=dict(arrowstyle='->', color=C[1], lw=0.8))
            ax.text(b + 1, 0.72, 'reach of RG 2 = max(end)', color=C[1], fontsize=6.5)
    if split:
        ax.add_patch(Rectangle((0, 0.05), 99.2, 0.55, fc='#f5b7b1', ec='#1b4f72', lw=0.6))
        ax.text(50, 0.32, 'Long record row group: its bounds span the contig, so every query reads it (one extra request)', ha='center', va='center', fontsize=6.5)
    ax.axvspan(*q, color='#2c3e50', alpha=0.15, lw=0)
    ax.text(sum(q) / 2, 2.05, 'query', ha='center', fontsize=6.5)
    ax.set_xlim(0, 100); ax.set_ylim(0, 2.3); ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    if not split:
        ax.spines['bottom'].set_visible(False); ax.tick_params(axis='x', length=0)
    ax.set_title(title, loc='left')
axes[1].set_xlabel('Position along contig (arbitrary units)')
save(fig, 'fig1_mechanism')


# Figure 2: separability against what the split achieved
pts = [  # (label, separability, best bytes reduction from splitting, source)
    ('DGV merged SVs', R('results/svsplit_dgv.json')['122880']['separability']['bimodality'],
     max(1 - t['megabytes'] / v['unsplit']['megabytes'] for v in R('results/svsplit_dgv.json').values() for t in v['thresholds']), 'results/svsplit_dgv.json'),
    ('chr1 annotation table', 75.1, 0.0, 'paper_out/crossover.json'),
    ('chr1, GRCh38 only', 87.6, 0.0, 'docs/paper.md'),
    ('gnomAD v4.1 window', 129, 0.0, 'docs/paper.md'),
    ('chr21 stress table', 407, 0.0, 'docs/paper.md'),
    ('Simulator', 39701, 0.69, 'docs/paper.md'),
]
c22 = R('results/split_1kg_chr22_sizes.json')
best22 = max(1 - r['split']['variant']['megabytes'] / r['sorted']['variant']['megabytes'] for r in c22['runs'])
pts.append(('1000 Genomes chr22 callset', c22['profile']['separability'], best22, 'results/split_1kg_chr22_sizes.json'))
fig, ax = plt.subplots(figsize=(3.4, 2.5))
for lab, s, y, _ in pts:
    real = lab != 'Simulator'
    ax.scatter(s, y * 100, s=22, marker='o' if real else 's', fc=C[0] if real else 'white', ec=C[0] if real else C[1], lw=0.8, zorder=3)
    if lab in ('Simulator', '1000 Genomes chr22 callset', 'DGV merged SVs'):
        ax.annotate(lab, (s, y * 100), xytext=(-5, 6) if lab != 'DGV merged SVs' else (4, -10), textcoords='offset points', fontsize=6.3,
                    ha='right' if lab != 'DGV merged SVs' else 'left')
ax.annotate('Four annotation and population\ntables (75 to 407): no saving', (120, 1), xytext=(25, 22), textcoords='offset points', fontsize=6.3, ha='left',
            arrowprops=dict(arrowstyle='-', lw=0.5, color='black'))
ax.set_xscale('log'); ax.axhline(0, color='black', lw=0.5)
ax.axvspan(1000, 2e5, color=C[3], alpha=0.07, lw=0)
ax.set_xlim(3, 3e5); ax.set_ylim(-15, 80)
ax.set_xlabel('Separability, p99.99 / p99 of record span (log scale)')
ax.set_ylabel('Best reduction in bytes read by splitting (%)')
save(fig, 'fig2_separability')


# Figure 3: the bimodal callset, by row group size and store
runs = {r['rg_rows']: r for f in ('results/split_1kg_chr22.json', 'results/split_1kg_chr22_sizes.json')
        for r in R(f)['runs'] if r['tau_bp'] == 500}
sizes = sorted(runs)
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
for k, (name, (lat, bw)) in enumerate(STORES.items()):
    t = lambda x: x['requests'] * lat + x['megabytes'] * 1e6 / bw
    axes[0].plot(sizes, [100 * (1 - t(runs[s]['split']['gene']) / t(runs[s]['sorted']['gene'])) for s in sizes],
                 marker='o', ms=3, lw=1, color=C[k], label=name)
axes[0].plot(sizes, [100 * (1 - runs[s]['split']['gene']['megabytes'] / runs[s]['sorted']['gene']['megabytes']) for s in sizes],
             ls='--', color='black', lw=0.9, label='Bytes read only')
axes[0].axhline(0, color='black', lw=0.5)
axes[0].set_xscale('log', base=2); axes[0].set_xlabel('Rows per row group (log scale)')
axes[0].set_ylabel('Saving from splitting (%)'); axes[0].set_title('A  Split against sorted, same row group size (gene queries)', loc='left')
axes[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(1.1, -0.28), ncol=2)
lat, bw = STORES['Remote cloud, measured (170 ms, 5.2 MB/s)']
for lay, col, lab in (('sorted', C[0], 'Sorted by start'), ('split', C[1], 'Split at 500 bp')):
    axes[1].plot(sizes, [1000 * (runs[s][lay]['gene']['requests'] * lat + runs[s][lay]['gene']['megabytes'] * 1e6 / bw) for s in sizes],
                 marker='o', ms=3, lw=1, color=col, label=lab)
axes[1].set_xscale('log', base=2); axes[1].set_xlabel('Rows per row group (log scale)')
axes[1].set_ylabel('Mean time per gene query (ms)'); axes[1].set_title('B  Remote cloud store', loc='left')
axes[1].legend(frameon=False)
save(fig, 'fig3_callset_split')


# Figure 4: row group sizing
d = R('results/rowgroups_dgv.json')['sets']['mixed']
lat, bw = 0.170, 5.2e6
xs = [r['rg_rows'] for r in d]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
axes[0].plot(xs, [1000 * (r['meas_requests'] * lat + r['meas_mb'] * 1e6 / bw) for r in d], 'o-', ms=3, lw=1, color=C[0], label='Measured')
axes[0].plot(xs, [1000 * (r['pred_requests'] * lat + r['pred_mb'] * 1e6 / bw) for r in d], 's--', ms=3, lw=1, color=C[1], mfc='white', label='Predicted from footers')
axes[0].set_xscale('log', base=2); axes[0].set_xlabel('Rows per row group (log scale)')
axes[0].set_ylabel('Mean time per query (ms)'); axes[0].set_title('A  DGV, 500 mixed queries', loc='left')
axes[0].legend(frameon=False)
labels, before, after, pred = [], [], [], []
for fmt in ('delta', 'iceberg'):
    f = R(f'results/format_{fmt}_gnomad_window.json')
    labels.append({'delta': 'Delta Lake', 'iceberg': 'Apache Iceberg'}[fmt])
    before.append(f['before']['ms']); after.append(f['after']['ms'])
x = range(len(labels))
axes[1].bar([i - 0.18 for i in x], before, 0.34, color=C[2], label='Writer default (one row group)')
axes[1].bar([i + 0.18 for i in x], after, 0.34, color=C[0], label='Recommended (65,536 rows)')
for i in x:
    axes[1].text(i + 0.18, after[i] + 60, f'{100 * (1 - after[i] / before[i]):.0f}% less', ha='center', fontsize=6.5)
axes[1].set_xticks(list(x)); axes[1].set_xticklabels(labels)
axes[1].set_ylabel('Mean time per query (ms)'); axes[1].set_title('B  gnomAD v4.1 window, 86 queries', loc='left')
axes[1].legend(frameon=False, loc='upper right')
save(fig, 'fig4_rowgroups')


# Figure 5: staleness
s = R('results/staleness_1kg_chr21.json')
g = [100 * v for v in s['growth']]
f = s['flip_rate_af_ge_0_1pct']
fig, ax = plt.subplots(figsize=(3.4, 2.3))
ax.fill_between(g, [100 * v for v in f['p10']], [100 * v for v in f['p90']], color=C[0], alpha=0.18, lw=0, label='10th to 90th percentile, 50 orders')
ax.plot(g, [100 * v for v in f['median']], 'o-', ms=3, lw=1, color=C[0], label='Median')
p = s['phase1_to_phase3']
ax.scatter([100 * p['growth']], [100 * p['flip_rate_af_ge_0_1pct']], marker='*', s=50, color=C[1], zorder=3, label='Phase 1 to phase 3 as released')
ax.set_xlabel('Cohort growth since materialization (%)')
ax.set_ylabel('Variants with AF ≥ 0.1% in a\nwrong frequency class (%)')
ax.legend(frameon=False, loc='lower right')
save(fig, 'fig5_staleness')
print('figures written to', OUT)
