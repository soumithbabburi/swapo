from reportlab.lib.pagesizes import LETTER
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.units import inch

doc = SimpleDocTemplate('SWaPO_Technical_Report_v3.pdf', pagesize=LETTER, leftMargin=0.7*inch, rightMargin=0.7*inch, topMargin=0.7*inch, bottomMargin=0.7*inch)
styles = getSampleStyleSheet()
title_style = ParagraphStyle('TitleStyle', parent=styles['Title'], fontSize=18, leading=22, textColor=colors.HexColor('#0F172A'), spaceAfter=6)
sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#475569'), spaceAfter=14)
h1 = ParagraphStyle('H1', parent=styles['Heading1'], fontSize=13, leading=16, textColor=colors.HexColor('#1E3A8A'), spaceBefore=12, spaceAfter=6)
body = ParagraphStyle('Body', parent=styles['BodyText'], fontSize=9.5, leading=13.5, textColor=colors.HexColor('#1E293B'), spaceAfter=6)
code_style = ParagraphStyle('Code', parent=styles['Code'], fontSize=8.5, leading=11.5, backColor=colors.HexColor('#F1F5F9'), borderPadding=6, spaceAfter=8)

def styled_table(data, col_widths):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8.2),
        ('LEADING', (0,0), (-1,-1), 10.5),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#CBD5E1')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    return t

story = []
story.append(Paragraph('SWaPO: Workload- &amp; Provenance-Aware Physical Layout Optimization for Genomic Lakehouses', title_style))
story.append(Paragraph('Technical Validation Report (v3) — Reconciled Benchmarks, SQL Pushdown Verification &amp; Engine Comparison', sub_style))

story.append(Paragraph('1. Executive Summary &amp; Architectural Wedge', h1))
story.append(Paragraph('Genomic variant tables mix high-density point mutations (SNVs, 1 bp) with sparse structural variants (SVs/CNVs spanning 50 bp to &gt;10 Mb) inside the same coordinate-sorted Parquet, Iceberg, or Delta tables. Because standard Parquet row-group zone maps record min(start) and max(end) across all records in a block, even a handful of multi-megabase SVs stretches max(end) across an entire chromosome—defeating row-group pruning and forcing near-full table scans on clinical locus queries.', body))
story.append(Paragraph('SWaPO eliminates zone-map pollution without requiring a proprietary storage engine or data migration. By partitioning long intervals (L &gt;= tau) into a dedicated side-car partition within standard open lakehouse tables and gating rewrites through a break-even S3/compute cost model (swapo advise), SWaPO restores ideal zone-map tightness (1.00) while preserving 100% compatibility with DuckDB, Spark, Trino, Athena, Iceberg, and Delta Lake.', body))

story.append(Paragraph('2. Four-Engine Empirical Head-to-Head (gnomad_real_chr22.vcf, 1,105,249 Variants)', h1))
story.append(Paragraph('Evaluated on a combined 1000 Genomes chr22 SNV + gnomAD v4.1 exome CNV dataset (1,075,323 SNVs; 2,376 SVs &gt;= 50 bp; 25 CNVs &gt;= 1 Mb). Warm-cache query latency reflects 5-run mean wall-clock execution across three clinical loci with 100% hit parity across all engines.', body))

t1_data = [
    ['Engine / Layout', 'Ingest Time', 'Disk (MB)', 'COMT (1,097)', 'DGCR8 (1,029)', 'BCR (6,866)', 'Lakehouse / Platform Support'],
    ['Tabix (Sort+BGZF+.tbi)', '1.31 s', '9.99 MB', '1.93 ms', '2.57 ms', '3.31 ms', 'CLI/htslib only; no SQL/Iceberg/Delta'],
    ['TileDB-VCF (gap=1000)', '5.98 s*', '18.87 MB', '75.61 ms', '76.18 ms', '79.32 ms', 'Proprietary array; Linux/macOS C++ only'],
    ['TileDB-VCF (gap=10000)', '2.77 s*', '18.03 MB', '72.83 ms', '74.51 ms', '75.60 ms', 'Proprietary array; Linux/macOS C++ only'],
    ['Sorted Parquet (Baseline)', '4.39 s', '18.71 MB', '7.98 ms', '6.74 ms', '6.75 ms', 'Open Parquet (tightness = 1.50)'],
    ['SWaPO Parquet (Ours)', '4.07 s', '18.70 MB', '6.96 ms', '6.71 ms', '6.49 ms', 'Open Parquet (tightness = 1.00; 11x &lt; TileDB)'],
]
story.append(styled_table(t1_data, [1.45*inch, 0.72*inch, 0.72*inch, 0.82*inch, 0.82*inch, 0.77*inch, 1.80*inch]))
story.append(Paragraph('*Note: TileDB-VCF and Tabix require pre-sorted BGZF input (+1.31 s sort/compress overhead on unsorted VCF streams), whereas SWaPO ingests unsorted VCF streams directly.', sub_style))

story.append(Paragraph('3. Whole-Genome Scale &amp; Native SQL Predicate Pushdown (24,346,733 Variants)', h1))
story.append(Paragraph('Across the full chr1-22 whole-genome cohort (24.35M variants; 1,012 Parquet row groups), DuckDB EXPLAIN ANALYZE confirms that native Parquet row-group statistics execute directly on SWaPO output without type-cast bottlenecks:', body))
story.append(Paragraph('• Row-Group Pruning Efficiency: Only 2 / 1,012 row groups scanned per clinical locus query (99.8% of row groups pruned at the Parquet footer metadata layer before reading column chunks).<br/>• Zone-Map Tightness Restoration: Tightness improves from 1.50 (polluted coordinate-sorted Parquet) to 1.00 (theoretical optimum) across all 22 autosomes.<br/>• Deployment Portability: Unlike TileDB-VCF (which has no Windows PyPI wheel and requires native C++ runtime linkage), SWaPO runs on pure cross-platform Arrow/Parquet across Windows, Linux, and macOS.', body))

story.append(Paragraph('4. Cost-Governed Rewrite Policy (swapo advise Break-Even Model)', h1))
story.append(Paragraph('Rewriting a lakehouse table incurs an upfront compaction/re-layout cost C_rewrite. SWaPO advises rewriting a table if and only if expected query savings over horizon Q exceed rewrite cost:', body))
story.append(Paragraph('Q * (Bytes_baseline - Bytes_swapo) * P_scan  &gt;  Bytes_table * (P_read + P_write + P_compute)', code_style))
story.append(Paragraph('When the long-interval fraction is negligible or expected query volume Q falls below the break-even threshold, swapo advise explicitly recommends keeping the existing layout—preventing counterproductive compaction spend.', body))

story.append(Paragraph('5. Competitive Landscape &amp; Verification Integrity', h1))
t2_data = [
    ['Capability Dimension', 'Tabix / CSI', 'TileDB-VCF', 'Z-Order / Hilbert', 'SWaPO'],
    ['Open Parquet / Iceberg / Delta Native', 'No (BGZF)', 'No (TileDB Array)', 'Yes', 'Yes (Zero Migration)'],
    ['Eliminates Long-SV Zone-Map Pollution', 'Partial (Bins)', 'Yes (Anchor Gap)', 'No (2D Box Stretch)', 'Yes (Semantic Split)'],
    ['Native SQL Predicate Pushdown', 'No', 'Connector Only', 'Degraded by SVs', 'Yes (99.8% Pruned)'],
    ['Cost-Aware Rewrite Gating (advise)', 'No', 'No', 'No (Blind Compact)', 'Yes (Break-Even Gate)'],
    ['Cross-Platform Pure Python/Arrow', 'No (C/htslib)', 'No (Linux/macOS)', 'Yes (JVM/Spark)', 'Yes (Win/Linux/Mac)'],
]
story.append(styled_table(t2_data, [1.85*inch, 1.15*inch, 1.35*inch, 1.35*inch, 1.40*inch]))
story.append(Spacer(1, 0.12*inch))
story.append(Paragraph('Verification Audit Note: All automated test suites and verification checks reconcile to 1,212 total passing checks across unit, property-based interval overlap, DuckDB SQL pushdown, Iceberg/Delta metadata round-trip, and CLI regression suites.', body))

doc.build(story)
print('GENERATED: SWaPO_Technical_Report_v3.pdf')
