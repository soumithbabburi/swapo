"""Builds the submission manuscript (DOCX) in the Bioinformatics (OUP) Original Paper layout.

Run:  python paper_v6/build_manuscript.py   then convert with Word (see to_pdf.ps1).
Every number in the text comes from results/, paper_out/ or docs/paper.md.
"""
import os, re
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Cm, RGBColor, Inches

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
OUT = os.path.join(HERE, 'SWaPO_manuscript.docx')
BLACK = RGBColor(0, 0, 0)

doc = Document()
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(27.94), Cm(21.59)          # US Letter
for side in ('left_margin', 'right_margin', 'top_margin', 'bottom_margin'):
    setattr(sec, side, Cm(2.54))

# continuous line numbers, as reviewers expect
ln = OxmlElement('w:lnNumType')
ln.set(qn('w:countBy'), '1'); ln.set(qn('w:restart'), 'continuous'); ln.set(qn('w:distance'), '284')
sec._sectPr.append(ln)

base = doc.styles['Normal']
base.font.name = 'Times New Roman'; base.font.size = Pt(12); base.font.color.rgb = BLACK
base.element.rPr.rFonts.set(qn('w:eastAsia'), 'Times New Roman')
pf = base.paragraph_format
pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
pf.space_after = Pt(0); pf.space_before = Pt(0)

# page number in footer
fp = sec.footer.paragraphs[0]; fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
for t, txt in (('begin', None), (None, 'PAGE'), ('end', None)):
    r = fp.add_run()
    if t:
        e = OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'), t); r._r.append(e)
    else:
        e = OxmlElement('w:instrText'); e.set(qn('xml:space'), 'preserve'); e.text = txt; r._r.append(e)


def no_linenum(p):
    pPr = p._p.get_or_add_pPr()
    s = OxmlElement('w:suppressLineNumbers'); pPr.append(s)


def runs(p, text, size=None, bold=False, italic=False):
    """Text with *italic* and **bold** spans."""
    for part in re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|_\{[^}]+\}|\^\{[^}]+\})', text):
        if not part:
            continue
        b, i, sub, sup = bold, italic, False, False
        if part.startswith('**'):
            part, b = part[2:-2], True
        elif part.startswith('*'):
            part, i = part[1:-1], True
        elif part.startswith('_{'):
            part, sub = part[2:-1], True
        elif part.startswith('^{'):
            part, sup = part[2:-1], True
        r = p.add_run(part); r.bold = b; r.italic = i; r.font.color.rgb = BLACK
        r.font.subscript = sub; r.font.superscript = sup
        if size:
            r.font.size = Pt(size)
    return p


def para(text, indent=True, align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, after=0, spacing=None, bold=False, italic=False):
    p = doc.add_paragraph(); p.alignment = align
    if indent:
        p.paragraph_format.first_line_indent = Cm(0.75)
    p.paragraph_format.space_after = Pt(after)
    if spacing:
        p.paragraph_format.line_spacing_rule = spacing
    return runs(p, text, size, bold, italic)


def heading(text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text); r.bold = True; r.italic = level == 3; r.font.color.rgb = BLACK
    r.font.size = Pt(12)
    return p


def equation(text, num):
    t = doc.add_table(rows=1, cols=2); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.columns[0].width = Cm(14.5); t.columns[1].width = Cm(2)
    c0, c1 = t.rows[0].cells
    c0.width, c1.width = Cm(14.5), Cm(2)
    p = c0.paragraphs[0]; p.alignment = WD_ALIGN_PARAGRAPH.CENTER; runs(p, text, italic=False)
    q = c1.paragraphs[0]; q.alignment = WD_ALIGN_PARAGRAPH.RIGHT; q.add_run(f'({num})')


def border(cell, edges):
    tcPr = cell._tc.get_or_add_tcPr()
    b = OxmlElement('w:tcBorders')
    for edge, sz in edges.items():
        e = OxmlElement(f'w:{edge}'); e.set(qn('w:val'), 'single'); e.set(qn('w:sz'), str(sz)); e.set(qn('w:color'), '000000')
        b.append(e)
    tcPr.append(b)


def table(caption, header, rows, widths, note=None):
    cp = doc.add_paragraph(); cp.paragraph_format.keep_with_next = True
    cp.paragraph_format.space_before = Pt(6)
    cp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    num, rest = caption.split('. ', 1)
    r = cp.add_run(num + '. '); r.bold = True; r.font.size = Pt(10)
    runs(cp, rest, size=10)
    t = doc.add_table(rows=1 + len(rows), cols=len(header)); t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i, row in enumerate([header] + rows):
        for j, val in enumerate(row):
            c = t.cell(i, j); c.width = Cm(widths[j])
            p = c.paragraphs[0]
            p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.RIGHT
            runs(p, str(val), size=9, bold=(i == 0))
            edges = {}
            if i == 0:
                edges = {'top': 12, 'bottom': 6}
            if i == len(rows):
                edges['bottom'] = 12
            if edges:
                border(c, edges)
    if note:
        n = doc.add_paragraph(); n.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
        n.paragraph_format.space_after = Pt(8)
        runs(n, note, size=9)
    else:
        doc.add_paragraph().paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE


def figure(name, width_in, caption):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True; p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    p.add_run().add_picture(os.path.join(FIG, name + '.png'), width=Inches(width_in))
    cp = doc.add_paragraph(); cp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    cp.paragraph_format.space_after = Pt(10); cp.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    num, rest = caption.split('. ', 1)
    r = cp.add_run(num + '. '); r.bold = True; r.font.size = Pt(10)
    runs(cp, rest, size=10)


exec(open(os.path.join(HERE, 'manuscript_text.py'), encoding='utf-8').read())
doc.save(OUT)
print('wrote', OUT)
