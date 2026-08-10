# -*- coding: utf-8 -*-
"""
Build a plain, single-column Word manuscript from en/_BaoCao_Full_EN.md,
following the CEA (Computers and Electronics in Agriculture) submission
rules in JOURNAL_TEMPLATE_RULES_CEA.md -- NOT the old Springer conference
template (build_conference_docx.py), which is for a different, numbered-
citation conference format and double-blind anonymization that no longer
match this manuscript.

Differences from build_conference_docx.py:
  - No template file required; starts from a blank python-docx Document
    (single column by construction -> FILE-02).
  - Real author name/affiliation on the title block (no double-blind
    anonymization -> FRONT-02/03), pulled from vn/11_title_page.md.
  - References are parsed in author-year (Harvard) style with no leading
    "[N]" bracket, matching the current en/09_references.md content; any
    URL found in a reference line is turned into a hyperlink (covers both
    "https://doi.org/..." at the end and mid-sentence dataset URLs).
  - Uses python-docx's built-in styles (Heading 1/2/3, List Bullet, List
    Number, Quote) instead of custom styles from a template.

USAGE:
  python build_journal_docx_en.py
Then open the .docx, Ctrl+A, F9 to recompute the SEQ Table/Figure/Equation
fields (same convention as the other build scripts in this folder).
"""
import os
import re
import sys

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor, Cm

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.dirname(HERE)  # outputs/
if ASSET_DIR not in sys.path:
    sys.path.insert(0, ASSET_DIR)

try:
    from mathomml import latex_to_omath, available as _omath_available
    HAS_OMATH = _omath_available()
except Exception:
    HAS_OMATH = False

from latex_png import render_latex_to_png, configure_cache
try:
    from PIL import Image as _PILImage
    HAS_PIL = True
except Exception:
    HAS_PIL = False

EQ_DPI = 400
configure_cache(os.path.join(HERE, ".latex_cache"))


def latex_png_size_inches(png_path):
    if not HAS_PIL:
        return None
    with _PILImage.open(png_path) as im:
        w, h = im.size
    return w / EQ_DPI, h / EQ_DPI


MD = os.path.join(HERE, "en", "_BaoCao_Full_EN_academic.md")
MD_DIR = os.path.dirname(MD)  # image paths in the .md are relative to en/, not outputs/
OUT = os.path.join(HERE, "en", "ChiliEdgeNet_CEA_Draft_EN_academic.docx")


def resolve_img_path(path):
    path = path.strip().strip("<>")
    return os.path.normpath(os.path.join(MD_DIR, path.replace("/", os.sep)))

CONTENT_WIDTH_IN = 6.3  # 8.5in page, 1in margins each side, single column
CONTENT_WIDTH_DXA = int(CONTENT_WIDTH_IN * 1440)
MIN_COL_DXA = 500

AUTHOR_NAME = "Tô Minh Hải\u00b9"
AFFILIATION = "\u00b9 University of Information Technology, Vietnam National University Ho Chi Minh City (VNUHCM-UIT)"
CORRESPONDING = "Corresponding author: Tô Minh Hải (tominhhai0801@gmail.com)"

# ---------------------------------------------------------------------------
doc = docx.Document()

# Page setup: US Letter, 1in margins, single column (default -> FILE-02 MUST).
sec = doc.sections[0]
sec.page_width = Inches(8.5)
sec.page_height = Inches(11)
sec.left_margin = sec.right_margin = Inches(1)
sec.top_margin = sec.bottom_margin = Inches(1)

def _force_style_font(style, font_name="Times New Roman"):
    """Set style.font.name AND strip the w:*Theme rFonts attributes.

    python-docx's blank Document() is based on Word's default template,
    whose built-in Heading 1/2/3 (and List Bullet/Number) styles carry a
    theme font reference (w:asciiTheme="majorHAnsi", i.e. Calibri Light)
    on top of the plain w:ascii/w:hAnsi attributes. Setting style.font.name
    only rewrites the plain attributes; many renderers (incl. LibreOffice,
    used to produce the reviewable PDF) still prefer the *Theme attribute
    when both are present, so headings silently render in the theme font
    instead of Times New Roman even though this script "set" them -- this
    was the visible font-mismatch bug between headings and body text.
    Removing the theme attributes forces the plain font to win everywhere.
    """
    style.font.name = font_name
    rpr = style.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rfonts.get(qn(attr)) is not None:
            del rfonts.attrib[qn(attr)]
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rfonts.set(qn(attr), font_name)


normal = doc.styles["Normal"]
_force_style_font(normal)
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15

for hname, size in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11.5)):
    st = doc.styles[hname]
    _force_style_font(st)
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_before = Pt(12)
    st.paragraph_format.space_after = Pt(6)

for lname in ("List Bullet", "List Number"):
    st = doc.styles[lname]
    _force_style_font(st)
    st.font.size = Pt(11)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def add_field_run(paragraph, field_instr, cached_text, bold=False):
    def _r():
        r = OxmlElement("w:r")
        if bold:
            rpr = OxmlElement("w:rPr")
            b = OxmlElement("w:b")
            rpr.append(b)
            r.append(rpr)
        return r

    r1 = _r(); fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin"); r1.append(fld1)
    r2 = _r(); instr = OxmlElement("w:instrText"); instr.set(qn("xml:space"), "preserve")
    instr.text = f" {field_instr} "; r2.append(instr)
    r3 = _r(); fld3 = OxmlElement("w:fldChar"); fld3.set(qn("w:fldCharType"), "separate"); r3.append(fld3)
    r4 = _r(); t4 = OxmlElement("w:t"); t4.text = cached_text; r4.append(t4)
    r5 = _r(); fld5 = OxmlElement("w:fldChar"); fld5.set(qn("w:fldCharType"), "end"); r5.append(fld5)
    for r in (r1, r2, r3, r4, r5):
        paragraph._p.append(r)


def simplify_math(s):
    s = s.strip()
    if s.startswith("$"):
        s = s.strip("$")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbb\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\operatorname\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\widetilde\{([^{}]*)\}", r"~\1", s)
    for _ in range(3):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    syms = {r"\left[": "[", r"\right]": "]", r"\left(": "(", r"\right)": ")",
            r"\lbrack": "[", r"\rbrack": "]", r"\approx": "≈", r"\cdot": "·",
            r"\times": "×", r"\sum": "Σ", r"\arg\min": "argmin", r"\alpha": "α",
            r"\phi": "φ", r"\pi": "π", r"\xi": "ξ", r"\Omega": "Ω", r"\varepsilon": "ε",
            r"\mathcal{L}": "L", r"\partial": "∂", r"\quad": "  ", r"\,": " ",
            r"\top": "T", r"\vphantom": "", r"\qquad": "   "}
    for k, v in syms.items():
        s = s.replace(k, v)
    s = s.replace("_{", "_").replace("^{", "^").replace("}", "")
    s = re.sub(r"\\operatorname", "", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "")
    return re.sub(r"\s+", " ", s).strip()


def clean_plain(t):
    t = t.replace("\\[", "[").replace("\\]", "]").replace("\\#", "#").replace("\\_", "_")
    return t


_INLINE = re.compile(r"(\*\*.+?\*\*)|(`[^`]+`)|(\$[^$]+\$)|(\*[^*\n]+?\*)")


def add_runs(par, text, base_italic=False, base_size=None):
    pos = 0
    for m in _INLINE.finditer(text):
        if m.start() > pos:
            r = par.add_run(clean_plain(text[pos:m.start()]))
            r.italic = base_italic
            if base_size:
                r.font.size = base_size
        tok = m.group(0)
        if tok.startswith("**"):
            r = par.add_run(clean_plain(tok[2:-2])); r.bold = True; r.italic = base_italic
        elif tok.startswith("`"):
            r = par.add_run(tok[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9.5)
        elif tok.startswith("$"):
            _ltx = tok.strip("$").strip()
            _done = False
            _size_pt = (base_size.pt if base_size else 11)
            png = render_latex_to_png(_ltx, fontsize_pt=_size_pt)
            if png:
                dims = latex_png_size_inches(png)
                if dims:
                    rr = par.add_run()
                    rr.add_picture(png, width=Inches(dims[0]), height=Inches(dims[1]))
                    _done = True
            if not _done and HAS_OMATH:
                try:
                    par._p.append(latex_to_omath(_ltx)); _done = True
                except Exception:
                    _done = False
            if not _done:
                r = par.add_run(simplify_math(tok)); r.italic = True
        elif tok.startswith("*"):
            r = par.add_run(clean_plain(tok[1:-1])); r.italic = True
        if base_size and tok.startswith(("**", "*")):
            r.font.size = base_size
        pos = m.end()
    if pos < len(text):
        r = par.add_run(clean_plain(text[pos:]))
        r.italic = base_italic
        if base_size:
            r.font.size = base_size


def add_hyperlink(par, url, text):
    r_id = par.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hl = OxmlElement("w:hyperlink")
    hl.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    rfonts = OxmlElement("w:rFonts"); rfonts.set(qn("w:ascii"), "Times New Roman"); rpr.append(rfonts)
    col = OxmlElement("w:color"); col.set(qn("w:val"), "0563C1"); rpr.append(col)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rpr.append(u)
    run.append(rpr)
    t = OxmlElement("w:t"); t.text = text; run.append(t)
    hl.append(run)
    par._p.append(hl)


_URL_RE = re.compile(r"https?://[^\s)]+")


def add_reference_paragraph(text):
    """One reference entry: hanging indent, any URL substring hyperlinked."""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.space_after = Pt(6)
    pos = 0
    for m in _URL_RE.finditer(text):
        if m.start() > pos:
            r = p.add_run(text[pos:m.start()])
        url = m.group(0).rstrip(".,;")
        trailing = m.group(0)[len(url):]
        add_hyperlink(p, url, url)
        if trailing:
            p.add_run(trailing)
        pos = m.end()
    if pos < len(text):
        p.add_run(text[pos:])
    for r in p.runs:
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)


def parse_img_width(attr):
    if attr:
        m = re.search(r"(\d+)\s*%", attr)
        if m:
            return Inches(CONTENT_WIDTH_IN * int(m.group(1)) / 100.0)
    return Inches(CONTENT_WIDTH_IN)


def suppress_style_numbering(paragraph):
    """Disable the paragraph style's built-in auto-numbering (numId=0), so a
    numbered list prints the literal digit from the source markdown instead.
    Word's "List Number" style shares one counter across the WHOLE document
    by default, so a later list would otherwise continue (e.g. start at 8)
    from wherever an earlier, unrelated numbered list left off."""
    pPr = paragraph._p.get_or_add_pPr()
    numPr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl"); ilvl.set(qn("w:val"), "0")
    numId = OxmlElement("w:numId"); numId.set(qn("w:val"), "0")
    numPr.append(ilvl); numPr.append(numId)
    pPr.append(numPr)


try:
    from PIL import ImageFont
    _FONT_CACHE = {}
    _FONT_SCALE = 200

    def _get_font(bold):
        key = "bold" if bold else "regular"
        if key not in _FONT_CACHE:
            for path in (
                ("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf" if bold
                 else "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"),
                "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
            ):
                if os.path.exists(path):
                    _FONT_CACHE[key] = ImageFont.truetype(path, _FONT_SCALE)
                    break
            else:
                raise OSError("no font found")
        return _FONT_CACHE[key]

    def text_width_pt(text, size_pt, bold=False):
        if not text:
            return 0.0
        font = _get_font(bold)
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0]) / _FONT_SCALE * size_pt
except Exception:
    def text_width_pt(text, size_pt, bold=False):
        mult = 0.58 if bold else 0.5
        return len(text) * size_pt * mult


def cell_longest_word_width_pt(raw_md_text, size_pt):
    is_bold_cell = "**" in raw_md_text
    plain = re.sub(r"[*`$]", "", raw_md_text)
    best = 0.0
    for tok in re.split(r"\s+", plain.strip()):
        if not tok:
            continue
        w = text_width_pt(tok, size_pt, bold=is_bold_cell)
        best = max(best, w)
    return best


def set_cell_borders(cell, top=None, bottom=None):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    if top:
        b = OxmlElement("w:top"); b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), str(top)); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "000000")
        borders.append(b)
    if bottom:
        b = OxmlElement("w:bottom"); b.set(qn("w:val"), "single")
        b.set(qn("w:sz"), str(bottom)); b.set(qn("w:space"), "0"); b.set(qn("w:color"), "000000")
        borders.append(b)
    tcPr.append(borders)


IMG = re.compile(r"^!\[(.*?)\]\((.*?)\)(\{.*\})?\s*$")
TABLE_CAP = re.compile(r"^\*\*Table\s+\d+\.\s*(.*?)\*\*\s*(.*)$")
FIG_CAP_PREFIX = re.compile(r"^Figure\s+\S+\.\s*(.*)$")

_table_seq = 0
_fig_seq = 0
_eq_seq = 0

# --------------------------------------------------------------------------
# read + split md
# --------------------------------------------------------------------------
raw = open(MD, encoding="utf-8").read().split("\n")
title_line = raw[0].lstrip("#").strip()
kw_idx = next(i for i, l in enumerate(raw) if l.strip().startswith("**Keywords:**"))
keywords_text = raw[kw_idx].strip()[len("**Keywords:**"):].strip()
abs_start = next(i for i, l in enumerate(raw) if l.strip() == "## Abstract")
next_h2 = next(i for i in range(abs_start + 1, len(raw)) if raw[i].strip().startswith("## "))
abstract_lines = raw[abs_start + 1:next_h2]
rest_lines = raw[next_h2:]

# --------------------------------------------------------------------------
# title block
# --------------------------------------------------------------------------
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run(title_line); r.bold = True; r.font.size = Pt(16); r.font.name = "Times New Roman"
p.paragraph_format.space_after = Pt(12)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run(AUTHOR_NAME); r.font.size = Pt(12); r.font.name = "Times New Roman"

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run(AFFILIATION); r.font.size = Pt(10.5); r.italic = True; r.font.name = "Times New Roman"

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run(CORRESPONDING); r.font.size = Pt(10.5); r.font.name = "Times New Roman"
p.paragraph_format.space_after = Pt(18)

p = doc.add_paragraph()
r = p.add_run("Abstract. "); r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(11)
add_runs(p, " ".join(x.strip() for x in abstract_lines if x.strip()))
for run in p.runs:
    if run.font.name is None:
        run.font.name = "Times New Roman"
p.paragraph_format.space_after = Pt(6)
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

p = doc.add_paragraph()
r = p.add_run("Keywords: "); r.bold = True; r.italic = True; r.font.name = "Times New Roman"
add_runs(p, keywords_text)
for run in p.runs:
    if run.font.name is None:
        run.font.name = "Times New Roman"
p.paragraph_format.space_after = Pt(18)

doc.add_page_break()


def emit_block_equation(ltx):
    """Centered display equation: real LaTeX-rendered PNG, falling back to
    OMML (never available outside Windows+Word) then plain simplified text."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _done = False
    png = render_latex_to_png(ltx, fontsize_pt=12)
    if png:
        dims = latex_png_size_inches(png)
        if dims:
            p.add_run().add_picture(png, width=Inches(dims[0]), height=Inches(dims[1]))
            _done = True
    if not _done and HAS_OMATH:
        try:
            p._p.append(latex_to_omath(ltx)); _done = True
        except Exception:
            _done = False
    if not _done:
        r = p.add_run(simplify_math(ltx)); r.italic = True; r.font.name = "Times New Roman"
    return p


# --------------------------------------------------------------------------
# body
# --------------------------------------------------------------------------
lines = rest_lines
n = len(lines)
i = 0
in_references = False

while i < n:
    line = lines[i]
    s = line.strip()

    if not s:
        i += 1
        continue
    if re.fullmatch(r"-{3,}", s):
        i += 1
        continue

    # IMG2 markers: previously laid images side by side in a table; now each
    # figure just falls through to the standard single-image handler below,
    # so every image gets its own line/row instead of sharing a row.
    if s in ("<!-- IMG2 -->", "<!-- /IMG2 -->"):
        i += 1
        continue

    # block math, "$$...$$" single line
    if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
        _eq_seq += 1
        emit_block_equation(s.strip("$").strip())
        i += 1
        continue

    # block math, multi-line "\[ ... \]" (used throughout the VN theory
    # section instead of "$$...$$"; same LaTeX rendering pipeline applies)
    if s == "\\[":
        i += 1
        eq_lines = []
        while i < n and lines[i].strip() != "\\]":
            eq_lines.append(lines[i].strip()); i += 1
        i += 1  # skip closing "\]"
        _eq_seq += 1
        emit_block_equation(" ".join(x for x in eq_lines if x))
        continue

    # image (figure)
    mimg = IMG.match(line)
    if mimg:
        caption_raw, path, attr = mimg.group(1), mimg.group(2), mimg.group(3)
        abspath = resolve_img_path(path)
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if os.path.exists(abspath):
            p.add_run().add_picture(abspath, width=parse_img_width(attr))
        else:
            r = p.add_run(f"[MISSING IMAGE: {path}]"); r.font.color.rgb = RGBColor(0xC0, 0, 0)
        mfig = FIG_CAP_PREFIX.match(caption_raw)
        desc = mfig.group(1) if mfig else caption_raw
        _fig_seq += 1
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = cap.add_run("Fig. "); r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(10)
        add_field_run(cap, 'SEQ Figure \\* MERGEFORMAT', str(_fig_seq), bold=True)
        r = cap.add_run("."); r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(10)
        cap.add_run(" ")
        add_runs(cap, desc, base_size=Pt(10))
        for run in cap.runs:
            if run.font.name is None:
                run.font.name = "Times New Roman"
        i += 1
        continue

    # table caption
    mtc = TABLE_CAP.match(s)
    if mtc:
        desc, trailing = mtc.group(1), mtc.group(2)
        _table_seq += 1
        cap = doc.add_paragraph()
        r = cap.add_run("Table "); r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(10.5)
        add_field_run(cap, 'SEQ Table \\* MERGEFORMAT', str(_table_seq), bold=True)
        r = cap.add_run("."); r.bold = True; r.font.name = "Times New Roman"; r.font.size = Pt(10.5)
        cap.add_run(" ")
        add_runs(cap, desc, base_size=Pt(10.5))
        if trailing.strip():
            add_runs(cap, " " + trailing.strip(), base_size=Pt(9.5))
        for run in cap.runs:
            if run.font.name is None:
                run.font.name = "Times New Roman"
        i += 1
        continue

    # heading
    mh = re.match(r"^(#{1,6})\s+(.*)$", line)
    if mh:
        level, txt = len(mh.group(1)), mh.group(2).strip()
        if txt.strip() == "References":
            in_references = True
        style_name = {2: "Heading 1", 3: "Heading 2"}.get(level, "Heading 3")
        h = doc.add_paragraph(style=style_name)
        add_runs(h, txt)
        for run in h.runs:
            run.font.name = "Times New Roman"  # belt-and-suspenders vs. theme-font fallback
        i += 1
        continue

    # pipe table
    if s.startswith("|"):
        tbl_lines = []
        while i < n and lines[i].strip().startswith("|"):
            tbl_lines.append(lines[i].strip()); i += 1
        sep_row = None
        rows = []
        for row in tbl_lines:
            if re.fullmatch(r"\|[\s:|-]+\|", row):
                sep_row = row; continue
            rows.append(row)
        aligns = []
        if sep_row:
            for c in sep_row.strip("|").split("|"):
                c = c.strip()
                if c.startswith(":") and c.endswith(":"):
                    aligns.append(WD_ALIGN_PARAGRAPH.CENTER)
                elif c.endswith(":"):
                    aligns.append(WD_ALIGN_PARAGRAPH.RIGHT)
                else:
                    aligns.append(WD_ALIGN_PARAGRAPH.LEFT)
        cells = [[c.strip() for c in row.strip("|").split("|")] for row in rows]
        ncol = max(len(r) for r in cells)
        while len(aligns) < ncol:
            aligns.append(WD_ALIGN_PARAGRAPH.LEFT)

        table_font_pt = 9 if ncol <= 5 else (8 if ncol == 6 else 7.5)
        CELL_MARGIN_PT = 14.0
        min_dxa = []
        for ci in range(ncol):
            vals = [row[ci] if ci < len(row) else "" for row in cells]
            max_w_pt = max((cell_longest_word_width_pt(v, table_font_pt) for v in vals), default=20.0)
            need = int((max_w_pt + CELL_MARGIN_PT) * 20)
            min_dxa.append(max(need, MIN_COL_DXA))
        total_min = sum(min_dxa)
        if total_min <= CONTENT_WIDTH_DXA:
            leftover = CONTENT_WIDTH_DXA - total_min
            col_widths_dxa = [d + int(leftover * d / total_min) for d in min_dxa]
        else:
            col_widths_dxa = [max(int(d * CONTENT_WIDTH_DXA / total_min), 300) for d in min_dxa]
        diff = CONTENT_WIDTH_DXA - sum(col_widths_dxa)
        col_widths_dxa[-1] += diff

        t = doc.add_table(rows=len(cells), cols=ncol)
        t.autofit = False
        tbl_pr = t._tbl.tblPr
        jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center"); tbl_pr.append(jc)
        layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed"); tbl_pr.append(layout)
        tblw = OxmlElement("w:tblW"); tblw.set(qn("w:type"), "dxa"); tblw.set(qn("w:w"), str(CONTENT_WIDTH_DXA)); tbl_pr.append(tblw)
        grid = t._tbl.find(qn("w:tblGrid"))
        for gc, w in zip(grid.findall(qn("w:gridCol")), col_widths_dxa):
            gc.set(qn("w:w"), str(w))

        for ri, row in enumerate(cells):
            if ri == 0 and len(cells) > 15:
                trPr = t.rows[0]._tr.get_or_add_trPr()
                th = OxmlElement("w:tblHeader"); trPr.append(th)
            for ci in range(ncol):
                cell = t.cell(ri, ci)
                cell.paragraphs[0].text = ""
                val = row[ci] if ci < len(row) else ""
                val_wrap = re.sub(r"([_-])(?=[^\s])", "\\1" + "\u200b", val)
                cp = cell.paragraphs[0]
                cp.alignment = aligns[ci]
                cp.paragraph_format.space_after = Pt(2)
                add_runs(cp, val_wrap, base_size=Pt(table_font_pt))
                for run in cp.runs:
                    run.font.name = "Times New Roman"
                    if run.font.size is None:
                        run.font.size = Pt(table_font_pt)
                    if ri == 0:
                        run.bold = True
                cell.width = Pt(col_widths_dxa[ci] / 20)
                if ri == 0:
                    set_cell_borders(cell, top=12, bottom=6)
                if ri == len(cells) - 1:
                    set_cell_borders(cell, bottom=6)
        doc.add_paragraph()
        continue

    # blockquote
    if s.startswith(">"):
        buf = []
        while i < n and lines[i].strip().startswith(">"):
            buf.append(lines[i].strip()[1:].strip()); i += 1
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.3)
        add_runs(p, " ".join(buf), base_italic=True)
        for r in p.runs:
            r.font.name = "Times New Roman"
        continue

    # bullet list
    mb = re.match(r"^\s*[-*]\s+(.*)$", line)
    if mb:
        p = doc.add_paragraph(style="List Bullet")
        add_runs(p, mb.group(1))
        for r in p.runs:
            if r.font.name is None:
                r.font.name = "Times New Roman"
        i += 1
        continue

    # numbered list -- literal digit from source, style auto-numbering off
    # (see suppress_style_numbering docstring: avoids cross-list continuation)
    mn = re.match(r"^\s*(\d+)\.\s+(.*)$", line)
    if mn:
        p = doc.add_paragraph(style="List Number")
        suppress_style_numbering(p)
        p.add_run(f"{mn.group(1)}. ")
        add_runs(p, mn.group(2))
        for r in p.runs:
            if r.font.name is None:
                r.font.name = "Times New Roman"
        i += 1
        continue

    # reference entry (author-year, no leading "[N]")
    if in_references:
        buf = [line]
        i += 1
        while i < n and lines[i].strip() and not lines[i].strip().startswith(("#",)):
            buf.append(lines[i]); i += 1
        add_reference_paragraph(clean_plain(" ".join(x.strip() for x in buf)))
        continue

    # normal paragraph
    buf = [line]
    i += 1
    while i < n and lines[i].strip() and not lines[i].strip().startswith(("|", ">", "#", "!", "-", "*")) \
            and not re.match(r"^\s*\d+\.\s", lines[i]):
        buf.append(lines[i]); i += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.25)
    add_runs(p, " ".join(x.strip() for x in buf))
    for r in p.runs:
        if r.font.name is None:
            r.font.name = "Times New Roman"

doc.save(OUT)
print(f"Saved: {OUT}")
print(f"Paragraphs: {len(doc.paragraphs)} | Tables: {len(doc.tables)} | "
      f"Images: {sum(1 for _ in doc.inline_shapes)}")
print(f"HAS_OMATH: {HAS_OMATH}")
print("\n>>> Open in Word: Ctrl+A then F9 to recompute Table/Figure/Equation numbers.")
