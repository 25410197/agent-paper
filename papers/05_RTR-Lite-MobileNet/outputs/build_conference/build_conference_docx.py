# -*- coding: utf-8 -*-
"""
Chuyen BaoCao_RTR_LiteFormer_v2_EN.md -> .docx dung dung style cua template
Springer LNCS (template/Springer_Template.docm).

CACH DUNG (workflow toi uu khi con sua nhieu lan):
  1. Sua noi dung trong file .md CON tuong ung trong en/ (00_title.md,
     01_abstract.md, ... - nguon chan ly duy nhat cho tung muc).
  2. Chay: python merge_sections.py   (sinh lai en/_BaoCao_Full_EN.md)
  3. Chay lai: python build_conference_docx.py
  4. Mo file .docx moi sinh, bam Ctrl+A roi F9 de Word tinh lai so
     Table/Figure/Equation (cac field SEQ) cho dung.
  5. Neu can sua sau khi da xuat ra .docx (them hinh, doi bang...), sua truc
     tiep trong Word cung duoc - nhung lan sau chay lai script se GHI DE file
     .docx, nen tot nhat la dua moi thay doi noi dung ve lai cac file .md con.

Quy uoc mapping (theo dung style co san trong template):
  - "# Title"                -> papertitle
  - "**Keywords:** ..."      -> keywords (dua xuong duoi Abstract, dung thu tu Springer)
  - "## Abstract" + doan sau -> abstract (khong hien tieu de "Abstract" rieng,
                                 giong mau: chu "Abstract." in dam ngay dau doan)
  - "## N. ten muc"          -> heading1  (giu nguyen so thu cong trong text,
                                 TAT auto-numbering cua style vi than bai co rat
                                 nhieu cho trich dan "muc 6.4", "Bang 4"... theo
                                 dung so hien co - de Word tu danh so lai se lech)
  - "### N.N ten muc"        -> heading2  (tuong tu, tat auto-numbering)
  - "**Table N. ...**"       -> tablecaption, so thay bang field SEQ Table
                                 (field nay AN TOAN de tu dong vi thu tu bang
                                 trong tai lieu dung voi so da ghi, va se tu
                                 cap nhat neu ban them/bot bang sau nay)
  - "![Figure N. ...](path)" -> anh (style "image") + figurecaption (field SEQ Figure)
  - "$$...$$"                -> equation (co field SEQ Equation o cuoi dong)
  - "- **...**"              -> bulletitem
  - "1.  **...**"            -> numitem
  - "> ..."                  -> Normal/p1a, in nghieng (khong co style rieng trong
                                 template cho callout/note nen dung in nghieng de
                                 phan biet)
  - "### A. ..." / "### B. ..." trong References -> heading2, so hieu [01]/[1]
                                 GIU NGUYEN dang literal text (khong dung auto-
                                 number cua style referenceitem) vi hai danh sach
                                 nay danh so doc lap va duoc trich dan cheo trong
                                 Bang 4 / phan than bai theo dung con so do.
"""
import os
import re

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

# Script nam trong build_conference/; tai nguyen dung chung (mathomml.py,
# template/, xai/) nam o thu muc CHA (outputs/). Them cha vao sys.path de
# import duoc mathomml, va dung ASSET_DIR de tim template + anh.
import sys as _sys

ASSET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ASSET_DIR not in _sys.path:
    _sys.path.insert(0, ASSET_DIR)

# LaTeX -> OMML (cong thuc that trong Word) qua module dung chung mathomml.py.
# Neu thieu latex2mathml / MML2OMML.XSL -> HAS_OMATH=False, tu dong fallback ve
# simplify_math (text tho) o moi cho chen cong thuc.
try:
    from mathomml import latex_to_omath
    from mathomml import available as _omath_available

    HAS_OMATH = _omath_available()
except Exception:
    HAS_OMATH = False

HERE = os.path.dirname(os.path.abspath(__file__))
# Nguon: cac file .md con trong en/ (00_title.md, 01_abstract.md, ...).
# Sua noi dung o do, chay `python merge_sections.py` de sinh lai file tong
# nay, roi moi chay script nay.
MD = os.path.join(HERE, "en", "_BaoCao_Full_EN.md")
TEMPLATE_DOCM = os.path.join(ASSET_DIR, "template", "Springer_Template.docm")
TEMPLATE_ASDOCX = os.path.join(ASSET_DIR, "template", "_Springer_Template_readable.docx")
OUT = os.path.join(HERE, "ChiliEdgeNet.docx")
CONTENT_WIDTH_IN = 4.9  # be rong vung noi dung cua template (khong ke le)
# be rong vung noi dung thuc te theo sectPr cua template: pgSz 11906 dxa,
# pgMar left/right 2494 dxa moi ben -> 11906 - 2*2494 = 6918 dxa (~4.8in)
CONTENT_WIDTH_DXA = 6918
MIN_COL_DXA = 500  # san ~0.35in, tranh cot bi bop ve 0

# DOUBLE-BLIND: an danh tac gia cho ban nop phan bien kin (ICEBA 2026).
AUTHOR_PLACEHOLDER = "Anonymous Author(s)"
ADDRESS_PLACEHOLDER = "Affiliation withheld for double-blind review"
EMAIL_PLACEHOLDER = ""


# --------------------------------------------------------------------------
# 0. File template doc duoc: _Springer_Template_readable.docx duoc tao 1 lan
#    bang chinh MS Word (Save As .docx tu file .docm goc) de dam bao sach
#    macro/customUI va giu nguyen toan bo style/numbering/header/footer.
#    Neu chua co, chay lenh PowerShell sau 1 lan (Word phai da cai san):
#
#    $word = New-Object -ComObject Word.Application
#    $doc = $word.Documents.Open("<duong dan>\Springer_Template.docm", $false, $true)
#    $doc.SaveAs([ref]"<duong dan>\_Springer_Template_readable.docx", [ref]12)
#    $doc.Close([ref]0); $word.Quit()
# --------------------------------------------------------------------------
if not os.path.exists(TEMPLATE_ASDOCX):
    raise SystemExit(
        f"Khong tim thay {TEMPLATE_ASDOCX}. Xem huong dan tao file o comment phia tren."
    )

doc = docx.Document(TEMPLATE_ASDOCX)

# xoa toan bo noi dung mau cua template (giu lai sectPr/header/footer)
body = doc.element.body
for child in list(body):
    if child.tag != qn("w:sectPr"):
        body.remove(child)


# --------------------------------------------------------------------------
# helpers OOXML cap thap: field code (SEQ), tat numbering cua style
# --------------------------------------------------------------------------
def _new_run_props(bold=False, italic=False):
    from docx.oxml import OxmlElement

    rpr = OxmlElement("w:rPr")
    if bold:
        b = OxmlElement("w:b")
        rpr.append(b)
    if italic:
        i = OxmlElement("w:i")
        rpr.append(i)
    return rpr


def add_field_run(paragraph, field_instr, cached_text, bold=False):
    """Chen field code kieu SEQ (vd 'SEQ Table \\* MERGEFORMAT') vao paragraph,
    hien thi tam thoi bang cached_text. Word se tinh lai khi bam Ctrl+A, F9."""
    from docx.oxml import OxmlElement

    def _r(tag_kwargs=None):
        r = OxmlElement("w:r")
        rpr = _new_run_props(bold=bold)
        if rpr.findall(qn("w:b")) or rpr.findall(qn("w:i")):
            r.append(rpr)
        return r

    r1 = _r(); fld1 = OxmlElement("w:fldChar"); fld1.set(qn("w:fldCharType"), "begin")
    r1.append(fld1)

    r2 = _r(); instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = f" {field_instr} "
    r2.append(instr)

    r3 = _r(); fld3 = OxmlElement("w:fldChar"); fld3.set(qn("w:fldCharType"), "separate")
    r3.append(fld3)

    r4 = _r()
    t4 = OxmlElement("w:t"); t4.text = cached_text
    r4.append(t4)

    r5 = _r(); fld5 = OxmlElement("w:fldChar"); fld5.set(qn("w:fldCharType"), "end")
    r5.append(fld5)

    for r in (r1, r2, r3, r4, r5):
        paragraph._p.append(r)


def suppress_style_numbering(paragraph):
    """Tat numbering tu dong ma style (heading1/heading2/referenceitem...) da
    gan san, de dung so thu cong da co san trong van ban (tranh vong so voi
    hang tram cho trich dan cheo 'muc 6.4', 'Bang 4' da viet co dinh)."""
    from docx.oxml import OxmlElement

    pPr = paragraph._p.get_or_add_pPr()
    numPr = OxmlElement("w:numPr")
    ilvl = OxmlElement("w:ilvl"); ilvl.set(qn("w:val"), "0")
    numId = OxmlElement("w:numId"); numId.set(qn("w:val"), "0")
    numPr.append(ilvl); numPr.append(numId)
    pPr.append(numPr)


def set_cell_borders(cell, top=None, bottom=None):
    from docx.oxml import OxmlElement

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


# --------------------------------------------------------------------------
# 1. Bo doi Latex don gian (tai su dung logic tu build_docx.py)
# --------------------------------------------------------------------------
def simplify_math(s):
    s = s.strip()
    if s.startswith("$"):
        s = s.strip("$")
    s = re.sub(r"\\text\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbb\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\widetilde\{([^{}]*)\}", r"~\1", s)
    for _ in range(3):
        s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    syms = {r"\left[": "[", r"\right]": "]", r"\left(": "(", r"\right)": ")",
            r"\lbrack": "[", r"\rbrack": "]", r"\approx": "≈", r"\cdot": "·",
            r"\times": "×", r"\sum": "Σ", r"\arg\min": "argmin", r"\alpha": "α",
            r"\phi": "φ", r"\pi": "π", r"\xi": "ξ", r"\Omega": "Ω",
            r"\mathcal{L}": "L", r"\partial": "∂", r"\quad": "  ", r"\,": " "}
    for k, v in syms.items():
        s = s.replace(k, v)
    s = s.replace("_{", "_").replace("^{", "^").replace("}", "")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "")
    return re.sub(r"\s+", " ", s).strip()


def clean_plain(t):
    # Xuat gach thuong "-" thay vi em/en dash de tranh cac ky tu dac biet
    # (— / –) thuong bi coi la dau hieu van ban AI. Nguon .md giu nguyen "---".
    t = t.replace("---", "-").replace("—", "-").replace("–", "-").replace("--", "-")
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
            r = par.add_run(tok[1:-1]); r.font.name = "Consolas"; r.font.size = Pt(9)
        elif tok.startswith("$"):
            _ltx = tok.strip("$").strip()
            _done = False
            if HAS_OMATH:
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


def _add_hyperlink(par, url, text):
    """Them 1 run hyperlink (mau xanh + gach chan) vao cuoi paragraph."""
    from docx.oxml import OxmlElement
    r_id = par.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hl = OxmlElement("w:hyperlink")
    hl.set(qn("r:id"), r_id)
    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    col = OxmlElement("w:color"); col.set(qn("w:val"), "0563C1"); rpr.append(col)
    u = OxmlElement("w:u"); u.set(qn("w:val"), "single"); rpr.append(u)
    run.append(rpr)
    t = OxmlElement("w:t"); t.text = text; run.append(t)
    hl.append(run)
    par._p.append(hl)


def add_ref_with_doi(par, text):
    """Render 1 dong reference; neu co 'DOI: 10.x' thi bien DOI thanh hyperlink.
    Neu khong co DOI nhung co 'Available: <url>' (nguon khong co DOI, vd NeurIPS
    proceedings) thi cung bien URL do thanh hyperlink."""
    m = re.search(r"^(.*DOI:\s*)(\S+)\s*$", text)
    if m:
        add_runs(par, m.group(1))
        doi = m.group(2).rstrip(".")
        _add_hyperlink(par, f"https://doi.org/{doi}", doi)
        par.add_run(".")
        return
    m2 = re.search(r"^(.*Available:\s*)(\S+)\s*$", text)
    if m2:
        add_runs(par, m2.group(1))
        url = m2.group(2).rstrip(".")
        _add_hyperlink(par, url, url)
        par.add_run(".")
        return
    add_runs(par, text)


def parse_img_width(attr):
    if attr:
        m = re.search(r"(\d+)\s*%", attr)
        if m:
            return Inches(CONTENT_WIDTH_IN * int(m.group(1)) / 100.0)
    return Inches(CONTENT_WIDTH_IN)


# --------------------------------------------------------------------------
# Do do rong chu THAT bang font Times New Roman (cung font Word dung), thay
# vi uoc luong tho theo so ky tu - chinh xac hon nhieu, tranh vo dong sai.
# --------------------------------------------------------------------------
_FONT_CACHE = {}
_FONT_SCALE = 200  # kich thuoc lon de do chinh xac roi quy doi ve pt

try:
    from PIL import ImageFont

    def _get_font(bold):
        key = "bold" if bold else "regular"
        if key not in _FONT_CACHE:
            path = "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"
            _FONT_CACHE[key] = ImageFont.truetype(path, _FONT_SCALE)
        return _FONT_CACHE[key]

    def text_width_pt(text, size_pt, bold=False):
        if not text:
            return 0.0
        font = _get_font(bold)
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0]) / _FONT_SCALE * size_pt
except Exception:
    def text_width_pt(text, size_pt, bold=False):
        # fallback tho neu khong co PIL/font (uoc luong ~0.5pt/ky tu)
        mult = 0.58 if bold else 0.5
        return len(text) * size_pt * mult


def cell_longest_word_width_pt(raw_md_text, size_pt):
    """Tim tu (token, tach theo khoang trang) dai nhat trong 1 o bang (co the
    chua **bold**), tra ve do rong pt thuc te cua tu do voi font tuong ung."""
    is_bold_cell = "**" in raw_md_text
    plain = re.sub(r"[*`$]", "", raw_md_text)
    best = 0.0
    for tok in re.split(r"\s+", plain.strip()):
        if not tok:
            continue
        w = text_width_pt(tok, size_pt, bold=is_bold_cell)
        best = max(best, w)
    return best


IMG = re.compile(r"^!\[(.*?)\]\((.*?)\)(\{.*\})?\s*$")
TABLE_CAP = re.compile(r"^\*\*Table\s+\d+\.\s*(.*?)\*\*\s*(.*)$")
FIG_CAP_PREFIX = re.compile(r"^Figure\s+\d+\.\s*(.*)$")

_table_seq = 0
_fig_seq = 0
_eq_seq = 0


# --------------------------------------------------------------------------
# 2. Doc file md, tach rieng Title / Keywords / Abstract-block / phan con lai
#    (Springer yeu cau thu tu: Title -> Author -> Address -> Abstract ->
#    Keywords -> Muc 1..., khac voi thu tu Keywords-truoc-Abstract trong .md)
# --------------------------------------------------------------------------
raw = open(MD, encoding="utf-8").read().split("\n")

title_line = raw[0].lstrip("#").strip()

kw_idx = next(i for i, l in enumerate(raw) if l.strip().startswith("**Keywords:**"))
keywords_text = raw[kw_idx].strip()[len("**Keywords:**"):].strip()

abs_start = next(i for i, l in enumerate(raw) if l.strip() == "## Abstract")
next_h2 = next(i for i in range(abs_start + 1, len(raw)) if raw[i].strip().startswith("## "))
abstract_lines = raw[abs_start + 1:next_h2]

rest_lines = raw[next_h2:]  # bat dau tu "## 1. Introduction" tro di


# --------------------------------------------------------------------------
# 3. Xuat Title / Author / Address / Abstract / Keywords
# --------------------------------------------------------------------------
p = doc.add_paragraph(style="papertitle")
add_runs(p, title_line)

p = doc.add_paragraph(style="author")
p.add_run(AUTHOR_PLACEHOLDER)

p = doc.add_paragraph(style="address")
p.add_run(ADDRESS_PLACEHOLDER + "\n" + EMAIL_PLACEHOLDER)

# Abstract: doan dau tien -> style abstract, chu "Abstract." in dam dau dong;
# cac bullet highlight -> bulletitem (font nho lai cho dong bo voi khoi abstract)
abs_body = [l for l in abstract_lines if l.strip()]
first_para_lines = []
i = 0
while i < len(abs_body) and not abs_body[i].strip().startswith(("-", "*")):
    first_para_lines.append(abs_body[i]); i += 1

p = doc.add_paragraph(style="abstract")
r = p.add_run("Abstract. "); r.bold = True
add_runs(p, " ".join(x.strip() for x in first_para_lines))

while i < len(abs_body):
    line = abs_body[i].strip()
    mb = re.match(r"^[-*]\s+(.*)$", line)
    if mb:
        p = doc.add_paragraph(style="bulletitem")
        add_runs(p, mb.group(1), base_size=Pt(9))
        for r in p.runs:
            if r.font.size is None:
                r.font.size = Pt(9)
    i += 1

p = doc.add_paragraph(style="keywords")
r = p.add_run("Keywords: "); r.bold = True
add_runs(p, keywords_text)


# --------------------------------------------------------------------------
# 4. Duyet phan con lai (Muc 1 ... het References + Data Availability)
# --------------------------------------------------------------------------
lines = rest_lines
n = len(lines)
i = 0
prev_is_block_boundary = True  # doan dau tien sau "1. Introduction" la p1a

while i < n:
    line = lines[i]
    s = line.strip()

    if not s:
        i += 1
        continue

    if re.fullmatch(r"-{3,}", s):
        i += 1
        continue

    # ghi chu editorial trong ngoac don o dau muc Related Work -> bo qua
    # (chi la huong dan noi bo khi anh xa template, khong dua vao ban Word)
    if s.startswith("*(") and s.endswith(")*") and "template" in s.lower():
        i += 1
        continue

    # side-by-side block: <!-- SBS --> text paragraph ... image line <!-- /SBS -->
    # -> bang 2 cot AN VIEN (text trai, anh+caption phai), tiet kiem chieu cao
    # khi hinh du nho de dat canh doan van thay vi chiem het 1 dong rieng.
    if s == "<!-- SBS -->":
        i += 1
        block = []
        while i < n and lines[i].strip() != "<!-- /SBS -->":
            block.append(lines[i])
            i += 1
        i += 1  # bo qua dong dong marker

        img_idx = next(k for k, l in enumerate(block) if IMG.match(l.strip()))
        text_lines = [l for l in block[:img_idx] if l.strip()]
        mimg2 = IMG.match(block[img_idx].strip())
        caption_raw2, path2, attr2 = mimg2.group(1), mimg2.group(2), mimg2.group(3)
        abspath2 = os.path.join(ASSET_DIR, path2.replace("/", os.sep))

        from docx.enum.table import WD_ALIGN_VERTICAL
        from docx.oxml import OxmlElement

        LEFT_DXA = int(CONTENT_WIDTH_DXA * 0.56)
        RIGHT_DXA = CONTENT_WIDTH_DXA - LEFT_DXA

        t = doc.add_table(rows=1, cols=2)
        t.autofit = False
        tbl_pr = t._tbl.tblPr
        layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed")
        tbl_pr.append(layout)
        tblw = tbl_pr.find(qn("w:tblW"))
        if tblw is None:
            tblw = OxmlElement("w:tblW"); tbl_pr.append(tblw)
        tblw.set(qn("w:type"), "dxa"); tblw.set(qn("w:w"), str(CONTENT_WIDTH_DXA))
        grid = t._tbl.find(qn("w:tblGrid"))
        for gc, w in zip(grid.findall(qn("w:gridCol")), [LEFT_DXA, RIGHT_DXA]):
            gc.set(qn("w:w"), str(w))

        left_cell, right_cell = t.cell(0, 0), t.cell(0, 1)
        left_cell.width = Pt(LEFT_DXA / 20)
        right_cell.width = Pt(RIGHT_DXA / 20)
        left_cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        right_cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

        left_cell.paragraphs[0].text = ""
        lp = left_cell.paragraphs[0]
        lp.paragraph_format.left_indent = Pt(0)
        lp.paragraph_format.first_line_indent = Pt(0)
        add_runs(lp, " ".join(x.strip() for x in text_lines))

        right_cell.paragraphs[0].text = ""
        ip = right_cell.paragraphs[0]
        ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if os.path.exists(abspath2):
            img_w = Inches(RIGHT_DXA / 1440 * 0.90)
            ip.add_run().add_picture(abspath2, width=img_w)
        else:
            r = ip.add_run(f"[MISSING IMAGE: {path2}]"); r.font.color.rgb = RGBColor(0xC0, 0, 0)

        mfig2 = FIG_CAP_PREFIX.match(caption_raw2)
        desc2 = mfig2.group(1) if mfig2 else caption_raw2
        _fig_seq += 1
        cap2 = right_cell.add_paragraph()
        cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap2.paragraph_format.left_indent = Pt(0)
        cap2.paragraph_format.first_line_indent = Pt(0)
        r = cap2.add_run("Fig. "); r.bold = True; r.font.size = Pt(9)
        add_field_run(cap2, 'SEQ Figure \\* MERGEFORMAT', str(_fig_seq), bold=True)
        r = cap2.add_run("."); r.bold = True; r.font.size = Pt(9)
        cap2.add_run(" ")
        add_runs(cap2, desc2, base_size=Pt(9))
        for run in cap2.runs:
            if run.font.size is None:
                run.font.size = Pt(9)

        doc.add_paragraph()
        prev_is_block_boundary = True
        continue

    # two-images-side-by-side block: <!-- IMG2 --> img1 line, img2 line <!-- /IMG2 -->
    # -> bang 2 cot AN VIEN, moi cot 1 anh + caption rieng, tiet kiem chieu cao
    # so voi xep 2 anh doc lap chong doc.
    if s == "<!-- IMG2 -->":
        i += 1
        block = []
        while i < n and lines[i].strip() != "<!-- /IMG2 -->":
            block.append(lines[i])
            i += 1
        i += 1  # bo qua dong marker dong

        img_lines = [l.strip() for l in block if IMG.match(l.strip())]

        from docx.oxml import OxmlElement

        HALF_DXA = CONTENT_WIDTH_DXA // 2
        t = doc.add_table(rows=1, cols=2)
        t.autofit = False
        tbl_pr = t._tbl.tblPr
        layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed")
        tbl_pr.append(layout)
        tblw = tbl_pr.find(qn("w:tblW"))
        if tblw is None:
            tblw = OxmlElement("w:tblW"); tbl_pr.append(tblw)
        tblw.set(qn("w:type"), "dxa"); tblw.set(qn("w:w"), str(CONTENT_WIDTH_DXA))
        grid = t._tbl.find(qn("w:tblGrid"))
        for gc in grid.findall(qn("w:gridCol")):
            gc.set(qn("w:w"), str(HALF_DXA))

        for col, img_line in enumerate(img_lines):
            mimg2 = IMG.match(img_line)
            caption_raw2, path2, attr2 = mimg2.group(1), mimg2.group(2), mimg2.group(3)
            abspath2 = os.path.join(ASSET_DIR, path2.replace("/", os.sep))
            cell = t.cell(0, col)
            cell.width = Pt(HALF_DXA / 20)

            cell.paragraphs[0].text = ""
            ip = cell.paragraphs[0]
            ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
            ip.paragraph_format.left_indent = Pt(0)
            ip.paragraph_format.first_line_indent = Pt(0)
            if os.path.exists(abspath2):
                img_w = Inches(HALF_DXA / 1440 * 0.90)
                ip.add_run().add_picture(abspath2, width=img_w)
            else:
                r = ip.add_run(f"[MISSING IMAGE: {path2}]"); r.font.color.rgb = RGBColor(0xC0, 0, 0)

            mfig2 = FIG_CAP_PREFIX.match(caption_raw2)
            desc2 = mfig2.group(1) if mfig2 else caption_raw2
            _fig_seq += 1
            cap2 = cell.add_paragraph()
            cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap2.paragraph_format.left_indent = Pt(0)
            cap2.paragraph_format.first_line_indent = Pt(0)
            r = cap2.add_run("Fig. "); r.bold = True; r.font.size = Pt(9)
            add_field_run(cap2, 'SEQ Figure \\* MERGEFORMAT', str(_fig_seq), bold=True)
            r = cap2.add_run("."); r.bold = True; r.font.size = Pt(9)
            cap2.add_run(" ")
            add_runs(cap2, desc2, base_size=Pt(9))
            for run in cap2.runs:
                if run.font.size is None:
                    run.font.size = Pt(9)

        doc.add_paragraph()
        prev_is_block_boundary = True
        continue

    # block math
    if s.startswith("$$") and s.endswith("$$") and len(s) > 4:
        _eq_seq += 1
        p = doc.add_paragraph(style="equation")
        p.add_run("\t")  # tab -> center-tab cua style equation: cong thuc CAN GIUA (Springer)
        _ltx = s.strip("$").strip()
        if HAS_OMATH:
            try:
                p._p.append(latex_to_omath(_ltx))
            except Exception:
                p.add_run(simplify_math(_ltx))
        else:
            p.add_run(simplify_math(_ltx))
        p.add_run("\t(")
        add_field_run(p, 'SEQ Equation \\* MERGEFORMAT', str(_eq_seq))
        p.add_run(")")
        prev_is_block_boundary = True
        i += 1
        continue

    # image (figure)
    mimg = IMG.match(line)
    if mimg:
        caption_raw, path, attr = mimg.group(1), mimg.group(2), mimg.group(3)
        abspath = os.path.join(ASSET_DIR, path.replace("/", os.sep))
        p = doc.add_paragraph(style="image")
        if os.path.exists(abspath):
            p.add_run().add_picture(abspath, width=parse_img_width(attr))
        else:
            r = p.add_run(f"[MISSING IMAGE: {path}]"); r.font.color.rgb = RGBColor(0xC0, 0, 0)

        mfig = FIG_CAP_PREFIX.match(caption_raw)
        desc = mfig.group(1) if mfig else caption_raw
        _fig_seq += 1
        cap = doc.add_paragraph(style="figurecaption")
        r = cap.add_run("Fig. "); r.bold = True
        add_field_run(cap, 'SEQ Figure \\* MERGEFORMAT', str(_fig_seq), bold=True)
        r = cap.add_run("."); r.bold = True
        cap.add_run(" ")
        add_runs(cap, desc)
        prev_is_block_boundary = True
        i += 1
        continue

    # table caption line ("**Table N. description**" [+ optional trailing note])
    mtc = TABLE_CAP.match(s)
    if mtc:
        desc, trailing = mtc.group(1), mtc.group(2)
        _table_seq += 1
        cap = doc.add_paragraph(style="tablecaption")
        r = cap.add_run("Table "); r.bold = True
        add_field_run(cap, 'SEQ Table \\* MERGEFORMAT', str(_table_seq), bold=True)
        r = cap.add_run("."); r.bold = True
        cap.add_run(" ")
        add_runs(cap, desc)
        if trailing.strip():
            add_runs(cap, " " + trailing.strip())
        i += 1
        continue

    # heading
    mh = re.match(r"^(#{1,6})\s+(.*)$", line)
    if mh:
        level, txt = len(mh.group(1)), mh.group(2).strip()
        if level == 2:
            h = doc.add_paragraph(style="heading1")
        else:
            h = doc.add_paragraph(style="heading2")
        suppress_style_numbering(h)
        add_runs(h, txt)
        prev_is_block_boundary = True
        i += 1
        continue

    # pipe table
    if s.startswith("|"):
        tbl_lines = []
        while i < n and lines[i].strip().startswith("|"):
            tbl_lines.append(lines[i].strip())
            i += 1
        sep_row = None
        rows = []
        for row in tbl_lines:
            if re.fullmatch(r"\|[\s:|-]+\|", row):
                sep_row = row
                continue
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

        # co chu nho lai khi bang nhieu cot, de co du cho tranh vo chu
        table_font_pt = 9 if ncol <= 5 else (8 if ncol == 6 else 7.5)

        # do rong TOI THIEU tuyet doi (dxa) de tu dai nhat trong cot khong bi
        # vo dong: do THAT bang font Times New Roman (regular/bold tuong ung)
        # thay vi uoc luong theo so ky tu, cong them dem cho margin trong o.
        CELL_MARGIN_PT = 14.0  # ~2 x margin trai/phai mac dinh cua Word + du phong kerning
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
            # khong du cho ngay ca o muc toi thieu -> co giam deu theo ty le
            col_widths_dxa = [max(int(d * CONTENT_WIDTH_DXA / total_min), 300) for d in min_dxa]
        diff = CONTENT_WIDTH_DXA - sum(col_widths_dxa)
        col_widths_dxa[-1] += diff

        t = doc.add_table(rows=len(cells), cols=ncol)
        t.autofit = False
        from docx.oxml import OxmlElement

        tbl_pr = t._tbl.tblPr
        jc = OxmlElement("w:jc"); jc.set(qn("w:val"), "center")
        tbl_pr.append(jc)
        layout = OxmlElement("w:tblLayout"); layout.set(qn("w:type"), "fixed")
        tbl_pr.append(layout)
        tblw = tbl_pr.find(qn("w:tblW"))
        if tblw is None:
            tblw = OxmlElement("w:tblW"); tbl_pr.append(tblw)
        tblw.set(qn("w:type"), "dxa"); tblw.set(qn("w:w"), str(CONTENT_WIDTH_DXA))

        # tblGrid: khai bao do rong tung cot cho dung layout "fixed"
        grid = t._tbl.find(qn("w:tblGrid"))
        for gc, w in zip(grid.findall(qn("w:gridCol")), col_widths_dxa):
            gc.set(qn("w:w"), str(w))

        for ri, row in enumerate(cells):
            # lap lai dong header o dau moi trang neu bang dai (Bang 4: 28 dong)
            if ri == 0 and len(cells) > 15:
                trPr = row_trPr = t.rows[0]._tr.get_or_add_trPr()
                th = OxmlElement("w:tblHeader"); trPr.append(th)
            for ci in range(ncol):
                cell = t.cell(ri, ci)
                cell.paragraphs[0].text = ""
                val = row[ci] if ci < len(row) else ""
                # cho phep ngat dong hop ly o dau "_"/"-" thay vi vo giua chu
                val_wrap = re.sub(r"([_-])(?=[^\s])", "\\1" + "​", val)
                cp = cell.paragraphs[0]
                cp.alignment = aligns[ci]
                cp.paragraph_format.left_indent = Pt(0)
                cp.paragraph_format.first_line_indent = Pt(0)  # Normal style ke thua firstLine=227dxa, phai tat trong o bang
                cp.paragraph_format.space_after = Pt(2)
                add_runs(cp, val_wrap, base_size=Pt(table_font_pt))
                for run in cp.runs:
                    if run.font.size is None:
                        run.font.size = Pt(table_font_pt)
                    if ri == 0:
                        run.bold = True
                # dat do rong ro rang tren tung o (fixed layout can ca hai)
                cell.width = Pt(col_widths_dxa[ci] / 20)
                if ri == 0:
                    set_cell_borders(cell, top=12, bottom=6)
                if ri == len(cells) - 1:
                    set_cell_borders(cell, bottom=6)
        doc.add_paragraph()
        prev_is_block_boundary = True
        continue

    # blockquote (framing note) -> in nghieng, khong co style callout rieng
    if s.startswith(">"):
        buf = []
        while i < n and lines[i].strip().startswith(">"):
            buf.append(lines[i].strip()[1:].strip())
            i += 1
        style_name = "p1a" if prev_is_block_boundary else "Normal"
        p = doc.add_paragraph(style=style_name)
        add_runs(p, " ".join(buf), base_italic=True)
        prev_is_block_boundary = True
        continue

    # bullet list
    mb = re.match(r"^\s*[-*]\s+(.*)$", line)
    if mb:
        p = doc.add_paragraph(style="bulletitem")
        add_runs(p, mb.group(1))
        prev_is_block_boundary = True
        i += 1
        continue

    # numbered list
    mn = re.match(r"^\s*\d+\.\s+(.*)$", line)
    if mn:
        p = doc.add_paragraph(style="numitem")
        add_runs(p, mn.group(1))
        prev_is_block_boundary = True
        i += 1
        continue

    # reference item: "[01] ..." hoac "[1] ..." -> giu nguyen so literal,
    # KHONG dung auto-number cua style referenceitem
    mref = re.match(r"^\[(\d+)\]\s*(.*)$", s)
    if mref:
        p = doc.add_paragraph(style="referenceitem")
        suppress_style_numbering(p)
        r = p.add_run(f"[{mref.group(1)}] ")
        add_ref_with_doi(p, mref.group(2))
        i += 1
        continue

    # normal paragraph (gom cac dong lien tuc den khi gap dong trong / block khac)
    buf = [line]
    i += 1
    while i < n and lines[i].strip() and not lines[i].strip().startswith(("|", ">", "#", "!", "-", "*", "[")) \
            and not re.match(r"^\s*\d+\.\s", lines[i]):
        buf.append(lines[i]); i += 1
    style_name = "p1a" if prev_is_block_boundary else "Normal"
    p = doc.add_paragraph(style=style_name)
    add_runs(p, " ".join(x.strip() for x in buf))
    prev_is_block_boundary = False

doc.save(OUT)
print(f"Saved: {OUT}")
print(f"Paragraphs: {len(doc.paragraphs)} | Tables: {len(doc.tables)} | "
      f"Images: {sum(1 for _ in doc.inline_shapes)}")
print("\n>>> Sau khi mo file trong Word: bam Ctrl+A roi F9 de tinh lai so Table/Figure/Equation.")
print(f">>> Dien ten tac gia/don vi thay cho '{AUTHOR_PLACEHOLDER}' / '{ADDRESS_PLACEHOLDER}'.")
