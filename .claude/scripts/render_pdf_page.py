"""Render PDF page(s) to high-res PNG for visual inspection (figures/diagrams pdftotext can't capture).

Usage:
    python render_pdf_page.py <pdf_path> <out_dir> <page1> [page2 page3 ...] [--zoom N]

Pages are 1-indexed (matching pdftotext's page count / form-feed pages). Output files are named
<out_dir>/page<N>.png. Requires PyMuPDF (`pip install pymupdf`), no poppler binaries needed.
"""
import sys
import os
import fitz  # PyMuPDF


def render_pages(pdf_path, out_dir, pages, zoom=3.0):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(zoom, zoom)
    saved = []
    for p in pages:
        if p < 1 or p > doc.page_count:
            print(f"[skip] page {p} out of range (doc has {doc.page_count} pages)")
            continue
        page = doc[p - 1]
        pix = page.get_pixmap(matrix=mat)
        out_path = os.path.join(out_dir, f"page{p}.png")
        pix.save(out_path)
        saved.append(out_path)
        print(f"[saved] {out_path}  ({pix.width}x{pix.height})")
    doc.close()
    return saved


if __name__ == "__main__":
    args = sys.argv[1:]
    zoom = 3.0
    if "--zoom" in args:
        i = args.index("--zoom")
        zoom = float(args[i + 1])
        del args[i:i + 2]
    if len(args) < 3:
        print(__doc__)
        sys.exit(1)
    pdf_path, out_dir = args[0], args[1]
    pages = [int(p) for p in args[2:]]
    render_pages(pdf_path, out_dir, pages, zoom=zoom)
