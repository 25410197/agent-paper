# -*- coding: utf-8 -*-
"""
Render real LaTeX to a tightly-cropped PNG via a full TeX install (pdflatex +
the `standalone` document class), so equations look like actual typeset LaTeX
in the docx instead of the plain-text `simplify_math` fallback.

Requires: pdflatex, amsmath/amssymb (both part of any standard texlive), and
pdftoppm (poppler-utils). Falls back gracefully (returns None) if any step
fails, so callers should keep the plain-text fallback path available.

Physical sizing: pdftoppm -r <DPI> renders the (points-based) PDF page at that
DPI, so inserting the resulting PNG into Word at width_px/DPI inches yields
the exact physical size implied by the LaTeX font size used -- no manual
tuning needed once DPI matches what image width/height are divided by.
"""
import hashlib
import os
import shutil
import subprocess
import tempfile

DPI = 400
CACHE_DIR = None  # set via configure_cache()


def configure_cache(cache_dir):
    global CACHE_DIR
    CACHE_DIR = cache_dir
    os.makedirs(cache_dir, exist_ok=True)


def _cache_key(latex, fontsize_pt):
    h = hashlib.md5(f"{fontsize_pt}::{latex}".encode("utf-8")).hexdigest()
    return h


def render_latex_to_png(latex, fontsize_pt=11):
    """Return an absolute path to a PNG rendering of `latex` (no leading/
    trailing $), or None if LaTeX/pdftoppm are unavailable or compilation
    fails. Cached by (latex, fontsize_pt) under CACHE_DIR if configured."""
    key = _cache_key(latex, fontsize_pt)
    if CACHE_DIR is not None:
        cached = os.path.join(CACHE_DIR, key + ".png")
        if os.path.exists(cached):
            return cached

    if shutil.which("pdflatex") is None or shutil.which("pdftoppm") is None:
        return None

    leading = round(fontsize_pt * 1.2, 1)
    tex_src = (
        "\\documentclass[10pt,preview,border=1pt]{standalone}\n"
        "\\usepackage{amsmath,amssymb,amsfonts}\n"
        "\\begin{document}\n"
        f"{{\\fontsize{{{fontsize_pt}}}{{{leading}}}\\selectfont "
        f"${latex}$}}\n"
        "\\end{document}\n"
    )

    with tempfile.TemporaryDirectory() as td:
        tex_path = os.path.join(td, "eq.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex_src)
        try:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "eq.tex"],
                cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=25, check=True,
            )
            pdf_path = os.path.join(td, "eq.pdf")
            if not os.path.exists(pdf_path):
                return None
            out_prefix = os.path.join(td, "eq_out")
            subprocess.run(
                ["pdftoppm", "-png", "-r", str(DPI), pdf_path, out_prefix],
                cwd=td, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=25, check=True,
            )
            produced = out_prefix + "-1.png"
            if not os.path.exists(produced):
                # single-page output without page suffix on some poppler builds
                alt = out_prefix + ".png"
                if os.path.exists(alt):
                    produced = alt
                else:
                    return None
            if CACHE_DIR is not None:
                dest = os.path.join(CACHE_DIR, key + ".png")
                shutil.copy(produced, dest)
                return dest
            # no cache configured: copy to a durable temp location
            dest = os.path.join(tempfile.gettempdir(), key + ".png")
            shutil.copy(produced, dest)
            return dest
        except Exception:
            return None
