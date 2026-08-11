# -*- coding: utf-8 -*-
"""
pdf_utils.py
------------
Extract text from an uploaded file's raw bytes (no temp files needed).
Supports PDF (pdfplumber -> pypdf fallback) and DOCX. Used by the web-app
endpoints to turn uploaded CV/JMP files into text for the pipeline.
"""

import io
import warnings


import os

# Only the CV (a few pages) and the JMP intro (early pages, up to Section 2) are
# ever used, so cap extraction. Override via ROOKIE_MAX_PDF_PAGES.
MAX_PDF_PAGES = int(os.environ.get("ROOKIE_MAX_PDF_PAGES", "40"))


def _pypdf_text(data: bytes, max_pages: int) -> str:
    """Lightweight extraction (pypdf). Low memory even on big/complex PDFs."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _pdfplumber_text(data: bytes, max_pages: int) -> str:
    """Higher-fidelity extraction (pdfplumber) but MUCH heavier memory — used
    only as a fallback. Cap pages + flush per-page cache to bound memory."""
    import pdfplumber
    parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            parts.append(page.extract_text() or "")
            try:
                page.flush_cache()
            except Exception:
                pass
    return "\n".join(parts)


def extract_pdf_bytes(data: bytes, max_pages: int = MAX_PDF_PAGES) -> str:
    """Extract text from the first `max_pages` pages. Try pypdf FIRST (light on
    memory so a big dissertation can't OOM a 512MB instance); fall back to
    pdfplumber only if pypdf yields little text."""
    text = ""
    try:
        text = _pypdf_text(data, max_pages)
    except Exception as e:
        warnings.warn(f"pypdf failed ({e}); trying pdfplumber.")
    if len(text.strip()) < 200:  # pypdf gave little/nothing -> try the heavier one
        try:
            alt = _pdfplumber_text(data, max_pages)
            if len(alt.strip()) > len(text.strip()):
                text = alt
        except Exception as e:
            warnings.warn(f"pdfplumber fallback failed ({e}).")
    return text


def extract_docx_bytes(data: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        warnings.warn(f"docx read failed ({e}).")
        return ""


def extract_upload(filename: str, data: bytes) -> str:
    """Dispatch on file extension."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return extract_pdf_bytes(data)
    if name.endswith(".docx"):
        return extract_docx_bytes(data)
    # last resort: try PDF then decode as text
    txt = extract_pdf_bytes(data)
    if txt.strip():
        return txt
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""
