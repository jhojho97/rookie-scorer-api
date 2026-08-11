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
# ever used, so cap extraction. pdfplumber holds per-page char data in memory —
# on a 150-page dissertation that alone can OOM a 512MB instance. Capping +
# flushing each page's cache bounds memory. Override via ROOKIE_MAX_PDF_PAGES.
MAX_PDF_PAGES = int(os.environ.get("ROOKIE_MAX_PDF_PAGES", "60"))


def extract_pdf_bytes(data: bytes, max_pages: int = MAX_PDF_PAGES) -> str:
    """Extract text from the first `max_pages` pages of a PDF (memory-bounded)."""
    text = ""
    try:
        import pdfplumber
        parts = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for i, page in enumerate(pdf.pages):
                if i >= max_pages:
                    break
                parts.append(page.extract_text() or "")
                # Release per-page cache so memory doesn't grow with page count.
                try:
                    page.flush_cache()
                except Exception:
                    pass
        text = "\n".join(parts)
    except Exception as e:
        warnings.warn(f"pdfplumber failed ({e}); trying pypdf.")
    if not text.strip():
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((p.extract_text() or "") for p in reader.pages[:max_pages])
        except Exception as e:
            warnings.warn(f"pypdf failed ({e}).")
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
