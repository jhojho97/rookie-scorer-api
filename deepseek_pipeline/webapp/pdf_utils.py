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


def extract_pdf_bytes(data: bytes) -> str:
    """Extract all text from PDF bytes."""
    text = ""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception as e:
        warnings.warn(f"pdfplumber failed ({e}); trying pypdf.")
    if not text.strip():
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
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
