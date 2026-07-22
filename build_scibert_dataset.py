# -*- coding: utf-8 -*-
"""
build_scibert_dataset.py
------------------------
Extract raw text from CV and JMP PDFs/DOCX for all candidates (2015-2018),
align to labels from the main dataset via alphabetical-order matching,
and write out data/scibert_dataset.csv plus a diagnostic match report.

Mapping rule (confirmed by user):
  Within each year, dataset rows sorted by ID correspond positionally to
  data/{year}/ subfolders sorted alphabetically.

Usage:
  python build_scibert_dataset.py
"""

import os
import re
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

# ── Optional imports (install if missing) ────────────────────────────────────
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    warnings.warn("pdfplumber not found; will fall back to pypdf for PDF extraction.")

try:
    from pypdf import PdfReader as PyPDFReader
    HAS_PYPDF = True
except ImportError:
    try:
        from PyPDF2 import PdfReader as PyPDFReader
        HAS_PYPDF = True
    except ImportError:
        HAS_PYPDF = False
        warnings.warn("Neither pdfplumber nor pypdf/PyPDF2 found. PDF extraction will fail.")

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    warnings.warn("python-docx not found; .docx files will be skipped.")

# .doc (legacy Word binary) — try win32com first (requires Microsoft Word on Windows),
# fall back to docx2txt if available
HAS_WIN32COM = False
HAS_DOCX2TXT = False
try:
    import win32com.client
    HAS_WIN32COM = True
except ImportError:
    pass
try:
    import docx2txt
    HAS_DOCX2TXT = True
except ImportError:
    pass

# wordninja: repairs space-stripped PDF text ("Iexaminetheinfluence" ->
# "I examine the influence"). pdfplumber drops inter-word spaces on some fonts
# (~20% of JMPs), which corrupts the Set E embedding. Optional: if not installed
# the repair is a no-op (pip install wordninja).
try:
    import wordninja
    HAS_WORDNINJA = True
except ImportError:
    HAS_WORDNINJA = False
    warnings.warn("wordninja not found; space-stripped PDF text will NOT be "
                  "repaired (pip install wordninja).")

# A maximal run of >= this many letters with no break is almost certainly glued
# words, not a real token -> re-segment it. Real words/hyphenates stay well under.
_GLUE_MIN_RUN = 18
_GLUE_RE = re.compile(r"[A-Za-z]{%d,}" % _GLUE_MIN_RUN)


def desegment_glued(text: str) -> str:
    """Re-insert spaces into glued word-runs while preserving line breaks and
    untouched clean tokens. Only tokens containing a long letter-run are passed
    through wordninja, so clean text, numbers, and punctuation are unaffected."""
    if not HAS_WORDNINJA or not text or not _GLUE_RE.search(text):
        return text                                   # fast path: nothing glued
    fixed_lines = []
    for line in text.split("\n"):
        toks = []
        for tok in line.split(" "):
            runs = re.findall(r"[A-Za-z]+", tok)
            if runs and max(len(r) for r in runs) >= _GLUE_MIN_RUN:
                toks.append(" ".join(wordninja.split(tok)))
            else:
                toks.append(tok)
        fixed_lines.append(" ".join(toks))
    return "\n".join(fixed_lines)


# ── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
YEARS = [2015, 2016, 2017, 2018]

# Regex patterns for JMP section detection (case-insensitive)
# Broad patterns to survive imperfect PDF text extraction:
#   - all-caps headers (ABSTRACT, INTRODUCTION)
#   - numbered variants (1. Introduction, I. Introduction, 1 Introduction)
#   - headers with trailing colon (Abstract:)
#   - headers that aren't on a line by themselves (word may be glued to newline)
ABSTRACT_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:abstract|ABSTRACT)\s*[:\n]', re.IGNORECASE
)
INTRO_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:[I1]\.?\s+)?(?:introduction|INTRODUCTION)\s*[:\n]',
    re.IGNORECASE
)
SECTION2_PATTERN = re.compile(
    r'(?:^|\n)\s*(?:2[\.\s]|II[\.\s])', re.MULTILINE
)

# --- Authors' Set E boundary: the Section 2 (or 1.2) heading -------------------
# The paper's dt_text_trim = contiguous text from the START of the JMP up to the
# Section-2 heading. This detector finds that heading as a line that starts with
# "2" / "II" / "Section 2" / "1.2" followed by a Title-Case heading word. It is
# anchored after the Introduction so a stray "2." in the abstract is not matched.
_SEC2_HEADING = re.compile(
    r'(?m)^[ \t ]{0,8}'
    r'(?:2(?:\.0)?|II|1\.2|Section[ \t]+2)'      # 2 / 2.0 / II / 1.2 / Section 2
    r'[\.\)\:]?[ \t ]+'                       # optional . ) : then space
    r'[A-Z][A-Za-z][A-Za-z \t\-,:]{2,70}$'         # a short Title-Case heading line
)


def find_section2_boundary(full_text: str):
    """Return the char index where Section 2 begins (start of its heading line),
    or None. Anchored after the Introduction to avoid false matches in the
    abstract; validates the match is a short heading line, not a sentence."""
    if not full_text:
        return None
    im = INTRO_PATTERN.search(full_text)
    start_from = (im.end() + 300) if im else 700
    m = _SEC2_HEADING.search(full_text, start_from)
    if not m:
        return None
    # back up to the beginning of the heading's line
    return full_text.rfind("\n", 0, m.start()) + 1


# ── PDF / DOCX text extraction ────────────────────────────────────────────────

def extract_text_pdfplumber(path: Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    pages = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
    except Exception as e:
        warnings.warn(f"pdfplumber failed on {path}: {e}")
        return ""
    return "\n".join(pages)


def extract_text_pypdf(path: Path) -> str:
    """Extract all text from a PDF using pypdf/PyPDF2 fallback."""
    pages = []
    try:
        reader = PyPDFReader(str(path))
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    except Exception as e:
        warnings.warn(f"pypdf failed on {path}: {e}")
        return ""
    return "\n".join(pages)


def extract_text_docx(path: Path) -> str:
    """Extract all text from a .docx file."""
    if not HAS_DOCX:
        warnings.warn(f"Skipping {path} — python-docx not installed.")
        return ""
    try:
        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        warnings.warn(f"python-docx failed on {path}: {e}")
        return ""


def extract_text_doc(path: Path) -> str:
    """
    Extract text from a legacy .doc (Word 97-2003 binary) file.

    Extraction order:
      1. antiword  — best for true binary .doc (Linux/Colab: apt-get install antiword)
      2. LibreOffice headless — converts to txt via temp file (already on Colab)
      3. docx2txt  — only works if the .doc is actually a renamed .docx (ZIP-based)
      4. win32com  — Windows only, requires Microsoft Word installed
    """
    import subprocess, platform, tempfile, shutil

    abs_path = str(path.resolve())

    # ── Method 1: antiword (Linux / Colab) ───────────────────────────────────
    if shutil.which("antiword"):
        try:
            result = subprocess.run(
                ["antiword", abs_path],
                capture_output=True, text=True, timeout=30,
                encoding="utf-8", errors="replace",
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout
        except Exception as e:
            warnings.warn(f"antiword failed on {path.name}: {e}")

    # ── Method 2: LibreOffice headless (already installed on Colab) ──────────
    if shutil.which("libreoffice") or shutil.which("soffice"):
        lo = shutil.which("libreoffice") or shutil.which("soffice")
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                subprocess.run(
                    [lo, "--headless", "--convert-to", "txt:Text",
                     "--outdir", tmp_dir, abs_path],
                    capture_output=True, timeout=60,
                )
                txt_file = Path(tmp_dir) / (path.stem + ".txt")
                if txt_file.exists():
                    text = txt_file.read_text(encoding="utf-8", errors="replace")
                    if text.strip():
                        return text
        except Exception as e:
            warnings.warn(f"LibreOffice conversion failed on {path.name}: {e}")

    # ── Method 3: docx2txt (only works if file is really a renamed .docx) ────
    if HAS_DOCX2TXT:
        try:
            text = docx2txt.process(abs_path)
            if text and text.strip():
                return text
        except Exception:
            pass   # silently skip — expected to fail on true binary .doc

    # ── Method 4: win32com (Windows + Microsoft Word only) ───────────────────
    if HAS_WIN32COM and platform.system() == "Windows":
        try:
            import pythoncom
            pythoncom.CoInitialize()
            word = win32com.client.Dispatch("Word.Application")
            word.Visible = False
            doc  = word.Documents.Open(abs_path, ReadOnly=True)
            text = doc.Content.Text
            doc.Close(False)
            word.Quit()
            pythoncom.CoUninitialize()
            if text.strip():
                return text
        except Exception as e:
            warnings.warn(f"win32com failed on {path.name}: {e}")

    warnings.warn(
        f"Cannot extract .doc file: {path.name}. "
        "On Colab run:  !apt-get install -q antiword"
    )
    return ""


def extract_text(path: Path) -> str:
    """Dispatch to the right extractor based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if HAS_PDFPLUMBER:
            text = extract_text_pdfplumber(path)
            if not text and HAS_PYPDF:
                text = extract_text_pypdf(path)
        elif HAS_PYPDF:
            text = extract_text_pypdf(path)
        else:
            text = ""
        return desegment_glued(text)     # repair space-stripped PDF text
    elif suffix == ".docx":
        return extract_text_docx(path)
    elif suffix == ".doc":
        return extract_text_doc(path)
    else:
        warnings.warn(f"Unknown file extension {suffix} for {path}")
        return ""


# ── File discovery helpers ────────────────────────────────────────────────────

DOC_EXTENSIONS = {".pdf", ".doc", ".docx"}

def all_docs(folder: Path) -> list[Path]:
    """
    All document files in folder (pdf, doc, docx), sorted by name.
    Searches RECURSIVELY (rglob) so documents tucked inside a nested
    subfolder are still found — this is what rescues the "both missing"
    candidates whose files sit one level deeper.
    """
    return sorted(
        [f for f in folder.rglob("*")
         if f.is_file() and f.suffix.lower() in DOC_EXTENSIONS
         # skip Word lock/owner temp files (~$Name.doc) and hidden files
         and not f.name.startswith("~$")
         and not f.name.startswith(".")],
        key=lambda p: p.name.lower(),
    )


def find_cv_file(folder: Path) -> Path | None:
    """
    Locate the CV file. Checks (in order):
      1. Name contains a CV keyword (broad list covering common naming styles)
      2. If exactly one document remains after the JMP is identified — not used
         here; handled in find_pair() below.
    """
    CV_KEYWORDS = [
        "cv", "c.v", "curriculum", "vitae", "vita",   # standard + "Vita*"
        "resume", "résumé",
    ]
    docs = all_docs(folder)
    # Exact / substring keyword match (case-insensitive, strip extension)
    for f in docs:
        stem_lower = f.stem.lower().replace(".", "").replace("_", "").replace("-", "").replace(" ", "")
        name_lower = f.name.lower()
        if any(kw.replace(".", "").replace(" ", "") in stem_lower for kw in CV_KEYWORDS):
            return f
        # Also check full name including extension
        if any(kw in name_lower for kw in CV_KEYWORDS):
            return f
    return None


def find_jmp_file(folder: Path, cv_file: Path | None = None) -> Path | None:
    """
    Locate the JMP file. Checks (in order):
      1. Name contains a known JMP keyword
      2. Exactly 2 docs and one is the CV → the other is the JMP
      3. Only 1 doc and no CV identified → that doc is the JMP
      4. 3+ docs, none keyword-matched → pick the LARGEST non-CV file.
         Rationale: a job market paper is a 40-60 page document, far larger
         than a CV, cover letter, or teaching statement. Size is a reliable
         tie-breaker when the filename gives no hint.
    """
    JMP_KEYWORDS = [
        "road", "dissertation", "job market", "jmp",
        "working paper", "paper", "draft", "essay",
    ]
    docs = all_docs(folder)

    # 1. Keyword match
    for f in docs:
        name_lower = f.name.lower()
        if any(kw in name_lower for kw in JMP_KEYWORDS):
            return f

    non_cv = [f for f in docs if f != cv_file]

    # 2. Exactly one non-CV doc → it's the JMP
    if len(non_cv) == 1:
        return non_cv[0]

    # 3. No CV identified but a single doc exists → treat it as JMP
    if cv_file is None and len(docs) == 1:
        return docs[0]

    # 4. Several non-CV docs, none keyword-matched → largest file wins
    if len(non_cv) >= 2:
        try:
            return max(non_cv, key=lambda f: f.stat().st_size)
        except OSError:
            return non_cv[0]

    return None


# ── JMP section extraction ────────────────────────────────────────────────────

def extract_jmp_sections(full_text: str) -> str:
    """
    Reproduce the paper's Set E input (dt_text_trim): the CONTIGUOUS text from the
    START of the JMP up to the Section 2 (or 1.2) heading. This matches the
    authors' manual boundary far better than extracting title+abstract+intro
    separately (which dropped ~65% of the text). See find_section2_boundary.

    Fallbacks, in order:
      1. Section-2 heading found -> text[:boundary]  (the authors' method).
      2. No heading, but Introduction found -> start .. intro + long window
         (covers a full intro even when Section 2 is undetected).
      3. Neither -> first 12000 chars (was 8000; the authors' trims run long).
    """
    if not full_text or not full_text.strip():
        return ""

    b = find_section2_boundary(full_text)
    if b and b > 400:
        return full_text[:b].strip()

    # Fallback 2: anchor on the Introduction, take a generous contiguous window.
    intro_match = INTRO_PATTERN.search(full_text)
    if intro_match:
        return full_text[:intro_match.end() + 11000].strip()

    # Fallback 3: no structural markers (broken PDF layout) -> generous head slice.
    return full_text[:12000].strip()


# ── Main extraction loop ──────────────────────────────────────────────────────

def build_dataset(csv_path: str = "data/2015-2018_rookie_dataset.csv") -> pd.DataFrame:
    print(f"Loading main dataset from {csv_path}...")
    data = pd.read_csv(csv_path)

    required_cols = {"ID", "year", "pub_top_5pct", "pub_w_top_5pct", "research_oriented"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Main dataset missing columns: {missing}")

    records = []
    match_report = []

    for year in YEARS:
        year_dir = DATA_DIR / str(year)
        if not year_dir.exists():
            warnings.warn(f"Year directory {year_dir} does not exist — skipping year {year}.")
            continue

        # Get dataset rows for this year, sorted by ID
        year_rows = data[data["year"] == year].sort_values("ID").reset_index(drop=True)

        # Get subfolders sorted alphabetically (ignore files like Readme.txt)
        year_folders = sorted(
            [d for d in year_dir.iterdir() if d.is_dir()],
            key=lambda p: p.name.lower()
        )

        n_rows = len(year_rows)
        n_folders = len(year_folders)

        if n_rows != n_folders:
            warnings.warn(
                f"Year {year}: {n_rows} dataset rows but {n_folders} folders. "
                f"Matching up to min({n_rows}, {n_folders}) candidates."
            )

        n_match = min(n_rows, n_folders)

        print(f"\nYear {year}: {n_rows} candidates, {n_folders} folders — processing {n_match}...")

        for i in range(n_match):
            row = year_rows.iloc[i]
            folder = year_folders[i]
            candidate_id = int(row["ID"])

            # Find CV and JMP files
            cv_file  = find_cv_file(folder)
            jmp_file = find_jmp_file(folder, cv_file=cv_file)

            # Reconciliation for generically-named docs (e.g. "Applicant Document 1/2"):
            # if the CV wasn't identified by keyword but a JMP was found and other
            # documents remain, the CV is the SMALLEST remaining doc (CVs are short,
            # papers are long).
            if cv_file is None and jmp_file is not None:
                leftovers = [f for f in all_docs(folder) if f != jmp_file]
                if leftovers:
                    try:
                        cv_file = min(leftovers, key=lambda f: f.stat().st_size)
                    except OSError:
                        cv_file = leftovers[0]

            cv_found = cv_file is not None
            jmp_found = jmp_file is not None

            match_report.append({
                "ID": candidate_id,
                "year": year,
                "folder_name": folder.name,
                "cv_file": cv_file.name if cv_found else "",
                "jmp_file": jmp_file.name if jmp_found else "",
                "cv_found": cv_found,
                "jmp_found": jmp_found,
            })

            # Extract text
            cv_text = extract_text(cv_file) if cv_found else ""
            if not cv_text and cv_found:
                warnings.warn(f"  [ID {candidate_id}] CV text extraction returned empty: {cv_file}")

            jmp_full = extract_text(jmp_file) if jmp_found else ""
            jmp_text = extract_jmp_sections(jmp_full) if jmp_full else ""
            if not jmp_text and jmp_found:
                warnings.warn(f"  [ID {candidate_id}] JMP text extraction returned empty: {jmp_file}")

            status = "OK"
            if not cv_found and not jmp_found:
                status = "NO_FILES"
            elif not cv_found:
                status = "NO_CV"
            elif not jmp_found:
                status = "NO_JMP"

            if i % 20 == 0:
                print(f"  [{i+1}/{n_match}] ID={candidate_id} folder='{folder.name}' "
                      f"cv={cv_found} jmp={jmp_found}")

            records.append({
                "ID": candidate_id,
                "year": year,
                "folder_name": folder.name,
                "cv_text": cv_text,
                "jmp_text": jmp_text,
                "pub_top_5pct": int(row["pub_top_5pct"]),
                "pub_w_top_5pct": int(row["pub_w_top_5pct"]),
                "research_oriented": int(row["research_oriented"]),
                "status": status,
            })

    df = pd.DataFrame(records)
    report_df = pd.DataFrame(match_report)

    # Save outputs
    out_path = DATA_DIR / "scibert_dataset.csv"
    report_path = DATA_DIR / "scibert_match_report.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    report_df.to_csv(report_path, index=False, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Saved {len(df)} candidates to {out_path}")
    print(f"Saved match report to {report_path}")
    print(f"\nLabel distribution:")
    for col in ["pub_top_5pct", "pub_w_top_5pct"]:
        vc = df[col].value_counts()
        print(f"  {col}: {vc.get(1,0)} positives / {len(df)} total ({vc.get(1,0)/len(df)*100:.1f}%)")
    print(f"\nText availability:")
    print(f"  CV text present:  {(df['cv_text'].str.len() > 0).sum()} / {len(df)}")
    print(f"  JMP text present: {(df['jmp_text'].str.len() > 0).sum()} / {len(df)}")

    return df


if __name__ == "__main__":
    build_dataset()
