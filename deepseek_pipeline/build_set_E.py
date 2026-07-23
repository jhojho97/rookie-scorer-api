# -*- coding: utf-8 -*-
"""
build_set_E.py
--------------
Build Set E (256 dissertation embedding dims, 0_dt..255_dt) - the JMP
title+abstract+intro embedded with OpenAI text-embedding-3-large, keeping the
first 256 numbers, EXACTLY per the paper's method:
  - <= 8191 tokens: embed the whole text directly
  - >  8191 tokens: overlapping sliding windows (100-token stride), averaged
  - no JMP: 256-dim zero vector
Requires tiktoken (pip install tiktoken).

Independent of DeepSeek. Extracts JMP text directly from the candidate folders.
If extract_deepseek.py has already cached jmp_text in raw_json, it reuses that
to avoid re-parsing PDFs (pass --json_dir).

Output: set_E.csv - columns: ID + 0_dt..255_dt.

Usage:
  set OPENAI_API_KEY=...
  python build_set_E.py --data_dir ../data --main_csv ../data/2015-2018_rookie_dataset.csv
  # reuse cached JMP text instead of re-parsing PDFs:
  python build_set_E.py ... --json_dir deepseek_output/raw_json
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from build_scibert_dataset import extract_text, find_cv_file, find_jmp_file, extract_jmp_sections
from common import SET_E_COLS, build_candidate_index, load_raw_json_by_folder

EMBED_MODEL = "text-embedding-3-large"
MAX_TOKENS  = 8191        # model input limit (paper's threshold)
STRIDE      = 100         # sliding-window stride (paper: 100-token stride)
REQ_TOKEN_BUDGET = 250000 # keep each embeddings request under the API token cap


def get_jmp_text(folder: Path, cached: dict | None) -> str:
    if cached and cached.get("jmp_text"):
        return cached["jmp_text"]
    cv_file = find_cv_file(folder)
    jmp_file = find_jmp_file(folder, cv_file=cv_file)
    if not jmp_file:
        return ""
    full = extract_text(jmp_file)
    return extract_jmp_sections(full) if full else ""


def _get_encoder():
    import tiktoken
    try:
        return tiktoken.encoding_for_model(EMBED_MODEL)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _embed_token_windows(client, windows, meter=None):
    """Embed a list of token-id lists; return list of first-256-dim vectors.
    Batches by a token budget so no single request exceeds the API cap."""
    vecs, i = [], 0
    while i < len(windows):
        batch, tok = [], 0
        while i < len(windows) and (not batch or tok + len(windows[i]) <= REQ_TOKEN_BUDGET):
            batch.append(windows[i]); tok += len(windows[i]); i += 1
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        if meter is not None and getattr(resp, "usage", None) is not None:
            meter.add("embedding", EMBED_MODEL,
                      getattr(resp.usage, "prompt_tokens", 0), 0)
        vecs.extend(np.asarray(d.embedding[:256], dtype=float) for d in resp.data)
    return vecs


def embed(texts, meter=None):
    """
    Reproduce the paper's JMP embedding exactly:
      - text-embedding-3-large, keep the first 256 dimensions
      - <= 8191 tokens: embed the whole title+abstract+intro directly
      - >  8191 tokens: overlapping sliding windows (100-token stride),
        average the resulting vectors
      - no JMP text: a 256-dim ZERO vector
    Returns (N, 256).
    """
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY for Set E embeddings.")
    client = OpenAI(api_key=key)
    enc = _get_encoder()

    out = np.zeros((len(texts), 256), dtype=float)   # missing JMP stays zero

    # Build the flat list of windows and remember which text each belongs to.
    windows, owner = [], []
    n_long = 0
    for i, t in enumerate(texts):
        # match the authors' pre-embedding whitespace cleanup exactly
        t = (t or "")
        for ch in ("\n", "\t", "\r", "\xa0"):
            t = t.replace(ch, " ")
        t = t.strip()
        if not t:
            continue                                  # zero vector
        toks = enc.encode(t)
        if len(toks) <= MAX_TOKENS:
            windows.append(toks); owner.append(i)
        else:
            n_long += 1
            starts = list(range(0, len(toks) - MAX_TOKENS + 1, STRIDE))
            if starts[-1] != len(toks) - MAX_TOKENS:
                starts.append(len(toks) - MAX_TOKENS)  # ensure the tail is covered
            for st in starts:
                windows.append(toks[st:st + MAX_TOKENS]); owner.append(i)

    print(f"  embedding {len(windows)} windows for {sum(1 for t in texts if (t or '').strip())} "
          f"papers ({n_long} papers exceeded {MAX_TOKENS} tokens -> sliding-window averaged)")
    vecs = _embed_token_windows(client, windows, meter=meter)

    # Average all windows belonging to the same paper.
    from collections import defaultdict
    acc = defaultdict(list)
    for o, v in zip(owner, vecs):
        acc[o].append(v)
    for i, vs in acc.items():
        out[i] = np.mean(vs, axis=0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--main_csv", required=True)
    ap.add_argument("--json_dir", default=None,
                    help="Optional raw_json dir to reuse cached jmp_text")
    ap.add_argument("--out", default="set_E.csv")
    args = ap.parse_args()

    index = build_candidate_index(args.data_dir, args.main_csv)
    json_by_folder = load_raw_json_by_folder(args.json_dir) if args.json_dir else {}

    ids, texts, n_empty = [], [], 0
    for r in index.itertuples(index=False):
        cid = int(r.ID)
        cached = json_by_folder.get(Path(r.folder).name)
        t = get_jmp_text(Path(r.folder), cached)
        if not t:
            n_empty += 1
        ids.append(cid)
        texts.append(t)

    if n_empty:
        warnings.warn(f"{n_empty} candidates had no JMP text - assigned a 256-dim "
                      f"ZERO vector (matches the paper).")

    emb = embed(texts)
    df = pd.DataFrame(emb, columns=SET_E_COLS)
    df.insert(0, "ID", ids)
    df.to_csv(args.out, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {args.out}")


if __name__ == "__main__":
    main()
