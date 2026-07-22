# -*- coding: utf-8 -*-
"""
canonicalize.py
---------------
Use DeepSeek to map messy extracted names (universities, journals) onto a fixed
reference list, so Set D joins survive abbreviations and alternate phrasings:
  "Wharton" / "UPenn"          -> "Pennsylvania"   (top-50 match)
  "JAR" / "J. Acc. Research"   -> "Journal of Accounting Research" (top-journal match)

The LLM is used ONLY for name disambiguation against a list YOU control - never
to invent rankings. Each unique name is resolved once and cached to JSON, so
build_set_D stays fast and deterministic.

Used by build_canonical_maps.py (the pre-pass). build_set_D.py then loads the
cached maps.
"""

import os
import json
import time
import warnings
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
BATCH = 40
MAX_RETRIES = 4


def get_client():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY environment variable.")
    return OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)


def _lk(s: str) -> str:
    """Light lookup key - strip + lowercase (stable join key for the cache)."""
    return str(s or "").strip().lower()


PROMPT = """You are matching extracted {kind} names onto a FIXED reference list.

Reference {kind} list (the only valid outputs):
{reference}

For each input name below, return the EXACT reference entry it refers to, or
null if it is NOT one of the reference entries (a different or lower-tier
{kind}). Account for abbreviations, acronyms, nicknames, and alternate
phrasings. Do not invent entries that are not in the reference list.

Return ONE JSON object mapping each input name (verbatim) to a reference entry
string or null. Output only the JSON object.

Input names:
{inputs}"""


def _call(client, kind, reference, names):
    ref_block = "\n".join(f"- {r}" for r in reference)
    in_block = "\n".join(f"- {n}" for n in names)
    prompt = PROMPT.format(kind=kind, reference=ref_block, inputs=in_block)
    content = ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0, max_tokens=2048,
            )
            content = resp.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError:
            warnings.warn("canonicalize: JSON parse failed; retrying.")
        except Exception as e:
            wait = 2 ** attempt
            warnings.warn(f"canonicalize call failed ({e}); retry in {wait}s.")
            time.sleep(wait)
    return {}


def canonicalize(names, reference, kind, cache_path, client=None) -> dict:
    """
    Resolve `names` against `reference`. Returns {lookup_key -> canonical or None}.
    Caches to cache_path; only previously-unseen names hit the API.
    """
    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))

    # unique, non-empty names not already cached
    todo = []
    seen = set()
    for n in names:
        k = _lk(n)
        if not k or k in cache or k in seen:
            continue
        seen.add(k)
        todo.append(n)

    if todo:
        client = client or get_client()
        ref_set = {r for r in reference}
        print(f"  canonicalizing {len(todo)} new {kind} names "
              f"(cache had {len(cache)}) ...")
        for start in range(0, len(todo), BATCH):
            chunk = todo[start:start + BATCH]
            result = _call(client, kind, reference, chunk)
            for raw in chunk:
                val = result.get(raw)
                # accept only outputs that are genuinely in the reference list
                cache[_lk(raw)] = val if (val in ref_set) else None
            print(f"    {min(start + BATCH, len(todo))}/{len(todo)} done")
            cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    else:
        print(f"  all {kind} names already cached ({len(cache)}).")

    return cache


def lookup(cache: dict, raw_name: str):
    """Return the canonical reference entry for a raw name, or None."""
    return cache.get(_lk(raw_name))
