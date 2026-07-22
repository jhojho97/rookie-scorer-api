# -*- coding: utf-8 -*-
"""
extract_gpt4.py
---------------
GPT-4 variant of the Stage-1 extraction (Set C + names for Set D). It reuses the
EXACT same prompts and parsing as extract_deepseek.py -- the only difference is
the provider: OpenAI's default endpoint + OPENAI_API_KEY + a GPT-4 model. This
lets you A/B GPT-4 vs DeepSeek on identical inputs to see whether the paper's
original extractor (GPT-4) recovers signal that DeepSeek loses.

The model is configurable so you can match whichever GPT-4 you want:
  PowerShell:  $env:GPT4_MODEL = "gpt-4o"        # default
               $env:GPT4_MODEL = "gpt-4-turbo"   # closer to the paper's vintage

Interface mirrors extract_deepseek: get_client() + extract_one(client, cv, jmp).
"""

import os

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai")

# Reuse the identical prompt-calling + parsing logic; only the model differs.
from extract_deepseek import extract_one as _extract_one_with_model

# Default to gpt-4o (current GPT-4-family workhorse, supports JSON mode).
GPT4_MODEL       = os.environ.get("GPT4_MODEL", "gpt-4o")
OPENAI_BASE_URL  = os.environ.get("OPENAI_BASE_URL")   # None => OpenAI default


def get_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Set OPENAI_API_KEY environment variable (GPT-4 extraction).")
    if OPENAI_BASE_URL:
        return OpenAI(api_key=key, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=key)


def extract_one(client, cv_text, jmp_text):
    """Same as extract_deepseek.extract_one but pinned to the GPT-4 model."""
    return _extract_one_with_model(client, cv_text, jmp_text, model=GPT4_MODEL)
