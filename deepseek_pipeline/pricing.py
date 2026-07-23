# -*- coding: utf-8 -*-
"""
pricing.py
----------
Turn LLM token usage into a USD cost, so each /predict can report what it cost
to score a candidate (extraction via DeepSeek/GPT-4 + JMP embedding via OpenAI).
The frontend uses this to meter per-user spend against a budget.

Token counts are EXACT (read from each API response's `usage`). Only the RATES
are assumptions — verify them against current provider pricing and override
without editing code via the ROOKIE_PRICES env var (JSON), e.g.
  ROOKIE_PRICES='{"deepseek-chat":[0.27,1.10],"text-embedding-3-large":[0.13,0]}'
Values are USD per 1,000,000 tokens: [input_rate, output_rate].
"""

import os
import json
import warnings

# USD per 1,000,000 tokens: model -> (input_rate, output_rate).
# As of 2026-07 — VERIFY before relying on these for real budgeting.
_DEFAULT_PRICES = {
    "deepseek-chat":          (0.27, 1.10),
    "deepseek-reasoner":      (0.55, 2.19),
    "gpt-4o":                 (2.50, 10.00),
    "gpt-4o-mini":            (0.15, 0.60),
    "gpt-4-turbo":            (10.00, 30.00),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-3-small": (0.02, 0.0),
}


def _load_prices():
    prices = dict(_DEFAULT_PRICES)
    raw = os.environ.get("ROOKIE_PRICES", "").strip()
    if raw:
        try:
            for model, pair in json.loads(raw).items():
                prices[model] = (float(pair[0]), float(pair[1]))
        except Exception as e:
            warnings.warn(f"[pricing] ignoring bad ROOKIE_PRICES ({e}).")
    return prices


_PRICES = _load_prices()


def rate(model):
    """(input_rate, output_rate) USD per 1M tokens for a model; (0,0) if unknown."""
    if model in _PRICES:
        return _PRICES[model]
    # tolerate versioned ids like "gpt-4o-2024-08-06"
    for k, v in _PRICES.items():
        if model and model.startswith(k):
            return v
    warnings.warn(f"[pricing] no rate for model {model!r}; cost counted as $0.")
    return (0.0, 0.0)


def cost_usd(model, input_tokens=0, output_tokens=0):
    pin, pout = rate(model)
    return (input_tokens or 0) / 1e6 * pin + (output_tokens or 0) / 1e6 * pout


class UsageMeter:
    """Accumulates per-call token usage during one score() and totals the cost.
    Create one per request (not shared) so it is safe under concurrency."""

    def __init__(self):
        self._calls = []   # list of dicts: stage, model, input_tokens, output_tokens

    def add(self, stage, model, input_tokens=0, output_tokens=0):
        self._calls.append({
            "stage": stage, "model": model,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        })

    def summary(self):
        """Compact, JSON-serialisable cost report for the API response."""
        # merge calls by (stage, model)
        merged = {}
        for c in self._calls:
            key = (c["stage"], c["model"])
            m = merged.setdefault(key, {"stage": c["stage"], "model": c["model"],
                                        "input_tokens": 0, "output_tokens": 0})
            m["input_tokens"] += c["input_tokens"]
            m["output_tokens"] += c["output_tokens"]
        breakdown, total = [], 0.0
        for m in merged.values():
            usd = cost_usd(m["model"], m["input_tokens"], m["output_tokens"])
            m["usd"] = round(usd, 6)
            total += usd
            breakdown.append(m)
        return {
            "usd": round(total, 6),
            "total_tokens": sum(m["input_tokens"] + m["output_tokens"] for m in breakdown),
            "breakdown": breakdown,
        }
