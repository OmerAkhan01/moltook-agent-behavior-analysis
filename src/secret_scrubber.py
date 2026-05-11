from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


# Keep patterns conservative to avoid false positives.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Stripe secret keys
    (re.compile(r"\bsk_live_[A-Za-z0-9]{16,}\b"), "sk_live_REDACTED"),
    (re.compile(r"\bsk_test_[A-Za-z0-9]{16,}\b"), "sk_test_REDACTED"),
    # Common “API key” style tokens (very conservative)
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AKIA_REDACTED"),  # AWS access key id
]


def redact_text(value: object) -> object:
    if not isinstance(value, str) or not value:
        return value
    s = value
    for pat, repl in _PATTERNS:
        s = pat.sub(repl, s)
    return s


def redact_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df
    out = df.copy()
    for c in cols:
        out[c] = out[c].map(redact_text)
    return out

