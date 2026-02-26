"""
Text features — lightweight NLP on the footnote_text field.

Phase 2 uses a simple rule-based approach (keyword matching + routine-plan
detection) that works without any ML model or internet access.

Phase 3+ can swap in FinBERT sentiment by enabling the FinBERT block below.
"""

import re

import numpy as np
import pandas as pd

# ── Routine / plan keywords that tend to appear in boilerplate disclosures ──
ROUTINE_PHRASES = [
    "10b5-1",
    "10b5 1",
    "rule 10b",
    "trading plan",
    "scheduled vesting",
    "vesting date",
    "tax withholding",
    "withholding",
    "rsu",
    "restricted stock unit",
    "performance award",
]

# ── Informative / potentially non-routine language ──
INFORMED_PHRASES = [
    "open market",
    "discretionary",
    "personally",
    "at market",
    "direct purchase",
    "direct sale",
    "additional shares",
]

# ── Option / derivative keywords ──
OPTION_PHRASES = [
    "option",
    "warrant",
    "conversion",
    "convertible",
    "exercise price",
]


def _count_keywords(text: str, phrases: list[str]) -> int:
    if not isinstance(text, str) or len(text) < 10:
        return 0
    t = text.lower()
    return sum(1 for p in phrases if p in t)


def _footnote_length(text) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.strip())


def _has_10b51_plan(text) -> int:
    if not isinstance(text, str):
        return 0
    return int(bool(re.search(r"10b5[-\s]?1", text, re.IGNORECASE)))


def _routine_language_score(text) -> float:
    """
    Heuristic score 0-1.  High → footnote reads like a boilerplate disclosure;
    Low → more unusual / discretionary language.
    """
    if not isinstance(text, str) or len(text.strip()) < 10:
        return 0.5  # unknown

    routine_hits = _count_keywords(text, ROUTINE_PHRASES)
    informed_hits = _count_keywords(text, INFORMED_PHRASES)
    total = routine_hits + informed_hits
    if total == 0:
        return 0.5
    return routine_hits / total


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all footnote-derived columns to the DataFrame.
    Safe to call even if footnote_text is empty or missing.
    """
    notes = df["footnote_text"].fillna("")

    df["footnote_length"] = notes.apply(_footnote_length)
    df["footnote_has_plan"] = notes.apply(_has_10b51_plan)
    df["footnote_routine_score"] = notes.apply(_routine_language_score)
    df["footnote_routine_hits"] = notes.apply(
        lambda t: _count_keywords(t, ROUTINE_PHRASES)
    )
    df["footnote_informed_hits"] = notes.apply(
        lambda t: _count_keywords(t, INFORMED_PHRASES)
    )
    df["footnote_has_option"] = notes.apply(
        lambda t: int(_count_keywords(t, OPTION_PHRASES) > 0)
    )

    # Reconcile with the structured column when available
    if "has_10b5_1_plan" in df.columns:
        df["has_plan"] = (
            df["has_10b5_1_plan"].astype(bool) | df["footnote_has_plan"].astype(bool)
        ).astype(int)
    else:
        df["has_plan"] = df["footnote_has_plan"]

    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all text features."""
    df = add_text_features(df)
    return df
