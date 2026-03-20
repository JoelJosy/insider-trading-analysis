"""
Insider-level features — aggregate statistics computed using only
past trades (expanding window), joined back onto each row.

These capture the historical "profile" of the insider, not just the current
trade. Run this BEFORE temporal features so temporal windows can build
on top of the enriched frame.

LOOK-AHEAD BIAS FIX (v2):
The original implementation computed per-insider aggregates from the FULL
dataset and joined them back to all rows — including early rows. This means
a trade from 2013 would show insider_total_trades=162 even though those
future trades hadn't happened yet.

This version uses a strictly expanding window: for each trade, only trades
that occurred BEFORE that trade's date are used to compute the insider's
profile. First trades get zeros/defaults since no history exists yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Rough seniority score for known officer titles / roles
ROLE_SENIORITY = {
    "ceo": 6,
    "cfo": 5,
    "coo": 4,
    "president": 4,
    "chief": 4,
    "general counsel": 3,
    "senior vice president": 3,
    "vice president": 2,
    "officer": 2,
    "director": 1,
}


def _seniority_score(role: str, title: str) -> int:
    """Assign a numeric seniority score from role + officer_title fields."""
    combined = f"{role} {title}".lower() if pd.notna(title) else str(role).lower()
    for keyword, score in ROLE_SENIORITY.items():
        if keyword in combined:
            return score
    return 1  # default: director / unknown


def add_role_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode insider role into numeric features.
    Flags for most senior roles help the model weight their trades more.
    """
    role = df["insider_role"].str.lower().fillna("")
    title = df["officer_title"].fillna("")

    df["role_seniority"] = [
        _seniority_score(r, t) for r, t in zip(df["insider_role"], title)
    ]
    df["is_ceo"] = (
        role.str.contains("ceo") | title.str.lower().str.contains("chief executive")
    ).astype(int)
    df["is_cfo"] = (
        role.str.contains("cfo") | title.str.lower().str.contains("chief financial")
    ).astype(int)
    df["is_coo"] = (
        role.str.contains("coo") | title.str.lower().str.contains("chief operating")
    ).astype(int)
    df["is_director_only"] = (
        (role == "director") & (df["is_officer"] == False)  # noqa: E712
    ).astype(int)
    return df


def _compute_insider_aggregates_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-insider aggregate stats using ONLY past trades (expanding window).

    For each trade at position i, only trades at positions 0..i-1 (strictly
    earlier dates for the same insider) are used. This eliminates look-ahead
    bias — early trades get zeros/defaults since no history exists yet.

    Only open-market trades (is_open_market=True) count toward buy/sell ratio
    to match the original logic.
    """
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    # Work on a sorted copy, track original index to join back
    df = df.sort_values(["insider_cik", "transaction_date"]).reset_index(drop=False)
    original_index_col = "index"  # reset_index creates this column

    records = []

    for insider_cik, group in df.groupby("insider_cik", sort=False):
        group = group.sort_values("transaction_date").reset_index(drop=True)

        n = len(group)
        is_open_mkt = group["is_open_market"].astype(bool).values
        directions = pd.to_numeric(
            group.get("trade_direction", pd.Series(0, index=group.index)),
            errors="coerce",
        ).fillna(0).astype(int).values
        values = pd.to_numeric(
            group.get("total_value", pd.Series(np.nan, index=group.index)),
            errors="coerce",
        ).values
        orig_indices = group[original_index_col].values

        for i in range(n):
            if i == 0:
                # No history yet — first trade for this insider
                records.append({
                    "orig_idx": orig_indices[i],
                    "insider_total_trades": 0,
                    "insider_avg_trade_value": 0.0,
                    "insider_buy_sell_ratio": 0.5,  # neutral prior
                    "insider_tenure_days": 0,
                    "insider_buy_count": 0,
                    "insider_sell_count": 0,
                })
                continue

            # Past trades: indices 0..i-1
            past_om_mask = is_open_mkt[:i]
            past_directions = directions[:i]
            past_values = values[:i]

            total_trades = i  # all past trades for this insider

            buy_count = int(np.sum((past_om_mask) & (past_directions == 1)))
            sell_count = int(np.sum((past_om_mask) & (past_directions == -1)))
            denom = buy_count + sell_count
            buy_sell_ratio = buy_count / denom if denom > 0 else 0.5

            valid_vals = past_values[
                ~np.isnan(past_values) & (past_values > 0)
            ]
            avg_trade_value = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0

            first_date = group["transaction_date"].iloc[0]
            current_date = group["transaction_date"].iloc[i]
            tenure_days = int((current_date - first_date).days)

            records.append({
                "orig_idx": orig_indices[i],
                "insider_total_trades": total_trades,
                "insider_avg_trade_value": avg_trade_value,
                "insider_buy_sell_ratio": buy_sell_ratio,
                "insider_tenure_days": tenure_days,
                "insider_buy_count": buy_count,
                "insider_sell_count": sell_count,
            })

    result_df = pd.DataFrame(records).set_index("orig_idx")
    return result_df


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Apply role features and join per-insider rolling aggregates onto the frame."""
    df = add_role_features(df)

    print("  Computing rolling insider aggregates (look-ahead-free)...")
    agg = _compute_insider_aggregates_rolling(df)

    cols = [
        "insider_total_trades",
        "insider_avg_trade_value",
        "insider_buy_sell_ratio",
        "insider_tenure_days",
        "insider_buy_count",
        "insider_sell_count",
    ]

    # Join back on original index
    df = df.join(agg[cols])

    # Fill any insiders that had no history at all (shouldn't happen but safe)
    df["insider_total_trades"] = df["insider_total_trades"].fillna(0).astype(int)
    df["insider_avg_trade_value"] = df["insider_avg_trade_value"].fillna(0.0)
    df["insider_buy_sell_ratio"] = df["insider_buy_sell_ratio"].fillna(0.5)
    df["insider_tenure_days"] = df["insider_tenure_days"].fillna(0).astype(int)
    df["insider_buy_count"] = df["insider_buy_count"].fillna(0).astype(int)
    df["insider_sell_count"] = df["insider_sell_count"].fillna(0).astype(int)

    return df