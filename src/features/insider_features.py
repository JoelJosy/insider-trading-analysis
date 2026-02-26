"""
Insider-level features — aggregate statistics computed across the full dataset
and joined back onto each row.

These capture the historical "profile" of the insider, not just the current
trade.  Run this BEFORE temporal features so temporal windows can build
on top of the enriched frame.
"""

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
    df["is_ceo"] = (role.str.contains("ceo") | title.str.lower().str.contains("chief executive")).astype(int)
    df["is_cfo"] = (role.str.contains("cfo") | title.str.lower().str.contains("chief financial")).astype(int)
    df["is_coo"] = (role.str.contains("coo") | title.str.lower().str.contains("chief operating")).astype(int)
    df["is_director_only"] = ((role == "director") & (df["is_officer"] == False)).astype(int)  # noqa: E712
    return df


def _compute_insider_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-insider aggregate stats from the FULL dataset."""
    # Only count open-market buys/sells for the ratio — exclude awards/taxes
    om = df[df["is_open_market"] == True].copy()  # noqa: E712

    buys = (
        om[om["trade_direction"] == 1]
        .groupby("insider_cik")["total_value"]
        .agg(buy_count="count", total_bought=("sum"))
    )
    sells = (
        om[om["trade_direction"] == -1]
        .groupby("insider_cik")["total_value"]
        .agg(sell_count="count", total_sold=("sum"))
    )

    # All trades (incl. awards) for total counts
    # avg_trade_value: only over rows that actually have a dollar value
    all_trades = df.groupby("insider_cik").agg(
        total_trades_hist=("accession_number", "count"),
        first_trade_date=("transaction_date", "min"),
        last_trade_date=("transaction_date", "max"),
    )
    avg_val = (
        df[df["total_value"].notna() & (df["total_value"] > 0)]
        .groupby("insider_cik")["total_value"]
        .mean()
        .rename("avg_trade_value_hist")
    )
    all_trades = all_trades.join(avg_val, how="left")

    agg = all_trades.join(buys, how="left").join(sells, how="left")
    agg["buy_count"] = agg["buy_count"].fillna(0).astype(int)
    agg["sell_count"] = agg["sell_count"].fillna(0).astype(int)
    agg["total_bought"] = agg["total_bought"].fillna(0)
    agg["total_sold"] = agg["total_sold"].fillna(0)

    denom = (agg["buy_count"] + agg["sell_count"]).replace(0, 1)
    agg["buy_sell_ratio_hist"] = agg["buy_count"] / denom

    # Approximate insider tenure from data
    agg["first_trade_date"] = pd.to_datetime(agg["first_trade_date"])
    agg["last_trade_date"] = pd.to_datetime(agg["last_trade_date"])
    agg["tenure_days_hist"] = (
        agg["last_trade_date"] - agg["first_trade_date"]
    ).dt.days.fillna(0)

    return agg.reset_index()


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Apply role features and join per-insider aggregates onto the frame."""
    df = add_role_features(df)

    agg = _compute_insider_aggregates(df)
    agg = agg.rename(columns={
        "total_trades_hist": "insider_total_trades",
        "avg_trade_value_hist": "insider_avg_trade_value",
        "buy_sell_ratio_hist": "insider_buy_sell_ratio",
        "tenure_days_hist": "insider_tenure_days",
        "buy_count": "insider_buy_count",
        "sell_count": "insider_sell_count",
    })

    cols = [
        "insider_cik",
        "insider_total_trades",
        "insider_avg_trade_value",
        "insider_buy_sell_ratio",
        "insider_tenure_days",
        "insider_buy_count",
        "insider_sell_count",
    ]
    df = df.merge(agg[cols], on="insider_cik", how="left")
    return df
