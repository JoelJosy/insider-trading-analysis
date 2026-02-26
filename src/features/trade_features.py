"""
Trade-level features — per-row transforms that require no external context.

Each function takes the full DataFrame and returns it with new columns added.
Order of application doesn't matter here.
"""

import numpy as np
import pandas as pd

# Transaction codes and their semantic direction
BUYS = {"A", "M"}          # Award/grant and exercise/conversion
SELLS = {"S", "F"}         # Open-market sale and tax-withholding sale
NEUTRAL = {"G", "J", "K"}  # Gift and other non-market transactions


def add_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Log1p-transform skewed numeric columns."""
    df["log_shares"] = np.log1p(df["shares"].fillna(0))
    df["log_total_value"] = np.log1p(df["total_value"].fillna(0))
    df["log_shares_owned_after"] = np.log1p(df["shares_owned_after"].fillna(0))
    # Flag rows where value is missing vs genuinely zero — M/A-code RSU vestings
    # have no filed price; the model needs to distinguish these from real zero-value trades.
    df["total_value_is_imputed"] = (
        (df["total_value"].fillna(0) == 0)
        & (df["transaction_code"].str.strip().str.upper().isin({"M", "A"}))
    ).astype(int)
    return df


def add_trade_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode transaction direction as an integer:
      +1 = buy/acquire  (A, M)
      -1 = sell/dispose (S, F)
       0 = other        (G, etc.)

    Also add boolean flags for downstream convenience.
    """
    code = df["transaction_code"].str.strip().str.upper()
    df["trade_direction"] = np.select(
        [code.isin(BUYS), code.isin(SELLS)],
        [1, -1],
        default=0,
    )
    df["is_buy"] = df["trade_direction"] == 1
    df["is_sell"] = df["trade_direction"] == -1
    df["is_open_market_buy"] = df["is_buy"] & df["is_open_market"]
    df["is_open_market_sell"] = df["is_sell"] & df["is_open_market"]
    return df


def add_signed_value(df: pd.DataFrame) -> pd.DataFrame:
    """Net dollar value: positive for buys, negative for sells."""
    df["signed_value"] = df["total_value"].fillna(0) * df["trade_direction"]
    return df


def add_days_to_filing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar days between the transaction and the SEC filing.
    Unusually long lags can signal delayed disclosure.
    """
    t = pd.to_datetime(df["transaction_date"], errors="coerce")
    f = pd.to_datetime(df["filing_date"], errors="coerce")
    df["days_to_filing"] = (f - t).dt.days.clip(lower=0)
    return df


def add_ownership_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate how much of the insider's known position was traded.
    Uses shares / (shares + shares_owned_after) so that full liquidations
    (shares_owned_after = 0) correctly return 1.0 instead of 0.0.
    """
    after = df["shares_owned_after"].fillna(0)
    traded = df["shares"].fillna(0)
    total_before = traded + after
    with np.errstate(divide="ignore", invalid="ignore"):
        pct = np.where(total_before > 0, traded / total_before, 0.0)
    df["pct_position_traded"] = np.clip(pct, 0, 1)
    return df


def add_value_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorical buckets for trade size (open-market only).
    Thresholds are rough market-practice tiers.
    """
    val = df["total_value"].fillna(0)
    bucket_labels = ["tiny", "small", "medium", "large", "mega"]
    df["value_bucket"] = pd.cut(
        val,
        bins=[0, 1e5, 5e5, 2e6, 1e7, np.inf],
        labels=bucket_labels,
        include_lowest=True,
    ).astype(str)
    # Numeric ordinal (0-4) for ML models that need continuous inputs
    bucket_order = {label: idx for idx, label in enumerate(bucket_labels)}
    df["value_bucket_num"] = df["value_bucket"].map(bucket_order).fillna(0).astype(int)
    return df


def add_filing_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week and month from transaction date — timing can matter."""
    t = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["txn_day_of_week"] = t.dt.dayofweek        # 0=Mon … 6=Sun
    df["txn_month"] = t.dt.month
    df["txn_quarter"] = t.dt.quarter
    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all trade-level feature functions in sequence."""
    df = add_log_transforms(df)
    df = add_trade_direction(df)
    df = add_signed_value(df)
    df = add_days_to_filing(df)
    df = add_ownership_metrics(df)
    df = add_value_bucket(df)
    df = add_filing_day_features(df)
    return df
