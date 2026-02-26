"""
Temporal features — rolling-window statistics per insider.

For each trade, we look BACK in time (excluding the current row) to
capture the insider's recent activity pattern.  Three look-back windows
are computed: 7 d, 30 d, and 90 d.

Requires: trade_features.build() already run so 'signed_value' exists.
"""

import warnings

import pandas as pd


WINDOWS = [7, 30, 90]  # calendar days


def _rolling_stats(group: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling stats for a single insider's sorted trade history.
    `closed='left'` means the current row is NOT included in its own window.
    """
    group = group.sort_values("transaction_date").copy()
    group = group.set_index("transaction_date")

    is_buy = (group["trade_direction"] == 1).astype(float)
    is_sell = (group["trade_direction"] == -1).astype(float)
    value = group["signed_value"].fillna(0)

    for days in WINDOWS:
        w = f"{days}D"
        roll_kwargs = dict(window=w, min_periods=0, closed="left")

        group[f"trades_{days}d"] = is_buy.rolling(**roll_kwargs).count() + \
                                    is_sell.rolling(**roll_kwargs).count()
        group[f"buy_count_{days}d"] = is_buy.rolling(**roll_kwargs).sum()
        group[f"sell_count_{days}d"] = is_sell.rolling(**roll_kwargs).sum()
        group[f"net_value_{days}d"] = value.rolling(**roll_kwargs).sum()

    return group.reset_index()


def add_rolling_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-insider rolling windows and merge back into the full frame."""
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = (
            df.groupby("insider_cik", group_keys=False)
            .apply(_rolling_stats)
            .reset_index(drop=True)
        )

    # Ensure the original row order is preserved
    result = result.sort_values(["transaction_date", "insider_cik"]).reset_index(drop=True)
    return result


def add_days_since_last_trade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calendar days since the same insider last filed any transaction.
    First trade per insider gets NaN → filled with 0.
    """
    df = df.sort_values(["insider_cik", "transaction_date"]).copy()
    df["days_since_last_trade"] = (
        df.groupby("insider_cik")["transaction_date"]
        .diff()
        .dt.days
        .fillna(0)
    )
    return df


def add_consecutive_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    How many consecutive trades (for this insider) have been in the same
    direction as the current trade?  Useful for detecting accumulation /
    distribution patterns.
    """
    df = df.sort_values(["insider_cik", "transaction_date"]).copy()

    streaks = []
    for _, group in df.groupby("insider_cik"):
        group = group.sort_values("transaction_date")
        dirs = group["trade_direction"].tolist()
        streak = [1]
        for i in range(1, len(dirs)):
            if dirs[i] == dirs[i - 1] and dirs[i] != 0:
                streak.append(streak[-1] + 1)
            else:
                streak.append(1)
        streaks.extend(zip(group.index, streak))

    streak_series = pd.Series(
        {idx: val for idx, val in streaks}, name="consecutive_direction"
    )
    df["consecutive_direction"] = df.index.map(streak_series).fillna(1).astype(int)
    return df


def add_trade_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trades per calendar day over the insider's 90-day rolling window.
    Higher frequency → more active (potentially more informed) insider.
    """
    df["trade_frequency_90d"] = df["trades_90d"] / 90.0
    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all temporal feature functions in sequence."""
    df = add_rolling_windows(df)
    df = add_days_since_last_trade(df)
    df = add_consecutive_direction(df)
    df = add_trade_frequency(df)
    return df
