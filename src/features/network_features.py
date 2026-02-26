"""
Network coordination features — detect clusters of insiders trading together.

For each transaction we look at a ±72-hour window and count how many OTHER
insiders at the same company traded, and whether they traded in the same
direction.  Coordinated buying/selling by multiple insiders is one of the
strongest signals of informed trading.

Note: With O(n) insiders and an event-based dataset (≤ a few thousand rows
per company), the inner join approach below is fast enough.
"""

import pandas as pd


WINDOW_HOURS = 72


def _count_cotraders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Self-join the frame on ticker + date proximity, then aggregate.

    Only open-market trades count as co-traders.  Awards and option exercises
    cluster around vesting dates and would create spurious coordination signals
    that have nothing to do with informed trading.

    Returns a DataFrame indexed like df with coordination stats.
    """
    df = df.copy()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    window = pd.Timedelta(hours=WINDOW_HOURS)

    rows = []

    # Vectorised inner: cross-join via numpy broadcasting works for ~1 k rows
    dates = df["transaction_date"].values
    insiders = df["insider_cik"].values
    tickers = df["ticker"].values
    directions = df["trade_direction"].values
    # Only count open-market trades as co-traders — awards/exercises happen on
    # vesting schedules and create spurious coordination signals.
    is_open_mkt = df["is_open_market"].values.astype(bool)

    for i in range(len(df)):
        # Peers = open-market trades within the window, same company, different insider
        time_mask = abs(dates - dates[i]) <= window.to_timedelta64()
        ticker_mask = tickers == tickers[i]
        other_mask = insiders != insiders[i]
        peers = time_mask & ticker_mask & other_mask & is_open_mkt

        total_peers = int(peers.sum())
        same_dir = int(
            (peers & (directions == directions[i]) & (directions[i] != 0)).sum()
        )
        opp_dir = int(
            (peers & (directions == -directions[i]) & (directions[i] != 0)).sum()
        )

        rows.append(
            {
                "other_insiders_72h": total_peers,
                "same_dir_insiders_72h": same_dir,
                "opp_dir_insiders_72h": opp_dir,
            }
        )

    stats = pd.DataFrame(rows, index=df.index)
    return stats


def add_coordination_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute coordination metrics and merge them onto the frame.
    """
    stats = _count_cotraders(df)
    df = df.join(stats)

    total = df["other_insiders_72h"].replace(0, 1)  # avoid div/0
    df["coordination_score"] = df["same_dir_insiders_72h"] / total
    df["cluster_flag"] = (df["same_dir_insiders_72h"] >= 2).astype(int)
    return df


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all network coordination features."""
    df = add_coordination_features(df)
    return df
