"""
Phase 3 labeling pipeline.

Builds ground-truth labels from:
1) Directional abnormal return vs benchmark (required)
2) Earnings-event proximity (optional)
3) SEC enforcement follow-up flag (optional)

Usage
-----
python -m src.labels.pipeline \
  --input data/processed/AAPL_form4_features.csv \
  --output data/processed/AAPL_form4_labeled.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.market import get_close_prices
from src.labels.quality import build_label_quality_summary, save_label_quality_report
from src.utils.config import get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/labels.log")


def _load_input(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    if "transaction_date" not in df.columns or "ticker" not in df.columns:
        raise ValueError("Input must contain 'transaction_date' and 'ticker' columns")
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce").dt.normalize()
    return df


def _normalize_code(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _trade_direction(df: pd.DataFrame) -> pd.Series:
    if "trade_direction" in df.columns:
        return pd.to_numeric(df["trade_direction"], errors="coerce").fillna(0).astype(int)

    code = _normalize_code(df.get("transaction_code", pd.Series(index=df.index, dtype=str)))
    buys = code.isin({"A", "M", "P", "L"})
    sells = code.isin({"S", "F", "D", "U"})
    return pd.Series(np.select([buys, sells], [1, -1], default=0), index=df.index)


def _map_price_on_dates(series: pd.Series, dates: pd.Series, method: str) -> pd.Series:
    if series.empty:
        return pd.Series(np.nan, index=dates.index)

    idx = pd.to_datetime(series.index).normalize()
    start = min(idx.min(), dates.min())
    end = max(idx.max(), dates.max())
    cal = pd.date_range(start=start, end=end, freq="D")

    expanded = series.reindex(cal)
    if method == "ffill":
        expanded = expanded.ffill()
    elif method == "bfill":
        expanded = expanded.bfill()
    else:
        raise ValueError("method must be 'ffill' or 'bfill'")

    return dates.map(expanded)


def _load_optional_dates(path: str | None, date_col: str, ticker_col: str = "ticker") -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=[ticker_col, date_col])
    p = Path(path)
    if not p.exists():
        logger.warning("Optional file not found: %s", path)
        return pd.DataFrame(columns=[ticker_col, date_col])

    df = pd.read_csv(p)
    if ticker_col not in df.columns or date_col not in df.columns:
        logger.warning("Optional file missing required columns '%s', '%s': %s", ticker_col, date_col, path)
        return pd.DataFrame(columns=[ticker_col, date_col])

    out = df[[ticker_col, date_col]].copy()
    out[ticker_col] = out[ticker_col].astype(str).str.upper()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    out = out.dropna(subset=[date_col])
    return out


def _nearest_day_diff(trade_dates: np.ndarray, event_dates: np.ndarray) -> np.ndarray:
    if len(event_dates) == 0:
        return np.full(len(trade_dates), np.inf)

    positions = np.searchsorted(event_dates, trade_dates)
    out = np.full(len(trade_dates), np.inf)

    left_idx = np.clip(positions - 1, 0, len(event_dates) - 1)
    right_idx = np.clip(positions, 0, len(event_dates) - 1)

    left_diff = np.abs((trade_dates - event_dates[left_idx]).astype("timedelta64[D]").astype(float))
    right_diff = np.abs((event_dates[right_idx] - trade_dates).astype("timedelta64[D]").astype(float))
    out = np.minimum(left_diff, right_diff)
    return out


def _has_future_event_within(trade_dates: np.ndarray, event_dates: np.ndarray, lookahead_days: int) -> np.ndarray:
    if len(event_dates) == 0:
        return np.zeros(len(trade_dates), dtype=bool)

    positions = np.searchsorted(event_dates, trade_dates)
    has_flag = np.zeros(len(trade_dates), dtype=bool)
    valid = positions < len(event_dates)
    if valid.any():
        next_events = event_dates[positions[valid]]
        days = (next_events - trade_dates[valid]).astype("timedelta64[D]").astype(int)
        has_flag[valid] = (days >= 0) & (days <= lookahead_days)
    return has_flag


def add_price_labels(
    df: pd.DataFrame,
    horizon_days: int,
    abnormal_threshold: float,
    benchmark_ticker: str,
    price_cache_dir: str,
    require_prices: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df["trade_direction"] = _trade_direction(df)

    txn_dates = pd.to_datetime(df["transaction_date"], errors="coerce").dt.normalize()
    target_dates = txn_dates + pd.to_timedelta(horizon_days, unit="D")

    df["txn_close"] = np.nan
    df["future_close"] = np.nan
    df["forward_return"] = np.nan

    min_date = txn_dates.min()
    max_target = target_dates.max()
    if pd.isna(min_date) or pd.isna(max_target):
        raise ValueError("transaction_date is empty or invalid")

    benchmark_prices = get_close_prices(
        benchmark_ticker,
        min_date - pd.Timedelta(days=10),
        max_target + pd.Timedelta(days=10),
        cache_dir=price_cache_dir,
    )
    if benchmark_prices.empty and require_prices:
        raise RuntimeError(
            f"Benchmark price download failed for {benchmark_ticker}. "
            "Check internet/proxy/VPN or run with --allow-missing-prices to continue."
        )

    bench_txn = _map_price_on_dates(benchmark_prices, txn_dates, method="ffill")
    bench_future = _map_price_on_dates(benchmark_prices, target_dates, method="bfill")
    df["benchmark_return"] = ((bench_future - bench_txn) / bench_txn).replace([np.inf, -np.inf], np.nan)

    missing_tickers: list[str] = []
    for ticker in df["ticker"].dropna().astype(str).str.upper().unique():
        mask = df["ticker"].astype(str).str.upper() == ticker
        t_dates = txn_dates[mask]
        f_dates = target_dates[mask]

        prices = get_close_prices(
            ticker,
            t_dates.min() - pd.Timedelta(days=10),
            f_dates.max() + pd.Timedelta(days=10),
            cache_dir=price_cache_dir,
        )
        if prices.empty:
            missing_tickers.append(ticker)
            continue

        txn_close = _map_price_on_dates(prices, t_dates, method="ffill")
        fut_close = _map_price_on_dates(prices, f_dates, method="bfill")
        forward = ((fut_close - txn_close) / txn_close).replace([np.inf, -np.inf], np.nan)

        df.loc[mask, "txn_close"] = txn_close.values
        df.loc[mask, "future_close"] = fut_close.values
        df.loc[mask, "forward_return"] = forward.values

    df["abnormal_return"] = df["forward_return"] - df["benchmark_return"]
    df["signed_abnormal_return"] = df["abnormal_return"] * df["trade_direction"]
    df["price_signal"] = (df["signed_abnormal_return"] >= abnormal_threshold).astype(int)
    df["label_horizon_days"] = horizon_days

    coverage_pct = float(df["forward_return"].notna().mean() * 100)
    logger.info("Price coverage for forward returns: %.1f%%", coverage_pct)

    if missing_tickers:
        msg = f"No market prices returned for tickers: {', '.join(sorted(set(missing_tickers)))}"
        if require_prices:
            raise RuntimeError(msg)
        logger.warning(msg)

    if coverage_pct == 0 and require_prices:
        raise RuntimeError(
            "Forward-return coverage is 0%. Labeling would be misleading. "
            "Fix market data connectivity or run with --allow-missing-prices."
        )

    return df


def add_earnings_confirmation(df: pd.DataFrame, earnings_csv: str | None, proximity_days: int) -> pd.DataFrame:
    df = df.copy()
    earnings = _load_optional_dates(earnings_csv, date_col="announcement_date")
    df["earnings_proximity_flag"] = 0

    if earnings.empty:
        return df

    for ticker in df["ticker"].dropna().astype(str).str.upper().unique():
        trade_mask = df["ticker"].astype(str).str.upper() == ticker
        event_dates = earnings.loc[earnings["ticker"] == ticker, "announcement_date"].dropna().sort_values().values
        if len(event_dates) == 0:
            continue

        trade_dates = pd.to_datetime(df.loc[trade_mask, "transaction_date"], errors="coerce").values
        diffs = _nearest_day_diff(trade_dates, event_dates)
        df.loc[trade_mask, "earnings_proximity_flag"] = (diffs <= proximity_days).astype(int)

    return df


def add_enforcement_confirmation(df: pd.DataFrame, enforcement_csv: str | None, lookahead_days: int = 365) -> pd.DataFrame:
    df = df.copy()
    enforcement = _load_optional_dates(enforcement_csv, date_col="action_date")
    df["enforcement_followup_flag"] = 0

    if enforcement.empty:
        return df

    for ticker in df["ticker"].dropna().astype(str).str.upper().unique():
        trade_mask = df["ticker"].astype(str).str.upper() == ticker
        event_dates = enforcement.loc[enforcement["ticker"] == ticker, "action_date"].dropna().sort_values().values
        if len(event_dates) == 0:
            continue

        trade_dates = pd.to_datetime(df.loc[trade_mask, "transaction_date"], errors="coerce").values
        flags = _has_future_event_within(trade_dates, event_dates, lookahead_days=lookahead_days)
        df.loc[trade_mask, "enforcement_followup_flag"] = flags.astype(int)

    return df


def _plan_flag(df: pd.DataFrame) -> pd.Series:
    has_plan = pd.to_numeric(df.get("has_plan", 0), errors="coerce").fillna(0)
    footnote_has_plan = pd.to_numeric(df.get("footnote_has_plan", 0), errors="coerce").fillna(0)
    return ((has_plan == 1) | (footnote_has_plan == 1)).astype(int)


def _cohen_routine_mask(df: pd.DataFrame) -> pd.Series:
    insider = df.get("insider_cik", pd.Series(index=df.index, dtype=object)).astype(str)
    txn_dates = pd.to_datetime(df.get("transaction_date", pd.Series(index=df.index)), errors="coerce")

    months_df = pd.DataFrame(
        {
            "row_id": df.index,
            "insider_cik": insider,
            "year": txn_dates.dt.year,
            "month": txn_dates.dt.month,
        }
    )
    months_df = months_df.dropna(subset=["year", "month"]).copy()
    if months_df.empty:
        return pd.Series(False, index=df.index)

    months_df["year"] = months_df["year"].astype(int)
    months_df["month"] = months_df["month"].astype(int)

    # Unique insider-month-year combinations to evaluate historical streaks.
    uniq = months_df[["insider_cik", "year", "month"]].drop_duplicates().copy()
    uniq = uniq.sort_values(["insider_cik", "month", "year"])

    routine_pairs: set[tuple[str, int, int]] = set()
    for (insider_cik, month), group in uniq.groupby(["insider_cik", "month"], sort=False):
        years = group["year"].tolist()
        years_set = set(years)
        for year in years:
            prior_streak = 0
            prev = year - 1
            while prev in years_set:
                prior_streak += 1
                prev -= 1
            if prior_streak >= 2:
                routine_pairs.add((insider_cik, month, year))

    current_pairs = list(zip(months_df["insider_cik"], months_df["month"], months_df["year"]))
    months_df["cohen_routine"] = [pair in routine_pairs for pair in current_pairs]

    out = pd.Series(False, index=df.index)
    flagged = months_df.loc[months_df["cohen_routine"], "row_id"]
    out.loc[flagged] = True
    return out


def combine_signals(df: pd.DataFrame, confidence_sources_required: int) -> pd.DataFrame:
    """
    Assign behavioral proxy labels.

    Kept under the original function name for backward compatibility with
    existing imports (e.g., calibration module).
    """
    df = df.copy()
    for col in ("price_signal", "earnings_proximity_flag", "enforcement_followup_flag"):
        if col not in df.columns:
            df[col] = 0

    df["label_source_count"] = (
        df["price_signal"].fillna(0).astype(int)
        + df["earnings_proximity_flag"].fillna(0).astype(int)
        + df["enforcement_followup_flag"].fillna(0).astype(int)
    )

    code = _normalize_code(df.get("transaction_code", pd.Series(index=df.index, dtype=str)))
    open_market_mask = code.isin({"P", "S"})
    excluded_mask = code.isin({"M", "A", "F", "G", "D"})

    # Only open-market P/S trades receive Cohen labels.
    # All non-open-market trades are uncertain for supervised labels.
    df["cohen_label"] = -1
    if open_market_mask.any():
        cohen_routine_open = _cohen_routine_mask(df.loc[open_market_mask])
        df.loc[open_market_mask, "cohen_label"] = np.where(cohen_routine_open, 0, 1).astype(int)

    # 10b5-1 override applies only to open-market rows that are label-eligible.
    df["plan_override"] = 0
    if open_market_mask.any():
        plan_flags_open = _plan_flag(df.loc[open_market_mask]).astype(int)
        df.loc[open_market_mask, "plan_override"] = plan_flags_open.values

    df["final_label"] = -1
    if open_market_mask.any():
        df.loc[open_market_mask, "final_label"] = np.where(
            df.loc[open_market_mask, "plan_override"] == 1,
            0,
            df.loc[open_market_mask, "cohen_label"],
        ).astype(int)

    # Explicitly keep listed excluded transaction codes as uncertain.
    df.loc[excluded_mask, "cohen_label"] = -1
    df.loc[excluded_mask, "final_label"] = -1

    # Compatibility aliases for downstream code expecting prior naming.
    df["informed_label"] = (df["final_label"] == 1).astype(int)
    df["routine_label"] = (df["final_label"] == 0).astype(int)
    df["label_name"] = np.where(
        df["final_label"] == 1,
        "opportunistic",
        np.where(df["final_label"] == 0, "routine", "uncertain"),
    )

    # Confidence reflects behavioral-label certainty.
    df["label_confidence"] = np.where(
        df["final_label"] == -1,
        "uncertain",
        np.where(df["plan_override"] == 1, "high", "medium"),
    )
    return df


def save(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved labeled dataset -> %s  (%d rows x %d cols)", out_path, len(df), len(df.columns))


def build_labels(
    input_csv: str,
    output_csv: str,
    horizon_days: int,
    benchmark_ticker: str,
    abnormal_threshold: float,
    confidence_sources_required: int,
    earnings_proximity_days: int,
    earnings_csv: str | None,
    enforcement_csv: str | None,
    enforcement_lookahead_days: int,
    price_cache_dir: str = "data/external/prices",
    require_prices: bool = True,
    report_dir: str = "reports",
) -> pd.DataFrame:
    df = _load_input(input_csv)
    df = add_price_labels(
        df,
        horizon_days=horizon_days,
        abnormal_threshold=abnormal_threshold,
        benchmark_ticker=benchmark_ticker,
        price_cache_dir=price_cache_dir,
        require_prices=require_prices,
    )
    df = add_earnings_confirmation(df, earnings_csv=earnings_csv, proximity_days=earnings_proximity_days)
    df = add_enforcement_confirmation(
        df,
        enforcement_csv=enforcement_csv,
        lookahead_days=enforcement_lookahead_days,
    )
    df = combine_signals(df, confidence_sources_required=confidence_sources_required)

    stem = Path(input_csv).stem
    summary = build_label_quality_summary(
        df,
        input_csv=input_csv,
        params={
            "horizon_days": horizon_days,
            "benchmark_ticker": benchmark_ticker,
            "abnormal_threshold": abnormal_threshold,
            "confidence_sources_required": confidence_sources_required,
            "earnings_proximity_days": earnings_proximity_days,
            "earnings_csv": earnings_csv,
            "enforcement_csv": enforcement_csv,
            "enforcement_lookahead_days": enforcement_lookahead_days,
        },
    )
    report_json = str(Path(report_dir) / f"{stem}_label_quality_report.json")
    report_md = str(Path(report_dir) / f"{stem}_label_quality_report.md")
    save_label_quality_report(summary, report_json, report_md)
    logger.info("Saved label quality report -> %s", report_json)

    save(df, output_csv)
    return df


def _parse_args() -> argparse.Namespace:
    cfg = get_config().labeling
    p = argparse.ArgumentParser(description="Build Phase 3 labels (routine vs informed)")
    p.add_argument("--input", required=True, help="Input CSV (processed or features CSV)")
    p.add_argument("--output", default=None, help="Output labeled CSV path")
    p.add_argument("--horizon-days", type=int, default=30, help="Forward return horizon in days")
    p.add_argument("--benchmark", default="SPY", help="Benchmark ticker for abnormal return")
    p.add_argument(
        "--abnormal-threshold",
        type=float,
        default=cfg.abnormal_return_threshold,
        help="Directional abnormal-return threshold for price_signal",
    )
    p.add_argument(
        "--confidence-sources-required",
        type=int,
        default=cfg.confidence_sources_required,
        help="Minimum confirming sources to mark informed",
    )
    p.add_argument(
        "--earnings-proximity-days",
        type=int,
        default=cfg.earnings_proximity_days,
        help="Window (days) around earnings announcements",
    )
    p.add_argument(
        "--earnings-csv",
        default="data/external/earnings/earnings_announcements.csv",
        help="Optional earnings CSV with columns: ticker, announcement_date",
    )
    p.add_argument(
        "--enforcement-csv",
        default="data/external/sec/sec_enforcement.csv",
        help="Optional enforcement CSV with columns: ticker, action_date",
    )
    p.add_argument(
        "--enforcement-lookahead-days",
        type=int,
        default=365,
        help="Look-ahead window for enforcement follow-up flag",
    )
    p.add_argument(
        "--price-cache-dir",
        default="data/external/prices",
        help="Directory for cached market prices",
    )
    p.add_argument(
        "--allow-missing-prices",
        action="store_true",
        help="Continue labeling even when market prices are missing (not recommended)",
    )
    p.add_argument("--report-dir", default="reports", help="Directory for label quality reports")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    input_path = args.input
    output_path = args.output or str(Path(input_path).with_name(f"{Path(input_path).stem}_labeled.csv"))

    logger.info("=== Label Pipeline START ===")
    logger.info("Input : %s", input_path)
    logger.info("Output: %s", output_path)

    df = build_labels(
        input_csv=input_path,
        output_csv=output_path,
        horizon_days=args.horizon_days,
        benchmark_ticker=args.benchmark,
        abnormal_threshold=args.abnormal_threshold,
        confidence_sources_required=args.confidence_sources_required,
        earnings_proximity_days=args.earnings_proximity_days,
        earnings_csv=args.earnings_csv,
        enforcement_csv=args.enforcement_csv,
        enforcement_lookahead_days=args.enforcement_lookahead_days,
        price_cache_dir=args.price_cache_dir,
        require_prices=not args.allow_missing_prices,
        report_dir=args.report_dir,
    )

    print(f"\nDone. Labeled dataset -> {output_path}")
    print(f"Rows: {len(df):,}")
    print("Label distribution:")
    print(df["label_name"].value_counts(dropna=False).to_string())
    logger.info("=== Label Pipeline DONE ===")
