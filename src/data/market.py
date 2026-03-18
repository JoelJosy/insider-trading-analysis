from datetime import timedelta
from io import StringIO
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/market.log")

_LAST_YF_REQUEST_TS = 0.0
_MIN_REQUEST_GAP_SEC = 1.2


def _throttle_yf_calls() -> None:
    global _LAST_YF_REQUEST_TS
    now = time.time()
    wait = _MIN_REQUEST_GAP_SEC - (now - _LAST_YF_REQUEST_TS)
    if wait > 0:
        time.sleep(wait)
    _LAST_YF_REQUEST_TS = time.time()


def _as_close_series(data: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(data, pd.Series):
        s = data
    elif isinstance(data, pd.DataFrame):
        if data.empty:
            return pd.Series(dtype=float)
        if "close" in data.columns:
            s = data["close"]
        elif "Close" in data.columns:
            s = data["Close"]
        elif data.shape[1] == 1:
            s = data.iloc[:, 0]
        else:
            return pd.Series(dtype=float)
    else:
        return pd.Series(dtype=float)

    s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    s.index = pd.to_datetime(s.index, errors="coerce").normalize()
    s = s[~s.index.isna()]
    return s.sort_index()


def _extract_close_from_history(history: pd.DataFrame) -> pd.Series:
    if history.empty:
        return pd.Series(dtype=float)

    if "Close" in history.columns:
        close = history["Close"]
    elif isinstance(history.columns, pd.MultiIndex) and "Close" in history.columns.get_level_values(0):
        close = history["Close"].iloc[:, 0]
    else:
        return pd.Series(dtype=float)

    return _as_close_series(close)


def _download_via_download(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    _throttle_yf_calls()
    history = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    return _extract_close_from_history(history)


def _download_via_ticker_history(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    _throttle_yf_calls()
    tk = yf.Ticker(ticker)
    history = tk.history(
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
    )
    return _extract_close_from_history(history)


def _download_via_stooq(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    symbol = f"{ticker.lower()}.us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    _throttle_yf_calls()
    response = requests.get(url, timeout=20)
    response.raise_for_status()

    if not response.text.strip() or response.text.strip().startswith("No data"):
        return pd.Series(dtype=float)

    df = pd.read_csv(
        StringIO(response.text),
        parse_dates=["Date"],
    )
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)

    df = df.rename(columns={"Date": "date", "Close": "close"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["date", "close"])
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    if df.empty:
        return pd.Series(dtype=float)

    close = df.set_index("date")["close"].astype(float).sort_index()
    return close


def _cache_path(cache_dir: str, ticker: str) -> Path:
    return Path(cache_dir) / f"{ticker.upper()}_daily_prices.csv"


def _read_cache(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float)

    cached = pd.read_csv(path, parse_dates=["date"])
    if cached.empty:
        return pd.Series(dtype=float)

    series = cached.set_index("date")["close"].sort_index()
    series.index = pd.to_datetime(series.index).normalize()
    return series


def _write_cache(path: Path, prices: pd.Series) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _as_close_series(prices)
    out = clean.rename("close").reset_index()
    out.columns = ["date", "close"]
    out.to_csv(path, index=False)


def _download_close_prices(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    methods = (
        ("download", _download_via_download),
        ("ticker.history", _download_via_ticker_history),
        ("stooq", _download_via_stooq),
    )
    for method_name, method in methods:
        for attempt in range(1, 4):
            try:
                close = method(ticker, start, end)
                if not close.empty:
                    if attempt > 1:
                        logger.info("Recovered market data for %s via %s on attempt %d", ticker, method_name, attempt)
                    return close
            except Exception as e:
                logger.warning("%s failed for %s (attempt %d/3): %s", method_name, ticker, attempt, e)
            backoff = 3 * attempt
            logger.info("Backing off %.1fs before retry for %s", backoff, ticker)
            time.sleep(backoff)
    return pd.Series(dtype=float)


def get_close_prices(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: str = "data/external/prices",
) -> pd.Series:
    path = _cache_path(cache_dir, ticker)
    cached = _read_cache(path)

    # Clamp end to today: future prices don't exist, so never attempt to fetch them.
    today = pd.Timestamp.today().normalize()
    effective_end = min(end, today)

    # Allow up to 5 calendar days of cache lag (covers weekends + holidays)
    # before triggering a network fetch on the upper edge.
    _LAG_TOLERANCE = pd.Timedelta(days=5)
    need_fetch = (
        cached.empty
        or start < cached.index.min()
        or effective_end > cached.index.max() + _LAG_TOLERANCE
    )
    if need_fetch:
        fetch_start = start if cached.empty else min(start, cached.index.min())
        fetch_end = effective_end if cached.empty else max(effective_end, cached.index.max())
        fresh = _download_close_prices(ticker, fetch_start, fetch_end)
        if fresh.empty and cached.empty:
            logger.warning("No market prices returned for %s", ticker)
            return pd.Series(dtype=float)

        parts = [_as_close_series(s) for s in (cached, fresh) if not s.empty]
        combined = pd.concat(parts).sort_index() if parts else pd.Series(dtype=float)
        combined = combined[~combined.index.duplicated(keep="last")]
        _write_cache(path, combined)
        cached = combined

    return cached.loc[(cached.index >= start) & (cached.index <= end)]


def add_close_price_on_txn_date(df: pd.DataFrame, cache_dir: str = "data/external/prices") -> pd.DataFrame:
    df = df.copy()
    if "transaction_date" not in df.columns or "ticker" not in df.columns:
        df["close_price_on_txn_date"] = np.nan
        return df

    txn_ts = pd.to_datetime(df["transaction_date"], errors="coerce").dt.normalize()
    df["close_price_on_txn_date"] = np.nan

    for ticker in df["ticker"].dropna().astype(str).str.upper().unique():
        ticker_mask = df["ticker"].astype(str).str.upper() == ticker
        ticker_dates = txn_ts[ticker_mask].dropna()
        if ticker_dates.empty:
            continue

        start, end = ticker_dates.min(), ticker_dates.max()
        close_prices = get_close_prices(ticker, start, end, cache_dir=cache_dir)
        if close_prices.empty:
            continue

        # Fill non-trading days with previous market close.
        calendar = pd.date_range(start=start, end=end, freq="D")
        daily = close_prices.reindex(calendar).ffill()

        mapped = txn_ts[ticker_mask].map(daily)
        df.loc[ticker_mask, "close_price_on_txn_date"] = mapped.values

    return df


def backfill_m_code_total_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "transaction_code" not in df.columns:
        return df

    code = df["transaction_code"].astype(str).str.strip().str.upper()
    mask = (code == "M") & (df["total_value"].fillna(0) == 0)
    df.loc[mask, "total_value"] = (
        df.loc[mask, "shares"] * df.loc[mask, "close_price_on_txn_date"]
    )

    logger.info("Backfilled total_value for %d M-code rows using close prices", int(mask.sum()))
    return df


def enrich_with_market_prices(df: pd.DataFrame, cache_dir: str = "data/external/prices") -> pd.DataFrame:
    if df.empty:
        return df
    df = add_close_price_on_txn_date(df, cache_dir=cache_dir)
    df = backfill_m_code_total_value(df)
    return df
