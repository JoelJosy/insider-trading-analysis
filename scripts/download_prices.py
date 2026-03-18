"""
download_prices.py
------------------
Downloads daily close prices for all tickers.

Default provider is Stooq (no Yahoo/yfinance dependency).
Yahoo chart API remains optional fallback if needed.

Saves to data/external/prices/<TICKER>_daily_prices.csv
in the exact format expected by src/data/market.py cache reader:
    date, close

Usage:
    python scripts/download_prices.py
    python scripts/download_prices.py --tickers SPY AAPL PFE
    python scripts/download_prices.py --start 2015-01-01 --end 2024-12-31
"""

import argparse
from io import StringIO
import time
from pathlib import Path

import pandas as pd
import requests

# ── Default settings ──────────────────────────────────────────────────────────

# DEFAULT_TICKERS = [
#     "SPY",   # benchmark — always needed
#     "AAPL", "PFE", "ABBV", "JPM", "LLY",
#     "MRK",  "XOM", "COP", "BAC", "WMT",
#     "BIIB", "REGN", "MRNA", "META",
# ]

DEFAULT_TICKERS = [
    "BMY", "AMGN", "GS", "BA", "LMT", "MS", "WFC",
]

DEFAULT_START = "2015-01-01"
DEFAULT_END   = "2024-12-31"
CACHE_DIR     = Path("data/external/prices")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# ── Core download function ────────────────────────────────────────────────────

def download_ticker_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily OHLCV from Yahoo Finance chart API and return
    a DataFrame with columns [date, close].
    """
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts   = int(pd.Timestamp(end).timestamp())

    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?interval=1d&period1={start_ts}&period2={end_ts}"
    )

    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()

    data = r.json()

    # Check for API-level errors
    chart = data.get("chart", {})
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo API error for {ticker}: {error}")

    results = chart.get("result")
    if not results:
        raise ValueError(f"No data returned for {ticker}")

    result     = results[0]
    timestamps = result.get("timestamp", [])
    closes     = result["indicators"]["quote"][0].get("close", [])

    if not timestamps or not closes:
        raise ValueError(f"Empty price data for {ticker}")

    df = pd.DataFrame({
        "date":  pd.to_datetime(timestamps, unit="s").normalize(),
        "close": closes,
    }).dropna(subset=["close"])

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def download_ticker_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()

    symbols = [f"{ticker.lower()}.us", ticker.lower()]
    for symbol in symbols:
        url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
        r = requests.get(url, timeout=30)
        r.raise_for_status()

        text = r.text.strip()
        if not text or text.startswith("No data"):
            continue

        df = pd.read_csv(StringIO(text), parse_dates=["Date"])
        if df.empty or "Close" not in df.columns:
            continue

        df = df.rename(columns={"Date": "date", "Close": "close"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df = df.dropna(subset=["date", "close"])
        df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
        if df.empty:
            continue

        return df[["date", "close"]].sort_values("date").reset_index(drop=True)

    raise ValueError(f"No data returned for {ticker} via Stooq")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download price cache for all project tickers."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="Space-separated list of tickers (default: all project tickers)"
    )
    parser.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Start date (default: {DEFAULT_START})"
    )
    parser.add_argument(
        "--end", default=DEFAULT_END,
        help=f"End date (default: {DEFAULT_END})"
    )
    parser.add_argument(
        "--cache-dir", default=str(CACHE_DIR),
        help=f"Cache directory (default: {CACHE_DIR})"
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Seconds to wait between requests (default: 0.5)"
    )
    parser.add_argument(
        "--provider", choices=["stooq", "yahoo"], default="stooq",
        help="Price provider to use (default: stooq)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip tickers that already have a cache file"
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tickers  = [t.upper() for t in args.tickers]
    success  = []
    failed   = []

    print(f"Downloading {len(tickers)} tickers: {args.start} → {args.end}")
    print(f"Cache dir: {cache_dir}")
    print(f"Provider: {args.provider}")
    print("-" * 50)

    for ticker in tickers:
        path = cache_dir / f"{ticker}_daily_prices.csv"

        if args.skip_existing and path.exists():
            existing = pd.read_csv(path)
            print(f"  {ticker:6s}  SKIPPED  ({len(existing)} rows already cached)")
            success.append(ticker)
            continue

        print(f"  {ticker:6s}  downloading...", end="  ", flush=True)

        try:
            if args.provider == "stooq":
                df = download_ticker_stooq(ticker, args.start, args.end)
            else:
                df = download_ticker_yahoo(ticker, args.start, args.end)
            df.to_csv(path, index=False)
            print(f"saved {len(df):,} rows  →  {path.name}")
            success.append(ticker)

        except Exception as e:
            print(f"FAILED  —  {e}")
            failed.append((ticker, str(e)))

        time.sleep(args.delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"Done.  {len(success)}/{len(tickers)} tickers downloaded successfully.")

    if failed:
        print(f"\nFailed tickers ({len(failed)}):")
        for ticker, reason in failed:
            print(f"  {ticker}: {reason}")
        print("\nRetry failed tickers with:")
        retry_list = " ".join(t for t, _ in failed)
        print(f"  python scripts/download_prices.py --tickers {retry_list}")
    else:
        print("All tickers cached. Labeling pipeline will now read from cache.")



if __name__ == "__main__":
    main()