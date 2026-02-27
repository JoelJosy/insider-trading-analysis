"""
External event ingestion for Phase 3 labeling.

Builds:
- data/external/earnings/earnings_announcements.csv
- data/external/sec/sec_enforcement.csv
"""

from __future__ import annotations

import argparse
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

from src.utils.config import get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/external_events.log")

_ENFORCEMENT_KEYWORDS = (
    "litigation",
    "charges",
    "charged",
    "settlement",
    "settled",
    "fraud",
    "insider trading",
    "enforcement",
)

_KNOWN_COMPANY_ALIASES: dict[str, list[str]] = {
    "AAPL": ["apple", "apple inc", "apple incorporated"],
    "MSFT": ["microsoft", "microsoft corp", "microsoft corporation"],
    "GOOGL": ["alphabet", "alphabet inc", "google", "google llc"],
    "AMZN": ["amazon", "amazon.com", "amazon com", "amazon.com inc"],
    "META": ["meta", "meta platforms", "facebook", "facebook inc"],
    "NVDA": ["nvidia", "nvidia corp", "nvidia corporation"],
    "TSLA": ["tesla", "tesla inc", "tesla motors"],
    "JPM": ["jpmorgan", "jp morgan", "jpmorgan chase", "jpmorgan chase & co"],
    "JNJ": ["johnson & johnson", "johnson and johnson"],
    "V": ["visa", "visa inc"],
}


def _normalize_tickers(tickers: list[str]) -> list[str]:
    return sorted({t.strip().upper() for t in tickers if t and t.strip()})


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


def _load_aliases_csv(path: str, tickers: list[str]) -> dict[str, set[str]]:
    p = Path(path)
    aliases = {t: set() for t in tickers}
    if not p.exists():
        return aliases

    df = pd.read_csv(p)
    if "ticker" not in df.columns or "alias" not in df.columns:
        logger.warning("Alias CSV missing required columns ticker,alias: %s", path)
        return aliases

    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        alias = _normalize_text(str(row.get("alias", "")).strip())
        if ticker in aliases and alias:
            aliases[ticker].add(alias)
    return aliases


def _fetch_company_aliases_from_yf(ticker: str) -> set[str]:
    alias_set: set[str] = set()
    for attempt in range(1, 3):
        try:
            tk = yf.Ticker(ticker)
            info = tk.info if isinstance(tk.info, dict) else {}
            for key in ("shortName", "longName"):
                v = info.get(key)
                if v:
                    alias_set.add(_normalize_text(str(v)))
            break
        except Exception as e:
            logger.warning("alias fetch failed for %s (attempt %d/2): %s", ticker, attempt, e)
            time.sleep(attempt)
    return {a for a in alias_set if a}


def _build_alias_map(tickers: list[str], alias_csv_path: str | None = None) -> dict[str, set[str]]:
    alias_map = {t: set() for t in tickers}
    custom = _load_aliases_csv(alias_csv_path, tickers) if alias_csv_path else {t: set() for t in tickers}

    for ticker in tickers:
        alias_map[ticker].add(_normalize_text(ticker))
        for a in _KNOWN_COMPANY_ALIASES.get(ticker, []):
            alias_map[ticker].add(_normalize_text(a))
        alias_map[ticker].update(custom.get(ticker, set()))
        alias_map[ticker].update(_fetch_company_aliases_from_yf(ticker))

        # prune tiny aliases that are likely noisy
        alias_map[ticker] = {a for a in alias_map[ticker] if len(a) >= 3}

    return alias_map


def _tickers_from_group(group: str) -> list[str]:
    cfg = get_config()
    if group not in cfg.tickers:
        raise ValueError(f"Unknown ticker group: {group}")
    return _normalize_tickers(cfg.tickers[group])


def _read_existing(path: str, required_cols: list[str]) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=required_cols)
    df = pd.read_csv(p)
    for c in required_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[required_cols]


def _save_deduped(path: str, df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in key_cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
    out = out.drop_duplicates(subset=key_cols, keep="last")
    out = out.sort_values(key_cols).reset_index(drop=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _fetch_earnings_for_ticker(ticker: str, max_rows: int = 40) -> list[dict]:
    for attempt in range(1, 4):
        try:
            tk = yf.Ticker(ticker)
            edf = tk.get_earnings_dates(limit=max_rows)
            if edf is None or edf.empty:
                return []

            idx = pd.to_datetime(edf.index, errors="coerce").tz_localize(None).normalize()
            idx = idx.dropna()
            return [{"ticker": ticker, "announcement_date": d.date().isoformat()} for d in idx.unique()]
        except Exception as e:
            logger.warning("earnings fetch failed for %s (attempt %d/3): %s", ticker, attempt, e)
            time.sleep(2 * attempt)
    return []


def build_earnings_csv(
    tickers: list[str],
    out_path: str = "data/external/earnings/earnings_announcements.csv",
    max_rows_per_ticker: int = 40,
) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        rows.extend(_fetch_earnings_for_ticker(ticker, max_rows=max_rows_per_ticker))

    existing = _read_existing(out_path, ["ticker", "announcement_date"])
    incoming = pd.DataFrame(rows, columns=["ticker", "announcement_date"])
    combined = pd.concat([existing, incoming], ignore_index=True)
    if not combined.empty:
        combined["ticker"] = combined["ticker"].astype(str).str.upper()
        combined["announcement_date"] = pd.to_datetime(combined["announcement_date"], errors="coerce").dt.date.astype(str)
        combined = combined[combined["announcement_date"] != "NaT"]

    final_df = _save_deduped(out_path, combined, ["ticker", "announcement_date"])
    logger.info("Saved earnings announcements -> %s (%d rows)", out_path, len(final_df))
    return final_df


def _match_tickers(text: str, tickers: list[str], alias_map: dict[str, set[str]]) -> list[tuple[str, str]]:
    if not text:
        return []

    found: list[tuple[str, str]] = []
    upper = text.upper()
    normalized = f" {_normalize_text(text)} "

    for ticker in tickers:
        if re.search(rf"\b{re.escape(ticker)}\b", upper):
            found.append((ticker, "ticker"))
            continue

        aliases = alias_map.get(ticker, set())
        for alias in aliases:
            if len(alias) < 3:
                continue
            if f" {alias} " in normalized:
                found.append((ticker, f"alias:{alias}"))
                break

    # unique by ticker (prefer ticker hit over alias)
    unique: dict[str, str] = {}
    for ticker, method in found:
        if ticker not in unique or unique[ticker].startswith("alias:"):
            unique[ticker] = method
    return [(ticker, method) for ticker, method in unique.items()]


def build_enforcement_csv(
    tickers: list[str],
    out_path: str = "data/external/sec/sec_enforcement.csv",
    rss_url: str = "https://www.sec.gov/rss/litigation/litreleases.xml",
    alias_csv_path: str | None = "data/external/sec/company_aliases.csv",
) -> pd.DataFrame:
    headers = {
        "User-Agent": get_config().sec_edgar.user_agent,
        "Accept": "application/rss+xml, application/xml, text/xml",
    }
    candidate_urls = [
        rss_url,
        "https://www.sec.gov/news/pressreleases.rss",
    ]

    response = None
    for url in candidate_urls:
        try:
            r = requests.get(url, headers=headers, timeout=30)
            if r.status_code == 200:
                response = r
                logger.info("Using SEC feed: %s", url)
                break
            logger.warning("SEC feed unavailable (%s): HTTP %s", url, r.status_code)
        except Exception as e:
            logger.warning("SEC feed request failed (%s): %s", url, e)

    if response is None:
        logger.warning("No SEC feed reachable; keeping existing enforcement CSV unchanged")
        existing = _read_existing(out_path, ["ticker", "action_date", "source_title", "source_link"])
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        existing.to_csv(out_path, index=False)
        return existing

    root = ET.fromstring(response.content)
    items = root.findall("./channel/item")
    rows: list[dict] = []
    alias_map = _build_alias_map(tickers, alias_csv_path=alias_csv_path)

    for item in items:
        title = (item.findtext("title") or "").strip()
        desc = (item.findtext("description") or "").strip()
        pub_date = (item.findtext("pubDate") or "").strip()
        link = (item.findtext("link") or "").strip()
        text = f"{title} {desc}"

        if not any(k in text.lower() for k in _ENFORCEMENT_KEYWORDS):
            continue

        matched = _match_tickers(text, tickers, alias_map=alias_map)
        if not matched:
            continue

        parsed = pd.to_datetime(pub_date, errors="coerce", utc=True)
        if pd.isna(parsed):
            continue
        action_date = parsed.tz_convert(None).date().isoformat()

        for ticker, matched_by in matched:
            rows.append(
                {
                    "ticker": ticker,
                    "action_date": action_date,
                    "source_title": title,
                    "source_link": link,
                    "matched_by": matched_by,
                }
            )

    existing = _read_existing(out_path, ["ticker", "action_date", "source_title", "source_link", "matched_by"])
    incoming = pd.DataFrame(rows, columns=["ticker", "action_date", "source_title", "source_link", "matched_by"])
    combined = pd.concat([existing, incoming], ignore_index=True)
    if not combined.empty:
        combined["ticker"] = combined["ticker"].astype(str).str.upper()
        combined["action_date"] = pd.to_datetime(combined["action_date"], errors="coerce").dt.date.astype(str)
        combined = combined[combined["action_date"] != "NaT"]

    final_df = _save_deduped(out_path, combined, ["ticker", "action_date", "source_link"])
    logger.info("Saved SEC enforcement events -> %s (%d rows)", out_path, len(final_df))
    return final_df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build external event CSVs for Phase 3 labeling")
    p.add_argument("--tickers", default="", help="Comma-separated tickers (e.g. AAPL,MSFT)")
    p.add_argument("--ticker-group", default="proof_of_concept", help="Config ticker group name")
    p.add_argument("--skip-earnings", action="store_true", help="Skip earnings ingestion")
    p.add_argument("--skip-enforcement", action="store_true", help="Skip SEC enforcement ingestion")
    p.add_argument("--max-earnings-rows", type=int, default=40)
    p.add_argument("--earnings-out", default="data/external/earnings/earnings_announcements.csv")
    p.add_argument("--enforcement-out", default="data/external/sec/sec_enforcement.csv")
    p.add_argument(
        "--alias-csv",
        default="data/external/sec/company_aliases.csv",
        help="Optional CSV with columns: ticker,alias for enforcement matching",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    from_cli = _normalize_tickers(args.tickers.split(",")) if args.tickers else []
    tickers = from_cli or _tickers_from_group(args.ticker_group)
    if not tickers:
        raise ValueError("No tickers provided. Use --tickers or a non-empty --ticker-group")

    logger.info("External events ingestion start for %d tickers", len(tickers))
    print(f"Tickers: {', '.join(tickers)}")

    if not args.skip_earnings:
        edf = build_earnings_csv(
            tickers=tickers,
            out_path=args.earnings_out,
            max_rows_per_ticker=args.max_earnings_rows,
        )
        print(f"Earnings rows: {len(edf):,} -> {args.earnings_out}")

    if not args.skip_enforcement:
        sdf = build_enforcement_csv(
            tickers=tickers,
            out_path=args.enforcement_out,
            alias_csv_path=args.alias_csv,
        )
        print(f"Enforcement rows: {len(sdf):,} -> {args.enforcement_out}")

    logger.info("External events ingestion complete")
