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

_EARNINGS_SEC_FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A"}

_KNOWN_TICKER_CIKS: dict[str, str] = {
    "AAPL": "0000320193",
    "ABBV": "0001551152",
    "AMGN": "0000318154",
    "BA": "0000012927",
    "BAC": "0000070858",
    "BIIB": "0000875045",
    "BMY": "0000014272",
    "COP": "0001163165",
    "GS": "0000886982",
    "JPM": "0000019617",
    "LLY": "0000059478",
    "LMT": "0000936468",
    "META": "0001326801",
    "MRK": "0000310158",
    "MRNA": "0001682852",
    "MS": "0000895421",
    "PFE": "0000078003",
    "REGN": "0000872589",
    "WFC": "0000072971",
    "WMT": "0000104169",
    "XOM": "0000034088",
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


def _build_alias_map(
    tickers: list[str],
    alias_csv_path: str | None = None,
    include_yf_aliases: bool = False,
) -> dict[str, set[str]]:
    alias_map = {t: set() for t in tickers}
    custom = _load_aliases_csv(alias_csv_path, tickers) if alias_csv_path else {t: set() for t in tickers}

    for ticker in tickers:
        alias_map[ticker].add(_normalize_text(ticker))
        for a in _KNOWN_COMPANY_ALIASES.get(ticker, []):
            alias_map[ticker].add(_normalize_text(a))
        alias_map[ticker].update(custom.get(ticker, set()))
        if include_yf_aliases:
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


def _sec_headers() -> dict[str, str]:
    configured_ua = str(get_config().sec_edgar.user_agent or "").strip()
    user_agent = configured_ua if "@" in configured_ua else "insider-trading-analysis/1.0 (research@local.invalid)"
    return {
        "User-Agent": user_agent,
        "Accept": "application/json,text/plain,*/*",
    }


def _fetch_sec_json(url: str, headers: dict[str, str], timeout: int = 30, retries: int = 3) -> dict:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            logger.warning("SEC request failed (%s): HTTP %s", url, r.status_code)
        except Exception as e:
            logger.warning("SEC request error (%s) attempt %d/%d: %s", url, attempt, retries, e)
        time.sleep(attempt)
    return {}


def _fetch_sec_ticker_cik_map(headers: dict[str, str]) -> dict[str, str]:
    payload = _fetch_sec_json("https://www.sec.gov/files/company_tickers.json", headers=headers)
    if not payload:
        return dict(_KNOWN_TICKER_CIKS)

    out: dict[str, str] = {}
    for _, v in payload.items():
        if not isinstance(v, dict):
            continue
        ticker = str(v.get("ticker", "")).strip().upper()
        cik_raw = v.get("cik_str")
        if not ticker or cik_raw is None:
            continue
        try:
            cik = str(int(cik_raw)).zfill(10)
        except Exception:
            continue
        out[ticker] = cik
    out.update(_KNOWN_TICKER_CIKS)
    return out


def _fetch_earnings_from_sec_filings(
    ticker: str,
    max_rows: int,
    headers: dict[str, str],
    ticker_cik_map: dict[str, str],
) -> list[dict]:
    cik = ticker_cik_map.get(ticker.upper())
    if not cik:
        return []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    payload = _fetch_sec_json(url, headers=headers)
    if not payload:
        return []

    recent = (payload.get("filings") or {}).get("recent") or {}
    forms = recent.get("form") or []
    dates = recent.get("filingDate") or []
    if not forms or not dates:
        return []

    rows: list[dict] = []
    for form, filing_date in zip(forms, dates):
        f = str(form or "").upper().strip()
        if f not in _EARNINGS_SEC_FORMS:
            continue
        d = pd.to_datetime(filing_date, errors="coerce")
        if pd.isna(d):
            continue
        rows.append({"ticker": ticker, "announcement_date": d.date().isoformat()})
        if len(rows) >= max_rows:
            break

    return rows


def build_earnings_csv(
    tickers: list[str],
    out_path: str = "data/external/earnings/earnings_announcements.csv",
    max_rows_per_ticker: int = 40,
    source: str = "auto",
) -> pd.DataFrame:
    source = (source or "auto").lower()
    if source not in {"auto", "yfinance", "sec"}:
        raise ValueError(f"Unknown earnings source: {source}")

    rows = []
    sec_headers = _sec_headers()
    sec_ticker_cik_map: dict[str, str] = {}
    sec_map_loaded = False

    def _get_sec_map() -> dict[str, str]:
        nonlocal sec_ticker_cik_map, sec_map_loaded
        if not sec_map_loaded:
            sec_ticker_cik_map = _fetch_sec_ticker_cik_map(sec_headers)
            sec_map_loaded = True
        return sec_ticker_cik_map

    for ticker in tickers:
        ticker_rows: list[dict] = []

        if source in {"auto", "sec"}:
            ticker_rows = _fetch_earnings_from_sec_filings(
                ticker=ticker,
                max_rows=max_rows_per_ticker,
                headers=sec_headers,
                ticker_cik_map=_get_sec_map(),
            )

        if not ticker_rows and source in {"auto", "yfinance"}:
            ticker_rows = _fetch_earnings_for_ticker(ticker, max_rows=max_rows_per_ticker)

        if not ticker_rows:
            logger.warning("No earnings dates found for %s using source=%s", ticker, source)

        rows.extend(ticker_rows)

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
    include_yf_aliases: bool = False,
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
    alias_map = _build_alias_map(
        tickers,
        alias_csv_path=alias_csv_path,
        include_yf_aliases=include_yf_aliases,
    )

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
    p.add_argument(
        "--earnings-source",
        choices=["auto", "sec", "yfinance"],
        default="auto",
        help="Earnings source: auto (SEC fallback to Yahoo), sec, or yfinance",
    )
    p.add_argument("--earnings-out", default="data/external/earnings/earnings_announcements.csv")
    p.add_argument("--enforcement-out", default="data/external/sec/sec_enforcement.csv")
    p.add_argument(
        "--alias-csv",
        default="data/external/sec/company_aliases.csv",
        help="Optional CSV with columns: ticker,alias for enforcement matching",
    )
    p.add_argument(
        "--include-yf-aliases",
        action="store_true",
        help="Include company aliases fetched from yfinance (slower and can be rate-limited)",
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
            source=args.earnings_source,
        )
        print(f"Earnings rows: {len(edf):,} -> {args.earnings_out}")

    if not args.skip_enforcement:
        sdf = build_enforcement_csv(
            tickers=tickers,
            out_path=args.enforcement_out,
            alias_csv_path=args.alias_csv,
            include_yf_aliases=args.include_yf_aliases,
        )
        print(f"Enforcement rows: {len(sdf):,} -> {args.enforcement_out}")

    logger.info("External events ingestion complete")
