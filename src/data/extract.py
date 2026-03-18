import re
import time
import xml.etree.ElementTree as ET
import os
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.data.market import enrich_with_market_prices
from src.utils.config import get_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/extract.log")


@dataclass
class InsiderInfo:
    name: str
    cik: Optional[str] = None
    is_director: bool = False
    is_officer: bool = False
    is_ten_percent_owner: bool = False
    is_other: bool = False
    officer_title: Optional[str] = None

    @property
    def role(self) -> str:
        if self.is_officer and self.officer_title:
            t = self.officer_title.lower()
            if "ceo" in t or "chief executive" in t:
                return "CEO"
            if "cfo" in t or "chief financial" in t:
                return "CFO"
            if "coo" in t or "chief operating" in t:
                return "COO"
            return "Officer"
        if self.is_director:
            return "Director"
        if self.is_ten_percent_owner:
            return "10%_Owner"
        return "Other"


@dataclass
class Transaction:
    security_title: str
    transaction_date: Optional[date] = None
    transaction_code: Optional[str] = None
    shares: Optional[float] = None
    price_per_share: Optional[float] = None
    acquired_disposed: Optional[str] = None
    shares_owned_after: Optional[float] = None
    direct_indirect: Optional[str] = None
    footnote_ids: List[str] = field(default_factory=list)

    @property
    def total_value(self) -> Optional[float]:
        return self.shares * self.price_per_share if self.shares and self.price_per_share else None

    @property
    def is_open_market(self) -> bool:
        return self.transaction_code in ("P", "S")


@dataclass
class DerivativeTransaction:
    security_title: str
    conversion_price: Optional[float] = None
    transaction_date: Optional[date] = None
    transaction_code: Optional[str] = None
    shares: Optional[float] = None
    price_per_share: Optional[float] = None
    exercise_date: Optional[date] = None
    expiration_date: Optional[date] = None
    underlying_shares: Optional[float] = None
    shares_owned_after: Optional[float] = None
    footnote_ids: List[str] = field(default_factory=list)


@dataclass
class Form4Filing:
    accession_number: str
    filing_date: date
    issuer_cik: Optional[str] = None
    issuer_name: Optional[str] = None
    issuer_ticker: Optional[str] = None
    insider: Optional[InsiderInfo] = None
    non_derivative_transactions: List[Transaction] = field(default_factory=list)
    derivative_transactions: List[DerivativeTransaction] = field(default_factory=list)
    non_derivative_holdings: List[Transaction] = field(default_factory=list)
    derivative_holdings: List[DerivativeTransaction] = field(default_factory=list)
    footnotes: Dict[str, str] = field(default_factory=dict)
    xml_path: Optional[str] = None
    is_amendment: bool = False

    @property
    def has_10b5_1_plan(self) -> bool:
        return any("10b5-1" in v.lower() or "10b-5" in v.lower() for v in self.footnotes.values())

    @property
    def all_footnotes_text(self) -> str:
        return " ".join(self.footnotes.values())


class Form4XMLParser:
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__, "logs/parser.log")

    def parse_file(self, xml_path: str) -> Optional[Form4Filing]:
        try:
            if xml_path.endswith(".txt"):
                content = Path(xml_path).read_text(encoding="utf-8")
                m = re.search(r"<XML>(.*?)</XML>", content, re.DOTALL)
                if m:
                    root = ET.fromstring(m.group(1).strip())
                else:
                    m = re.search(r"<ownershipDocument>(.*?)</ownershipDocument>", content, re.DOTALL)
                    if m:
                        root = ET.fromstring(f"<ownershipDocument>{m.group(1)}</ownershipDocument>")
                    else:
                        self.logger.warning(f"No XML found in: {xml_path}")
                        return None
            else:
                root = ET.parse(xml_path).getroot()

            ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
            return self._parse_root(root, xml_path, ns)
        except ET.ParseError as e:
            self.logger.warning(f"XML parse error in {xml_path}: {e}")
        except Exception as e:
            self.logger.error(f"Error parsing {xml_path}: {e}")
        return None

    def _parse_root(self, root: ET.Element, xml_path: str, ns: str) -> Form4Filing:
        issuer = root.find(f"{ns}issuer")

        nd_table = root.find(f"{ns}nonDerivativeTable")
        transactions, holdings = [], []
        if nd_table is not None:
            transactions = [t for t in (self._parse_non_deriv(e, ns) for e in nd_table.findall(f"{ns}nonDerivativeTransaction")) if t]
            holdings = [t for t in (self._parse_non_deriv(e, ns, True) for e in nd_table.findall(f"{ns}nonDerivativeHolding")) if t]

        d_table = root.find(f"{ns}derivativeTable")
        d_transactions, d_holdings = [], []
        if d_table is not None:
            d_transactions = [t for t in (self._parse_deriv(e, ns) for e in d_table.findall(f"{ns}derivativeTransaction")) if t]
            d_holdings = [t for t in (self._parse_deriv(e, ns) for e in d_table.findall(f"{ns}derivativeHolding")) if t]

        return Form4Filing(
            accession_number=self._accession_number(xml_path),
            filing_date=self._parse_date(self._v(ns, root, "ownerSignature", "signatureDate"))
                        or self._parse_date(self._v(ns, root, "periodOfReport"))
                        or date.today(),
            issuer_cik=self._v(ns, issuer, "issuerCik"),
            issuer_name=self._v(ns, issuer, "issuerName"),
            issuer_ticker=self._v(ns, issuer, "issuerTradingSymbol"),
            insider=self._parse_owner(root, ns),
            non_derivative_transactions=transactions,
            non_derivative_holdings=holdings,
            derivative_transactions=d_transactions,
            derivative_holdings=d_holdings,
            footnotes=self._parse_footnotes(root, ns),
            xml_path=xml_path,
            is_amendment=(self._v(ns, root, "documentType") == "4/A"),
        )

    def _parse_owner(self, root: ET.Element, ns: str) -> Optional[InsiderInfo]:
        owner = root.find(f"{ns}reportingOwner")
        if owner is None:
            return None
        oid = owner.find(f"{ns}reportingOwnerId")
        rel = owner.find(f"{ns}reportingOwnerRelationship")
        rb = lambda tag: self._v(ns, rel, tag) == "1" if rel is not None else False
        return InsiderInfo(
            name=self._v(ns, oid, "rptOwnerName") or "Unknown",
            cik=self._v(ns, oid, "rptOwnerCik"),
            is_director=rb("isDirector"),
            is_officer=rb("isOfficer"),
            is_ten_percent_owner=rb("isTenPercentOwner"),
            is_other=rb("isOther"),
            officer_title=self._v(ns, rel, "officerTitle"),
        )

    def _parse_non_deriv(self, elem: ET.Element, ns: str, is_holding: bool = False) -> Optional[Transaction]:
        security_title = self._v(ns, elem, "securityTitle", "value") or "Unknown"

        trans_date = trans_code = None
        if not is_holding:
            trans_date = self._parse_date(self._v(ns, elem, "transactionDate", "value"))
            trans_code = self._v(ns, elem, "transactionCoding", "transactionCode")

        amounts_tag = "postTransactionAmounts" if is_holding else "transactionAmounts"
        amounts = elem.find(f"{ns}{amounts_tag}")
        shares_tag = "sharesOwnedFollowingTransaction" if is_holding else "transactionShares"
        shares = self._parse_float(self._v(ns, amounts, shares_tag, "value"))
        price = self._parse_float(self._v(ns, amounts, "transactionPricePerShare", "value")) if not is_holding else None
        ad = self._v(ns, amounts, "transactionAcquiredDisposedCode", "value") if not is_holding else None

        post = elem.find(f"{ns}postTransactionAmounts")
        shares_after = self._parse_float(self._v(ns, post, "sharesOwnedFollowingTransaction", "value"))

        own = elem.find(f"{ns}ownershipNature")
        di_elem = own.find(f"{ns}directOrIndirectOwnership") if own is not None else None
        di = self._v(ns, di_elem, "value")

        return Transaction(security_title, trans_date, trans_code, shares, price, ad, shares_after, di, self._footnote_ids(elem, ns))

    def _parse_deriv(self, elem: ET.Element, ns: str) -> Optional[DerivativeTransaction]:
        amounts = elem.find(f"{ns}transactionAmounts")
        post = elem.find(f"{ns}postTransactionAmounts")
        underlying = elem.find(f"{ns}underlyingSecurity")
        return DerivativeTransaction(
            security_title=self._v(ns, elem, "securityTitle", "value") or "Unknown",
            conversion_price=self._parse_float(self._v(ns, elem, "conversionOrExercisePrice", "value")),
            transaction_date=self._parse_date(self._v(ns, elem, "transactionDate", "value")),
            transaction_code=self._v(ns, elem, "transactionCoding", "transactionCode"),
            shares=self._parse_float(self._v(ns, amounts, "transactionShares", "value")),
            price_per_share=self._parse_float(self._v(ns, amounts, "transactionPricePerShare", "value")),
            exercise_date=self._parse_date(self._v(ns, elem, "exerciseDate", "value")),
            expiration_date=self._parse_date(self._v(ns, elem, "expirationDate", "value")),
            underlying_shares=self._parse_float(self._v(ns, underlying, "underlyingSecurityShares", "value")),
            shares_owned_after=self._parse_float(self._v(ns, post, "sharesOwnedFollowingTransaction", "value")),
            footnote_ids=self._footnote_ids(elem, ns),
        )

    def _parse_footnotes(self, root: ET.Element, ns: str) -> Dict[str, str]:
        fns = root.find(f"{ns}footnotes")
        if fns is None:
            return {}
        return {fn.get("id", ""): " ".join((fn.text or "").split()) for fn in fns.findall(f"{ns}footnote")}

    def _footnote_ids(self, elem: ET.Element, ns: str) -> List[str]:
        return [e.get("id", "") for e in elem.iter() if e.tag.endswith("footnoteId") and e.get("id")]

    @staticmethod
    def _v(ns: str, parent: Optional[ET.Element], *tags: str) -> Optional[str]:
        e = parent
        for tag in tags:
            if e is None:
                return None
            e = e.find(f"{ns}{tag}")
        return e.text.strip() if e is not None and e.text else None

    def _parse_date(self, s: Optional[str]) -> Optional[date]:
        if not s:
            return None
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                pass
        return None

    def _parse_float(self, s: Optional[str]) -> Optional[float]:
        if not s:
            return None
        try:
            return float(s.replace(",", "").replace("$", "").strip())
        except ValueError:
            return None

    def _accession_number(self, xml_path: str) -> str:
        for part in Path(xml_path).parts:
            m = re.search(r"\d{10}-\d{2}-\d{6}", part)
            if m:
                return m.group()
        return Path(xml_path).stem


class SECEdgarDownloader:
    def __init__(self, download_dir: str = "data/raw"):
        self.config = get_config()
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.__class__.__name__, "logs/downloader.log")
        self._last_request = 0.0
        self.app_name = os.getenv("SEC_EDGAR_APP_NAME", "InsiderTradingAnalysis")
        self.contact_email = os.getenv("SEC_EDGAR_CONTACT_EMAIL", "joeljosy449@gmail.com")
        self.max_retries = int(os.getenv("SEC_EDGAR_MAX_RETRIES", "5"))
        self.base_backoff_seconds = float(os.getenv("SEC_EDGAR_BACKOFF_SECONDS", "2"))

    @staticmethod
    def _is_retryable_error(error: Exception) -> bool:
        message = str(error).lower()
        retry_markers = ("429", "500", "502", "503", "504", "timeout", "temporar", "connection")
        return any(marker in message for marker in retry_markers)

    def _throttle(self):
        interval = 1.0 / self.config.sec_edgar.rate_limit_per_second
        elapsed = time.time() - self._last_request
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request = time.time()

    def download(self, ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None, limit: int = 1000) -> Path:
        from sec_edgar_downloader import Downloader
        cfg = self.config.sec_edgar
        dl = Downloader(self.app_name, self.contact_email, download_folder=str(self.download_dir))

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                self._throttle()
                dl.get("4", ticker, after=start_date or cfg.start_date, before=end_date or cfg.end_date, limit=limit)
                self.logger.info(f"Downloaded Form 4 filings for {ticker}")
                return self.download_dir / ticker
            except Exception as e:
                last_error = e
                if attempt >= self.max_retries or not self._is_retryable_error(e):
                    break
                backoff = self.base_backoff_seconds * (2 ** (attempt - 1))
                self.logger.warning(
                    f"SEC download failed for {ticker} (attempt {attempt}/{self.max_retries}): {e}. "
                    f"Retrying in {backoff:.1f}s"
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"Failed to download Form 4 filings for {ticker} after {self.max_retries} attempts"
        ) from last_error

    def find_files(self, ticker: str) -> List[Path]:
        base = self.download_dir / "sec-edgar-filings" / ticker / "4"
        if not base.exists():
            self.logger.warning(f"No filings directory for {ticker}")
            return []
        files = []
        for d in base.iterdir():
            if not d.is_dir():
                continue
            xmls = list(d.glob("*.xml"))
            files.extend(xmls if xmls else d.glob("*.txt"))
        self.logger.info(f"Found {len(files)} files for {ticker}")
        return files


def _filing_to_records(filing: Form4Filing) -> List[dict]:
    base = {
        "accession_number": filing.accession_number,
        "filing_date": filing.filing_date,
        "issuer_cik": filing.issuer_cik,
        "issuer_name": filing.issuer_name,
        "ticker": filing.issuer_ticker,
        "insider_name": filing.insider.name if filing.insider else None,
        "insider_cik": filing.insider.cik if filing.insider else None,
        "insider_role": filing.insider.role if filing.insider else None,
        "is_director": filing.insider.is_director if filing.insider else False,
        "is_officer": filing.insider.is_officer if filing.insider else False,
        "is_ten_percent_owner": filing.insider.is_ten_percent_owner if filing.insider else False,
        "officer_title": filing.insider.officer_title if filing.insider else None,
        "has_10b5_1_plan": filing.has_10b5_1_plan,
        "footnote_text": filing.all_footnotes_text,
        "is_amendment": filing.is_amendment,
        "xml_path": filing.xml_path,
    }
    records = []
    for t in filing.non_derivative_transactions:
        records.append({**base, "security_title": t.security_title, "transaction_date": t.transaction_date,
                        "transaction_code": t.transaction_code, "shares": t.shares, "price_per_share": t.price_per_share,
                        "total_value": t.total_value, "acquired_disposed": t.acquired_disposed,
                        "shares_owned_after": t.shares_owned_after, "direct_indirect": t.direct_indirect,
                        "is_open_market": t.is_open_market, "is_derivative": False})
    for t in filing.derivative_transactions:
        # For exercise/conversion (M code), price_per_share is often 0 or null;
        # use conversion_price as the cost basis instead.
        deriv_price = t.price_per_share or t.conversion_price
        tv = t.shares * deriv_price if t.shares and deriv_price else None
        records.append({**base, "security_title": t.security_title, "transaction_date": t.transaction_date,
                        "transaction_code": t.transaction_code, "shares": t.shares, "price_per_share": t.price_per_share,
                        "total_value": tv, "acquired_disposed": None, "shares_owned_after": t.shares_owned_after,
                        "direct_indirect": None, "is_open_market": False, "is_derivative": True,
                        "conversion_price": t.conversion_price, "exercise_date": t.exercise_date,
                        "expiration_date": t.expiration_date, "underlying_shares": t.underlying_shares})
    return records


def parse_filings_to_dataframe(xml_files: List[Path], parser: Optional[Form4XMLParser] = None) -> pd.DataFrame:
    parser = parser or Form4XMLParser()
    records = []
    for f in tqdm(xml_files, desc="Parsing XML files"):
        filing = parser.parse_file(str(f))
        if filing:
            records.extend(_filing_to_records(filing))
    df = pd.DataFrame(records)
    for col in ["filing_date", "transaction_date", "exercise_date", "expiration_date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def download_and_parse_ticker(
    ticker: str,
    download_dir: str = "data/raw",
    output_dir: str = "data/processed",
    num_filings: int = 1000,
    enrich_market_prices: bool = True,
) -> Tuple[Optional[Path], pd.DataFrame]:
    dl = SECEdgarDownloader(download_dir)
    dl.download(ticker, limit=num_filings)
    xml_files = dl.find_files(ticker)
    if not xml_files:
        logger.warning(f"No files found for {ticker}")
        return None, pd.DataFrame()
    df = parse_filings_to_dataframe(xml_files)
    if enrich_market_prices and not df.empty:
        try:
            df = enrich_with_market_prices(df)
        except Exception as e:
            logger.warning(f"Market price enrichment skipped due to error: {e}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{ticker}_form4.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df)} transactions to {csv_path}")
    return csv_path, df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Download and parse SEC Form 4 filings")
    p.add_argument("--ticker", default="AAPL")
    p.add_argument("--num-filings", type=int, default=100)
    p.add_argument("--download-dir", default="data/raw")
    p.add_argument("--output-dir", default="data/processed")
    p.add_argument("--skip-market-prices", action="store_true", help="Skip yfinance close-price enrichment")
    args = p.parse_args()
    csv_path, df = download_and_parse_ticker(
        args.ticker,
        args.download_dir,
        args.output_dir,
        args.num_filings,
        enrich_market_prices=not args.skip_market_prices,
    )
    if not df.empty:
        print(f"Parsed {len(df)} transactions -> {csv_path}")
        print(df.head())
