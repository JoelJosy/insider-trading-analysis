"""
SEC EDGAR Form 4 Data Extraction Module.

Downloads and parses SEC Form 4 filings (insider trading disclosures)
using the sec-edgar-downloader library with comprehensive XML parsing.
"""

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from src.utils.config import get_config
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__, "logs/extract.log")


@dataclass
class InsiderInfo:
    """Represents insider information from Form 4."""
    name: str
    cik: Optional[str] = None
    is_director: bool = False
    is_officer: bool = False
    is_ten_percent_owner: bool = False
    is_other: bool = False
    officer_title: Optional[str] = None
    
    @property
    def role(self) -> str:
        """Determine the primary role of the insider."""
        if self.is_officer and self.officer_title:
            title_lower = self.officer_title.lower()
            if 'ceo' in title_lower or 'chief executive' in title_lower:
                return 'CEO'
            elif 'cfo' in title_lower or 'chief financial' in title_lower:
                return 'CFO'
            elif 'coo' in title_lower or 'chief operating' in title_lower:
                return 'COO'
            else:
                return 'Officer'
        elif self.is_director:
            return 'Director'
        elif self.is_ten_percent_owner:
            return '10%_Owner'
        else:
            return 'Other'


@dataclass
class Transaction:
    """Represents a single transaction from Form 4."""
    security_title: str
    transaction_date: Optional[date] = None
    transaction_code: Optional[str] = None  # P=Purchase, S=Sale, A=Grant, etc.
    shares: Optional[float] = None
    price_per_share: Optional[float] = None
    acquired_disposed: Optional[str] = None  # A=Acquired, D=Disposed
    shares_owned_after: Optional[float] = None
    direct_indirect: Optional[str] = None  # D=Direct, I=Indirect
    footnote_ids: List[str] = field(default_factory=list)
    
    @property
    def total_value(self) -> Optional[float]:
        """Calculate total transaction value."""
        if self.shares and self.price_per_share:
            return self.shares * self.price_per_share
        return None
    
    @property
    def is_open_market(self) -> bool:
        """Check if this is an open market transaction."""
        # P=Open Market Purchase, S=Open Market Sale
        return self.transaction_code in ('P', 'S')


@dataclass
class DerivativeTransaction:
    """Represents a derivative transaction (options, warrants, etc.)."""
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
    """Represents a complete parsed Form 4 filing."""
    accession_number: str
    filing_date: date
    accepted_datetime: Optional[datetime] = None
    
    # Company info
    issuer_cik: Optional[str] = None
    issuer_name: Optional[str] = None
    issuer_ticker: Optional[str] = None
    
    # Insider info
    insider: Optional[InsiderInfo] = None
    
    # Transactions
    non_derivative_transactions: List[Transaction] = field(default_factory=list)
    derivative_transactions: List[DerivativeTransaction] = field(default_factory=list)
    
    # Holdings (not transactions, just current holdings)
    non_derivative_holdings: List[Transaction] = field(default_factory=list)
    derivative_holdings: List[DerivativeTransaction] = field(default_factory=list)
    
    # Footnotes
    footnotes: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    xml_path: Optional[str] = None
    is_amendment: bool = False
    
    @property
    def has_10b5_1_plan(self) -> bool:
        """Check if any footnote mentions 10b5-1 plan."""
        for footnote_text in self.footnotes.values():
            if '10b5-1' in footnote_text.lower() or '10b-5' in footnote_text.lower():
                return True
        return False
    
    @property
    def all_footnotes_text(self) -> str:
        """Concatenate all footnotes into a single string."""
        return ' '.join(self.footnotes.values())


class Form4XMLParser:
    """Parser for SEC Form 4 XML files."""
    
    # XML namespaces used in Form 4 filings
    NAMESPACES = {
        '': 'http://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000320193&type=4&dateb=&owner=include&count=40'
    }
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__, "logs/parser.log")
    
    def parse_file(self, xml_path: str) -> Optional[Form4Filing]:
        """
        Parse a Form 4 XML file or full-submission text file.

        Args:
            xml_path: Path to the XML or TXT file.

        Returns:
            Form4Filing object if parsing succeeded, None otherwise.
        """
        try:
            if xml_path.endswith('.txt'):
                with open(xml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract XML block
                match = re.search(r'<XML>(.*?)</XML>', content, re.DOTALL)
                if match:
                    xml_content = match.group(1).strip()
                    root = ET.fromstring(xml_content)
                else:
                    # Try finding ownershipDocument directly if XML tags are missing
                    match = re.search(r'<ownershipDocument>(.*?)</ownershipDocument>', content, re.DOTALL)
                    if match:
                        xml_content = f"<ownershipDocument>{match.group(1)}</ownershipDocument>"
                        root = ET.fromstring(xml_content)
                    else:
                        self.logger.warning(f"No XML found in text file: {xml_path}")
                        return None
            else:
                tree = ET.parse(xml_path)
                root = tree.getroot()
            
            # Handle namespace in root element
            if root.tag.startswith('{'):
                # Extract namespace and strip it for easier parsing
                ns = root.tag.split('}')[0] + '}'
            else:
                ns = ''
            
            return self._parse_root(root, xml_path, ns)
            
        except ET.ParseError as e:
            self.logger.warning(f"XML parse error in {xml_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error parsing {xml_path}: {e}")
            return None

    # ... (rest of parsing methods)

    def find_form4_xml_files(self, ticker: str) -> List[Path]:
        """
        Find all Form 4 XML or TXT files for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of paths to files.
        """
        # The sec-edgar-downloader saves files in a specific structure
        # sec-edgar-filings/{ticker}/4/{accession_number}/
        
        base_path = self.download_dir / "sec-edgar-filings" / ticker / "4"
        
        if not base_path.exists():
            self.logger.warning(f"No filings directory found for {ticker}")
            return []
        
        files = []
        for accession_dir in base_path.iterdir():
            if accession_dir.is_dir():
                # Find XML files or full-submission.txt
                found = False
                for xml_file in accession_dir.glob("*.xml"):
                    files.append(xml_file)
                    found = True
                
                if not found:
                    for txt_file in accession_dir.glob("*.txt"):
                        files.append(txt_file)
        
        self.logger.info(f"Found {len(files)} filing files for {ticker}")
        return files
    
    def _parse_root(self, root: ET.Element, xml_path: str, ns: str = '') -> Form4Filing:
        """Parse the root element of Form 4 XML."""
        
        # Get accession number and filing date from file path or metadata
        accession_number = self._extract_accession_number(xml_path)
        
        # Parse schema version
        schema_version = root.find(f'{ns}schemaVersion')
        
        # Parse document type to check for amendments
        doc_type = self._get_text(root, f'{ns}documentType', '')
        is_amendment = doc_type == '4/A'
        
        # Parse period of report (filing date)
        period_text = self._get_text(root, f'{ns}periodOfReport', '')
        filing_date = self._parse_date(period_text) or date.today()
        
        # Parse issuer information
        issuer = root.find(f'{ns}issuer')
        issuer_cik = self._get_text(issuer, f'{ns}issuerCik') if issuer is not None else None
        issuer_name = self._get_text(issuer, f'{ns}issuerName') if issuer is not None else None
        issuer_ticker = self._get_text(issuer, f'{ns}issuerTradingSymbol') if issuer is not None else None
        
        # Parse reporting owner (insider)
        insider = self._parse_reporting_owner(root, ns)
        
        # Parse non-derivative transactions
        non_deriv_table = root.find(f'{ns}nonDerivativeTable')
        non_derivative_transactions = []
        non_derivative_holdings = []
        
        if non_deriv_table is not None:
            for trans_elem in non_deriv_table.findall(f'{ns}nonDerivativeTransaction'):
                trans = self._parse_non_derivative_transaction(trans_elem, ns)
                if trans:
                    non_derivative_transactions.append(trans)
            
            for holding_elem in non_deriv_table.findall(f'{ns}nonDerivativeHolding'):
                holding = self._parse_non_derivative_transaction(holding_elem, ns, is_holding=True)
                if holding:
                    non_derivative_holdings.append(holding)
        
        # Parse derivative transactions
        deriv_table = root.find(f'{ns}derivativeTable')
        derivative_transactions = []
        derivative_holdings = []
        
        if deriv_table is not None:
            for trans_elem in deriv_table.findall(f'{ns}derivativeTransaction'):
                trans = self._parse_derivative_transaction(trans_elem, ns)
                if trans:
                    derivative_transactions.append(trans)
            
            for holding_elem in deriv_table.findall(f'{ns}derivativeHolding'):
                holding = self._parse_derivative_transaction(holding_elem, ns)
                if holding:
                    derivative_holdings.append(holding)
        
        # Parse footnotes
        footnotes = self._parse_footnotes(root, ns)
        
        return Form4Filing(
            accession_number=accession_number,
            filing_date=filing_date,
            issuer_cik=issuer_cik,
            issuer_name=issuer_name,
            issuer_ticker=issuer_ticker,
            insider=insider,
            non_derivative_transactions=non_derivative_transactions,
            derivative_transactions=derivative_transactions,
            non_derivative_holdings=non_derivative_holdings,
            derivative_holdings=derivative_holdings,
            footnotes=footnotes,
            xml_path=xml_path,
            is_amendment=is_amendment
        )
    
    def _parse_reporting_owner(self, root: ET.Element, ns: str) -> Optional[InsiderInfo]:
        """Parse the reporting owner (insider) information."""
        owner_elem = root.find(f'{ns}reportingOwner')
        if owner_elem is None:
            return None
        
        # Owner ID
        owner_id = owner_elem.find(f'{ns}reportingOwnerId')
        name = self._get_text(owner_id, f'{ns}rptOwnerName') if owner_id is not None else 'Unknown'
        cik = self._get_text(owner_id, f'{ns}rptOwnerCik') if owner_id is not None else None
        
        # Owner relationship
        relationship = owner_elem.find(f'{ns}reportingOwnerRelationship')
        is_director = False
        is_officer = False
        is_ten_pct = False
        is_other = False
        officer_title = None
        
        if relationship is not None:
            is_director = self._get_text(relationship, f'{ns}isDirector', '0') == '1'
            is_officer = self._get_text(relationship, f'{ns}isOfficer', '0') == '1'
            is_ten_pct = self._get_text(relationship, f'{ns}isTenPercentOwner', '0') == '1'
            is_other = self._get_text(relationship, f'{ns}isOther', '0') == '1'
            officer_title = self._get_text(relationship, f'{ns}officerTitle')
        
        return InsiderInfo(
            name=name,
            cik=cik,
            is_director=is_director,
            is_officer=is_officer,
            is_ten_percent_owner=is_ten_pct,
            is_other=is_other,
            officer_title=officer_title
        )
    
    def _parse_non_derivative_transaction(
        self, elem: ET.Element, ns: str, is_holding: bool = False
    ) -> Optional[Transaction]:
        """Parse a non-derivative transaction or holding."""
        
        security_title = self._get_text(
            elem.find(f'{ns}securityTitle'), 
            f'{ns}value', 
            'Unknown'
        )
        
        # Transaction specifics
        trans_date = None
        trans_code = None
        
        if not is_holding:
            trans_date_elem = elem.find(f'{ns}transactionDate')
            if trans_date_elem is not None:
                date_str = self._get_text(trans_date_elem, f'{ns}value')
                trans_date = self._parse_date(date_str)
            
            coding_elem = elem.find(f'{ns}transactionCoding')
            if coding_elem is not None:
                trans_code = self._get_text(coding_elem, f'{ns}transactionCode')
        
        # Transaction amounts
        amounts = elem.find(f'{ns}transactionAmounts') if not is_holding else elem.find(f'{ns}postTransactionAmounts')
        shares = None
        price = None
        acquired_disposed = None
        
        if amounts is not None:
            shares_elem = amounts.find(f'{ns}transactionShares') or amounts.find(f'{ns}sharesOwnedFollowingTransaction')
            if shares_elem is not None:
                shares = self._parse_float(self._get_text(shares_elem, f'{ns}value'))
            
            price_elem = amounts.find(f'{ns}transactionPricePerShare')
            if price_elem is not None:
                price = self._parse_float(self._get_text(price_elem, f'{ns}value'))
            
            ad_elem = amounts.find(f'{ns}transactionAcquiredDisposedCode')
            if ad_elem is not None:
                acquired_disposed = self._get_text(ad_elem, f'{ns}value')
        
        # Post-transaction ownership
        post_amounts = elem.find(f'{ns}postTransactionAmounts')
        shares_after = None
        if post_amounts is not None:
            shares_after_elem = post_amounts.find(f'{ns}sharesOwnedFollowingTransaction')
            if shares_after_elem is not None:
                shares_after = self._parse_float(self._get_text(shares_after_elem, f'{ns}value'))
        
        # Direct/indirect ownership
        ownership = elem.find(f'{ns}ownershipNature')
        direct_indirect = None
        if ownership is not None:
            di_elem = ownership.find(f'{ns}directOrIndirectOwnership')
            if di_elem is not None:
                direct_indirect = self._get_text(di_elem, f'{ns}value')
        
        # Footnote references
        footnote_ids = self._extract_footnote_ids(elem, ns)
        
        return Transaction(
            security_title=security_title,
            transaction_date=trans_date,
            transaction_code=trans_code,
            shares=shares,
            price_per_share=price,
            acquired_disposed=acquired_disposed,
            shares_owned_after=shares_after,
            direct_indirect=direct_indirect,
            footnote_ids=footnote_ids
        )
    
    def _parse_derivative_transaction(
        self, elem: ET.Element, ns: str
    ) -> Optional[DerivativeTransaction]:
        """Parse a derivative transaction or holding."""
        
        security_title = self._get_text(
            elem.find(f'{ns}securityTitle'),
            f'{ns}value',
            'Unknown'
        )
        
        # Conversion/exercise price
        conv_elem = elem.find(f'{ns}conversionOrExercisePrice')
        conversion_price = None
        if conv_elem is not None:
            conversion_price = self._parse_float(self._get_text(conv_elem, f'{ns}value'))
        
        # Transaction date
        trans_date_elem = elem.find(f'{ns}transactionDate')
        trans_date = None
        if trans_date_elem is not None:
            trans_date = self._parse_date(self._get_text(trans_date_elem, f'{ns}value'))
        
        # Transaction code
        coding_elem = elem.find(f'{ns}transactionCoding')
        trans_code = None
        if coding_elem is not None:
            trans_code = self._get_text(coding_elem, f'{ns}transactionCode')
        
        # Transaction amounts
        amounts = elem.find(f'{ns}transactionAmounts')
        shares = None
        price = None
        
        if amounts is not None:
            shares_elem = amounts.find(f'{ns}transactionShares')
            if shares_elem is not None:
                shares = self._parse_float(self._get_text(shares_elem, f'{ns}value'))
            
            price_elem = amounts.find(f'{ns}transactionPricePerShare')
            if price_elem is not None:
                price = self._parse_float(self._get_text(price_elem, f'{ns}value'))
        
        # Exercise/expiration dates
        exercise_date_elem = elem.find(f'{ns}exerciseDate')
        exercise_date = None
        if exercise_date_elem is not None:
            exercise_date = self._parse_date(self._get_text(exercise_date_elem, f'{ns}value'))
        
        expiry_elem = elem.find(f'{ns}expirationDate')
        expiration_date = None
        if expiry_elem is not None:
            expiration_date = self._parse_date(self._get_text(expiry_elem, f'{ns}value'))
        
        # Underlying security
        underlying = elem.find(f'{ns}underlyingSecurity')
        underlying_shares = None
        if underlying is not None:
            und_shares_elem = underlying.find(f'{ns}underlyingSecurityShares')
            if und_shares_elem is not None:
                underlying_shares = self._parse_float(self._get_text(und_shares_elem, f'{ns}value'))
        
        # Post-transaction ownership
        post_amounts = elem.find(f'{ns}postTransactionAmounts')
        shares_after = None
        if post_amounts is not None:
            shares_after_elem = post_amounts.find(f'{ns}sharesOwnedFollowingTransaction')
            if shares_after_elem is not None:
                shares_after = self._parse_float(self._get_text(shares_after_elem, f'{ns}value'))
        
        # Footnote references
        footnote_ids = self._extract_footnote_ids(elem, ns)
        
        return DerivativeTransaction(
            security_title=security_title,
            conversion_price=conversion_price,
            transaction_date=trans_date,
            transaction_code=trans_code,
            shares=shares,
            price_per_share=price,
            exercise_date=exercise_date,
            expiration_date=expiration_date,
            underlying_shares=underlying_shares,
            shares_owned_after=shares_after,
            footnote_ids=footnote_ids
        )
    
    def _parse_footnotes(self, root: ET.Element, ns: str) -> Dict[str, str]:
        """Parse all footnotes from the filing."""
        footnotes = {}
        
        footnotes_elem = root.find(f'{ns}footnotes')
        if footnotes_elem is not None:
            for fn in footnotes_elem.findall(f'{ns}footnote'):
                fn_id = fn.get('id', '')
                fn_text = fn.text or ''
                # Clean up whitespace
                fn_text = ' '.join(fn_text.split())
                footnotes[fn_id] = fn_text
        
        return footnotes
    
    def _extract_footnote_ids(self, elem: ET.Element, ns: str) -> List[str]:
        """Extract footnote IDs referenced in an element."""
        ids = []
        for fn_ref in elem.iter():
            if fn_ref.tag.endswith('footnoteId'):
                fn_id = fn_ref.get('id', '')
                if fn_id:
                    ids.append(fn_id)
        return ids
    
    def _get_text(
        self, elem: Optional[ET.Element], tag: str, default: Optional[str] = None
    ) -> Optional[str]:
        """Safely get text from a child element."""
        if elem is None:
            return default
        child = elem.find(tag)
        if child is not None and child.text:
            return child.text.strip()
        return default
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[date]:
        """Parse a date string in various formats."""
        if not date_str:
            return None
        
        formats = ['%Y-%m-%d', '%m/%d/%Y', '%Y%m%d']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        return None
    
    def _parse_float(self, value_str: Optional[str]) -> Optional[float]:
        """Parse a float value, handling edge cases."""
        if not value_str:
            return None
        try:
            # Remove commas and other formatting
            cleaned = value_str.replace(',', '').replace('$', '').strip()
            return float(cleaned)
        except ValueError:
            return None
    
    def _extract_accession_number(self, xml_path: str) -> str:
        """Extract accession number from file path."""
        path = Path(xml_path)
        # Try to find accession number pattern in path
        pattern = r'\d{10}-\d{2}-\d{6}'
        for part in path.parts:
            match = re.search(pattern, part)
            if match:
                return match.group()
        return path.stem


class SECEdgarDownloader:
    """
    Download SEC Form 4 filings from EDGAR.
    
    Uses sec-edgar-downloader library with rate limiting and error handling.
    """
    
    def __init__(self, download_dir: str = "data/raw"):
        """
        Initialize the downloader.

        Args:
            download_dir: Directory to save downloaded filings.
        """
        self.config = get_config()
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(self.__class__.__name__, "logs/downloader.log")
        
        # Rate limiting
        self.rate_limit = self.config.sec_edgar.rate_limit_per_second
        self.last_request_time = 0.0
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self.last_request_time = time.time()
    
    def download_form4_filings(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        num_filings: int = 1000
    ) -> Path:
        """
        Download Form 4 filings for a company.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.
            num_filings: Maximum number of filings to download.

        Returns:
            Path to the download directory for this ticker.
        """
        try:
            from sec_edgar_downloader import Downloader
        except ImportError:
            self.logger.error("sec-edgar-downloader not installed. Run: pip install sec-edgar-downloader")
            raise
        
        # Use config defaults if not specified
        if start_date is None:
            start_date = self.config.sec_edgar.start_date
        if end_date is None:
            end_date = self.config.sec_edgar.end_date
        
        ticker_dir = self.download_dir / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Downloading Form 4 filings for {ticker} ({start_date} to {end_date})")
        
        # Initialize downloader with user agent
        dl = Downloader(
            company_name="InsiderTradingAnalysis",
            email_address="research@example.com",
            download_folder=str(self.download_dir)
        )
        
        try:
            self._rate_limit()
            
            # Download Form 4 filings
            dl.get(
                "4",  # Form type
                ticker,
                after=start_date,
                before=end_date,
                limit=num_filings
            )
            
            self.logger.info(f"Successfully downloaded filings for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error downloading filings for {ticker}: {e}")
            raise
        
        return ticker_dir
    
    def find_form4_xml_files(self, ticker: str) -> List[Path]:
        """
        Find all Form 4 XML or TXT files for a ticker.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            List of paths to files.
        """
        # The sec-edgar-downloader saves files in a specific structure
        # sec-edgar-filings/{ticker}/4/{accession_number}/
        
        base_path = self.download_dir / "sec-edgar-filings" / ticker / "4"
        
        if not base_path.exists():
            self.logger.warning(f"No filings directory found for {ticker}")
            return []
        
        files = []
        for accession_dir in base_path.iterdir():
            if accession_dir.is_dir():
                # Find XML files or full-submission.txt
                found = False
                for xml_file in accession_dir.glob("*.xml"):
                    files.append(xml_file)
                    found = True
                
                if not found:
                    for txt_file in accession_dir.glob("*.txt"):
                        files.append(txt_file)
        
        self.logger.info(f"Found {len(files)} filing files for {ticker}")
        return files


def parse_filings_to_dataframe(
    xml_files: List[Path],
    parser: Optional[Form4XMLParser] = None
) -> pd.DataFrame:
    """
    Parse multiple Form 4 XML files and return a DataFrame.

    Args:
        xml_files: List of paths to XML files.
        parser: Optional parser instance (creates new one if not provided).

    Returns:
        DataFrame with parsed transaction data.
    """
    if parser is None:
        parser = Form4XMLParser()
    
    records = []
    
    for xml_path in tqdm(xml_files, desc="Parsing XML files"):
        filing = parser.parse_file(str(xml_path))
        
        if filing is None:
            continue
        
        # Extract non-derivative transactions
        for trans in filing.non_derivative_transactions:
            record = {
                'accession_number': filing.accession_number,
                'filing_date': filing.filing_date,
                'issuer_cik': filing.issuer_cik,
                'issuer_name': filing.issuer_name,
                'ticker': filing.issuer_ticker,
                'insider_name': filing.insider.name if filing.insider else None,
                'insider_cik': filing.insider.cik if filing.insider else None,
                'insider_role': filing.insider.role if filing.insider else None,
                'is_director': filing.insider.is_director if filing.insider else False,
                'is_officer': filing.insider.is_officer if filing.insider else False,
                'is_ten_percent_owner': filing.insider.is_ten_percent_owner if filing.insider else False,
                'officer_title': filing.insider.officer_title if filing.insider else None,
                'security_title': trans.security_title,
                'transaction_date': trans.transaction_date,
                'transaction_code': trans.transaction_code,
                'shares': trans.shares,
                'price_per_share': trans.price_per_share,
                'total_value': trans.total_value,
                'acquired_disposed': trans.acquired_disposed,
                'shares_owned_after': trans.shares_owned_after,
                'direct_indirect': trans.direct_indirect,
                'is_open_market': trans.is_open_market,
                'has_10b5_1_plan': filing.has_10b5_1_plan,
                'footnote_text': filing.all_footnotes_text,
                'is_derivative': False,
                'is_amendment': filing.is_amendment,
                'xml_path': str(xml_path)
            }
            records.append(record)
        
        # Extract derivative transactions
        for trans in filing.derivative_transactions:
            record = {
                'accession_number': filing.accession_number,
                'filing_date': filing.filing_date,
                'issuer_cik': filing.issuer_cik,
                'issuer_name': filing.issuer_name,
                'ticker': filing.issuer_ticker,
                'insider_name': filing.insider.name if filing.insider else None,
                'insider_cik': filing.insider.cik if filing.insider else None,
                'insider_role': filing.insider.role if filing.insider else None,
                'is_director': filing.insider.is_director if filing.insider else False,
                'is_officer': filing.insider.is_officer if filing.insider else False,
                'is_ten_percent_owner': filing.insider.is_ten_percent_owner if filing.insider else False,
                'officer_title': filing.insider.officer_title if filing.insider else None,
                'security_title': trans.security_title,
                'transaction_date': trans.transaction_date,
                'transaction_code': trans.transaction_code,
                'shares': trans.shares,
                'price_per_share': trans.price_per_share,
                'total_value': trans.shares * trans.price_per_share if trans.shares and trans.price_per_share else None,
                'acquired_disposed': None,
                'shares_owned_after': trans.shares_owned_after,
                'direct_indirect': None,
                'is_open_market': False,
                'has_10b5_1_plan': filing.has_10b5_1_plan,
                'footnote_text': filing.all_footnotes_text,
                'is_derivative': True,
                'is_amendment': filing.is_amendment,
                'conversion_price': trans.conversion_price,
                'exercise_date': trans.exercise_date,
                'expiration_date': trans.expiration_date,
                'underlying_shares': trans.underlying_shares,
                'xml_path': str(xml_path)
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    
    # Convert date columns
    date_cols = ['filing_date', 'transaction_date', 'exercise_date', 'expiration_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df


def download_and_parse_ticker(
    ticker: str,
    download_dir: str = "data/raw",
    output_dir: str = "data/processed",
    num_filings: int = 1000
) -> Tuple[Path, pd.DataFrame]:
    """
    Download and parse Form 4 filings for a single ticker.

    Args:
        ticker: Stock ticker symbol.
        download_dir: Directory to save raw filings.
        output_dir: Directory to save parsed CSV.
        num_filings: Maximum number of filings to download.

    Returns:
        Tuple of (csv_path, dataframe).
    """
    logger = setup_logger(__name__, "logs/extract.log")
    
    # Download
    downloader = SECEdgarDownloader(download_dir)
    downloader.download_form4_filings(ticker, num_filings=num_filings)
    
    # Find XML files
    xml_files = downloader.find_form4_xml_files(ticker)
    
    if not xml_files:
        logger.warning(f"No XML files found for {ticker}")
        return None, pd.DataFrame()
    
    # Parse
    parser = Form4XMLParser()
    df = parse_filings_to_dataframe(xml_files, parser)
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{ticker}_form4.csv"
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Saved {len(df)} transactions to {csv_path}")
    
    return csv_path, df


if __name__ == "__main__":
    import argparse
    
    arg_parser = argparse.ArgumentParser(description="Download and parse SEC Form 4 filings")
    arg_parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    arg_parser.add_argument("--num-filings", type=int, default=100, help="Number of filings to download")
    arg_parser.add_argument("--download-dir", type=str, default="data/raw", help="Download directory")
    arg_parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    
    args = arg_parser.parse_args()
    
    csv_path, df = download_and_parse_ticker(
        args.ticker,
        args.download_dir,
        args.output_dir,
        args.num_filings
    )
    
    if not df.empty:
        print(f"\nDownloaded and parsed {len(df)} transactions")
        print(f"\nSample data:")
        print(df.head())
        print(f"\nData saved to: {csv_path}")
