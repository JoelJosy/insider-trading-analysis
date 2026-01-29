"""
Database Loading Module for Insider Trading Analysis.

Handles loading parsed SEC Form 4 data into PostgreSQL database
with proper dimension and fact table management.
"""

import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_batch, execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.utils.config import get_config, DatabaseConfig
from src.utils.logger import setup_logger

# Setup logger
logger = setup_logger(__name__, "logs/load.log")


class DatabaseManager:
    """
    Manages database connections and operations for the insider trading warehouse.
    """
    
    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize database manager.

        Args:
            config: Database configuration. Uses global config if not provided.
        """
        if config is None:
            config = get_config().database
        
        self.config = config
        self.logger = setup_logger(self.__class__.__name__, "logs/db.log")
        self._connection = None
        self._engine = None
    
    @property
    def connection(self) -> psycopg2.extensions.connection:
        """Get or create database connection."""
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(**self.config.psycopg2_params)
        return self._connection
    
    @property
    def engine(self):
        """Get or create SQLAlchemy engine."""
        if self._engine is None:
            self._engine = create_engine(self.config.connection_string)
        return self._engine
    
    def close(self):
        """Close database connections."""
        if self._connection and not self._connection.closed:
            self._connection.close()
        if self._engine:
            self._engine.dispose()
    
    def execute_sql_file(self, sql_path: str) -> None:
        """
        Execute a SQL file against the database.

        Args:
            sql_path: Path to the SQL file.
        """
        with open(sql_path, 'r') as f:
            sql_content = f.read()
        
        conn = self.connection
        try:
            with conn.cursor() as cur:
                cur.execute(sql_content)
            conn.commit()
            self.logger.info(f"Successfully executed SQL file: {sql_path}")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error executing SQL file {sql_path}: {e}")
            raise
    
    def create_database(self) -> None:
        """
        Create the database if it doesn't exist.
        
        Note: This connects to the default 'postgres' database first.
        """
        # Connect to postgres database
        conn_params = self.config.psycopg2_params.copy()
        db_name = conn_params.pop('dbname')
        conn_params['dbname'] = 'postgres'
        
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        
        try:
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (db_name,)
                )
                exists = cur.fetchone() is not None
                
                if not exists:
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(db_name)
                    ))
                    self.logger.info(f"Created database: {db_name}")
                else:
                    self.logger.info(f"Database already exists: {db_name}")
        finally:
            conn.close()
    
    def initialize_schema(self, schema_path: str = "sql/schema.sql") -> None:
        """
        Initialize the database schema.

        Args:
            schema_path: Path to the schema SQL file.
        """
        self.create_database()
        
        # Execute schema SQL
        full_path = Path(__file__).parent.parent.parent / schema_path
        if full_path.exists():
            self.execute_sql_file(str(full_path))
        else:
            self.logger.warning(f"Schema file not found: {full_path}")


class InsiderTradingLoader:
    """
    Loads parsed insider trading data into the database.
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize the loader.

        Args:
            db_manager: Database manager instance.
        """
        self.db = db_manager or DatabaseManager()
        self.logger = setup_logger(self.__class__.__name__, "logs/loader.log")
        
        # Cache for dimension lookups
        self._company_cache: Dict[str, int] = {}
        self._insider_cache: Dict[str, int] = {}
        self._time_cache: Dict[date, int] = {}
        self._transaction_type_cache: Dict[str, int] = {}
    
    def _get_or_create_company(
        self, 
        cik: str, 
        ticker: Optional[str] = None,
        company_name: Optional[str] = None
    ) -> int:
        """
        Get or create a company dimension record.
        """
        # Ensure CIK is a string
        cik = str(cik).split('.')[0]  # Handle float-like strings "123.0"
        
        cache_key = cik
        if cache_key in self._company_cache:
            return self._company_cache[cache_key]
        
        conn = self.db.connection
        with conn.cursor() as cur:
            # Try to find existing
            cur.execute(
                "SELECT company_key FROM dim_company WHERE cik = %s",
                (cik,)
            )
            result = cur.fetchone()
            
            if result:
                company_key = result[0]
            else:
                # Insert new company
                cur.execute(
                    """
                    INSERT INTO dim_company (cik, ticker, company_name, is_current)
                    VALUES (%s, %s, %s, TRUE)
                    RETURNING company_key
                    """,
                    (cik, ticker, company_name)
                )
                company_key = cur.fetchone()[0]
                conn.commit()
                self.logger.debug(f"Created company: {ticker} (CIK: {cik})")
            
            self._company_cache[cache_key] = company_key
            return company_key
    
    def _get_or_create_insider(
        self,
        cik: Optional[str],
        name: str,
        role: Optional[str] = None
    ) -> int:
        """
        Get or create an insider dimension record.
        """
        # Ensure CIK is a string if present
        if cik:
            cik = str(cik).split('.')[0]
        
        # Use CIK if available, otherwise use name hash
        cache_key = cik if cik else hashlib.md5(name.encode()).hexdigest()[:20]
        
        if cache_key in self._insider_cache:
            return self._insider_cache[cache_key]
        
        conn = self.db.connection
        with conn.cursor() as cur:
            # Try to find existing by CIK
            if cik:
                cur.execute(
                    "SELECT insider_key FROM dim_insider WHERE cik = %s",
                    (cik,)
                )
            else:
                cur.execute(
                    "SELECT insider_key FROM dim_insider WHERE name = %s",
                    (name,)
                )
            
            result = cur.fetchone()
            
            if result:
                insider_key = result[0]
            else:
                # Insert new insider
                cur.execute(
                    """
                    INSERT INTO dim_insider (cik, name, typical_role, first_seen_date)
                    VALUES (%s, %s, %s, CURRENT_DATE)
                    RETURNING insider_key
                    """,
                    (cik, name, role)
                )
                insider_key = cur.fetchone()[0]
                conn.commit()
                self.logger.debug(f"Created insider: {name}")
            
            self._insider_cache[cache_key] = insider_key
            return insider_key
    
    def _get_time_key(self, trade_date: date) -> int:
        """
        Get the time dimension key for a date.

        Args:
            trade_date: The date to look up.

        Returns:
            time_key for the dimension.
        """
        if trade_date in self._time_cache:
            return self._time_cache[trade_date]
        
        conn = self.db.connection
        with conn.cursor() as cur:
            cur.execute(
                "SELECT time_key FROM dim_time WHERE date = %s",
                (trade_date,)
            )
            result = cur.fetchone()
            
            if result:
                self._time_cache[trade_date] = result[0]
                return result[0]
            else:
                # Time dimension should be pre-populated, but handle missing dates
                self.logger.warning(f"Missing time dimension for date: {trade_date}")
                return None
    
    def _get_transaction_type_key(self, code: Optional[str]) -> Optional[int]:
        """
        Get the transaction type dimension key.

        Args:
            code: Transaction code (P, S, A, etc.).

        Returns:
            type_key for the dimension, or None if code is invalid.
        """
        if not code:
            return None
        
        if code in self._transaction_type_cache:
            return self._transaction_type_cache[code]
        
        conn = self.db.connection
        with conn.cursor() as cur:
            cur.execute(
                "SELECT type_key FROM dim_transaction_type WHERE code = %s",
                (code,)
            )
            result = cur.fetchone()
            
            if result:
                self._transaction_type_cache[code] = result[0]
                return result[0]
            
            return None
    
    def load_dataframe(self, df: pd.DataFrame, batch_size: int = 1000) -> int:
        """
        Load a DataFrame of parsed trades into the database.

        Args:
            df: DataFrame with parsed trade data.
            batch_size: Number of records to insert per batch.

        Returns:
            Number of records loaded.
        """
        if df.empty:
            self.logger.warning("Empty DataFrame provided, nothing to load")
            return 0
        
        loaded_count = 0
        conn = self.db.connection
        
        # Prepare records for insertion
        records = []
        
        for idx, row in df.iterrows():
            try:
                # Get dimension keys
                company_key = self._get_or_create_company(
                    cik=str(row.get('issuer_cik', '')),
                    ticker=row.get('ticker'),
                    company_name=row.get('issuer_name')
                )
                
                insider_key = self._get_or_create_insider(
                    cik=row.get('insider_cik'),
                    name=row.get('insider_name', 'Unknown'),
                    role=row.get('insider_role')
                )
                
                # Get time key - use transaction_date if available, else filing_date
                trade_date = row.get('transaction_date')
                if pd.isna(trade_date):
                    trade_date = row.get('filing_date')
                
                if pd.isna(trade_date):
                    self.logger.warning(f"Missing date for row {idx}, skipping")
                    continue
                
                if isinstance(trade_date, str):
                    trade_date = pd.to_datetime(trade_date).date()
                elif isinstance(trade_date, datetime):
                    trade_date = trade_date.date()
                elif isinstance(trade_date, pd.Timestamp):
                    trade_date = trade_date.date()
                
                time_key = self._get_time_key(trade_date)
                if time_key is None:
                    continue
                
                transaction_type_key = self._get_transaction_type_key(
                    row.get('transaction_code')
                )
                
                # Sanitize values (convert NaN/NaT to None)
                def sanitize(val):
                    if pd.isna(val) or val == 'NaN':
                        return None
                    return val

                # Prepare record tuple
                record = (
                    insider_key,
                    company_key,
                    time_key,
                    transaction_type_key,
                    row.get('accession_number', ''),
                    sanitize(row.get('filing_date')),
                    sanitize(row.get('transaction_date')),
                    sanitize(row.get('shares')),
                    sanitize(row.get('price_per_share')),
                    sanitize(row.get('total_value')),
                    sanitize(row.get('shares_owned_after')),
                    sanitize(row.get('acquired_disposed')),
                    sanitize(row.get('direct_indirect')),
                    bool(row.get('is_derivative', False)),
                    bool(row.get('is_amendment', False)),
                    bool(row.get('has_10b5_1_plan', False)),
                    sanitize(row.get('security_title')),
                    sanitize(row.get('conversion_price')),
                    sanitize(row.get('exercise_date')),
                    sanitize(row.get('expiration_date')),
                    sanitize(row.get('underlying_shares')),
                    sanitize(row.get('footnote_text')),
                    sanitize(row.get('xml_path'))
                )
                records.append(record)
                
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {e}")
                conn.rollback()
                continue
        
        # Batch insert
        if records:
            try:
                with conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO fact_insider_trades (
                            insider_key, company_key, time_key, transaction_type_key,
                            accession_number, filing_date, transaction_date,
                            shares_traded, price_per_share, total_value,
                            shares_owned_after, acquired_disposed, direct_indirect,
                            is_derivative, is_amendment, has_10b5_1_plan,
                            security_title, conversion_price, exercise_date,
                            expiration_date, underlying_shares, footnote_text, xml_path
                        ) VALUES %s
                        ON CONFLICT ON CONSTRAINT unique_transaction DO NOTHING
                        """,
                        records,
                        page_size=batch_size
                    )
                    loaded_count = cur.rowcount
                
                conn.commit()
                self.logger.info(f"Loaded {loaded_count} records to database")
                
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error loading batch: {e}")
                raise
        
        return loaded_count
    
    def load_csv(self, csv_path: str, batch_size: int = 1000) -> int:
        """
        Load a CSV file into the database.

        Args:
            csv_path: Path to the CSV file.
            batch_size: Number of records to insert per batch.

        Returns:
            Number of records loaded.
        """
        self.logger.info(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path, parse_dates=['filing_date', 'transaction_date'])
        return self.load_dataframe(df, batch_size)
    
    def update_insider_statistics(self) -> None:
        """Update aggregate statistics in dim_insider."""
        conn = self.db.connection
        
        query = """
        UPDATE dim_insider i
        SET 
            total_trades_historical = stats.total_trades,
            total_buys_historical = stats.total_buys,
            total_sells_historical = stats.total_sells,
            total_value_bought = stats.value_bought,
            total_value_sold = stats.value_sold,
            avg_trade_size = stats.avg_value,
            first_seen_date = stats.first_date,
            last_seen_date = stats.last_date,
            updated_at = CURRENT_TIMESTAMP
        FROM (
            SELECT 
                t.insider_key,
                COUNT(*) as total_trades,
                SUM(CASE WHEN tt.is_buy_equivalent THEN 1 ELSE 0 END) as total_buys,
                SUM(CASE WHEN tt.is_buy_equivalent = FALSE THEN 1 ELSE 0 END) as total_sells,
                SUM(CASE WHEN tt.is_buy_equivalent THEN t.total_value ELSE 0 END) as value_bought,
                SUM(CASE WHEN tt.is_buy_equivalent = FALSE THEN t.total_value ELSE 0 END) as value_sold,
                AVG(t.total_value) as avg_value,
                MIN(t.transaction_date) as first_date,
                MAX(t.transaction_date) as last_date
            FROM fact_insider_trades t
            LEFT JOIN dim_transaction_type tt ON t.transaction_type_key = tt.type_key
            GROUP BY t.insider_key
        ) stats
        WHERE i.insider_key = stats.insider_key
        """
        
        try:
            with conn.cursor() as cur:
                cur.execute(query)
            conn.commit()
            self.logger.info("Updated insider statistics")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error updating insider statistics: {e}")
    
    def refresh_materialized_views(self) -> None:
        """Refresh all materialized views."""
        conn = self.db.connection
        
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT refresh_all_materialized_views()")
            conn.commit()
            self.logger.info("Refreshed materialized views")
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error refreshing views: {e}")


def create_database_schema(schema_path: str = "sql/schema.sql") -> None:
    """
    Create the database and initialize schema.

    Args:
        schema_path: Path to the schema SQL file.
    """
    db = DatabaseManager()
    try:
        db.initialize_schema(schema_path)
    finally:
        db.close()


def load_parsed_data(csv_path: str) -> int:
    """
    Load parsed CSV data into the database.

    Args:
        csv_path: Path to the parsed CSV file.

    Returns:
        Number of records loaded.
    """
    db = DatabaseManager()
    loader = InsiderTradingLoader(db)
    
    try:
        count = loader.load_csv(csv_path)
        loader.update_insider_statistics()
        return count
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load insider trading data to database")
    parser.add_argument("--init-schema", action="store_true", help="Initialize database schema")
    parser.add_argument("--load-csv", type=str, help="Path to CSV file to load")
    parser.add_argument("--update-stats", action="store_true", help="Update insider statistics")
    parser.add_argument("--refresh-views", action="store_true", help="Refresh materialized views")
    
    args = parser.parse_args()
    
    if args.init_schema:
        print("Initializing database schema...")
        create_database_schema()
        print("Schema initialized successfully")
    
    if args.load_csv:
        print(f"Loading CSV: {args.load_csv}")
        count = load_parsed_data(args.load_csv)
        print(f"Loaded {count} records")
    
    if args.update_stats or args.refresh_views:
        db = DatabaseManager()
        loader = InsiderTradingLoader(db)
        
        if args.update_stats:
            print("Updating insider statistics...")
            loader.update_insider_statistics()
        
        if args.refresh_views:
            print("Refreshing materialized views...")
            loader.refresh_materialized_views()
        
        db.close()
        print("Done")
