import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from sqlalchemy import create_engine

from src.utils.config import get_config, DatabaseConfig
from src.utils.logger import setup_logger

logger = setup_logger(__name__, "logs/load.log")

def _san(v):
    try:
        return None if v == "NaN" or (not isinstance(v, str) and pd.isna(v)) else v
    except (TypeError, ValueError):
        return v


class DatabaseManager:
    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or get_config().database
        self.logger = setup_logger(self.__class__.__name__, "logs/db.log")
        self._connection = None
        self._engine = None

    @property
    def connection(self) -> psycopg2.extensions.connection:
        if self._connection is None or self._connection.closed:
            self._connection = psycopg2.connect(**self.config.psycopg2_params)
        return self._connection

    @property
    def engine(self):
        if self._engine is None:
            self._engine = create_engine(self.config.connection_string)
        return self._engine

    def close(self):
        if self._connection and not self._connection.closed:
            self._connection.close()
        if self._engine:
            self._engine.dispose()

    def execute_sql_file(self, sql_path: str) -> None:
        content = Path(sql_path).read_text()
        conn = self.connection
        try:
            with conn.cursor() as cur:
                cur.execute(content)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def create_database(self) -> None:
        params = {**self.config.psycopg2_params, "dbname": "postgres"}
        db_name = self.config.psycopg2_params["dbname"]
        conn = psycopg2.connect(**params)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
                if not cur.fetchone():
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
                    self.logger.info(f"Created database: {db_name}")
        finally:
            conn.close()

    def initialize_schema(self, schema_path: str = "sql/schema.sql") -> None:
        self.create_database()
        full_path = Path(__file__).parent.parent.parent / schema_path
        if full_path.exists():
            self.execute_sql_file(str(full_path))


class InsiderTradingLoader:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.logger = setup_logger(self.__class__.__name__, "logs/loader.log")
        self._company_cache: Dict[str, int] = {}
        self._insider_cache: Dict[str, int] = {}
        self._time_cache: Dict[date, int] = {}
        self._type_cache: Dict[str, int] = {}

    def _get_or_create_company(self, cik: str, ticker: Optional[str], name: Optional[str]) -> int:
        cik = str(cik).split(".")[0]
        if cik in self._company_cache:
            return self._company_cache[cik]
        conn = self.db.connection
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO dim_company (cik, ticker, company_name, is_current)
                VALUES (%s, %s, %s, TRUE)
                ON CONFLICT (cik) DO UPDATE SET ticker = EXCLUDED.ticker
                RETURNING company_key
            """, (cik, ticker, name))
            key = cur.fetchone()[0]
            conn.commit()
        self._company_cache[cik] = key
        return key

    def _get_or_create_insider(self, cik: Optional[str], name: str, role: Optional[str]) -> int:
        if cik:
            cik = str(cik).split(".")[0]
        cache_key = cik if cik else hashlib.md5(name.encode()).hexdigest()[:20]
        if cache_key in self._insider_cache:
            return self._insider_cache[cache_key]
        conn = self.db.connection
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO dim_insider (cik, name, typical_role, first_seen_date)
                VALUES (%s, %s, %s, CURRENT_DATE)
                ON CONFLICT (cik) DO UPDATE SET typical_role = EXCLUDED.typical_role
                RETURNING insider_key
            """, (cik, name, role))
            key = cur.fetchone()[0]
            conn.commit()
        self._insider_cache[cache_key] = key
        return key

    def _get_time_key(self, d: date) -> Optional[int]:
        if d in self._time_cache:
            return self._time_cache[d]
        with self.db.connection.cursor() as cur:
            cur.execute("SELECT time_key FROM dim_time WHERE date = %s", (d,))
            row = cur.fetchone()
        if row:
            self._time_cache[d] = row[0]
            return row[0]
        self.logger.warning(f"Missing time dimension for: {d}")
        return None

    def _get_type_key(self, code: Optional[str]) -> Optional[int]:
        if not code:
            return None
        if code in self._type_cache:
            return self._type_cache[code]
        with self.db.connection.cursor() as cur:
            cur.execute("SELECT type_key FROM dim_transaction_type WHERE code = %s", (code,))
            row = cur.fetchone()
        if row:
            self._type_cache[code] = row[0]
            return row[0]
        return None

    def _coerce_date(self, val) -> Optional[date]:
        if val is None or (not isinstance(val, str) and pd.isna(val)):
            return None
        if isinstance(val, (datetime, pd.Timestamp)):
            return val.date()
        return val

    def load_dataframe(self, df: pd.DataFrame, batch_size: int = 1000) -> int:
        if df.empty:
            return 0
        conn = self.db.connection
        records = []
        for idx, row in df.iterrows():
            try:
                company_key = self._get_or_create_company(
                    str(row.get("issuer_cik", "")), row.get("ticker"), row.get("issuer_name")
                )
                insider_key = self._get_or_create_insider(
                    row.get("insider_cik"), row.get("insider_name", "Unknown"), row.get("insider_role")
                )
                trade_date = self._coerce_date(row.get("transaction_date")) or self._coerce_date(row.get("filing_date"))
                if trade_date is None:
                    continue
                time_key = self._get_time_key(trade_date)
                if time_key is None:
                    continue
                records.append((
                    insider_key, company_key, time_key,
                    self._get_type_key(row.get("transaction_code")),
                    row.get("accession_number", ""),
                    _san(row.get("filing_date")), _san(row.get("transaction_date")),
                    _san(row.get("shares")), _san(row.get("price_per_share")),
                    _san(row.get("total_value")), _san(row.get("shares_owned_after")),
                    _san(row.get("acquired_disposed")), _san(row.get("direct_indirect")),
                    bool(row.get("is_derivative", False)), bool(row.get("is_amendment", False)),
                    bool(row.get("has_10b5_1_plan", False)),
                    _san(row.get("security_title")), _san(row.get("conversion_price")),
                    _san(row.get("exercise_date")), _san(row.get("expiration_date")),
                    _san(row.get("underlying_shares")), _san(row.get("footnote_text")),
                    _san(row.get("xml_path")),
                ))
            except Exception as e:
                self.logger.error(f"Error processing row {idx}: {e}")
                conn.rollback()
        if not records:
            return 0
        try:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """INSERT INTO fact_insider_trades (
                        insider_key, company_key, time_key, transaction_type_key,
                        accession_number, filing_date, transaction_date,
                        shares_traded, price_per_share, total_value,
                        shares_owned_after, acquired_disposed, direct_indirect,
                        is_derivative, is_amendment, has_10b5_1_plan,
                        security_title, conversion_price, exercise_date,
                        expiration_date, underlying_shares, footnote_text, xml_path
                    ) VALUES %s ON CONFLICT ON CONSTRAINT unique_transaction DO NOTHING""",
                    records,
                    page_size=batch_size,
                )
                count = cur.rowcount
            conn.commit()
            self.logger.info(f"Loaded {count} records")
            return count
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Batch insert failed: {e}")
            raise

    def load_csv(self, csv_path: str, batch_size: int = 1000) -> int:
        df = pd.read_csv(csv_path, parse_dates=["filing_date", "transaction_date"])
        return self.load_dataframe(df, batch_size)

    def update_insider_statistics(self) -> None:
        conn = self.db.connection
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE dim_insider i
                    SET total_trades_historical = s.total_trades,
                        total_buys_historical   = s.total_buys,
                        total_sells_historical  = s.total_sells,
                        total_value_bought      = s.value_bought,
                        total_value_sold        = s.value_sold,
                        avg_trade_size          = s.avg_value,
                        first_seen_date         = s.first_date,
                        last_seen_date          = s.last_date,
                        updated_at              = CURRENT_TIMESTAMP
                    FROM (
                        SELECT t.insider_key,
                            COUNT(*) as total_trades,
                            SUM(CASE WHEN tt.is_buy_equivalent THEN 1 ELSE 0 END) as total_buys,
                            SUM(CASE WHEN NOT tt.is_buy_equivalent THEN 1 ELSE 0 END) as total_sells,
                            SUM(CASE WHEN tt.is_buy_equivalent THEN t.total_value ELSE 0 END) as value_bought,
                            SUM(CASE WHEN NOT tt.is_buy_equivalent THEN t.total_value ELSE 0 END) as value_sold,
                            AVG(t.total_value) as avg_value,
                            MIN(t.transaction_date) as first_date,
                            MAX(t.transaction_date) as last_date
                        FROM fact_insider_trades t
                        LEFT JOIN dim_transaction_type tt ON t.transaction_type_key = tt.type_key
                        GROUP BY t.insider_key
                    ) s WHERE i.insider_key = s.insider_key
                """)
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to update insider stats: {e}")

    def refresh_materialized_views(self) -> None:
        conn = self.db.connection
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT refresh_all_materialized_views()")
            conn.commit()
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Failed to refresh views: {e}")


def create_database_schema(schema_path: str = "sql/schema.sql") -> None:
    db = DatabaseManager()
    try:
        db.initialize_schema(schema_path)
    finally:
        db.close()


def load_parsed_data(csv_path: str) -> int:
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
    p = argparse.ArgumentParser(description="Load insider trading data to database")
    p.add_argument("--init-schema", action="store_true")
    p.add_argument("--load-csv", type=str)
    p.add_argument("--update-stats", action="store_true")
    p.add_argument("--refresh-views", action="store_true")
    args = p.parse_args()
    if args.init_schema:
        create_database_schema()
        print("Schema initialized")
    if args.load_csv:
        print(f"Loaded {load_parsed_data(args.load_csv)} records")
    if args.update_stats or args.refresh_views:
        db = DatabaseManager()
        loader = InsiderTradingLoader(db)
        if args.update_stats:
            loader.update_insider_statistics()
        if args.refresh_views:
            loader.refresh_materialized_views()
        db.close()
