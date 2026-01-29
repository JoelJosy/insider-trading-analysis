-- Insider Trading Analysis Database Schema
-- PostgreSQL Star Schema Data Warehouse
-- ========================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- DIMENSION TABLES
-- ============================================

-- Dimension: Time
-- Contains date-related attributes for temporal analysis
CREATE TABLE IF NOT EXISTS dim_time (
    time_key SERIAL PRIMARY KEY,
    date DATE UNIQUE NOT NULL,
    year INTEGER NOT NULL,
    quarter INTEGER NOT NULL,
    month INTEGER NOT NULL,
    week_of_year INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,  -- 0=Monday, 6=Sunday
    day_of_month INTEGER NOT NULL,
    day_name VARCHAR(10) NOT NULL,
    month_name VARCHAR(10) NOT NULL,
    is_weekend BOOLEAN NOT NULL,
    is_quarter_end BOOLEAN NOT NULL,
    is_month_end BOOLEAN NOT NULL,
    is_earnings_week BOOLEAN DEFAULT FALSE,
    days_to_next_earnings INTEGER,
    days_from_last_earnings INTEGER,
    
    -- Indexes for common queries
    CONSTRAINT valid_quarter CHECK (quarter BETWEEN 1 AND 4),
    CONSTRAINT valid_month CHECK (month BETWEEN 1 AND 12),
    CONSTRAINT valid_day_of_week CHECK (day_of_week BETWEEN 0 AND 6)
);

CREATE INDEX IF NOT EXISTS idx_dim_time_date ON dim_time(date);
CREATE INDEX IF NOT EXISTS idx_dim_time_year_month ON dim_time(year, month);


-- Dimension: Company (Issuer)
-- Information about companies whose insiders are trading
CREATE TABLE IF NOT EXISTS dim_company (
    company_key SERIAL PRIMARY KEY,
    cik VARCHAR(20) UNIQUE NOT NULL,
    ticker VARCHAR(10),
    company_name VARCHAR(500),
    sector VARCHAR(100),
    industry VARCHAR(200),
    market_cap_category VARCHAR(20),  -- 'Large', 'Mid', 'Small', 'Micro'
    market_cap_value NUMERIC(20, 2),
    sp100_member BOOLEAN DEFAULT FALSE,
    sp500_member BOOLEAN DEFAULT FALSE,
    nasdaq100_member BOOLEAN DEFAULT FALSE,
    exchange VARCHAR(20),
    employee_count INTEGER,
    
    -- SCD Type 2 fields (for tracking changes over time)
    effective_start_date DATE NOT NULL DEFAULT CURRENT_DATE,
    effective_end_date DATE,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dim_company_cik ON dim_company(cik);
CREATE INDEX IF NOT EXISTS idx_dim_company_ticker ON dim_company(ticker);
CREATE INDEX IF NOT EXISTS idx_dim_company_sector ON dim_company(sector);
CREATE INDEX IF NOT EXISTS idx_dim_company_current ON dim_company(is_current) WHERE is_current = TRUE;


-- Dimension: Insider
-- Information about company insiders
CREATE TABLE IF NOT EXISTS dim_insider (
    insider_key SERIAL PRIMARY KEY,
    cik VARCHAR(20) UNIQUE,
    name VARCHAR(500) NOT NULL,
    typical_role VARCHAR(50),  -- 'CEO', 'CFO', 'Director', '10%_Owner', 'Officer', 'Other'
    
    -- Historical aggregates (updated periodically)
    first_seen_date DATE,
    last_seen_date DATE,
    total_trades_historical INTEGER DEFAULT 0,
    total_buys_historical INTEGER DEFAULT 0,
    total_sells_historical INTEGER DEFAULT 0,
    total_value_bought NUMERIC(20, 2) DEFAULT 0,
    total_value_sold NUMERIC(20, 2) DEFAULT 0,
    avg_trade_size NUMERIC(20, 2),
    
    -- Unique companies traded
    num_companies_traded INTEGER DEFAULT 1,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dim_insider_cik ON dim_insider(cik);
CREATE INDEX IF NOT EXISTS idx_dim_insider_name ON dim_insider(name);
CREATE INDEX IF NOT EXISTS idx_dim_insider_role ON dim_insider(typical_role);


-- Dimension: Transaction Type
-- Lookup table for transaction codes
CREATE TABLE IF NOT EXISTS dim_transaction_type (
    type_key SERIAL PRIMARY KEY,
    code CHAR(1) UNIQUE NOT NULL,
    description VARCHAR(200) NOT NULL,
    category VARCHAR(50) NOT NULL,  -- 'Open Market', 'Grant/Award', 'Exercise', 'Other'
    is_open_market BOOLEAN NOT NULL DEFAULT FALSE,
    is_buy_equivalent BOOLEAN,  -- TRUE for buys, FALSE for sells, NULL for neutral
    is_derivative BOOLEAN NOT NULL DEFAULT FALSE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed transaction type dimension
INSERT INTO dim_transaction_type (code, description, category, is_open_market, is_buy_equivalent, is_derivative)
VALUES
    ('P', 'Open Market Purchase', 'Open Market', TRUE, TRUE, FALSE),
    ('S', 'Open Market Sale', 'Open Market', TRUE, FALSE, FALSE),
    ('A', 'Award/Grant of Securities', 'Grant/Award', FALSE, TRUE, FALSE),
    ('D', 'Disposition to Issuer', 'Disposition', FALSE, FALSE, FALSE),
    ('F', 'Payment of Exercise Price or Tax Liability', 'Tax/Exercise', FALSE, FALSE, FALSE),
    ('I', 'Discretionary Transaction', 'Discretionary', FALSE, NULL, FALSE),
    ('M', 'Exercise of Derivative Security', 'Exercise', FALSE, NULL, TRUE),
    ('C', 'Conversion of Derivative Security', 'Conversion', FALSE, NULL, TRUE),
    ('E', 'Expiration of Short Derivative Position', 'Expiration', FALSE, NULL, TRUE),
    ('H', 'Expiration of Long Derivative Position', 'Expiration', FALSE, NULL, TRUE),
    ('O', 'Exercise of Out-of-Money Derivative', 'Exercise', FALSE, NULL, TRUE),
    ('X', 'Exercise of In-the-Money Derivative', 'Exercise', FALSE, NULL, TRUE),
    ('G', 'Gift', 'Other', FALSE, NULL, FALSE),
    ('L', 'Small Acquisition', 'Other', FALSE, TRUE, FALSE),
    ('W', 'Acquisition or Disposition by Will/Laws', 'Other', FALSE, NULL, FALSE),
    ('Z', 'Deposit into Voting Trust', 'Other', FALSE, NULL, FALSE),
    ('J', 'Other Acquisition or Disposition', 'Other', FALSE, NULL, FALSE),
    ('K', 'Equity Swap or Similar Instrument', 'Derivative', FALSE, NULL, TRUE),
    ('U', 'Disposition pursuant to tender offer', 'Tender', FALSE, FALSE, FALSE),
    ('V', 'Transaction Voluntarily Reported Early', 'Other', FALSE, NULL, FALSE)
ON CONFLICT (code) DO NOTHING;


-- ============================================
-- FACT TABLES
-- ============================================

-- Fact: Insider Trades
-- Main fact table containing all insider transactions
CREATE TABLE IF NOT EXISTS fact_insider_trades (
    trade_id SERIAL, -- Removed PRIMARY KEY from here
    
    -- Foreign keys to dimensions
    insider_key INTEGER NOT NULL REFERENCES dim_insider(insider_key),
    company_key INTEGER NOT NULL REFERENCES dim_company(company_key),
    time_key INTEGER NOT NULL REFERENCES dim_time(time_key),
    transaction_type_key INTEGER REFERENCES dim_transaction_type(type_key),
    
    -- SEC Filing identifiers
    accession_number VARCHAR(50) NOT NULL,
    filing_date DATE NOT NULL,
    transaction_date DATE,
    
    -- Transaction details (measures)
    shares_traded NUMERIC(20, 4),
    price_per_share NUMERIC(20, 6),
    total_value NUMERIC(20, 2),
    
    -- Ownership details
    ownership_percentage_before NUMERIC(10, 6),
    ownership_percentage_after NUMERIC(10, 6),
    shares_owned_after NUMERIC(20, 4),
    
    -- Transaction characteristics
    acquired_disposed CHAR(1),  -- 'A' or 'D'
    direct_indirect CHAR(1),    -- 'D' or 'I'
    is_derivative BOOLEAN NOT NULL DEFAULT FALSE,
    is_amendment BOOLEAN NOT NULL DEFAULT FALSE,
    
    -- 10b5-1 plan flag
    has_10b5_1_plan BOOLEAN DEFAULT FALSE,
    
    -- Security info
    security_title VARCHAR(500),
    
    -- Derivative-specific fields
    conversion_price NUMERIC(20, 6),
    exercise_date DATE,
    expiration_date DATE,
    underlying_shares NUMERIC(20, 4),
    
    -- Footnote text (for NLP analysis)
    footnote_text TEXT,
    
    -- Source file reference
    xml_path VARCHAR(1000),
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints for partitioning
    PRIMARY KEY (trade_id, filing_date),
    CONSTRAINT unique_transaction UNIQUE (accession_number, insider_key, transaction_date, shares_traded, price_per_share, filing_date)
) PARTITION BY RANGE (filing_date);

-- Create partitions by year
CREATE TABLE IF NOT EXISTS fact_insider_trades_2020 PARTITION OF fact_insider_trades
    FOR VALUES FROM ('2020-01-01') TO ('2021-01-01');

CREATE TABLE IF NOT EXISTS fact_insider_trades_2021 PARTITION OF fact_insider_trades
    FOR VALUES FROM ('2021-01-01') TO ('2022-01-01');

CREATE TABLE IF NOT EXISTS fact_insider_trades_2022 PARTITION OF fact_insider_trades
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');

CREATE TABLE IF NOT EXISTS fact_insider_trades_2023 PARTITION OF fact_insider_trades
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE IF NOT EXISTS fact_insider_trades_2024 PARTITION OF fact_insider_trades
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS fact_insider_trades_2025 PARTITION OF fact_insider_trades
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

-- Indexes on fact table
CREATE INDEX IF NOT EXISTS idx_fact_trades_insider ON fact_insider_trades(insider_key);
CREATE INDEX IF NOT EXISTS idx_fact_trades_company ON fact_insider_trades(company_key);
CREATE INDEX IF NOT EXISTS idx_fact_trades_time ON fact_insider_trades(time_key);
CREATE INDEX IF NOT EXISTS idx_fact_trades_filing_date ON fact_insider_trades(filing_date);
CREATE INDEX IF NOT EXISTS idx_fact_trades_transaction_date ON fact_insider_trades(transaction_date);
CREATE INDEX IF NOT EXISTS idx_fact_trades_accession ON fact_insider_trades(accession_number);


-- Fact: Daily Summary
-- Aggregate fact table for daily insider trading activity per company
CREATE TABLE IF NOT EXISTS fact_daily_summary (
    summary_id SERIAL PRIMARY KEY,
    company_key INTEGER NOT NULL REFERENCES dim_company(company_key),
    time_key INTEGER NOT NULL REFERENCES dim_time(time_key),
    date DATE NOT NULL,
    
    -- Buy aggregates
    total_insider_buys INTEGER DEFAULT 0,
    total_shares_bought NUMERIC(20, 4) DEFAULT 0,
    total_value_bought NUMERIC(20, 2) DEFAULT 0,
    avg_buy_price NUMERIC(20, 6),
    
    -- Sell aggregates
    total_insider_sells INTEGER DEFAULT 0,
    total_shares_sold NUMERIC(20, 4) DEFAULT 0,
    total_value_sold NUMERIC(20, 2) DEFAULT 0,
    avg_sell_price NUMERIC(20, 6),
    
    -- Trading activity
    num_unique_insiders_trading INTEGER DEFAULT 0,
    num_ceo_trades INTEGER DEFAULT 0,
    num_cfo_trades INTEGER DEFAULT 0,
    num_director_trades INTEGER DEFAULT 0,
    
    -- Net metrics
    net_shares NUMERIC(20, 4) DEFAULT 0,  -- bought - sold
    net_value NUMERIC(20, 2) DEFAULT 0,
    
    -- Sentiment score (-1 to 1, based on buy/sell ratio)
    net_sentiment_score NUMERIC(5, 4),
    
    -- 10b5-1 plan trades
    num_10b5_1_trades INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_daily_summary UNIQUE (company_key, date)
);

CREATE INDEX IF NOT EXISTS idx_fact_daily_company ON fact_daily_summary(company_key);
CREATE INDEX IF NOT EXISTS idx_fact_daily_date ON fact_daily_summary(date);


-- ============================================
-- SUPPLEMENTARY TABLES
-- ============================================

-- Stock price data
CREATE TABLE IF NOT EXISTS stock_prices (
    price_id SERIAL PRIMARY KEY,
    company_key INTEGER NOT NULL REFERENCES dim_company(company_key),
    date DATE NOT NULL,
    
    open_price NUMERIC(20, 6),
    high_price NUMERIC(20, 6),
    low_price NUMERIC(20, 6),
    close_price NUMERIC(20, 6),
    adjusted_close NUMERIC(20, 6),
    volume BIGINT,
    
    -- Calculated metrics
    daily_return NUMERIC(10, 6),
    volatility_20d NUMERIC(10, 6),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_price UNIQUE (company_key, date)
);

CREATE INDEX IF NOT EXISTS idx_stock_prices_company_date ON stock_prices(company_key, date);


-- Earnings announcements
CREATE TABLE IF NOT EXISTS earnings_announcements (
    earnings_id SERIAL PRIMARY KEY,
    company_key INTEGER NOT NULL REFERENCES dim_company(company_key),
    announcement_date DATE NOT NULL,
    
    fiscal_quarter VARCHAR(10),
    fiscal_year INTEGER,
    
    eps_estimate NUMERIC(10, 4),
    eps_actual NUMERIC(10, 4),
    eps_surprise NUMERIC(10, 4),
    eps_surprise_percent NUMERIC(10, 4),
    
    revenue_estimate NUMERIC(20, 2),
    revenue_actual NUMERIC(20, 2),
    revenue_surprise NUMERIC(20, 2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT unique_earnings UNIQUE (company_key, announcement_date, fiscal_quarter)
);

CREATE INDEX IF NOT EXISTS idx_earnings_company_date ON earnings_announcements(company_key, announcement_date);


-- SEC Enforcement Actions (for validation)
CREATE TABLE IF NOT EXISTS sec_enforcement (
    enforcement_id SERIAL PRIMARY KEY,
    case_number VARCHAR(100),
    case_title VARCHAR(1000),
    case_date DATE,
    case_url VARCHAR(2000),
    
    -- Parsed details
    involves_insider_trading BOOLEAN DEFAULT TRUE,
    defendant_names TEXT[],
    company_names TEXT[],
    company_tickers TEXT[],
    
    -- Linked entities
    insider_keys INTEGER[],
    company_keys INTEGER[],
    
    -- Case details
    case_summary TEXT,
    settlement_amount NUMERIC(20, 2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_enforcement_date ON sec_enforcement(case_date);


-- ============================================
-- MATERIALIZED VIEWS
-- ============================================

-- Materialized view: Recent high-value trades
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_recent_high_value_trades AS
SELECT 
    t.trade_id,
    c.ticker,
    c.company_name,
    i.name AS insider_name,
    i.typical_role,
    d.date AS trade_date,
    tt.description AS transaction_type,
    t.shares_traded,
    t.price_per_share,
    t.total_value,
    t.has_10b5_1_plan
FROM fact_insider_trades t
JOIN dim_company c ON t.company_key = c.company_key
JOIN dim_insider i ON t.insider_key = i.insider_key
JOIN dim_time d ON t.time_key = d.time_key
LEFT JOIN dim_transaction_type tt ON t.transaction_type_key = tt.type_key
WHERE t.total_value >= 100000
  AND t.filing_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY t.total_value DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_recent_trades ON mv_recent_high_value_trades(trade_id);


-- Materialized view: Insider trading activity by company
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_company_insider_activity AS
SELECT 
    c.company_key,
    c.ticker,
    c.company_name,
    c.sector,
    COUNT(t.trade_id) AS total_trades,
    COUNT(DISTINCT i.insider_key) AS unique_insiders,
    SUM(CASE WHEN tt.is_buy_equivalent THEN 1 ELSE 0 END) AS total_buys,
    SUM(CASE WHEN tt.is_buy_equivalent = FALSE THEN 1 ELSE 0 END) AS total_sells,
    SUM(CASE WHEN tt.is_buy_equivalent THEN t.total_value ELSE 0 END) AS value_bought,
    SUM(CASE WHEN tt.is_buy_equivalent = FALSE THEN t.total_value ELSE 0 END) AS value_sold,
    AVG(t.total_value) AS avg_trade_value
FROM dim_company c
LEFT JOIN fact_insider_trades t ON c.company_key = t.company_key
LEFT JOIN dim_insider i ON t.insider_key = i.insider_key
LEFT JOIN dim_transaction_type tt ON t.transaction_type_key = tt.type_key
WHERE c.is_current = TRUE
GROUP BY c.company_key, c.ticker, c.company_name, c.sector;

CREATE UNIQUE INDEX IF NOT EXISTS idx_mv_company_activity ON mv_company_insider_activity(company_key);


-- ============================================
-- UTILITY FUNCTIONS
-- ============================================

-- Function to populate dim_time for a date range
CREATE OR REPLACE FUNCTION populate_dim_time(start_date DATE, end_date DATE)
RETURNS INTEGER AS $$
DECLARE
    current_date_iter DATE := start_date;
    rows_inserted INTEGER := 0;
BEGIN
    WHILE current_date_iter <= end_date LOOP
        INSERT INTO dim_time (
            date, year, quarter, month, week_of_year, day_of_week, day_of_month,
            day_name, month_name, is_weekend, is_quarter_end, is_month_end
        )
        VALUES (
            current_date_iter,
            EXTRACT(YEAR FROM current_date_iter)::INTEGER,
            EXTRACT(QUARTER FROM current_date_iter)::INTEGER,
            EXTRACT(MONTH FROM current_date_iter)::INTEGER,
            EXTRACT(WEEK FROM current_date_iter)::INTEGER,
            EXTRACT(DOW FROM current_date_iter)::INTEGER,
            EXTRACT(DAY FROM current_date_iter)::INTEGER,
            TO_CHAR(current_date_iter, 'Day'),
            TO_CHAR(current_date_iter, 'Month'),
            EXTRACT(DOW FROM current_date_iter) IN (0, 6),
            current_date_iter = DATE_TRUNC('quarter', current_date_iter) + INTERVAL '3 months' - INTERVAL '1 day',
            current_date_iter = DATE_TRUNC('month', current_date_iter + INTERVAL '1 month') - INTERVAL '1 day'
        )
        ON CONFLICT (date) DO NOTHING;
        
        rows_inserted := rows_inserted + 1;
        current_date_iter := current_date_iter + INTERVAL '1 day';
    END LOOP;
    
    RETURN rows_inserted;
END;
$$ LANGUAGE plpgsql;


-- Function to refresh all materialized views
CREATE OR REPLACE FUNCTION refresh_all_materialized_views()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_recent_high_value_trades;
    REFRESH MATERIALIZED VIEW CONCURRENTLY mv_company_insider_activity;
END;
$$ LANGUAGE plpgsql;


-- Function to update daily summary
CREATE OR REPLACE FUNCTION update_daily_summary(target_date DATE)
RETURNS VOID AS $$
BEGIN
    INSERT INTO fact_daily_summary (
        company_key, time_key, date,
        total_insider_buys, total_shares_bought, total_value_bought, avg_buy_price,
        total_insider_sells, total_shares_sold, total_value_sold, avg_sell_price,
        num_unique_insiders_trading, net_shares, net_value, net_sentiment_score
    )
    SELECT 
        t.company_key,
        d.time_key,
        target_date,
        SUM(CASE WHEN tt.is_buy_equivalent THEN 1 ELSE 0 END),
        SUM(CASE WHEN tt.is_buy_equivalent THEN t.shares_traded ELSE 0 END),
        SUM(CASE WHEN tt.is_buy_equivalent THEN t.total_value ELSE 0 END),
        AVG(CASE WHEN tt.is_buy_equivalent THEN t.price_per_share END),
        SUM(CASE WHEN tt.is_buy_equivalent = FALSE THEN 1 ELSE 0 END),
        SUM(CASE WHEN tt.is_buy_equivalent = FALSE THEN t.shares_traded ELSE 0 END),
        SUM(CASE WHEN tt.is_buy_equivalent = FALSE THEN t.total_value ELSE 0 END),
        AVG(CASE WHEN tt.is_buy_equivalent = FALSE THEN t.price_per_share END),
        COUNT(DISTINCT t.insider_key),
        SUM(CASE WHEN tt.is_buy_equivalent THEN t.shares_traded 
                 WHEN tt.is_buy_equivalent = FALSE THEN -t.shares_traded 
                 ELSE 0 END),
        SUM(CASE WHEN tt.is_buy_equivalent THEN t.total_value 
                 WHEN tt.is_buy_equivalent = FALSE THEN -t.total_value 
                 ELSE 0 END),
        CASE 
            WHEN SUM(1) = 0 THEN 0
            ELSE (SUM(CASE WHEN tt.is_buy_equivalent THEN 1 ELSE 0 END)::NUMERIC - 
                  SUM(CASE WHEN tt.is_buy_equivalent = FALSE THEN 1 ELSE 0 END)::NUMERIC) / 
                  NULLIF(SUM(1)::NUMERIC, 0)
        END
    FROM fact_insider_trades t
    JOIN dim_time d ON d.date = target_date
    LEFT JOIN dim_transaction_type tt ON t.transaction_type_key = tt.type_key
    WHERE t.transaction_date = target_date
    GROUP BY t.company_key, d.time_key
    ON CONFLICT (company_key, date) DO UPDATE SET
        total_insider_buys = EXCLUDED.total_insider_buys,
        total_shares_bought = EXCLUDED.total_shares_bought,
        total_value_bought = EXCLUDED.total_value_bought,
        avg_buy_price = EXCLUDED.avg_buy_price,
        total_insider_sells = EXCLUDED.total_insider_sells,
        total_shares_sold = EXCLUDED.total_shares_sold,
        total_value_sold = EXCLUDED.total_value_sold,
        avg_sell_price = EXCLUDED.avg_sell_price,
        num_unique_insiders_trading = EXCLUDED.num_unique_insiders_trading,
        net_shares = EXCLUDED.net_shares,
        net_value = EXCLUDED.net_value,
        net_sentiment_score = EXCLUDED.net_sentiment_score,
        updated_at = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;


-- Initialize dim_time with dates from 2019 to 2026
SELECT populate_dim_time('2019-01-01'::DATE, '2026-12-31'::DATE);
