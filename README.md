# Insider Trading Analysis

> ML system that analyzes SEC Form 4 insider trading filings to classify routine vs informed trades and predict future stock movement signals.

Pattern Recognition Project | Data Warehousing Course

---

## 🎯 Project Overview

This research project implements an **Explainable Anomaly Detection** system for insider trading patterns using multi-source validation. The system combines:

- Temporal Sequence Modeling (LSTM) on insider trading history
- Network Coordination Features (graph analysis of multiple insiders)
- Explainability Layer (SHAP values for each prediction)
- Multi-source Validation (SEC enforcement + earnings surprises + stock returns)
- Ablation Study (which features/methods contribute most)

---

## 🏗️ Project Structure

```
insider-trading-analysis/
├── config/
│   └── config.yaml          # Central configuration
├── data/
│   ├── raw/                  # SEC XML filings
│   ├── processed/            # Parsed CSVs
│   └── external/             # Stock prices, earnings
├── src/
│   ├── data/
│   │   ├── extract.py        # SEC EDGAR downloader & parser
│   │   ├── load.py           # Database loader
│   │   └── quality.py        # Data quality checks
│   ├── features/             # Feature engineering (coming soon)
│   ├── models/               # ML models (coming soon)
│   ├── evaluation/           # Metrics & explainability (coming soon)
│   └── utils/
│       ├── config.py         # Configuration loader
│       └── logger.py         # Logging utilities
├── sql/
│   └── schema.sql            # PostgreSQL star schema
├── app/                      # Streamlit dashboard (coming soon)
├── tests/                    # Unit tests
├── notebooks/                # Jupyter notebooks
├── reports/                  # Data quality reports
├── .env                      # Environment variables
├── requirements.txt          # Base dependencies
├── requirements-lock.txt     # Exact dependency versions (generated via pip freeze)
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.11 or 3.12 (recommended)**
  > Python 3.13 is not recommended as some ML libraries (Torch, SciPy, etc.) may not be fully supported.
- PostgreSQL 14+ (running on localhost:5432) — **only needed for Steps 3–5**
- Git

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd insider-trading-analysis
```

### 2. Create a virtual environment

**Mac/Linux**

```bash
python3.11 -m venv venv
source venv/bin/activate
```

**Windows**

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

For exact reproducibility across machines:

```bash
pip install --upgrade pip
pip install -r requirements-lock.txt
```

If you only need base dependencies:

```bash
pip install -r requirements.txt
```

### 4. Configure environment

Copy `.env.example` to `.env` and fill in your PostgreSQL credentials:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=insider_trading_db
DB_USER=postgres
DB_PASSWORD=your_password
```

Review `config/config.yaml` for rate limits, date ranges, and model hyperparameters.

---

## 🔁 Pipeline — How It Works

The pipeline has **5 sequential steps**. Steps 1–2 run entirely on disk (no database needed). Steps 3–5 require PostgreSQL.

```
[SEC EDGAR API]
      │
      ▼
Step 1 — EXTRACT (extract.py)
  Downloads raw XML Form 4 filings to data/raw/
  Parses them into a flat DataFrame
  Saves to data/processed/<TICKER>_form4.csv
      │
      ▼
Step 2 — QUALITY CHECK (quality.py)   ← optional, can run anytime after Step 1
  Reads the processed CSV
  Generates warnings, statistics, recommendations
  Saves reports/  ← JSON + Markdown quality report
      │
      ▼
Step 3 — INITIALIZE DATABASE (load.py --init-schema)
  Creates the PostgreSQL database (if it doesn't exist)
  Runs sql/schema.sql to create all tables
      │
      ▼
Step 4 — LOAD (load.py --load-csv)
  Reads the processed CSV
  Populates dim_company, dim_insider, dim_time, dim_transaction_type
  Inserts rows into fact_insider_trades
  Updates dim_insider aggregate statistics
      │
      ▼
Step 5 — (Future) FEATURES / MODELS
  Feature engineering, labeling, ML training
```

---

## ▶️ Running the Pipeline

> Run all commands from the project root with the venv activated.

### Step 1 — Download & Parse SEC filings

```bash
python -m src.data.extract --ticker AAPL --num-filings 500
```

**What it does:**

- Fetches up to 100 Form 4 filings for AAPL from SEC EDGAR (respects the 10 req/s rate limit)
- Saves raw XML files under `data/raw/sec-edgar-filings/AAPL/4/<accession>/`
- Parses all XML files into a flat table of transactions
- Writes `data/processed/AAPL_form4.csv` (~294 rows for 100 filings)

**What it updates:**
| Location | Detail |
|---|---|
| `data/raw/sec-edgar-filings/` | Raw `.txt` XML filing bundles downloaded from SEC |
| `data/processed/AAPL_form4.csv` | Parsed transactions (one row per transaction, not per filing) |
| `logs/extract.log`, `logs/parser.log`, `logs/downloader.log` | Execution logs |

---

### Step 2 — Generate quality report _(optional)_

```bash
python -m src.data.quality --csv data/processed/AAPL_form4.csv
```

**What it does:**

- Reads the processed CSV
- Computes per-column stats (missing %, outliers, cardinality, distributions)
- Runs domain-specific checks (negative prices, future dates, transaction imbalance)

**What it updates:**
| Location | Detail |
|---|---|
| `reports/AAPL_form4_quality_report.json` | Full machine-readable report |
| `reports/AAPL_form4_quality_report.md` | Human-readable Markdown report |
| `logs/quality.log` | Execution log |

> You can re-run this at any time after Step 1, without touching the database.

---

### Step 3 — Initialize the database _(one-time setup)_

> **Requires PostgreSQL running.** You do NOT need pgAdmin — this runs from the terminal.

```bash
python -m src.data.load --init-schema
```

**What it does:**

- Connects to PostgreSQL as the user in `.env`
- Creates the `insider_trading_db` database if it doesn't exist
- Executes `sql/schema.sql` to create all tables, indexes, and constraints

**What it updates:**
| Location | Detail |
|---|---|
| PostgreSQL | Creates `insider_trading_db` database |
| PostgreSQL tables | `dim_time`, `dim_company`, `dim_insider`, `dim_transaction_type`, `fact_insider_trades`, `stock_prices`, `earnings_announcements`, `sec_enforcement` |
| `logs/db.log` | Execution log |

> Only run this once (or after dropping the database to reset).

---

### Step 4 — Load data into the database

```bash
python -m src.data.load --load-csv data/processed/AAPL_form4.csv
```

**What it does:**

- Reads the processed CSV
- Upserts companies into `dim_company` (keyed by CIK)
- Upserts insiders into `dim_insider` (keyed by CIK)
- Looks up date keys from `dim_time`
- Looks up transaction type keys from `dim_transaction_type`
- Batch-inserts transactions into `fact_insider_trades` (skips duplicates via `ON CONFLICT`)
- Runs an aggregate `UPDATE` on `dim_insider` to refresh total trade counts, buy/sell totals, average trade size, first/last seen dates

**What it updates:**
| Table | What changes |
|---|---|
| `dim_company` | New company row inserted or ticker updated |
| `dim_insider` | New insider row inserted or role updated |
| `fact_insider_trades` | New transaction rows inserted (duplicates skipped) |
| `dim_insider` (stats) | `total_trades_historical`, `total_buys_historical`, `total_sells_historical`, `total_value_bought`, `total_value_sold`, `avg_trade_size`, `first_seen_date`, `last_seen_date`, `updated_at` |
| `logs/loader.log` | Execution log |

---

### Optional standalone commands

```bash
# Only refresh insider aggregate stats (after bulk loads)
python -m src.data.load --update-stats

# Refresh materialized views (if defined in schema)
python -m src.data.load --refresh-views
```

---

**No, pgAdmin is not required to run the pipeline.** All steps run from the terminal.

pgAdmin (or any SQL client like TablePlus, DBeaver, psql) is useful **only if** you want to:

- Browse the loaded data visually
- Write ad-hoc SQL queries against the star schema
- Inspect tables after `--init-schema` or `--load-csv`

To connect in any SQL client use: `host=localhost port=5432 db=insider_trading_db user=postgres`

---

## 📦 Dependency Management (Team Workflow)

After installing or adding new libraries:

```bash
pip freeze > requirements-lock.txt
```

Commit this file so all collaborators use **identical package versions**.

**Team recommendations:**

- Python: 3.11 (preferred) or 3.12
- Avoid Python 3.13 unless all libraries support it

---

## 📊 Database Schema

The system uses a PostgreSQL star schema with:

### Fact Tables

- `fact_insider_trades` – Main transaction fact table (partitioned by year)
- `fact_daily_summary` – Aggregated daily insider activity

### Dimension Tables

- `dim_time` – Date attributes
- `dim_company` – Company information
- `dim_insider` – Insider details
- `dim_transaction_type` – Transaction code lookup

### Supplementary Tables

- `stock_prices` – Daily OHLCV data
- `earnings_announcements` – Earnings surprises
- `sec_enforcement` – SEC litigation cases

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📈 Development Phases

- [x] Phase 1: Data Infrastructure (ETL, Database)
- [ ] Phase 2: Feature Engineering
- [ ] Phase 3: Labeling & Ground Truth
- [ ] Phase 4: Model Development
- [ ] Phase 5: Validation & Evaluation
- [ ] Phase 6: Dashboard & Demo
- [ ] Phase 7: Research Paper

---

## 📝 Configuration

Key settings in `config/config.yaml`:

```yaml
database:
  host: localhost
  port: 5432
  name: insider_trading_db
  password: supa # Update via .env

sec_edgar:
  rate_limit_per_second: 10
  date_range:
    start: "2020-01-01"
    end: "2024-12-31"

models:
  random_seed: 42
  lstm:
    hidden_size: 64
    dropout: 0.3
```

---

## 📄 License

This project is for academic research purposes.

---

## 🙏 Acknowledgments

- SEC EDGAR for providing public filing data
- Yahoo Finance for market data
