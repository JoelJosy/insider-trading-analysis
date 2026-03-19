# Insider Trading Analysis

> ML system that analyzes SEC Form 4 insider trading filings to classify routine vs opportunistic trades using behavioral proxies and validate with market/regulatory signals.

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
│   ├── features/             # Feature engineering pipeline
│   ├── labels/               # Phase 3 labeling pipeline
│   ├── models/               # Model training pipelines (NB, XGBoost, Isolation Forest)
│   ├── evaluation/           # Metrics & explainability outputs
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

The pipeline has **6 sequential steps**. Steps 1–2 and Step 6 run entirely on disk (no database needed). Steps 3–5 require PostgreSQL.

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
Step 5 — FEATURES (src/features/pipeline.py)
  Deduplicates, runs 5 feature modules (trade, insider, temporal, network, text)
  Writes data/processed/<TICKER>_form4_features.csv (47 feature columns)
      │
      ▼
Step 6 — LABELING (src/labels/pipeline.py)
  Builds behavioral-proxy labels (routine vs opportunistic)
  using Cohen heuristic + 10b5-1 plan override
  while keeping price/earnings/enforcement signals for validation
      │
      ▼
Step 7 — (Future) MODELS
  ML training, SHAP explainability
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
- Fetches daily close prices from Yahoo Finance (Phase 3 prep)
- Back-fills `total_value` for `M`-code rows when SEC price is missing: `shares * close_price_on_txn_date`
- Writes `data/processed/AAPL_form4.csv` (~294 rows for 100 filings)

**What it updates:**
| Location | Detail |
|---|---|
| `data/raw/sec-edgar-filings/` | Raw `.txt` XML filing bundles downloaded from SEC |
| `data/processed/AAPL_form4.csv` | Parsed transactions (one row per transaction, not per filing) |
| `data/external/prices/` | Cached daily close prices downloaded via `yfinance` |
| `logs/extract.log`, `logs/parser.log`, `logs/downloader.log` | Execution logs |

To skip market-price enrichment (offline or faster parsing):

```bash
python -m src.data.extract --ticker AAPL --num-filings 500 --skip-market-prices
```

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

### Step 5 — Feature Engineering

```bash
python -m src.features.pipeline --input data/processed/AAPL_form4.csv
```

**What it does:**

- Deduplicates the processed CSV
- Runs 5 feature modules: trade → insider → temporal → network → text
- Writes `data/processed/AAPL_form4_features.csv` (724 rows × 47 feature columns)

**Add `--full` to keep all 80+ columns instead of just the feature matrix.**

**What it updates:**
| Location | Detail |
|---|---|
| `data/processed/AAPL_form4_features.csv` | Feature matrix |
| `logs/features.log` | Execution log |

---

### Step 6 — Labeling & Ground Truth (Phase 3)

```bash
python -m src.labels.pipeline --input data/processed/AAPL_form4_features.csv
```

**What it does:**

- Loads your features (or processed) CSV with `ticker` + `transaction_date`
- Computes behavioral labels with open-market filtering:
  - Applies Cohen heuristic only on `transaction_code` in `{P, S}`
  - `cohen_label = 0` (routine) if same insider traded in the same calendar month in 2+ prior consecutive years, else `1` (opportunistic)
  - Applies 10b5-1 override (`has_plan` or `footnote_has_plan` = 1) to force `final_label = 0`
  - Sets `cohen_label = -1` and `final_label = -1` for excluded/non-open-market codes (`M, A, F, G, D`) as **uncertain**
- Also computes outcome validation signals (not training labels):
  - directional abnormal return vs `SPY`
  - `price_signal` from `config.labeling.abnormal_return_threshold`
  - optional earnings/enforcement confirmation flags from:
  - `data/external/earnings/earnings_announcements.csv` (`ticker`, `announcement_date`)
  - `data/external/sec/sec_enforcement.csv` (`ticker`, `action_date`)
- Writes label columns:
  - `cohen_label`, `plan_override`, `final_label`
  - compatibility aliases: `informed_label`, `routine_label`, `label_name`, `label_confidence`
  - validation context: `price_signal`, `abnormal_return`, `label_source_count`, etc.
- Fails fast if benchmark/ticker market prices are unavailable (prevents silent all-routine labels)
- Generates Phase 3 label quality report (JSON + Markdown)

**What it updates:**
| Location | Detail |
|---|---|
| `data/processed/*_labeled.csv` | Final Phase 3 labeled dataset |
| `logs/labels.log` | Labeling execution log |
| `reports/*_label_quality_report.json` | Label-quality metrics (coverage, signal counts, recommendations) |
| `reports/*_label_quality_report.md` | Human-readable label quality summary |

**Useful options:**

```bash
python -m src.labels.pipeline \
  --input data/processed/AAPL_form4_features.csv \
  --output data/processed/AAPL_form4_labeled.csv \
  --horizon-days 30 \
  --benchmark SPY
```

Training-ready subset (exclude uncertain labels):

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("data/processed/AAPL_form4_features_labeled_behavioral.csv")
train = df[df["final_label"].isin([0, 1])].copy()
train.to_csv("data/processed/AAPL_form4_features_labeled_behavioral_train.csv", index=False)
print(train["final_label"].value_counts().to_dict())
PY
```

If you must continue without market prices (debug only):

```bash
python -m src.labels.pipeline --input data/processed/AAPL_form4_features.csv --allow-missing-prices
```

If Yahoo fetch fails repeatedly, upgrade `yfinance` in your environment:

```bash
pip install --upgrade yfinance
```

### Step 6.1 — Validation Signal Calibration (optional)

```bash
python -m src.labels.calibrate --input data/processed/AAPL_form4_features.csv
```

**What it does:**

- Sweeps outcome-signal thresholds (`abnormal_threshold`, confidence-source settings)
- Reports how outcome-driven label proportions would change under those settings
- Useful for validation diagnostics; behavioral labels in Step 6 are still driven by Cohen + plan override

**What it updates:**
| Location | Detail |
|---|---|
| `reports/*_label_calibration_grid.csv` | Full threshold/confidence sweep table |
| `reports/*_label_calibration_summary.json` | Machine-readable recommended settings |
| `reports/*_label_calibration_summary.md` | Human-readable calibration summary |

### Step 6.2 — Build External Confirmation Data (earnings + SEC enforcement)

```bash
python -m src.data.external_events --tickers AAPL
```

For larger ticker batches or when Yahoo Finance rate-limits, use SEC-based earnings source:

```bash
python -m src.data.external_events \
  --tickers AAPL,ABBV,AMGN,BA,BIIB,BMY,COP,LMT,MRK,REGN,XOM \
  --earnings-source sec \
  --skip-enforcement
```

**What it does:**

- Fetches historical earnings announcement dates (`--earnings-source auto|sec|yfinance`)
  - `auto` (default): SEC filings fallback + Yahoo where needed
  - `sec`: SEC submission filing dates for 10-Q / 10-K forms
  - `yfinance`: Yahoo Finance earnings calendar only
- Fetches SEC litigation RSS releases and maps rows to your tickers (when ticker is mentioned)
- Applies enforcement matching using ticker + company-name aliases
- Appends + deduplicates records in external CSVs

**What it updates:**
| Location | Detail |
|---|---|
| `data/external/earnings/earnings_announcements.csv` | `ticker,announcement_date` rows |
| `data/external/sec/sec_enforcement.csv` | `ticker,action_date,source_title,source_link,matched_by` rows |
| `logs/external_events.log` | Ingestion execution log |

Optional: add custom aliases in `data/external/sec/company_aliases.csv` with columns `ticker,alias`.

Useful options:

```bash
# earnings only
python -m src.data.external_events --tickers AAPL,MSFT --skip-enforcement --earnings-source sec

# enforcement only
python -m src.data.external_events --tickers AAPL,MSFT --skip-earnings

# include yfinance alias enrichment for enforcement matching (can be slower / rate-limited)
python -m src.data.external_events --tickers AAPL,MSFT --include-yf-aliases
```

---

## 🤖 Model Training & Evaluation (Current)

Model scripts expect time-based split files in `data/processed/train.csv`, `data/processed/val.csv`, and `data/processed/test.csv`.

### 1) Build training splits from labeled datasets

```bash
python scripts/create_splits.py
```

This reads `data/processed/*_form4_features_labeled.csv` files and writes:

- `data/processed/master_labeled.csv`
- `data/processed/train.csv` (<= 2022)
- `data/processed/val.csv` (2023)
- `data/processed/test.csv` (2024)

### 2) Train Naive Bayes baseline

```bash
python -m src.models.naive_bayes_model
```

Outputs:

- `models/naive_bayes_model.joblib`
- `models/naive_bayes_scaler.joblib`
- `reports/model_results_naive_bayes.json`
- `reports/nb_pr_curve.png`
- `reports/nb_feature_separation.png`

### 3) Train Isolation Forest anomaly model

```bash
python -m src.models.isolation_forest_model
```

Outputs:

- `models/isolation_forest_model.joblib`
- `models/isolation_forest_scaler.joblib`
- `reports/model_results_isolation_forest.json`
- `reports/iforest_score_distribution.png`
- `reports/iforest_feature_importance.png`

### 4) Train XGBoost + SHAP explainability

```bash
python -m src.models.xgb_model
```

Outputs:

- `models/xgb_model.joblib`
- `models/xgb_scaler.joblib`
- `reports/model_results_xgb.json`
- `reports/shap_xgb.png`
- `reports/shap_xgb_beeswarm.png`

### Notes

- All three model scripts automatically filter to open-market transactions (`P`, `S`) and supervised labels (`final_label` in `{0,1}`) where required.
- Feature leakage checks run before training; training aborts if leakage columns are accidentally included.
- Run with Python 3.11/3.12 to avoid package-compatibility issues.

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
- [x] Phase 2: Feature Engineering
- [x] Phase 3: Labeling & Ground Truth
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
