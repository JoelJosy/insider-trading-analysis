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
- PostgreSQL 14+ (running on localhost:5432)
- Git

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd insider-trading-analysis
```

---

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

---

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

---

### 4. Configure environment

- Copy `.env` and update with your PostgreSQL credentials
- Review `config/config.yaml` for additional settings

---

### 5. Initialize database

```bash
python -m src.data.load --init-schema
```

---

## 📦 Dependency Management (Team Workflow)

After installing or adding new libraries:

```bash
pip freeze > requirements-lock.txt
```

Commit this file so all collaborators use **identical package versions**.

**Team recommendations**

- Python: 3.11 (preferred) or 3.12
- Avoid Python 3.13 unless all libraries support it

---

## 📥 Download and Parse SEC Filings

### Proof of Concept (Apple only)

```bash
python -m src.data.extract --ticker AAPL --num-filings 100
```

### Load to database

```bash
python -m src.data.load --load-csv data/processed/AAPL_form4.csv
```

### Generate quality report

```bash
python -m src.data.quality --csv data/processed/AAPL_form4.csv --report
```

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
