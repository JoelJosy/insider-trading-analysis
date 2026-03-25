# Insider Trading Analysis

Machine learning pipeline for SEC Form 4 insider-trading analysis, from raw filings to labeled datasets and multi-model evaluation.

## Project Status

Implemented and used in this repo:

- SEC Form 4 extraction and parsing
- Feature engineering pipeline (trade, insider, temporal, network, text)
- Phase 3 behavioral labeling (Cohen pattern + 10b5-1 override)
- External confirmation data ingestion (earnings + SEC enforcement)
- Time-based train/val/test split generation
- Multiple model tracks:
  - XGBoost + SHAP
  - Naive Bayes
  - Isolation Forest
  - K-means
  - Fuzzy scoring
  - Apriori association rules
- Validation and analysis utilities (cross-validation, ablation, post-hoc checks)

## Repository Layout

```
insider-trading-analysis/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/                        # SEC filing downloads
│   ├── external/
│   │   ├── earnings/
│   │   ├── prices/
│   │   └── sec/
│   └── processed/                  # Parsed, feature, labeled, and split CSVs
├── logs/
├── models/                         # Saved model/scaler artifacts (.joblib)
├── reports/
│   ├── dataset/
│   └── model/
├── scripts/
│   ├── run_pipeline.sh
│   ├── create_splits.py
│   ├── download_prices.py
│   └── test.py
├── sql/
│   └── schema.sql
├── src/
│   ├── data/
│   ├── features/
│   ├── labels/
│   ├── models/
│   └── utils/
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

## Setup

1. Create and activate a Python environment (3.11/3.12 recommended).

2. Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements-lock.txt
```

3. Configure environment variables in .env (for database use):

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=insider_trading_db
DB_USER=postgres
DB_PASSWORD=your_password
```

4. Review config/config.yaml for ticker groups, SEC settings, and labeling parameters.

## End-to-End Pipeline

Run from project root with the environment activated.

### 1) Extract SEC filings

```bash
python -m src.data.extract --ticker AAPL --num-filings 500
```

Useful flags:

- --skip-market-prices: skip price enrichment at extraction time
- --refresh: force SEC re-download

Outputs:

- data/raw/sec-edgar-filings/<TICKER>/4/... (raw filing files)
- data/processed/<TICKER>\_form4.csv
- logs/extract.log, logs/parser.log, logs/downloader.log

### 2) Data quality report (optional)

```bash
python -m src.data.quality --csv data/processed/AAPL_form4.csv
```

Outputs:

- reports/AAPL_form4_quality_report.json
- reports/AAPL_form4_quality_report.md
- logs/quality.log

### 3) Feature engineering

```bash
python -m src.features.pipeline --input data/processed/AAPL_form4.csv
```

Useful flags:

- --skip-preflight
- --full
- --output <custom_path>

Output:

- data/processed/AAPL_form4_features.csv

### 4) External events (earnings + enforcement)

```bash
python -m src.data.external_events --tickers AAPL
```

Useful options:

- --earnings-source auto|sec|yfinance
- --skip-earnings
- --skip-enforcement
- --include-yf-aliases

Outputs:

- data/external/earnings/earnings_announcements.csv
- data/external/sec/sec_enforcement.csv
- logs/external_events.log

### 5) Labeling (Phase 3)

```bash
python -m src.labels.pipeline --input data/processed/AAPL_form4_features.csv
```

Useful options:

- --output data/processed/AAPL_form4_features_labeled.csv
- --benchmark SPY
- --horizon-days 30
- --focus-ticker AAPL
- --allow-missing-prices (debug only)

Outputs:

- data/processed/\*\_labeled.csv
- reports/\*\_label_quality_report.json
- reports/\*\_label_quality_report.md
- logs/labels.log

### 6) Build modeling splits

```bash
python scripts/create_splits.py
```

This reads data/processed/\*\_form4_features_labeled.csv and writes:

- data/processed/master_labeled.csv
- data/processed/train.csv
- data/processed/val.csv
- data/processed/test.csv

## Batch Run Script

For multi-ticker pipeline runs, use:

```bash
bash scripts/run_pipeline.sh
```

Current script flow per ticker:

1. src.data.extract (with --skip-market-prices)
2. src.features.pipeline (with --skip-preflight)
3. src.data.external_events (earnings only in current script config)
4. src.labels.pipeline

Note: because extraction is currently called with --skip-market-prices in this script, ensure price cache files exist in data/external/prices for labeling.

## Price Cache Utility

Populate or refresh cached daily prices:

```bash
python scripts/download_prices.py --tickers SPY AAPL
```

Defaults to Stooq provider and writes:

- data/external/prices/<TICKER>\_daily_prices.csv

## Model Training

All model modules assume split files exist:

- data/processed/train.csv
- data/processed/val.csv
- data/processed/test.csv

### Core supervised/anomaly models

```bash
python -m src.models.naive_bayes_model
python -m src.models.isolation_forest_model
python -m src.models.xgb_model
python -m src.models.kmeans_model
python -m src.models.fuzzy_model
python -m src.models.apriori_model
```

### Validation and analysis modules

```bash
python -m src.models.cross_validate
python -m src.models.ablation
python -m src.models.posthoc_validation
```

Additional analysis scripts in src/models:

- ensemble_analysis.py
- sector_analysis.py
- posthoc_by_sector.py

## Typical Output Artifacts

Model artifacts in models/:

- xgb_model.joblib, xgb_scaler.joblib
- naive_bayes_model.joblib, naive_bayes_scaler.joblib
- isolation_forest_model.joblib, isolation_forest_scaler.joblib
- kmeans_model.joblib, kmeans_scaler.joblib
- fuzzy_model.joblib
- apriori_model.joblib

Report artifacts in reports/ (examples):

- model_results_xgb.json
- model_results_naive_bayes.json
- model_results_isolation_forest.json
- model_results_kmeans.json
- model_results_fuzzy.json
- model_results_apriori.json
- shap_xgb.png, shap_xgb_beeswarm.png
- nb_pr_curve.png, nb_feature_separation.png
- iforest_score_distribution.png, iforest_feature_importance.png
- fuzzy_score_distribution.png
- apriori_rules.csv
- cross_validation_results.json
- ablation_results.json, ablation_auc_pr_drop.png
- posthoc_validation.json, posthoc_boxplot.png, posthoc_histogram.png

## Optional Database Flow

Database is optional for feature/label/model workflows, but supported.

Initialize schema:

```bash
python -m src.data.load --init-schema
```

Load parsed CSV:

```bash
python -m src.data.load --load-csv data/processed/AAPL_form4.csv
```

Maintenance:

```bash
python -m src.data.load --update-stats
python -m src.data.load --refresh-views
```

## Notes

- Most modeling code filters to open-market transactions (P/S) and final_label in {0,1} for supervised evaluation.
- Leakage-related columns are explicitly excluded in model feature definitions.
- If market data is unavailable, labeling will fail fast unless --allow-missing-prices is set.

## License

Academic/research use.
