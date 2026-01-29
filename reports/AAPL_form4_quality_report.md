# Data Quality Report

**Generated:** 2026-01-29T22:46:09.071893

**Source:** data/processed/AAPL_form4.csv

---

## Dataset Overview

- **Total Rows:** 600
- **Total Columns:** 31
- **Memory Usage:** 1.56 MB
- **Date Range:** 2020-02-26 to 2024-12-16
- **Duplicate Rows:** 22

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | str | 600 | 0.0% | 200 |
| filing_date | datetime64[us] | 600 | 0.0% | 83 |
| issuer_cik | int64 | 600 | 0.0% | 1 |
| issuer_name | str | 600 | 0.0% | 1 |
| ticker | str | 600 | 0.0% | 1 |
| insider_name | str | 600 | 0.0% | 15 |
| insider_cik | int64 | 600 | 0.0% | 15 |
| insider_role | str | 600 | 0.0% | 5 |
| is_director | bool | 600 | 0.0% | 2 |
| is_officer | bool | 600 | 0.0% | 2 |
| is_ten_percent_owner | bool | 600 | 0.0% | 1 |
| officer_title | str | 600 | 18.5% | 6 |
| security_title | str | 600 | 0.0% | 3 |
| transaction_date | datetime64[us] | 600 | 0.0% | 117 |
| transaction_code | str | 600 | 0.0% | 5 |
| shares | float64 | 600 | 0.0% | 336 |
| price_per_share | float64 | 600 | 41.8% | 199 |
| total_value | float64 | 600 | 60.5% | 230 |
| acquired_disposed | str | 600 | 41.8% | 2 |
| shares_owned_after | float64 | 600 | 0.0% | 405 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| issuer_cik | 320193.00 | 0.00 | 320193.00 | 320193.00 | 320193.00 | 0.0% |
| insider_cik | 1471541.67 | 200737.24 | 1051401.00 | 1496686.00 | 1767094.00 | 0.0% |
| shares | 84220.99 | 329502.73 | 165.00 | 18115.00 | 5040000.00 | 10.3% |
| price_per_share | 128.85 | 108.27 | 0.00 | 146.75 | 503.43 | 2.6% |
| total_value | 11045486.26 | 33281333.01 | 27934.50 | 3004144.00 | 397025647.20 | 5.1% |
| shares_owned_after | 411883.52 | 975485.07 | 0.00 | 89064.00 | 8319726.00 | 8.8% |
| conversion_price | 48.95 | nan | 48.95 | 48.95 | 48.95 | 0.0% |
| underlying_shares | 78515.51 | 330697.79 | 281.00 | 22292.00 | 5040000.00 | 8.8% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 253 |
| S | 175 |
| A | 92 |
| F | 61 |
| G | 19 |

---

## Insider Role Distribution

| Role | Count |
|------|-------|
| Officer | 278 |
| Director | 111 |
| CFO | 83 |
| COO | 75 |
| CEO | 53 |

---

## Warnings

- Found 22 duplicate rows (3.7%)
- High missing values (>50%) in columns: total_value, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns (single value): issuer_cik, issuer_name, ticker, is_ten_percent_owner, is_amendment, conversion_price

---

## Recommendations

- Remove duplicate rows before analysis using df.drop_duplicates()
- Implement imputation strategy for: price_per_share, total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares