# Data Quality Report

**Generated:** 2026-02-10T15:07:39.092049

**Source:** data/processed/AAPL_form4.csv

---

## Dataset Overview

- **Total Rows:** 294
- **Total Columns:** 31
- **Memory Usage:** 0.79 MB
- **Date Range:** 2022-09-25 to 2024-12-16
- **Duplicate Rows:** 11

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 294 | 0.0% | 100 |
| filing_date | datetime64[ns] | 294 | 0.0% | 42 |
| issuer_cik | int64 | 294 | 0.0% | 1 |
| issuer_name | object | 294 | 0.0% | 1 |
| ticker | object | 294 | 0.0% | 1 |
| insider_name | object | 294 | 0.0% | 15 |
| insider_cik | int64 | 294 | 0.0% | 15 |
| insider_role | object | 294 | 0.0% | 5 |
| is_director | bool | 294 | 0.0% | 2 |
| is_officer | bool | 294 | 0.0% | 2 |
| is_ten_percent_owner | bool | 294 | 0.0% | 1 |
| officer_title | object | 294 | 18.4% | 6 |
| security_title | object | 294 | 0.0% | 2 |
| transaction_date | datetime64[ns] | 294 | 0.0% | 58 |
| transaction_code | object | 294 | 0.0% | 5 |
| shares | float64 | 294 | 0.0% | 161 |
| price_per_share | float64 | 294 | 42.5% | 91 |
| total_value | float64 | 294 | 62.2% | 107 |
| acquired_disposed | object | 294 | 42.2% | 2 |
| shares_owned_after | float64 | 294 | 0.0% | 196 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| issuer_cik | 320193.00 | 0.00 | 320193.00 | 320193.00 | 320193.00 | 0.0% |
| insider_cik | 1467817.63 | 198158.41 | 1051401.00 | 1496686.00 | 1767094.00 | 0.0% |
| shares | 60747.37 | 88852.59 | 165.00 | 30792.00 | 511000.00 | 7.5% |
| price_per_share | 120.09 | 90.78 | 0.00 | 164.90 | 251.10 | 0.0% |
| total_value | 9037770.20 | 9144856.67 | 27934.50 | 8657719.40 | 57302386.15 | 2.7% |
| shares_owned_after | 478054.81 | 1050076.51 | 0.00 | 73749.50 | 4534576.00 | 9.5% |
| underlying_shares | 56029.24 | 92433.96 | 1516.00 | 29688.00 | 511000.00 | 7.3% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 125 |
| S | 81 |
| A | 46 |
| F | 30 |
| G | 12 |

---

## Insider Role Distribution

| Role | Count |
|------|-------|
| Officer | 133 |
| Director | 54 |
| CFO | 41 |
| COO | 36 |
| CEO | 30 |

---

## Warnings

- Found 11 duplicate rows (3.7%)
- High missing values (>50%) in columns: total_value, conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns (single value): issuer_cik, issuer_name, ticker, is_ten_percent_owner, is_amendment

---

## Recommendations

- Remove duplicate rows before analysis using df.drop_duplicates()
- Implement imputation strategy for: price_per_share, total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares