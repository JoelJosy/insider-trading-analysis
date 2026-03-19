# Data Quality Report

**Generated:** 2026-03-18T21:56:02.297669

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 3,684
- **Total Columns:** 31
- **Memory Usage:** 31.66 MB
- **Date Range:** 2024-03-14 to 2025-12-30
- **Duplicate Rows:** 17

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 3,684 | 0.0% | 469 |
| filing_date | datetime64[ns] | 3,684 | 0.0% | 204 |
| issuer_cik | object | 3,684 | 0.0% | 1 |
| issuer_name | object | 3,684 | 0.0% | 1 |
| ticker | object | 3,684 | 0.0% | 1 |
| insider_name | object | 3,684 | 0.0% | 24 |
| insider_cik | object | 3,684 | 0.0% | 23 |
| insider_role | object | 3,684 | 0.0% | 5 |
| is_director | bool | 3,684 | 0.0% | 2 |
| is_officer | bool | 3,684 | 0.0% | 2 |
| is_ten_percent_owner | bool | 3,684 | 0.0% | 2 |
| officer_title | object | 3,684 | 4.8% | 8 |
| has_10b5_1_plan | bool | 3,684 | 0.0% | 2 |
| footnote_text | object | 3,684 | 0.0% | 257 |
| is_amendment | bool | 3,684 | 0.0% | 1 |
| xml_path | object | 3,684 | 0.0% | 469 |
| security_title | object | 3,684 | 0.0% | 4 |
| transaction_date | datetime64[ns] | 3,684 | 0.0% | 264 |
| transaction_code | object | 3,684 | 0.0% | 6 |
| shares | float64 | 3,684 | 0.0% | 1,708 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 4580.29 | 26375.79 | 9.00 | 1060.00 | 639347.00 | 10.2% |
| price_per_share | 479.15 | 256.63 | 0.00 | 535.16 | 793.31 | 20.7% |
| total_value | 888940.72 | 1151277.43 | 5753.84 | 513604.33 | 17418921.39 | 7.4% |
| shares_owned_after | 4294729.30 | 29073906.46 | 0.00 | 37803.50 | 295129194.00 | 2.3% |
| underlying_shares | 12267.32 | 40153.91 | 102.00 | 4720.00 | 600000.00 | 10.5% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| S | 2,876 |
| M | 390 |
| C | 298 |
| A | 49 |
| F | 46 |
| G | 25 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| CEO | 2,715 |
| Officer | 462 |
| CFO | 184 |
| Director | 176 |
| COO | 147 |

---

## Warnings

- Found 17 duplicate rows (0.5%)
- High missing values (>50%) in: conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: issuer_cik, issuer_name, ticker, is_amendment

---

## Recommendations

- Remove duplicates with df.drop_duplicates()
- Consider winsorizing or log-transforming price_per_share
- Implement imputation for: total_value, conversion_price, exercise_date, expiration_date, underlying_shares
- Transaction types are imbalanced - consider stratified sampling