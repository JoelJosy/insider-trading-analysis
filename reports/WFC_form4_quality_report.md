# Data Quality Report

**Generated:** 2026-03-18T23:02:49.361446

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 1,246
- **Total Columns:** 32
- **Memory Usage:** 2.83 MB
- **Date Range:** 2020-06-22 to 2025-12-05
- **Duplicate Rows:** 13

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 1,246 | 0.0% | 495 |
| filing_date | datetime64[ns] | 1,246 | 0.0% | 120 |
| issuer_cik | object | 1,246 | 0.0% | 23 |
| issuer_name | object | 1,246 | 0.0% | 25 |
| ticker | object | 1,246 | 0.0% | 23 |
| insider_name | object | 1,246 | 0.0% | 42 |
| insider_cik | object | 1,246 | 0.0% | 42 |
| insider_role | object | 1,246 | 0.0% | 7 |
| is_director | bool | 1,246 | 0.0% | 2 |
| is_officer | bool | 1,246 | 0.0% | 2 |
| is_ten_percent_owner | bool | 1,246 | 0.0% | 2 |
| officer_title | object | 1,246 | 18.0% | 12 |
| has_10b5_1_plan | bool | 1,246 | 0.0% | 1 |
| footnote_text | object | 1,246 | 0.7% | 271 |
| is_amendment | bool | 1,246 | 0.0% | 1 |
| xml_path | object | 1,246 | 0.0% | 495 |
| security_title | object | 1,246 | 0.0% | 30 |
| transaction_date | datetime64[ns] | 1,246 | 0.0% | 138 |
| transaction_code | object | 1,246 | 0.0% | 8 |
| shares | float64 | 1,246 | 0.0% | 832 |

*... and 12 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 24222.38 | 49117.18 | 1.00 | 9848.74 | 1046000.00 | 10.7% |
| price_per_share | 21.72 | 53.66 | 0.00 | 0.00 | 1276.63 | 0.2% |
| total_value | 1187834.67 | 3348971.39 | 75.00 | 482083.60 | 86451900.00 | 10.0% |
| shares_owned_after | 86202.45 | 147697.75 | 0.00 | 38296.90 | 1225951.67 | 8.7% |
| conversion_price | 82.65 | nan | 82.65 | 82.65 | 82.65 | 0.0% |
| underlying_shares | 31723.41 | 64132.50 | 1.71 | 13205.00 | 1046000.00 | 8.7% |
| close_price_on_txn_date | 54.08 | 16.69 | 7.25 | 48.70 | 89.83 | 0.0% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 571 |
| A | 278 |
| F | 277 |
| G | 55 |
| J | 51 |
| P | 8 |
| S | 5 |
| D | 1 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 791 |
| Director | 174 |
| Other | 85 |
| COO | 70 |
| CEO | 66 |
| CFO | 42 |
| 10%_Owner | 18 |

---

## Warnings

- Found 13 duplicate rows (1.0%)
- High missing values (>50%) in: conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: has_10b5_1_plan, is_amendment, conversion_price, expiration_date

---

## Recommendations

- Remove duplicates with df.drop_duplicates()
- Implement imputation for: acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic