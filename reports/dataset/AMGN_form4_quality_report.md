# Data Quality Report

**Generated:** 2026-03-18T23:02:08.785400

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 643
- **Total Columns:** 32
- **Memory Usage:** 1.25 MB
- **Date Range:** 2019-11-01 to 2025-11-20
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 643 | 0.0% | 469 |
| filing_date | datetime64[ns] | 643 | 0.0% | 188 |
| issuer_cik | object | 643 | 0.0% | 4 |
| issuer_name | object | 643 | 0.0% | 4 |
| ticker | object | 643 | 0.0% | 4 |
| insider_name | object | 643 | 0.0% | 32 |
| insider_cik | object | 643 | 0.0% | 32 |
| insider_role | object | 643 | 0.0% | 6 |
| is_director | bool | 643 | 0.0% | 2 |
| is_officer | bool | 643 | 0.0% | 2 |
| is_ten_percent_owner | bool | 643 | 0.0% | 2 |
| officer_title | object | 643 | 25.7% | 15 |
| has_10b5_1_plan | bool | 643 | 0.0% | 2 |
| footnote_text | object | 643 | 0.0% | 433 |
| is_amendment | bool | 643 | 0.0% | 1 |
| xml_path | object | 643 | 0.0% | 469 |
| security_title | object | 643 | 0.0% | 7 |
| transaction_date | datetime64[ns] | 643 | 0.0% | 138 |
| transaction_code | object | 643 | 0.0% | 7 |
| shares | float64 | 643 | 0.0% | 458 |

*... and 12 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 170428.36 | 2192118.14 | 13.53 | 1000.00 | 33603948.00 | 14.5% |
| price_per_share | 132.98 | 128.04 | 0.00 | 198.76 | 337.26 | 0.0% |
| total_value | 2959372.32 | 21179599.99 | 3541.50 | 228905.00 | 421442751.23 | 15.5% |
| shares_owned_after | 857923.18 | 9789795.47 | 0.00 | 18156.00 | 236249845.00 | 9.5% |
| conversion_price | 197.36 | 81.57 | 54.69 | 235.97 | 300.30 | 0.0% |
| underlying_shares | 487531.87 | 3695394.86 | 1000.00 | 18603.00 | 32754291.00 | 11.4% |
| close_price_on_txn_date | 252.54 | 37.04 | 11.98 | 239.74 | 338.45 | 0.6% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| A | 276 |
| F | 252 |
| M | 52 |
| S | 47 |
| G | 9 |
| C | 5 |
| P | 2 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 399 |
| Director | 147 |
| CEO | 52 |
| CFO | 27 |
| 10%_Owner | 14 |
| Other | 4 |

---

## Warnings

- High missing values (>50%) in: conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: is_amendment

---

## Recommendations

- Implement imputation for: total_value, conversion_price, exercise_date, expiration_date, underlying_shares