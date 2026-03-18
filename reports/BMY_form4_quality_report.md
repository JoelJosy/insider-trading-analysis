# Data Quality Report

**Generated:** 2026-03-18T23:01:59.257607

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 1,012
- **Total Columns:** 32
- **Memory Usage:** 2.48 MB
- **Date Range:** 2019-12-04 to 2026-02-01
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 1,012 | 0.0% | 263 |
| filing_date | datetime64[ns] | 1,012 | 0.0% | 87 |
| issuer_cik | object | 1,012 | 0.0% | 1 |
| issuer_name | object | 1,012 | 0.0% | 1 |
| ticker | object | 1,012 | 0.0% | 1 |
| insider_name | object | 1,012 | 0.0% | 45 |
| insider_cik | object | 1,012 | 0.0% | 45 |
| insider_role | object | 1,012 | 0.0% | 4 |
| is_director | bool | 1,012 | 0.0% | 2 |
| is_officer | bool | 1,012 | 0.0% | 2 |
| is_ten_percent_owner | bool | 1,012 | 0.0% | 1 |
| officer_title | object | 1,012 | 14.6% | 28 |
| has_10b5_1_plan | bool | 1,012 | 0.0% | 2 |
| footnote_text | object | 1,012 | 0.3% | 114 |
| is_amendment | bool | 1,012 | 0.0% | 1 |
| xml_path | object | 1,012 | 0.0% | 263 |
| security_title | object | 1,012 | 0.0% | 6 |
| transaction_date | datetime64[ns] | 1,012 | 0.0% | 101 |
| transaction_code | object | 1,012 | 0.0% | 7 |
| shares | float64 | 1,012 | 0.0% | 656 |

*... and 12 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 10711.97 | 22872.26 | 10.00 | 3562.34 | 300000.00 | 15.8% |
| price_per_share | 21.62 | 29.65 | 0.00 | 0.00 | 78.88 | 0.0% |
| total_value | 549023.10 | 1012539.16 | 1397.24 | 225487.68 | 9957964.65 | 14.5% |
| shares_owned_after | 71889.86 | 118192.67 | 0.00 | 21954.00 | 712285.33 | 10.9% |
| conversion_price | 53.07 | 8.65 | 38.41 | 54.25 | 72.42 | 0.0% |
| underlying_shares | 11697.11 | 20856.05 | 18.15 | 4189.00 | 148758.00 | 12.9% |
| close_price_on_txn_date | 61.11 | 6.55 | 41.30 | 63.11 | 79.19 | 0.0% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 418 |
| A | 228 |
| F | 171 |
| J | 155 |
| S | 28 |
| G | 7 |
| P | 5 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 689 |
| Director | 148 |
| CEO | 93 |
| CFO | 82 |

---

## Warnings

- High missing values (>50%) in: conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: issuer_cik, issuer_name, ticker, is_ten_percent_owner, is_amendment

---

## Recommendations

- Implement imputation for: total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic