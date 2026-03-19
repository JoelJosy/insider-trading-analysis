# Data Quality Report

**Generated:** 2026-03-18T21:56:26.508668

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 2,454
- **Total Columns:** 31
- **Memory Usage:** 8.24 MB
- **Date Range:** 2021-01-24 to 2025-12-11
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 2,454 | 0.0% | 479 |
| filing_date | datetime64[ns] | 2,454 | 0.0% | 270 |
| issuer_cik | object | 2,454 | 0.0% | 1 |
| issuer_name | object | 2,454 | 0.0% | 1 |
| ticker | object | 2,454 | 0.0% | 1 |
| insider_name | object | 2,454 | 0.0% | 18 |
| insider_cik | object | 2,454 | 0.0% | 18 |
| insider_role | object | 2,454 | 0.0% | 5 |
| is_director | bool | 2,454 | 0.0% | 2 |
| is_officer | bool | 2,454 | 0.0% | 2 |
| is_ten_percent_owner | bool | 2,454 | 0.0% | 1 |
| officer_title | object | 2,454 | 12.0% | 7 |
| has_10b5_1_plan | bool | 2,454 | 0.0% | 2 |
| footnote_text | object | 2,454 | 0.0% | 231 |
| is_amendment | bool | 2,454 | 0.0% | 1 |
| xml_path | object | 2,454 | 0.0% | 479 |
| security_title | object | 2,454 | 0.0% | 5 |
| transaction_date | datetime64[ns] | 2,454 | 0.0% | 424 |
| transaction_code | object | 2,454 | 0.0% | 7 |
| shares | float64 | 2,454 | 0.0% | 929 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 10389.96 | 25684.29 | 26.00 | 5000.00 | 688073.00 | 12.4% |
| price_per_share | 102.82 | 104.50 | 0.00 | 120.96 | 461.85 | 2.4% |
| total_value | 796548.30 | 1326994.29 | 2531.16 | 213144.00 | 21884345.28 | 5.2% |
| shares_owned_after | 2673120.54 | 2460110.33 | 0.00 | 1630891.00 | 11467664.00 | 0.0% |
| conversion_price | 19.64 | 38.01 | 0.00 | 10.90 | 149.52 | 11.7% |
| underlying_shares | 16766.21 | 39582.83 | 178.00 | 5000.00 | 688073.00 | 2.1% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| S | 1,314 |
| M | 940 |
| A | 101 |
| F | 47 |
| G | 46 |
| J | 3 |
| P | 3 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| CEO | 1,053 |
| Officer | 1,007 |
| Director | 171 |
| Other | 123 |
| CFO | 100 |

---

## Warnings

- High missing values (>50%) in: conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: issuer_cik, issuer_name, ticker, is_ten_percent_owner, is_amendment

---

## Recommendations

- Implement imputation for: acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares