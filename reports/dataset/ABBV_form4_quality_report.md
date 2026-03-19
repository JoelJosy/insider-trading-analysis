# Data Quality Report

**Generated:** 2026-03-18T21:26:03.457541

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 586
- **Total Columns:** 31
- **Memory Usage:** 1.42 MB
- **Date Range:** 2017-09-30 to 2025-09-30
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 586 | 0.0% | 266 |
| filing_date | datetime64[ns] | 586 | 0.0% | 106 |
| issuer_cik | object | 586 | 0.0% | 1 |
| issuer_name | object | 586 | 0.0% | 1 |
| ticker | object | 586 | 0.0% | 1 |
| insider_name | object | 586 | 0.0% | 32 |
| insider_cik | object | 586 | 0.0% | 32 |
| insider_role | object | 586 | 0.0% | 5 |
| is_director | bool | 586 | 0.0% | 2 |
| is_officer | bool | 586 | 0.0% | 2 |
| is_ten_percent_owner | bool | 586 | 0.0% | 1 |
| officer_title | object | 586 | 16.9% | 33 |
| has_10b5_1_plan | bool | 586 | 0.0% | 2 |
| footnote_text | object | 586 | 7.8% | 126 |
| is_amendment | bool | 586 | 0.0% | 1 |
| xml_path | object | 586 | 0.0% | 266 |
| security_title | object | 586 | 0.0% | 9 |
| transaction_date | datetime64[ns] | 586 | 0.0% | 137 |
| transaction_code | object | 586 | 0.0% | 8 |
| shares | float64 | 586 | 0.0% | 422 |

*... and 11 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 21155.47 | 31930.12 | 33.00 | 10427.00 | 229132.00 | 8.7% |
| price_per_share | 53.62 | 63.66 | 0.00 | 0.00 | 231.54 | 0.0% |
| total_value | 2719430.77 | 3503419.98 | 6479.57 | 1468099.54 | 21423842.00 | 4.7% |
| shares_owned_after | 94539.78 | 107849.88 | 0.00 | 55570.00 | 625294.00 | 6.5% |
| conversion_price | 75.70 | 59.21 | 0.00 | 79.02 | 231.54 | 0.0% |
| underlying_shares | 32510.51 | 44345.33 | 33.00 | 16205.00 | 229132.00 | 4.6% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| A | 321 |
| S | 99 |
| F | 64 |
| M | 64 |
| G | 24 |
| P | 12 |
| D | 1 |
| W | 1 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 339 |
| Other | 111 |
| Director | 68 |
| CEO | 41 |
| CFO | 27 |

---

## Warnings

- High missing values (>50%) in: conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: issuer_cik, issuer_name, ticker, is_ten_percent_owner, is_amendment

---

## Recommendations

- Implement imputation for: total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic