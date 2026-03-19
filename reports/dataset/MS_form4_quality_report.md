# Data Quality Report

**Generated:** 2026-03-18T23:02:40.554199

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 977
- **Total Columns:** 32
- **Memory Usage:** 1.60 MB
- **Date Range:** 2013-10-28 to 2025-12-01
- **Duplicate Rows:** 6

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 977 | 0.0% | 498 |
| filing_date | datetime64[ns] | 977 | 0.0% | 178 |
| issuer_cik | object | 977 | 0.0% | 11 |
| issuer_name | object | 977 | 0.0% | 11 |
| ticker | object | 977 | 0.0% | 11 |
| insider_name | object | 977 | 0.0% | 42 |
| insider_cik | object | 977 | 0.0% | 40 |
| insider_role | object | 977 | 0.0% | 7 |
| is_director | bool | 977 | 0.0% | 2 |
| is_officer | bool | 977 | 0.0% | 2 |
| is_ten_percent_owner | bool | 977 | 0.0% | 2 |
| officer_title | object | 977 | 44.5% | 21 |
| has_10b5_1_plan | bool | 977 | 0.0% | 2 |
| footnote_text | object | 977 | 6.1% | 196 |
| is_amendment | bool | 977 | 0.0% | 1 |
| xml_path | object | 977 | 0.0% | 498 |
| security_title | object | 977 | 0.0% | 22 |
| transaction_date | datetime64[ns] | 977 | 0.0% | 334 |
| transaction_code | object | 977 | 0.0% | 13 |
| shares | float64 | 977 | 0.0% | 700 |

*... and 12 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 167857.00 | 1152191.57 | 0.00 | 14520.00 | 16000000.00 | 12.6% |
| price_per_share | 167.69 | 2092.66 | 0.00 | 38.55 | 50000.00 | 0.6% |
| total_value | 6456645.74 | 30342111.17 | 0.04 | 825426.88 | 326921135.86 | 8.4% |
| shares_owned_after | 48330435.48 | 132304871.31 | 0.00 | 199274.08 | 432019499.05 | 12.7% |
| conversion_price | 32.61 | 14.97 | 6.25 | 30.01 | 53.17 | 0.0% |
| underlying_shares | 1054256.78 | 3663801.15 | 100.00 | 130000.00 | 16000000.00 | 14.8% |
| close_price_on_txn_date | 67.80 | 34.82 | 3.00 | 54.48 | 365.40 | 0.6% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| A | 346 |
| S | 228 |
| F | 155 |
| P | 79 |
| G | 74 |
| M | 34 |
| J | 31 |
| D | 21 |
| L | 3 |
| X | 2 |
| E | 2 |
| C | 1 |
| I | 1 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 335 |
| Director | 239 |
| 10%_Owner | 185 |
| CEO | 101 |
| CFO | 83 |
| COO | 23 |
| Other | 11 |

---

## Warnings

- Found 6 duplicate rows (0.6%)
- High missing values (>50%) in: conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: is_amendment
- Found 4 suspiciously high prices (>$10,000/share)

---

## Recommendations

- Remove duplicates with df.drop_duplicates()
- Implement imputation for: total_value, conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic