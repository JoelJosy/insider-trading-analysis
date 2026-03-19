# Data Quality Report

**Generated:** 2026-03-18T23:02:33.329517

**Source:** DataFrame

---

## Dataset Overview

- **Total Rows:** 1,158
- **Total Columns:** 32
- **Memory Usage:** 2.34 MB
- **Date Range:** 2016-02-01 to 2025-12-05
- **Duplicate Rows:** 0

---

## Column Statistics

| Column | Type | Count | Missing % | Unique |
|--------|------|-------|-----------|--------|
| accession_number | object | 1,158 | 0.0% | 500 |
| filing_date | datetime64[ns] | 1,158 | 0.0% | 152 |
| issuer_cik | object | 1,158 | 0.0% | 2 |
| issuer_name | object | 1,158 | 0.0% | 2 |
| ticker | object | 1,158 | 0.0% | 2 |
| insider_name | object | 1,158 | 0.0% | 46 |
| insider_cik | object | 1,158 | 0.0% | 46 |
| insider_role | object | 1,158 | 0.0% | 6 |
| is_director | bool | 1,158 | 0.0% | 2 |
| is_officer | bool | 1,158 | 0.0% | 2 |
| is_ten_percent_owner | bool | 1,158 | 0.0% | 2 |
| officer_title | object | 1,158 | 21.4% | 20 |
| has_10b5_1_plan | bool | 1,158 | 0.0% | 1 |
| footnote_text | object | 1,158 | 0.5% | 283 |
| is_amendment | bool | 1,158 | 0.0% | 1 |
| xml_path | object | 1,158 | 0.0% | 500 |
| security_title | object | 1,158 | 0.0% | 6 |
| transaction_date | datetime64[ns] | 1,158 | 0.0% | 180 |
| transaction_code | object | 1,158 | 0.0% | 9 |
| shares | float64 | 1,158 | 0.0% | 616 |

*... and 12 more columns*

---

## Numeric Column Distributions

| Column | Mean | Std | Min | Median | Max | Outliers % |
|--------|------|-----|-----|--------|-----|------------|
| shares | 104190.16 | 2981528.09 | 0.75 | 501.50 | 100000000.00 | 11.1% |
| price_per_share | 182.09 | 199.43 | 0.00 | 0.00 | 564.21 | 0.0% |
| total_value | 1404050.24 | 10582122.33 | 282.69 | 227344.97 | 289800000.00 | 9.1% |
| shares_owned_after | 25230.68 | 507300.40 | 0.00 | 3378.00 | 17253279.00 | 10.2% |
| conversion_price | 72.86 | 32.57 | 2.90 | 82.01 | 106.87 | 38.5% |
| underlying_shares | 103684.51 | 1708547.03 | 0.75 | 417.60 | 34506556.00 | 7.5% |
| close_price_on_txn_date | 365.56 | 91.27 | 173.21 | 362.24 | 584.56 | 0.0% |

---

## Transaction Type Distribution

| Type | Count |
|------|-------|
| M | 420 |
| A | 367 |
| F | 179 |
| S | 124 |
| P | 19 |
| G | 17 |
| I | 17 |
| D | 14 |
| W | 1 |

---

## Insider Role Distribution

| Type | Count |
|------|-------|
| Officer | 669 |
| Director | 245 |
| COO | 89 |
| CEO | 88 |
| CFO | 64 |
| 10%_Owner | 3 |

---

## Warnings

- High missing values (>50%) in: conversion_price, exercise_date, expiration_date, underlying_shares
- Constant columns: has_10b5_1_plan, is_amendment

---

## Recommendations

- Implement imputation for: price_per_share, total_value, acquired_disposed, direct_indirect, conversion_price, exercise_date, expiration_date, underlying_shares
- Low 10b5-1 plan coverage - review footnote parsing logic